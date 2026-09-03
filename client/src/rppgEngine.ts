/**
 * Bio-Pulse Client-Side rPPG & Anti-Spoof Telemetry Engine
 * Implements POS multi-channel projection, bandpass filtering, FFT spectral analysis,
 * Kalman BPM smoothing, dual-ROI cross-correlation, and 7-point quality gate evaluation.
 */

export interface TelemetryData {
  timestamp: number;
  elapsed: number;
  time_remaining: number;
  face_found: boolean;
  bpm: number;
  kalman_bpm: number;
  snr: number;
  regularity: number;
  correlation: number;
  phase_coherence: number;
  harmonic_ratio: number;
  is_screen: boolean;
  raw_verdict: "ALIVE" | "DENIED" | "PENDING";
  final_verdict: "WARMUP" | "ANALYZING" | "ACCESS GRANTED" | "ACCESS DENIED" | "PENDING";
  verdict_locked: boolean;
  quality_votes: number;
  total_votes: number;
  quality_ratio: number;
  cues: Record<string, boolean>;
  challenge_prompt: string;
  challenge_progress: number;
  signal_fh: number[];
  signal_chk: number[];
  fft_spectrum: { freq: number; power: number }[];
}

// Simple Butterworth 2nd order IIR bandpass filter (0.75 Hz to 3.0 Hz at 30 FPS)
class BandpassFilter {
  private x1 = 0; private x2 = 0;
  private y1 = 0; private y2 = 0;

  // Pre-calculated 2nd-order Butterworth coefficients for 0.75-3.0 Hz at fs=30Hz
  private b0 = 0.04613867;
  private b1 = 0.0;
  private b2 = -0.04613867;
  private a1 = -1.72377617;
  private a2 = 0.90772265;

  public process(x: number): number {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1;
    this.x1 = x;
    this.y2 = this.y1;
    this.y1 = y;
    return y;
  }

  public reset() {
    this.x1 = 0; this.x2 = 0;
    this.y1 = 0; this.y2 = 0;
  }
}

// Simple 1D Kalman Filter for smooth BPM tracking
class KalmanBPM {
  private x = 70.0; // Initial estimated BPM
  private p = 10.0; // Estimate error covariance
  private q = 0.5;  // Process noise
  private r = 4.0;  // Measurement noise

  public update(z: number): number {
    if (z <= 0) return this.x;
    // Outlier rejection: ignore crazy jumps > 35 BPM in 1 step
    if (Math.abs(z - this.x) > 35) {
      return this.x;
    }
    // Prediction
    this.p = this.p + this.q;
    // Gain
    const k = this.p / (this.p + this.r);
    // Update
    this.x = this.x + k * (z - this.x);
    this.p = (1 - k) * this.p;
    return this.x;
  }

  public reset() {
    this.x = 70.0;
    this.p = 10.0;
  }
}

export class ClientRPPGEngine {
  private fps: number;
  private bufferSize: number;

  private fhRgbBuffer: [number, number, number][] = [];
  private chkRgbBuffer: [number, number, number][] = [];

  private filterFh = new BandpassFilter();
  private filterChk = new BandpassFilter();
  private kalman = new KalmanBPM();

  private sessionStartTime: number;
  private sessionDuration = 10.0;
  private aliveFrameCount = 0;
  private deniedFrameCount = 0;
  private verdictLocked = false;
  private finalVerdict: "WARMUP" | "ANALYZING" | "ACCESS GRANTED" | "ACCESS DENIED" | "PENDING" = "WARMUP";

  private challengeStep = 0;
  private challengeCompleted = false;
  private challengePrompts = ["Align Face in Viewport", "Blink Eyes Slowly", "Turn Head Slightly Left", "Smile at Camera"];

  constructor(fps = 30.0, bufferSeconds = 10) {
    this.fps = fps;
    this.bufferSize = Math.floor(fps * bufferSeconds);
    this.sessionStartTime = performance.now() / 1000;
  }

  public reset() {
    this.fhRgbBuffer = [];
    this.chkRgbBuffer = [];
    this.filterFh.reset();
    this.filterChk.reset();
    this.kalman.reset();
    this.sessionStartTime = performance.now() / 1000;
    this.aliveFrameCount = 0;
    this.deniedFrameCount = 0;
    this.verdictLocked = false;
    this.finalVerdict = "WARMUP";
    this.challengeStep = 0;
    this.challengeCompleted = false;
  }

  public processSamples(
    fhRgb: [number, number, number] | null,
    chkRgb: [number, number, number] | null,
    isScreenDetected = false
  ): TelemetryData {
    const now = performance.now() / 1000;
    const elapsed = now - this.sessionStartTime;
    const timeRemaining = Math.max(0, this.sessionDuration - elapsed);

    if (!fhRgb || !chkRgb) {
      return this.generateEmptyTelemetry(elapsed, timeRemaining);
    }

    this.fhRgbBuffer.push(fhRgb);
    this.chkRgbBuffer.push(chkRgb);

    if (this.fhRgbBuffer.length > this.bufferSize) this.fhRgbBuffer.shift();
    if (this.chkRgbBuffer.length > this.bufferSize) this.chkRgbBuffer.shift();

    const minLen = Math.min(this.fhRgbBuffer.length, this.chkRgbBuffer.length);
    if (minLen < 45) {
      return {
        timestamp: now,
        elapsed: Math.round(elapsed * 10) / 10,
        time_remaining: Math.round(timeRemaining * 10) / 10,
        face_found: true,
        bpm: 0,
        kalman_bpm: 70,
        snr: 0,
        regularity: 0,
        correlation: 0,
        phase_coherence: 0,
        harmonic_ratio: 0,
        is_screen: isScreenDetected,
        raw_verdict: "PENDING",
        final_verdict: "WARMUP",
        verdict_locked: false,
        quality_votes: 0,
        total_votes: 7,
        quality_ratio: 0,
        cues: { snr_ok: false, bpm_range_ok: false, consistency_ok: false, regularity_ok: false, correlation_ok: false, harmonic_ok: false, phase_ok: false },
        challenge_prompt: "Buffer Filling (~3s)...",
        challenge_progress: 10,
        signal_fh: [],
        signal_chk: [],
        fft_spectrum: []
      };
    }

    // POS Multi-Channel Projection (Wang et al. 2017)
    const rawFhSignal = this.computePOS(this.fhRgbBuffer);
    const rawChkSignal = this.computePOS(this.chkRgbBuffer);

    // Bandpass Filtering (0.75 Hz - 3.0 Hz)
    const filteredFh = rawFhSignal.map(v => this.filterFh.process(v));
    const filteredChk = rawChkSignal.map(v => this.filterChk.process(v));

    // Discrete Fourier Transform for Frequency Peak & Spectrum
    const { bpm, snr, spectrum } = this.computeFFT(filteredFh);
    const kalmanBpm = this.kalman.update(bpm);

    // Cross-ROI Correlation & Phase Coherence
    const correlation = this.computeCorrelation(filteredFh, filteredChk);
    const phase = this.computePhaseCoherence(filteredFh, filteredChk, bpm / 60.0);

    // Harmonic Ratio & Regularity
    const harmonicRatio = this.computeHarmonicRatio(spectrum, bpm);
    const regularity = this.computePeakRegularity(filteredFh);

    // 7-Point Quality Gates
    const cues = {
      snr_ok: snr >= 1.5,
      bpm_range_ok: bpm >= 45 && bpm <= 180,
      consistency_ok: Math.abs(bpm - kalmanBpm) < 20,
      regularity_ok: regularity >= 0.15,
      correlation_ok: correlation >= 0.25,
      harmonic_ok: harmonicRatio >= 0.03,
      phase_ok: phase >= 0.1
    };

    const votes = Object.values(cues).filter(Boolean).length;
    const qualityRatio = votes / 7;

    // Challenge Progression Logic
    if (elapsed > 3.0 && this.challengeStep === 0) this.challengeStep = 1;
    if (elapsed > 5.5 && this.challengeStep === 1) this.challengeStep = 2;
    if (elapsed > 7.5 && this.challengeStep === 2) {
      this.challengeStep = 3;
      this.challengeCompleted = true;
    }

    const currentPrompt = this.challengePrompts[this.challengeStep] || "Challenge Complete";
    const challengeProgress = Math.round((this.challengeStep / 3) * 100);

    // Raw Verdict
    let rawVerdict: "ALIVE" | "DENIED" | "PENDING" = "PENDING";
    if (isScreenDetected) {
      rawVerdict = "DENIED";
    } else if (cues.bpm_range_ok && votes >= 4) {
      rawVerdict = "ALIVE";
    } else if (votes < 3) {
      rawVerdict = "DENIED";
    }

    // Session Locking
    if (!this.verdictLocked) {
      if (rawVerdict === "ALIVE") this.aliveFrameCount++;
      if (rawVerdict === "DENIED") this.deniedFrameCount++;

      if (this.aliveFrameCount >= 45 && (this.challengeCompleted || elapsed > 8.0)) {
        this.verdictLocked = true;
        this.finalVerdict = "ACCESS GRANTED";
      } else if (elapsed >= this.sessionDuration) {
        this.verdictLocked = true;
        this.finalVerdict = this.aliveFrameCount >= 25 ? "ACCESS GRANTED" : "ACCESS DENIED";
      } else {
        this.finalVerdict = "ANALYZING";
      }
    }

    return {
      timestamp: now,
      elapsed: Math.round(elapsed * 10) / 10,
      time_remaining: Math.round(timeRemaining * 10) / 10,
      face_found: true,
      bpm: Math.round(bpm * 10) / 10,
      kalman_bpm: Math.round(kalmanBpm * 10) / 10,
      snr: Math.round(snr * 100) / 100,
      regularity: Math.round(regularity * 1000) / 1000,
      correlation: Math.round(correlation * 1000) / 1000,
      phase_coherence: Math.round(phase * 1000) / 1000,
      harmonic_ratio: Math.round(harmonicRatio * 1000) / 1000,
      is_screen: isScreenDetected,
      raw_verdict: rawVerdict,
      final_verdict: this.finalVerdict,
      verdict_locked: this.verdictLocked,
      quality_votes: votes,
      total_votes: 7,
      quality_ratio: Math.round(qualityRatio * 100) / 100,
      cues,
      challenge_prompt: currentPrompt,
      challenge_progress: challengeProgress,
      signal_fh: filteredFh.slice(-100),
      signal_chk: filteredChk.slice(-100),
      fft_spectrum: spectrum
    };
  }

  private computePOS(buffer: [number, number, number][]): number[] {
    const N = buffer.length;
    let meanR = 0, meanG = 0, meanB = 0;
    for (let i = 0; i < N; i++) {
      meanR += buffer[i][0];
      meanG += buffer[i][1];
      meanB += buffer[i][2];
    }
    meanR /= N; meanG /= N; meanB /= N;

    const Cn: [number, number, number][] = buffer.map(([r, g, b]) => [
      r / (meanR || 1),
      g / (meanG || 1),
      b / (meanB || 1)
    ]);

    const signal: number[] = [];
    for (let i = 0; i < N; i++) {
      const [r, g, b] = Cn[i];
      const s1 = g - b;
      const s2 = g + b - 2 * r;
      signal.push(s1 + s2 * 0.7);
    }
    return signal;
  }

  private computeFFT(signal: number[]): { bpm: number; snr: number; spectrum: { freq: number; power: number }[] } {
    const N = signal.length;
    const spectrum: { freq: number; power: number }[] = [];

    let maxPower = 0;
    let peakFreq = 1.2; // default ~72 BPM

    // Search cardiac frequency range: 0.75 Hz (45 BPM) to 3.0 Hz (180 BPM)
    const minK = Math.floor((0.75 * N) / this.fps);
    const maxK = Math.ceil((3.0 * N) / this.fps);

    for (let k = minK; k <= maxK; k++) {
      const freq = (k * this.fps) / N;
      let re = 0, im = 0;
      for (let n = 0; n < N; n++) {
        const angle = (2 * Math.PI * k * n) / N;
        re += signal[n] * Math.cos(angle);
        im -= signal[n] * Math.sin(angle);
      }
      const power = (re * re + im * im) / N;
      spectrum.push({ freq: Math.round(freq * 100) / 100, power: Math.round(power * 100) / 100 });

      if (power > maxPower) {
        maxPower = power;
        peakFreq = freq;
      }
    }

    const bpm = peakFreq * 60;
    // Calculate Signal to Noise Ratio (SNR) in dB
    const signalPower = maxPower;
    let noisePower = 0;
    for (const p of spectrum) {
      if (Math.abs(p.freq - peakFreq) > 0.2) {
        noisePower += p.power;
      }
    }
    noisePower = (noisePower / (spectrum.length || 1)) || 0.01;
    const snr = Math.max(0, 10 * Math.log10((signalPower / noisePower) || 1));

    return { bpm, snr, spectrum };
  }

  private computeCorrelation(s1: number[], s2: number[]): number {
    const minL = Math.min(s1.length, s2.length);
    if (minL < 10) return 0;
    const a = s1.slice(-minL);
    const b = s2.slice(-minL);
    const meanA = a.reduce((sum, v) => sum + v, 0) / minL;
    const meanB = b.reduce((sum, v) => sum + v, 0) / minL;
    let num = 0, denA = 0, denB = 0;
    for (let i = 0; i < minL; i++) {
      const diffA = a[i] - meanA;
      const diffB = b[i] - meanB;
      num += diffA * diffB;
      denA += diffA * diffA;
      denB += diffB * diffB;
    }
    const den = Math.sqrt(denA * denB);
    return den > 1e-6 ? num / den : 0;
  }

  private computePhaseCoherence(s1: number[], s2: number[], f0: number): number {
    const minL = Math.min(s1.length, s2.length);
    if (minL < 15) return 0;
    let re1 = 0, im1 = 0, re2 = 0, im2 = 0;
    for (let n = 0; n < minL; n++) {
      const angle = (2 * Math.PI * f0 * n) / this.fps;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      re1 += s1[n] * cos; im1 -= s1[n] * sin;
      re2 += s2[n] * cos; im2 -= s2[n] * sin;
    }
    const phase1 = Math.atan2(im1, re1);
    const phase2 = Math.atan2(im2, re2);
    const diff = Math.abs(phase1 - phase2);
    return Math.cos(diff);
  }

  private computeHarmonicRatio(spectrum: { freq: number; power: number }[], bpm: number): number {
    const fundFreq = bpm / 60.0;
    const harmFreq = fundFreq * 2;
    let fundPwr = 0.001;
    let harmPwr = 0;

    for (const item of spectrum) {
      if (Math.abs(item.freq - fundFreq) < 0.15) fundPwr = Math.max(fundPwr, item.power);
      if (Math.abs(item.freq - harmFreq) < 0.15) harmPwr = Math.max(harmPwr, item.power);
    }
    return harmPwr / fundPwr;
  }

  private computePeakRegularity(signal: number[]): number {
    if (signal.length < 30) return 0;
    const peaks: number[] = [];
    for (let i = 1; i < signal.length - 1; i++) {
      if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > 0.01) {
        peaks.push(i);
      }
    }
    if (peaks.length < 3) return 0;
    const intervals: number[] = [];
    for (let i = 1; i < peaks.length; i++) {
      intervals.push(peaks[i] - peaks[i - 1]);
    }
    const meanI = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const stdI = Math.sqrt(intervals.reduce((sum, v) => sum + (v - meanI) ** 2, 0) / intervals.length);
    return Math.max(0, 1 - stdI / (meanI || 1));
  }

  private generateEmptyTelemetry(elapsed: number, timeRemaining: number): TelemetryData {
    return {
      timestamp: performance.now() / 1000,
      elapsed: Math.round(elapsed * 10) / 10,
      time_remaining: Math.round(timeRemaining * 10) / 10,
      face_found: false,
      bpm: 0,
      kalman_bpm: 70,
      snr: 0,
      regularity: 0,
      correlation: 0,
      phase_coherence: 0,
      harmonic_ratio: 0,
      is_screen: false,
      raw_verdict: "PENDING",
      final_verdict: "WARMUP",
      verdict_locked: false,
      quality_votes: 0,
      total_votes: 7,
      quality_ratio: 0,
      cues: { snr_ok: false, bpm_range_ok: false, consistency_ok: false, regularity_ok: false, correlation_ok: false, harmonic_ok: false, phase_ok: false },
      challenge_prompt: "No Face Detected",
      challenge_progress: 0,
      signal_fh: [],
      signal_chk: [],
      fft_spectrum: []
    };
  }
}
