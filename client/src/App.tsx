import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  Camera,
  Download,
  Eye,
  Heart,
  RefreshCw,
  ShieldCheck,
  ShieldX,
  Volume2,
  VolumeX,
  Wifi,
  WifiOff
} from 'lucide-react';
import { ClientRPPGEngine, TelemetryData } from './rppgEngine';

// Forehead & Cheek ROI Mediapipe indices
const FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288];
const CHEEK_INDICES = [205, 207, 214, 212, 138, 215, 177, 137, 227, 34, 143];

// Web Audio API Synthesizer for Heartbeat & Verdict Chimes
class HeartbeatSynth {
  private ctx: AudioContext | null = null;

  private init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new AudioCtx();
    }
  }

  public playPulseClick(bpm: number) {
    try {
      this.init();
      if (!this.ctx || this.ctx.state !== 'running') return;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(120, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(40, this.ctx.currentTime + 0.08);
      gain.gain.setValueAtTime(0.15, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.08);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.08);
    } catch {
      // Audio context policy fallback
    }
  }

  public playVerdictChime(success: boolean) {
    try {
      this.init();
      if (!this.ctx) return;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = success ? 'triangle' : 'sawtooth';
      osc.frequency.setValueAtTime(success ? 523.25 : 220, this.ctx.currentTime); // C5 or A3
      if (success) {
        osc.frequency.setValueAtTime(659.25, this.ctx.currentTime + 0.15); // E5
      }
      gain.gain.setValueAtTime(0.2, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.4);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.4);
    } catch {
      // Audio context policy fallback
    }
  }
}

const synth = new HeartbeatSynth();

export default function App() {
  const [engineMode, setEngineMode] = useState<'browser' | 'python_ws' | 'video'>('browser');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCam, setSelectedCam] = useState<string>('');
  const [isAudioMuted, setIsAudioMuted] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(30);
  const [wsConnected, setWsConnected] = useState<boolean>(false);

  const [telemetry, setTelemetry] = useState<TelemetryData>({
    timestamp: 0,
    elapsed: 0,
    time_remaining: 10,
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
    challenge_prompt: "Align Face in Viewport",
    challenge_progress: 0,
    signal_fh: [],
    signal_chk: [],
    fft_spectrum: []
  });

  const [auditLogs, setAuditLogs] = useState<{ time: string; msg: string; type: 'info' | 'success' | 'warn' | 'error' }[]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const oscCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fftCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const engineRef = useRef<ClientRPPGEngine>(new ClientRPPGEngine(30, 10));
  const wsRef = useRef<WebSocket | null>(null);
  const faceMeshRef = useRef<unknown>(null);
  const animFrameRef = useRef<number | null>(null);
  const lastPulseTimeRef = useRef<number>(0);
  const previousVerdictRef = useRef<string>('WARMUP');

  // Log helper
  const addLog = (msg: string, type: 'info' | 'success' | 'warn' | 'error' = 'info') => {
    const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setAuditLogs(prev => [{ time, msg, type }, ...prev.slice(0, 49)]);
  };

  // Enumerate video devices
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then(devices => {
      const cams = devices.filter(d => d.kind === 'videoinput');
      setCameras(cams);
      if (cams.length > 0) setSelectedCam(cams[0].deviceId);
    });

    addLog("Bio-Pulse Telemetry Console initialized", "info");
    addLog("POS multi-channel rPPG engine active", "info");
  }, []);

  // Web Audio Pulse Trigger
  useEffect(() => {
    if (isAudioMuted || telemetry.bpm <= 0 || !telemetry.face_found) return;
    const intervalMs = (60 / telemetry.bpm) * 1000;
    const now = performance.now();
    if (now - lastPulseTimeRef.current >= intervalMs) {
      synth.playPulseClick(telemetry.bpm);
      lastPulseTimeRef.current = now;
    }
  }, [telemetry.bpm, telemetry.face_found, isAudioMuted]);

  // Verdict Lock Sound & Log Trigger
  useEffect(() => {
    if (telemetry.final_verdict !== previousVerdictRef.current) {
      if (telemetry.final_verdict === "ACCESS GRANTED") {
        synth.playVerdictChime(true);
        addLog(`VERDICT LOCKED: ACCESS GRANTED (${telemetry.bpm} BPM, SNR ${telemetry.snr}dB)`, "success");
      } else if (telemetry.final_verdict === "ACCESS DENIED") {
        synth.playVerdictChime(false);
        addLog(`VERDICT LOCKED: ACCESS DENIED (${telemetry.quality_votes}/7 Quality Cues Passed)`, "error");
      } else if (telemetry.final_verdict === "ANALYZING") {
        addLog("Pulse buffer ready. Executing active anti-spoof challenge...", "warn");
      }
      previousVerdictRef.current = telemetry.final_verdict;
    }
  }, [telemetry.final_verdict, telemetry.bpm, telemetry.snr, telemetry.quality_votes]);

  // Handle Python WebSocket Connection
  useEffect(() => {
    if (engineMode === 'python_ws') {
      const wsUrl = `ws://${window.location.hostname}:8000/ws/stream`;
      addLog(`Connecting to Python WebSocket backend: ${wsUrl}...`, "info");
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setWsConnected(true);
        addLog("Connected to Python Bio-Pulse Engine WebSocket", "success");
      };

      ws.onmessage = (event) => {
        try {
          const res = JSON.parse(event.data);
          if (res.type === 'telemetry') {
            setTelemetry(res.data);
          }
        } catch {
          // JSON parse err
        }
      };

      ws.onerror = () => {
        setWsConnected(false);
        addLog("Python WebSocket connection failed. Ensure server.py is running on port 8000.", "error");
      };

      ws.onclose = () => {
        setWsConnected(false);
        addLog("Python WebSocket connection closed", "warn");
      };

      wsRef.current = ws;

      return () => {
        ws.close();
      };
    }
  }, [engineMode]);

  // Setup Camera & MediaPipe / Frame processing loop
  useEffect(() => {
    let stream: MediaStream | null = null;
    let frameCount = 0;
    let lastFpsTime = performance.now();

    const startCamera = async () => {
      try {
        const constraints = {
          video: selectedCam ? { deviceId: { exact: selectedCam }, width: 640, height: 480 } : { width: 640, height: 480 }
        };
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      } catch (err) {
        addLog(`Camera Access Error: ${(err as Error).message}`, "error");
      }
    };

    startCamera();

    // Setup MediaPipe Face Mesh JS for In-Browser mode
    if (typeof window !== 'undefined' && (window as unknown as { FaceMesh: unknown }).FaceMesh) {
      const FaceMeshClass = (window as unknown as { FaceMesh: new (config: unknown) => { setOptions: (opts: unknown) => void; onResults: (cb: (results: unknown) => void) => void; send: (input: { image: HTMLVideoElement }) => Promise<void> } }).FaceMesh;
      const faceMesh = new FaceMeshClass({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
      });
      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      faceMesh.onResults((results: unknown) => {
        const res = results as { multiFaceLandmarks?: { x: number; y: number; z: number }[][] };
        processMeshResults(res);
      });

      faceMeshRef.current = faceMesh;
    }

    const processMeshResults = (results: { multiFaceLandmarks?: { x: number; y: number; z: number }[][] }) => {
      const v = videoRef.current;
      const c = videoCanvasRef.current;
      if (!v || !c || v.readyState < 2) return;

      const ctx = c.getContext('2d');
      if (!ctx) return;

      c.width = v.videoWidth || 640;
      c.height = v.videoHeight || 480;

      // Draw Raw Frame
      ctx.drawImage(v, 0, 0, c.width, c.height);

      let fhRgb: [number, number, number] | null = null;
      let chkRgb: [number, number, number] | null = null;
      let faceFound = false;

      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        faceFound = true;
        const lms = results.multiFaceLandmarks[0];

        // Draw Contour Mesh Wireframe
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < lms.length; i += 4) {
          const x = lms[i].x * c.width;
          const y = lms[i].y * c.height;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Extract ROI Points & Draw Bounding Boxes
        const getRoiPolygon = (indices: number[]) => {
          return indices.map(idx => ({
            x: Math.floor(lms[idx].x * c.width),
            y: Math.floor(lms[idx].y * c.height)
          }));
        };

        const getMeanRgb = (pts: { x: number; y: number }[]): [number, number, number] => {
          if (pts.length === 0) return [128, 128, 128];
          const imgData = ctx.getImageData(0, 0, c.width, c.height);
          let sumR = 0, sumG = 0, sumB = 0, count = 0;
          for (let i = 0; i < pts.length; i++) {
            const px = (pts[i].y * c.width + pts[i].x) * 4;
            if (px >= 0 && px < imgData.data.length) {
              sumR += imgData.data[px];
              sumG += imgData.data[px + 1];
              sumB += imgData.data[px + 2];
              count++;
            }
          }
          return [sumR / (count || 1), sumG / (count || 1), sumB / (count || 1)];
        };

        const fhPts = getRoiPolygon(FOREHEAD_INDICES);
        const chkPts = getRoiPolygon(CHEEK_INDICES);

        fhRgb = getMeanRgb(fhPts);
        chkRgb = getMeanRgb(chkPts);

        // Draw Forehead ROI Box (Cyan)
        ctx.strokeStyle = '#06B6D4';
        ctx.lineWidth = 2;
        ctx.strokeRect(fhPts[0]?.x - 20, fhPts[0]?.y - 15, 60, 30);
        ctx.fillStyle = '#06B6D4';
        ctx.font = '10px JetBrains Mono';
        ctx.fillText('ROI: FOREHEAD', fhPts[0]?.x - 20, fhPts[0]?.y - 20);

        // Draw Cheek ROI Box (Emerald)
        ctx.strokeStyle = '#10B981';
        ctx.strokeRect(chkPts[0]?.x - 15, chkPts[0]?.y - 10, 45, 25);
        ctx.fillStyle = '#10B981';
        ctx.fillText('ROI: CHEEK', chkPts[0]?.x - 15, chkPts[0]?.y - 15);
      }

      // If Mode is Python WS, send image frame to server
      if (engineMode === 'python_ws' && wsRef.current?.readyState === WebSocket.OPEN) {
        const frameData = c.toDataURL('image/jpeg', 0.6);
        wsRef.current.send(JSON.stringify({ type: 'frame', data: frameData }));
      } else if (engineMode === 'browser') {
        // Run Client TypeScript rPPG Engine
        const telem = engineRef.current.processSamples(fhRgb, chkRgb, false);
        setTelemetry(telem);
      }
    };

    // Frame processing loop tick
    const loop = async () => {
      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        setFps(Math.round((frameCount * 1000) / (now - lastFpsTime)));
        frameCount = 0;
        lastFpsTime = now;
      }

      if (videoRef.current && videoRef.current.readyState >= 2 && faceMeshRef.current) {
        await (faceMeshRef.current as { send: (input: { image: HTMLVideoElement }) => Promise<void> }).send({ image: videoRef.current });
      }

      animFrameRef.current = requestAnimationFrame(loop);
    };

    loop();

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (stream) stream.getTracks().forEach(t => t.stop());
    };
  }, [selectedCam, engineMode]);

  // Render Dual-Channel Oscilloscope Waveform Canvas
  useEffect(() => {
    const c = oscCanvasRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;

    c.width = c.clientWidth || 400;
    c.height = c.clientHeight || 120;

    ctx.clearRect(0, 0, c.width, c.height);

    // Draw Oscilloscope Grid Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < c.width; x += 30) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c.height); ctx.stroke();
    }
    for (let y = 0; y < c.height; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(c.width, y); ctx.stroke();
    }

    const drawSignal = (sig: number[], color: string) => {
      if (!sig || sig.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const step = c.width / (sig.length - 1);
      const midY = c.height / 2;
      for (let i = 0; i < sig.length; i++) {
        const x = i * step;
        const y = midY - sig[i] * (c.height * 0.4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    drawSignal(telemetry.signal_fh, '#06B6D4');  // Forehead rPPG (Cyan)
    drawSignal(telemetry.signal_chk, '#10B981'); // Cheek rPPG (Emerald)
  }, [telemetry.signal_fh, telemetry.signal_chk]);

  // Render FFT Spectrum Canvas
  useEffect(() => {
    const c = fftCanvasRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;

    c.width = c.clientWidth || 400;
    c.height = c.clientHeight || 100;

    ctx.clearRect(0, 0, c.width, c.height);

    if (!telemetry.fft_spectrum || telemetry.fft_spectrum.length === 0) return;

    const maxPwr = Math.max(...telemetry.fft_spectrum.map(s => s.power), 1);
    const barWidth = c.width / telemetry.fft_spectrum.length;

    telemetry.fft_spectrum.forEach((item, i) => {
      const h = (item.power / maxPwr) * (c.height - 15);
      const x = i * barWidth;
      const y = c.height - h - 15;

      const isCardiacPeak = Math.abs(item.freq - telemetry.bpm / 60) < 0.15;
      ctx.fillStyle = isCardiacPeak ? '#10B981' : '#3B4259';
      ctx.fillRect(x, y, barWidth - 1, h);
    });

    // Draw Frequency Scale Baseline
    ctx.fillStyle = '#64748B';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('0.75Hz (45 BPM)', 5, c.height - 3);
    ctx.fillText('1.5Hz (90 BPM)', c.width / 2 - 25, c.height - 3);
    ctx.fillText('3.0Hz (180 BPM)', c.width - 70, c.height - 3);
  }, [telemetry.fft_spectrum, telemetry.bpm]);

  // Reset Session Handler
  const handleReset = () => {
    engineRef.current.reset();
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset' }));
    }
    setTelemetry({
      timestamp: 0,
      elapsed: 0,
      time_remaining: 10,
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
      challenge_prompt: "Align Face in Viewport",
      challenge_progress: 0,
      signal_fh: [],
      signal_chk: [],
      fft_spectrum: []
    });
    addLog("Session reset by operator", "info");
  };

  // Export JSON Report Handler
  const handleExportAudit = () => {
    const reportData = {
      system: "Bio-Pulse Authenticator v1.0",
      timestamp: new Date().toISOString(),
      final_verdict: telemetry.final_verdict,
      heart_rate_bpm: telemetry.bpm,
      kalman_smoothed_bpm: telemetry.kalman_bpm,
      snr_db: telemetry.snr,
      quality_votes_passed: `${telemetry.quality_votes}/${telemetry.total_votes}`,
      anti_spoof_cues: telemetry.cues,
      audit_logs: auditLogs
    };
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bio-pulse-audit-${Date.now()}.json`;
    a.click();
    addLog("Exported verification audit log JSON", "success");
  };

  return (
    <div className="min-h-screen bg-[#08090E] text-[#E2E8F0] flex flex-col font-sans">
      {/* ──────────────────────────────────────────────
          1. System Header & Mode Controls
      ────────────────────────────────────────────── */}
      <header className="h-16 border-b border-[#1E2433] bg-[#0E111A] px-6 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-[#10B981]/15 border border-[#10B981]/40 rounded text-[#10B981]">
            <Activity className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <h1 className="font-semibold tracking-wide text-white text-base flex items-center gap-2">
              BIO-PULSE AUTHENTICATOR
              <span className="text-[10px] font-mono font-medium px-2 py-0.5 bg-[#1E2433] text-[#94A3B8] rounded border border-[#2E364D]">
                rPPG TELEMETRY v1.0
              </span>
            </h1>
            <p className="text-xs text-[#64748B] font-mono">NON-INVASIVE CARDIAC ANTI-SPOOF ENGINE</p>
          </div>
        </div>

        {/* Right Controls */}
        <div className="flex items-center space-x-3 text-xs font-mono">
          {/* Camera Selector */}
          <div className="flex items-center bg-[#141824] border border-[#1E2433] rounded px-2 py-1">
            <Camera className="w-3.5 h-3.5 text-[#64748B] mr-2" />
            <select
              value={selectedCam}
              onChange={e => setSelectedCam(e.target.value)}
              className="bg-transparent text-white focus:outline-none text-xs"
            >
              {cameras.map(c => (
                <option key={c.deviceId} value={c.deviceId} className="bg-[#141824] text-white">
                  {c.label || `Camera ${c.deviceId.slice(0, 5)}`}
                </option>
              ))}
            </select>
          </div>

          {/* Audio Synthesizer Toggle */}
          <button
            onClick={() => setIsAudioMuted(!isAudioMuted)}
            className="p-2 bg-[#141824] border border-[#1E2433] rounded text-[#94A3B8] hover:text-white transition"
            title={isAudioMuted ? "Unmute Heartbeat Synth" : "Mute Heartbeat Synth"}
          >
            {isAudioMuted ? <VolumeX className="w-4 h-4 text-red-400" /> : <Volume2 className="w-4 h-4 text-[#10B981]" />}
          </button>

          {/* FPS Counter */}
          <div className="px-2.5 py-1 bg-[#141824] border border-[#1E2433] rounded text-[#06B6D4] font-bold">
            {fps} FPS
          </div>

          {/* Reset Button */}
          <button
            onClick={handleReset}
            className="p-2 bg-[#141824] border border-[#1E2433] rounded text-[#94A3B8] hover:text-white transition"
            title="Reset Scan Session"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ──────────────────────────────────────────────
          2. Main Telemetry Dashboard Content
      ────────────────────────────────────────────── */}
      <main className="flex-1 p-5 grid grid-cols-12 gap-5 max-w-[1700px] w-full mx-auto">
        {/* Left Column (7 cols): Biometric Viewport & Live Overlay */}
        <div className="col-span-12 lg:col-span-7 flex flex-col space-y-4">
          {/* Main Video Canvas Viewport */}
          <div className="telemetry-panel p-3 flex-1 flex flex-col justify-between overflow-hidden min-h-[440px]">
            <div className="flex items-center justify-between border-b border-[#1E2433] pb-2 mb-2">
              <div className="flex items-center space-x-2 text-xs font-mono">
                <Eye className="w-4 h-4 text-[#06B6D4]" />
                <span className="text-white font-semibold">LIVE CAMERA FEED & FACIAL ROI REGIONS</span>
              </div>

              {/* Status Badge */}
              <div className={`badge-status ${telemetry.final_verdict === "ACCESS GRANTED" ? "badge-granted" :
                telemetry.final_verdict === "ACCESS DENIED" ? "badge-denied" :
                  telemetry.final_verdict === "ANALYZING" ? "badge-analyzing" : "badge-warmup"
                }`}>
                {telemetry.final_verdict}
              </div>
            </div>

            {/* Video Canvas Container */}
            <div className="relative flex-1 bg-black rounded overflow-hidden flex items-center justify-center border border-[#1E2433]">
              <video ref={videoRef} className="hidden" playsInline muted />
              <canvas ref={videoCanvasRef} className="w-full h-full object-cover" />

              {/* Viewport Overlay: Active Challenge Instruction Banner */}
              <div className="absolute top-4 left-4 right-4 bg-[#0E111A]/85 backdrop-blur border border-[#1E2433] rounded p-3 flex items-center justify-between text-xs">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-full bg-[#06B6D4]/20 border border-[#06B6D4] flex items-center justify-center font-mono font-bold text-[#06B6D4]">
                    {telemetry.challenge_progress > 0 ? `${Math.round(telemetry.challenge_progress / 33)}` : '!'}
                  </div>
                  <div>
                    <div className="text-[10px] text-[#64748B] font-mono tracking-wider">ACTIVE CHALLENGE GATE</div>
                    <div className="text-white font-semibold tracking-wide text-sm">{telemetry.challenge_prompt}</div>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-28 bg-[#1A202C] h-2 rounded-full overflow-hidden border border-[#2D3748]">
                  <div className="bg-[#06B6D4] h-full transition-all duration-300" style={{ width: `${telemetry.challenge_progress}%` }} />
                </div>
              </div>

              {/* Viewport Overlay: Final Verdict Stamp */}
              {telemetry.verdict_locked && (
                <div className="absolute inset-0 bg-black/65 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center animate-fade-in">
                  {telemetry.final_verdict === "ACCESS GRANTED" ? (
                    <>
                      <div className="w-16 h-16 rounded-full bg-[#10B981]/20 border-2 border-[#10B981] flex items-center justify-center text-[#10B981] mb-3">
                        <ShieldCheck className="w-10 h-10" />
                      </div>
                      <h2 className="text-2xl font-bold text-white tracking-widest font-mono">ACCESS GRANTED</h2>
                      <p className="text-xs text-[#10B981] font-mono mt-1">LIVENESS VERIFIED • RPPG PULSE CONFIRMED ({telemetry.bpm} BPM)</p>
                    </>
                  ) : (
                    <>
                      <div className="w-16 h-16 rounded-full bg-[#EF4444]/20 border-2 border-[#EF4444] flex items-center justify-center text-[#EF4444] mb-3">
                        <ShieldX className="w-10 h-10" />
                      </div>
                      <h2 className="text-2xl font-bold text-white tracking-widest font-mono">ACCESS DENIED</h2>
                      <p className="text-xs text-[#EF4444] font-mono mt-1">SPOOF DETECTED OR INSUFFICIENT BIOMETRIC CUES</p>
                    </>
                  )}
                  <button onClick={handleReset} className="mt-4 px-4 py-2 bg-[#1E2433] hover:bg-[#2E364D] text-white font-mono text-xs rounded border border-[#3B4259] transition">
                    START NEW SCAN
                  </button>
                </div>
              )}
            </div>

            {/* Bottom Bar: Session Countdown Timer */}
            <div className="mt-3 flex items-center justify-between text-xs font-mono text-[#94A3B8]">
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-[#10B981] animate-ping" />
                <span>SESSION TIME ELAPSED: {telemetry.elapsed}s</span>
              </div>
              <div>WINDOW REMAINING: {telemetry.time_remaining}s</div>
            </div>
          </div>

          {/* Dual Channel Oscilloscope (rPPG Waveforms) */}
          <div className="telemetry-panel p-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2 text-xs font-mono text-[#06B6D4]">
                <Activity className="w-4 h-4" />
                <span className="font-semibold text-white">DUAL-ROI RPPG OSCILLOSCOPE (POS PROLECTION)</span>
              </div>
              <div className="flex items-center space-x-4 text-[11px] font-mono">
                <span className="flex items-center text-[#06B6D4]"><span className="w-2.5 h-2.5 rounded-full bg-[#06B6D4] mr-1.5" /> FOREHEAD ROI</span>
                <span className="flex items-center text-[#10B981]"><span className="w-2.5 h-2.5 rounded-full bg-[#10B981] mr-1.5" /> CHEEK ROI</span>
              </div>
            </div>
            <div className="h-28 oscilloscope-bg border border-[#1E2433] rounded overflow-hidden relative">
              <canvas ref={oscCanvasRef} className="w-full h-full" />
            </div>
          </div>
        </div>

        {/* Right Column (5 cols): Biometric Telemetry & Quality Matrix */}
        <div className="col-span-12 lg:col-span-5 flex flex-col space-y-4">
          {/* Heart Rate & Signal Quality Gauges */}
          <div className="grid grid-cols-2 gap-3">
            {/* Cardiac Rate Counter */}
            <div className="telemetry-panel p-4 flex flex-col justify-between">
              <div className="text-[11px] font-mono text-[#64748B] tracking-wider flex items-center justify-between">
                <span>ESTIMATED HEART RATE</span>
                <Heart className={`w-4 h-4 text-[#EF4444] ${telemetry.bpm > 0 ? 'animate-pulse' : ''}`} />
              </div>
              <div className="my-2 flex items-baseline space-x-2">
                <span className="text-4xl font-bold font-mono text-white tracking-tight">
                  {telemetry.bpm > 0 ? telemetry.bpm : '--'}
                </span>
                <span className="text-xs font-mono text-[#10B981]">BPM</span>
              </div>
              <div className="text-[10px] font-mono text-[#64748B]">
                KALMAN SMOOTHED: <span className="text-white">{telemetry.kalman_bpm} BPM</span>
              </div>
            </div>

            {/* Signal-to-Noise Ratio (SNR) Gauge */}
            <div className="telemetry-panel p-4 flex flex-col justify-between">
              <div className="text-[11px] font-mono text-[#64748B] tracking-wider">PULSE SIGNAL QUALITY (SNR)</div>
              <div className="my-2 flex items-baseline space-x-2">
                <span className="text-4xl font-bold font-mono text-[#06B6D4] tracking-tight">
                  {telemetry.snr}
                </span>
                <span className="text-xs font-mono text-[#64748B]">dB</span>
              </div>
              <div className="text-[10px] font-mono text-[#64748B]">
                THRESHOLD: <span className="text-white">&gt; 1.50 dB</span>
              </div>
            </div>
          </div>

          {/* FFT Cardiac Spectrum Analyzer */}
          <div className="telemetry-panel p-3">
            <div className="text-xs font-mono font-semibold text-white mb-2 flex items-center justify-between">
              <span>FFT CARDIAC POWER SPECTRUM (0.75 - 3.0 Hz)</span>
              <span className="text-[10px] text-[#06B6D4]">PEAK FREQ: {(telemetry.bpm / 60).toFixed(2)} Hz</span>
            </div>
            <div className="h-24 bg-[#0C0E15] border border-[#1E2433] rounded overflow-hidden">
              <canvas ref={fftCanvasRef} className="w-full h-full" />
            </div>
          </div>

          {/* 7-Point Anti-Spoofing Quality Gate Matrix */}
          <div className="telemetry-panel p-4 flex-1">
            <div className="flex items-center justify-between border-b border-[#1E2433] pb-2 mb-3">
              <div className="text-xs font-mono font-semibold text-white flex items-center space-x-2">
                <ShieldCheck className="w-4 h-4 text-[#10B981]" />
                <span>7-POINT ANTI-SPOOF QUALITY MATRIX</span>
              </div>
              <div className="text-xs font-mono text-[#10B981] font-bold">
                {telemetry.quality_votes} / {telemetry.total_votes} PASSED
              </div>
            </div>

            <div className="space-y-2 text-xs font-mono">
              {/* Cue 1: SNR */}
              <div className="flex items-center justify-between p-2 telemetry-card">
                <span className="text-[#94A3B8]">1. Cardiac SNR (&gt;1.5 dB)</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${telemetry.cues.snr_ok ? 'bg-[#10B981]/20 text-[#34D399] border border-[#10B981]' : 'bg-[#EF4444]/20 text-[#F87171] border border-[#EF4444]'}`}>
                  {telemetry.snr} dB
                </span>
              </div>

              {/* Cue 2: BPM Range */}
              <div className="flex items-center justify-between p-2 telemetry-card">
                <span className="text-[#94A3B8]">2. Physiological Range (45-180)</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${telemetry.cues.bpm_range_ok ? 'bg-[#10B981]/20 text-[#34D399] border border-[#10B981]' : 'bg-[#EF4444]/20 text-[#F87171] border border-[#EF4444]'}`}>
                  {telemetry.bpm} BPM
                </span>
              </div>

              {/* Cue 3: Temporal Stability */}
              <div className="flex items-center justify-between p-2 telemetry-card">
                <span className="text-[#94A3B8]">3. Temporal Stability</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${telemetry.cues.consistency_ok ? 'bg-[#10B981]/20 text-[#34D399] border border-[#10B981]' : 'bg-[#EF4444]/20 text-[#F87171] border border-[#EF4444]'}`}>
                  {telemetry.cues.consistency_ok ? 'STABLE' : 'UNSTABLE'}
                </span>
              </div>

              {/* Cue 4: Peak Regularity */}
              <div className="flex items-center justify-between p-2 telemetry-card">
                <span className="text-[#94A3B8]">4. Peak Regularity (&gt;0.15)</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${telemetry.cues.regularity_ok ? 'bg-[#10B981]/20 text-[#34D399] border border-[#10B981]' : 'bg-[#EF4444]/20 text-[#F87171] border border-[#EF4444]'}`}>
                  {telemetry.regularity}
                </span>
              </div>

              {/* Cue 5: Spatial Cross-ROI Correlation */}
              <div className="flex items-center justify-between p-2 telemetry-card">
                <span className="text-[#94A3B8]">5. Spatial Cross-Correlation (&gt;0.25)</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${telemetry.cues.correlation_ok ? 'bg-[#10B981]/20 text-[#34D399] border border-[#10B981]' : 'bg-[#EF4444]/20 text-[#F87171] border border-[#EF4444]'}`}>
                  {telemetry.correlation}
                </span>
              </div>

              {/* Cue 6: Phase Coherence */}
              <div className="flex items-center justify-between p-2 telemetry-card">
                <span className="text-[#94A3B8]">6. Spatial Phase Coherence (&gt;0.10)</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${telemetry.cues.phase_ok ? 'bg-[#10B981]/20 text-[#34D399] border border-[#10B981]' : 'bg-[#EF4444]/20 text-[#F87171] border border-[#EF4444]'}`}>
                  {telemetry.phase_coherence}
                </span>
              </div>

              {/* Cue 7: 2nd Harmonic Ratio */}
              <div className="flex items-center justify-between p-2 telemetry-card">
                <span className="text-[#94A3B8]">7. 2nd Harmonic Ratio (&gt;0.03)</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${telemetry.cues.harmonic_ok ? 'bg-[#10B981]/20 text-[#34D399] border border-[#10B981]' : 'bg-[#EF4444]/20 text-[#F87171] border border-[#EF4444]'}`}>
                  {telemetry.harmonic_ratio}
                </span>
              </div>
            </div>
          </div>

          {/* Audit Log Stream Console */}
          <div className="telemetry-panel p-3">
            <div className="flex items-center justify-between border-b border-[#1E2433] pb-2 mb-2">
              <span className="text-xs font-mono font-semibold text-white">SYSTEM AUDIT TRAIL LOG</span>
              <button
                onClick={handleExportAudit}
                className="text-[10px] font-mono text-[#06B6D4] hover:underline flex items-center gap-1"
              >
                <Download className="w-3 h-3" /> EXPORT REPORT
              </button>
            </div>
            <div className="h-24 overflow-y-auto space-y-1 font-mono text-[11px]">
              {auditLogs.map((log, i) => (
                <div key={i} className="flex items-start space-x-2">
                  <span className="text-[#64748B]">{log.time}</span>
                  <span className={
                    log.type === 'success' ? 'text-[#34D399]' :
                      log.type === 'error' ? 'text-[#F87171]' :
                        log.type === 'warn' ? 'text-[#FBBF24]' : 'text-[#94A3B8]'
                  }>
                    {log.msg}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
