"""
PulseExtractor — multi-channel rPPG signal processing pipeline.

Collects R/G/B spatial means from a face ROI, projects them to a 1-D
pulse signal using POS (Plane-Orthogonal-to-Skin, Wang et al. 2017) or
CHROM (de Haan & Jeanne 2013), applies a bandpass Butterworth filter to
isolate the cardiac band (0.75–3 Hz ≈ 45–180 BPM), then computes BPM via
FFT peak detection.

Why multi-channel instead of green-only: green alone carries the strongest
pulse, but it is also corrupted by shared illumination/motion noise that
hits all three channels equally. POS/CHROM project the RGB signal onto a
direction orthogonal to that shared specular component, so the recovered
pulse rejects illumination noise a green-only mean cannot. A screen-replayed
or synthetic "fake pulse" injected as a flat luminance change is suppressed
by the same projection.

Beyond BPM, this module exposes several anti-spoof cues consumed by
LivenessDetector:
  - get_harmonic_ratio(): real pulses have a 2nd harmonic; a synthetic
    single tone does not.
  - get_hrv_metrics(): real pulses have natural heart-rate variability
    (a jitter BAND, not perfect periodicity); too-perfect = synthetic,
    too-irregular = noise.
  - phase_coherence(): forehead and cheek pulses are near in-phase on a
    real face; uncorrelated noise has random phase.
  - update_bpm_track(): Kalman smoothing for temporal continuity, with an
    outlier gate that resists sudden spoof-induced jumps.

Usage:
    pe = PulseExtractor(fps=30.0, buffer_seconds=10, channel_mode="pos")
    # In your frame loop (frame is BGR, so map R<-[:,:,2], B<-[:,:,0]):
    pe.add_sample((r_mean, g_mean, b_mean))
    bpm, snr, filtered_signal = pe.get_bpm()
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque


def phase_coherence(sig_a: np.ndarray, sig_b: np.ndarray,
                    fps: float, f0: float) -> float:
    """
    Cross-spectral phase coherence between two signals at frequency f0.

    Real forehead/cheek pulses share a common cardiac source and are close
    to in-phase → returns ~ +1. Uncorrelated noise (e.g. a photo where each
    region's pixel noise is independent) has random relative phase → ~ 0.

    This complements (does NOT replace) magnitude cross-correlation: a replay
    can inject the same tone into both ROIs and score high magnitude
    correlation, but maintaining stable in-phase coherence at the exact pulse
    frequency across regions is much harder to fake.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Two filtered pulse signals (e.g. forehead and cheek).
    fps : float
        Sampling rate.
    f0 : float
        Frequency of interest in Hz (the pulse fundamental).

    Returns
    -------
    coherence : float
        cos(phase difference) at f0, in [-1, 1]. Returns 0.0 if inputs are
        too short or f0 is invalid.
    """
    if sig_a is None or sig_b is None:
        return 0.0
    n = min(len(sig_a), len(sig_b))
    if n < 30 or f0 <= 0:
        return 0.0

    a = np.asarray(sig_a[-n:], dtype=np.float64)
    b = np.asarray(sig_b[-n:], dtype=np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0

    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # Bin nearest the pulse fundamental
    k = int(np.argmin(np.abs(freqs - f0)))
    cross = fa[k] * np.conj(fb[k])
    if abs(cross) < 1e-12:
        return 0.0

    phase_diff = np.angle(cross)
    return float(np.cos(phase_diff))


class PulseExtractor:
    """
    Real-time multi-channel rPPG pulse extractor.

    Parameters
    ----------
    fps : float
        Actual measured frames-per-second of the webcam.
    buffer_seconds : int
        How many seconds of signal to keep in the rolling buffer.
        Longer = more stable BPM but slower to react to changes.
        10 seconds is a good balance for demo.
    channel_mode : str
        "pos"   — Plane-Orthogonal-to-Skin projection (default, most robust).
        "chrom" — Chrominance-based projection (alternative).
        "green" — legacy green-channel-only (backward compatibility).
    """

    def __init__(self, fps: float = 30.0, buffer_seconds: int = 10,
                 channel_mode: str = "pos", detrend_mode: str = "smoothness"):
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.channel_mode = channel_mode

        # Signal-conditioning chain (ported from prouast/heartbeat-js):
        #   "smoothness" — standardize → smoothness-priors detrend (Tarvainen
        #                  2002) → moving average. Removes baseline wander far
        #                  better than a linear fit.
        #   "linear"     — legacy: subtract a linear polyfit trend.
        self.detrend_mode = detrend_mode
        self.detrend_lambda = float(fps)   # heartbeat-js uses lambda = fps
        # heartbeat-js applies a 3-pass moving average because it has no IIR
        # bandpass. We keep a Butterworth bandpass, which already low-passes the
        # signal; measured against synthetic data, adding the MA on top only
        # concentrates broadband noise in-band and SHRINKS the real-vs-noise SNR
        # separation. So the MA is disabled by default (passes=0) while the
        # method remains available for faithfulness/experimentation.
        self._ma_passes = 0
        self._ma_kernel = max(int(fps / 6), 2)
        self._detrend_M = None             # cached detrend matrix, keyed on length
        self._filtered_cache = None        # per-frame cache of the filtered signal

        # Three parallel channel buffers (vectorize cleaner than a deque of tuples)
        self._r_buf = deque(maxlen=self.buffer_size)
        self._g_buf = deque(maxlen=self.buffer_size)
        self._b_buf = deque(maxlen=self.buffer_size)

        # Cardiac frequency band: 0.75 Hz (45 BPM) to 3.0 Hz (180 BPM)
        self.low_hz = 0.75
        self.high_hz = 3.0

        # Pre-compute Butterworth filter coefficients
        # Order 3 is a good trade-off: enough rolloff without ringing
        nyquist = fps / 2.0
        low_norm = self.low_hz / nyquist
        high_norm = self.high_hz / nyquist

        # Clamp to valid range (0, 1) exclusive
        low_norm = max(low_norm, 0.01)
        high_norm = min(high_norm, 0.99)

        self.b, self.a = butter(3, [low_norm, high_norm], btype='band')

        # Minimum samples needed before we can produce a BPM estimate.
        # Need at least 3 seconds of data for a reasonable FFT.
        self.min_samples = int(fps * 3)

        # Kalman (1-D random-walk) BPM tracker state
        self._bpm_est = None
        self._bpm_var = 1e3
        self._kal_q = 2.0          # process noise (BPM^2 per step)
        self._kal_r = 25.0         # base measurement noise (BPM^2)
        self._kal_snr_ref = 6.0    # SNR at which measurement noise ≈ base

    # ──────────────────────────────────────────────
    # Sample intake
    # ──────────────────────────────────────────────
    def add_sample(self, rgb_mean):
        """
        Add one frame's (R, G, B) spatial means to the buffers.

        Parameters
        ----------
        rgb_mean : tuple/list/ndarray of length 3
            (red_mean, green_mean, blue_mean) of the ROI. NOTE the caller is
            responsible for color order — OpenCV frames are BGR, so build the
            tuple as (roi[:,:,2].mean(), roi[:,:,1].mean(), roi[:,:,0].mean()).
        """
        r, g, b = rgb_mean
        self._r_buf.append(float(r))
        self._g_buf.append(float(g))
        self._b_buf.append(float(b))
        self._filtered_cache = None  # invalidate: buffer changed

    def add_sample_green(self, green: float):
        """
        Legacy shim: add a single green value (replicated to all channels).

        Kept so old callers/tests that only have a green mean still work.
        With channel_mode="green" this reproduces the original behavior.
        """
        v = float(green)
        self._r_buf.append(v)
        self._g_buf.append(v)
        self._b_buf.append(v)
        self._filtered_cache = None  # invalidate: buffer changed

    def get_signal_length(self) -> int:
        """Return the current number of samples in the buffer."""
        return len(self._g_buf)

    def is_ready(self) -> bool:
        """Return True if we have enough samples for a BPM estimate."""
        return len(self._g_buf) >= self.min_samples

    def get_raw_signal(self) -> np.ndarray:
        """Return the raw (unprojected) green channel as a numpy array."""
        return np.array(self._g_buf)

    # ──────────────────────────────────────────────
    # Channel projection (POS / CHROM / green)
    # ──────────────────────────────────────────────
    def _channel_matrix(self) -> np.ndarray:
        """Return a 3×N matrix [R; G; B] from the buffers."""
        return np.array([
            np.asarray(self._r_buf, dtype=np.float64),
            np.asarray(self._g_buf, dtype=np.float64),
            np.asarray(self._b_buf, dtype=np.float64),
        ])

    def _project_pos(self, C: np.ndarray) -> np.ndarray:
        """
        POS projection (Wang et al. 2017, "Algorithmic Principles of Remote PPG").

        Temporal-normalize each channel by its mean, then build two
        projections orthogonal to the skin-tone/specular direction and
        combine them with an amplitude-balancing alpha:
            S1 = G - B            = [ 0,  1, -1] · Cn
            S2 = G + B - 2R       = [-2,  1,  1] · Cn
            alpha = std(S1) / std(S2)
            pulse = S1 + alpha * S2
        """
        means = np.mean(C, axis=1, keepdims=True)
        means[means < 1e-8] = 1e-8
        Cn = C / means  # temporal normalization

        r, g, b = Cn[0], Cn[1], Cn[2]
        s1 = g - b
        s2 = g + b - 2.0 * r

        std_s2 = np.std(s2)
        alpha = (np.std(s1) / std_s2) if std_s2 > 1e-8 else 0.0
        pulse = s1 + alpha * s2
        return pulse - np.mean(pulse)

    def _project_chrom(self, C: np.ndarray) -> np.ndarray:
        """
        CHROM projection (de Haan & Jeanne 2013).

            Xc = 3R - 2G
            Yc = 1.5R + G - 1.5B   (on temporally-normalized channels)
            alpha = std(Xc) / std(Yc)
            S = Xc - alpha * Yc
        """
        means = np.mean(C, axis=1, keepdims=True)
        means[means < 1e-8] = 1e-8
        Cn = C / means

        r, g, b = Cn[0], Cn[1], Cn[2]
        xc = 3.0 * r - 2.0 * g
        yc = 1.5 * r + g - 1.5 * b

        std_yc = np.std(yc)
        alpha = (np.std(xc) / std_yc) if std_yc > 1e-8 else 0.0
        s = xc - alpha * yc
        return s - np.mean(s)

    def _project(self) -> np.ndarray:
        """Project the buffered RGB channels to a 1-D pulse per channel_mode."""
        C = self._channel_matrix()
        if self.channel_mode == "pos":
            return self._project_pos(C)
        if self.channel_mode == "chrom":
            return self._project_chrom(C)
        # green: just the green channel (legacy behavior)
        return C[1] - np.mean(C[1])

    # ──────────────────────────────────────────────
    # Signal conditioning (ported from prouast/heartbeat-js)
    # ──────────────────────────────────────────────
    @staticmethod
    def _standardize(sig: np.ndarray) -> np.ndarray:
        """Z-score the signal (subtract mean, divide by std)."""
        std = np.std(sig)
        if std < 1e-8:
            return sig - np.mean(sig)
        return (sig - np.mean(sig)) / std

    def _detrend_smoothness(self, sig: np.ndarray) -> np.ndarray:
        """
        Smoothness-priors detrending (Tarvainen et al. 2002), as used by
        heartbeat-js. Removes baseline wander while preserving the pulse:

            z_detrended = (I − (I + λ²·D₂ᵀD₂)⁻¹) · z

        where D₂ is the 2nd-order difference operator and λ controls the
        cutoff (heartbeat-js uses λ = fps). The (I − inv) matrix depends only
        on the signal length and λ, so it is built once and cached.
        """
        n = len(sig)
        if n < 3:
            return sig - np.mean(sig)
        if self._detrend_M is None or self._detrend_M.shape[0] != n:
            ident = np.eye(n)
            d2 = np.diff(ident, n=2, axis=0)              # (n-2, n)
            lam2 = self.detrend_lambda ** 2
            self._detrend_M = ident - np.linalg.inv(ident + lam2 * (d2.T @ d2))
        return self._detrend_M @ sig

    def _moving_average(self, sig: np.ndarray) -> np.ndarray:
        """Apply an N-pass moving-average smoother (heartbeat-js: 3 passes)."""
        k = self._ma_kernel
        if k < 2:
            return sig
        kernel = np.ones(k) / k
        for _ in range(self._ma_passes):
            sig = np.convolve(sig, kernel, mode="same")
        return sig

    def get_filtered_signal(self) -> np.ndarray:
        """
        Return the bandpass-filtered pulse signal.

        Steps:
        1. Project RGB → 1-D pulse (POS/CHROM/green)
        2. Condition the signal:
             - detrend_mode="smoothness" (default, from heartbeat-js):
               standardize → smoothness-priors detrend → moving average
             - detrend_mode="linear": subtract a linear polyfit trend (legacy)
        3. Apply Butterworth bandpass filter to enforce the cardiac band

        The result is cached per frame (invalidated by add_sample) because the
        smoothness detrend's matrix multiply is reused by get_bpm /
        get_harmonic_ratio / get_hrv_metrics / get_peak_regularity each frame.
        """
        if self._filtered_cache is not None:
            return self._filtered_cache
        if not self.is_ready():
            return np.array([])

        signal = self._project()

        if self.detrend_mode == "smoothness":
            signal = self._standardize(signal)
            signal = self._detrend_smoothness(signal)
            signal = self._moving_average(signal)
        else:
            # Legacy: remove a linear trend via polyfit.
            n = len(signal)
            x = np.arange(n, dtype=np.float64)
            trend = np.polyval(np.polyfit(x, signal, 1), x)
            signal = signal - trend

        # Apply bandpass filter to enforce the cardiac band for SNR/harmonics
        try:
            filtered = filtfilt(self.b, self.a, signal)
        except ValueError:
            # filtfilt can fail if signal is too short relative to filter order
            return np.array([])

        self._filtered_cache = filtered
        return filtered

    # ──────────────────────────────────────────────
    # Spectrum / BPM / SNR
    # ──────────────────────────────────────────────
    def _compute_spectrum(self, filtered: np.ndarray) -> tuple:
        """
        Shared FFT helper. Returns (freqs, fft_mag) for the full rfft.
        """
        n = len(filtered)
        fft_mag = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        return freqs, fft_mag

    def get_bpm(self) -> tuple:
        """
        Compute BPM from the filtered signal using FFT.

        Returns
        -------
        bpm : float
            Estimated heart rate in beats per minute. 0.0 if not enough data.
        snr : float
            Signal-to-noise ratio in dB. Real pulse should be > ~3 dB.
        filtered : np.ndarray
            The filtered signal for visualization.
        """
        filtered = self.get_filtered_signal()
        if len(filtered) == 0:
            return 0.0, 0.0, np.array([])

        freqs, fft_mag = self._compute_spectrum(filtered)

        # Only look at frequencies in the cardiac band
        band_mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        band_freqs = freqs[band_mask]
        band_mags = fft_mag[band_mask]

        if len(band_mags) == 0:
            return 0.0, 0.0, filtered

        # Peak frequency in the cardiac band → BPM
        peak_idx = np.argmax(band_mags)
        peak_freq = band_freqs[peak_idx]
        bpm = peak_freq * 60.0

        # SNR: ratio of peak power to average power in the band
        peak_power = band_mags[peak_idx] ** 2
        avg_power = np.mean(band_mags ** 2) + 1e-10
        snr = 10.0 * np.log10(peak_power / avg_power + 1e-10)

        return bpm, snr, filtered

    def get_harmonic_ratio(self) -> float:
        """
        Ratio of 2nd-harmonic energy to fundamental energy.

        A real photoplethysmographic waveform is not a pure sinusoid — the
        sharp systolic upstroke produces a measurable 2nd harmonic. A
        synthetic single-tone "fake pulse" (or a flat replay) has almost no
        harmonic, so this ratio collapses toward 0.

        Returns
        -------
        ratio : float
            E(2*f0) / E(f0), summed over a ±0.1 Hz bin around each. Returns
            0.0 if not ready or if 2*f0 exceeds Nyquist.
        """
        filtered = self.get_filtered_signal()
        if len(filtered) == 0:
            return 0.0

        freqs, fft_mag = self._compute_spectrum(filtered)
        power = fft_mag ** 2

        band_mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        if not np.any(band_mask):
            return 0.0
        band_freqs = freqs[band_mask]
        band_power = power[band_mask]
        f0 = band_freqs[int(np.argmax(band_power))]

        nyquist = self.fps / 2.0
        if 2.0 * f0 > nyquist:
            return 0.0

        def band_energy(center, half=0.1):
            m = (freqs >= center - half) & (freqs <= center + half)
            return float(np.sum(power[m]))

        e_fund = band_energy(f0)
        e_harm = band_energy(2.0 * f0)
        if e_fund < 1e-12:
            return 0.0
        return e_harm / e_fund

    # ──────────────────────────────────────────────
    # Peak / HRV analysis
    # ──────────────────────────────────────────────
    def _find_pulse_peaks(self, filtered: np.ndarray):
        """
        Shared peak finder for regularity + HRV. Returns (peaks, sig_std).
        """
        min_distance = max(int(self.fps / 3), 3)  # 180 BPM cap → fps/3 spacing
        sig_std = np.std(filtered)
        if sig_std < 1e-8:
            return np.array([], dtype=int), 0.0
        peaks, _ = find_peaks(
            filtered,
            distance=min_distance,
            height=0.15 * sig_std,
            prominence=0.1 * sig_std,
        )
        return peaks, sig_std

    def get_peak_regularity(self) -> tuple:
        """
        Time-domain peak regularity score (kept for the dashboard).

        NOTE: this score REWARDS perfect periodicity (CV→0 → 1.0). It is a
        good lower-bound anti-noise indicator, but on its own it would let a
        synthetic perfectly-periodic tone pass. The liveness gate therefore
        pairs it with get_hrv_metrics()['jitter_ok'] (an upper bound).

        Returns (peak_count, regularity[0–1], avg_peak_amplitude).
        """
        filtered = self.get_filtered_signal()
        if len(filtered) < self.min_samples:
            return 0, 0.0, 0.0

        peaks, sig_std = self._find_pulse_peaks(filtered)
        peak_count = len(peaks)
        if peak_count < 2 or sig_std < 1e-8:
            return peak_count, 0.0, 0.0

        intervals = np.diff(peaks)
        mean_interval = np.mean(intervals)
        if mean_interval < 1e-6:
            return peak_count, 0.0, 0.0

        interval_cv = np.std(intervals) / mean_interval
        # CV=0 → 1.0, CV~0.4 → 0.5, CV>0.8 → ~0
        regularity = max(0.0, 1.0 - 1.2 * interval_cv)

        avg_peak_amplitude = np.mean(np.abs(filtered[peaks])) / sig_std
        return peak_count, regularity, avg_peak_amplitude

    def get_hrv_metrics(self) -> dict:
        """
        Heart-rate-variability metrics for anti-spoof.

        Real cardiac rhythm has natural beat-to-beat variability (a jitter
        BAND), and often respiratory sinus arrhythmia (RSA): a ~0.15–0.4 Hz
        modulation of the inter-beat intervals from breathing. A synthetic
        injected tone is unnaturally perfect (CV ≈ 0); random noise is
        unnaturally irregular (CV large). `jitter_ok` accepts only the
        physiological middle band.

        Returns
        -------
        dict with:
            ibi_cv : float        coefficient of variation of inter-beat intervals
            rmssd : float         root-mean-square of successive IBI differences (s)
            jitter_ok : bool      CV_MIN <= ibi_cv <= CV_MAX
            rsa_power : float|None spectral energy of IBI series in 0.15–0.4 Hz
        """
        # Lower bound rejects synthetic "too perfect" tones; upper bound
        # rejects pure noise. The upper bound is generous (0.50) because real
        # rPPG peak detection on a short noisy buffer naturally inflates IBI
        # CV well past true physiological HRV — too tight here falsely rejects
        # real people. The lower bound (0.035) sits between a synthetic tone's
        # quantization-limited CV (~0.02) and real human HRV (~0.05–0.10).
        CV_MIN, CV_MAX = 0.035, 0.50
        result = {"ibi_cv": 0.0, "rmssd": 0.0, "jitter_ok": False, "rsa_power": None}

        filtered = self.get_filtered_signal()
        if len(filtered) < self.min_samples:
            return result

        peaks, sig_std = self._find_pulse_peaks(filtered)
        if len(peaks) < 3 or sig_std < 1e-8:
            return result

        ibi = np.diff(peaks) / self.fps  # seconds
        mean_ibi = np.mean(ibi)
        if mean_ibi < 1e-6:
            return result

        ibi_cv = float(np.std(ibi) / mean_ibi)
        rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) >= 2 else 0.0
        jitter_ok = CV_MIN <= ibi_cv <= CV_MAX

        # RSA: spectral energy of the (detrended) IBI series in 0.15–0.4 Hz.
        # IBI series is sampled at one value per beat; its sampling rate is the
        # mean heart rate in Hz.
        rsa_power = None
        if len(ibi) >= 6:
            ibi_dt = ibi - np.mean(ibi)
            fs_ibi = 1.0 / mean_ibi  # beats per second
            mag = np.abs(np.fft.rfft(ibi_dt))
            f = np.fft.rfftfreq(len(ibi_dt), d=1.0 / fs_ibi)
            rsa_mask = (f >= 0.15) & (f <= 0.40)
            if np.any(rsa_mask):
                total = np.sum(mag ** 2) + 1e-12
                rsa_power = float(np.sum(mag[rsa_mask] ** 2) / total)

        result.update(ibi_cv=ibi_cv, rmssd=rmssd, jitter_ok=jitter_ok,
                      rsa_power=rsa_power)
        return result

    # ──────────────────────────────────────────────
    # Kalman BPM tracking
    # ──────────────────────────────────────────────
    def update_bpm_track(self, bpm_meas: float, snr: float) -> float:
        """
        1-D random-walk Kalman smoothing of BPM with an outlier gate.

        Exposed separately from get_bpm() so the raw measurement contract is
        preserved. The measurement noise R scales inversely with SNR (trust
        strong-SNR readings more), and a hard outlier gate makes the tracker
        coast through sudden spoof-induced jumps instead of following them.

        Returns the smoothed BPM estimate.
        """
        # No usable measurement → coast on the prediction
        if bpm_meas <= 0:
            if self._bpm_est is not None:
                self._bpm_var += self._kal_q
            return self._bpm_est if self._bpm_est is not None else 0.0

        if self._bpm_est is None:
            self._bpm_est = bpm_meas
            self._bpm_var = self._kal_r
            return self._bpm_est

        # Predict
        self._bpm_var += self._kal_q

        # Outlier gate: a jump >25 BPM from the estimate is treated as a
        # spurious/spoof reading — coast instead of jumping.
        if abs(bpm_meas - self._bpm_est) > 25.0:
            return self._bpm_est

        # Adaptive measurement noise: lower SNR → larger R → less trust
        r = self._kal_r * max(1.0, self._kal_snr_ref / max(snr, 1e-6))
        k = self._bpm_var / (self._bpm_var + r)
        self._bpm_est += k * (bpm_meas - self._bpm_est)
        self._bpm_var = (1.0 - k) * self._bpm_var
        return self._bpm_est

    def reset(self):
        """Clear the signal buffers and Kalman state (e.g. when face is lost)."""
        self._r_buf.clear()
        self._g_buf.clear()
        self._b_buf.clear()
        self._bpm_est = None
        self._bpm_var = 1e3
        self._filtered_cache = None
        # keep _detrend_M: it depends only on length+lambda, safe to reuse
