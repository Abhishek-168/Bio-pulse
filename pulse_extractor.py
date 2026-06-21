"""
PulseExtractor — rPPG signal processing pipeline.

Collects green-channel spatial means from the forehead ROI,
applies a bandpass Butterworth filter to isolate the cardiac
frequency band (0.75–3 Hz ≈ 45–180 BPM), then computes BPM
via FFT peak detection.

Usage:
    pe = PulseExtractor(fps=30.0, buffer_seconds=10)
    # In your frame loop:
    pe.add_sample(green_mean_value)
    bpm, snr, filtered_signal = pe.get_bpm()
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque


class PulseExtractor:
    """
    Real-time rPPG pulse extractor.

    Parameters
    ----------
    fps : float
        Actual measured frames-per-second of the webcam.
    buffer_seconds : int
        How many seconds of signal to keep in the rolling buffer.
        Longer = more stable BPM but slower to react to changes.
        10 seconds is a good balance for demo.
    """

    def __init__(self, fps: float = 30.0, buffer_seconds: int = 10):
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.signal_buffer = deque(maxlen=self.buffer_size)

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

    def add_sample(self, green_mean: float):
        """Add one frame's green-channel spatial mean to the buffer."""
        self.signal_buffer.append(green_mean)

    def get_signal_length(self) -> int:
        """Return the current number of samples in the buffer."""
        return len(self.signal_buffer)

    def is_ready(self) -> bool:
        """Return True if we have enough samples for a BPM estimate."""
        return len(self.signal_buffer) >= self.min_samples

    def get_raw_signal(self) -> np.ndarray:
        """Return the raw (unfiltered) signal buffer as a numpy array."""
        return np.array(self.signal_buffer)

    def get_filtered_signal(self) -> np.ndarray:
        """
        Return the bandpass-filtered signal.

        Steps:
        1. Convert deque to numpy array
        2. Detrend (remove DC offset + linear drift from lighting changes)
        3. Apply Butterworth bandpass filter
        """
        if not self.is_ready():
            return np.array([])

        signal = np.array(self.signal_buffer, dtype=np.float64)

        # Detrend: remove mean and linear trend using numpy polyfit
        # (numerically stable — avoids overflow from manual regression)
        # This kills slow lighting changes that would otherwise
        # dominate the signal and hide the pulse.
        n = len(signal)
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, signal, 1)  # linear fit: [slope, intercept]
        trend = np.polyval(coeffs, x)
        signal = signal - trend

        # Apply bandpass filter
        try:
            filtered = filtfilt(self.b, self.a, signal)
        except ValueError:
            # filtfilt can fail if signal is too short relative to filter order
            return np.array([])

        return filtered

    def get_bpm(self) -> tuple:
        """
        Compute BPM from the filtered signal using FFT.

        Returns
        -------
        bpm : float
            Estimated heart rate in beats per minute.
            Returns 0.0 if not enough data.
        snr : float
            Signal-to-noise ratio in dB. Higher = more confident.
            A real pulse should have SNR > ~3 dB.
            Returns 0.0 if not enough data.
        filtered : np.ndarray
            The filtered signal for visualization.
        """
        filtered = self.get_filtered_signal()
        if len(filtered) == 0:
            return 0.0, 0.0, np.array([])

        # FFT
        n = len(filtered)
        fft_vals = np.fft.rfft(filtered)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)

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

    def get_peak_regularity(self) -> tuple:
        """
        Analyze time-domain peak regularity of the filtered signal.

        A real heartbeat produces evenly-spaced peaks. Random noise
        from a photo/screen produces irregular, inconsistent peaks.

        Returns
        -------
        peak_count : int
            Number of peaks found in the signal.
        regularity : float
            0.0 to 1.0 — how evenly spaced the peaks are.
            1.0 = perfectly regular (like a real heartbeat).
            Low values = irregular noise.
        avg_peak_amplitude : float
            Average amplitude of detected peaks relative to signal std.
        """
        filtered = self.get_filtered_signal()
        if len(filtered) < self.min_samples:
            return 0, 0.0, 0.0

        # Find peaks with minimum distance based on expected HR range
        # At 180 BPM max, minimum peak distance = fps * 60/180 = fps/3
        min_distance = max(int(self.fps / 3), 3)

        # Adaptive height threshold: peaks should be above median
        sig_std = np.std(filtered)
        if sig_std < 1e-8:
            return 0, 0.0, 0.0

        peaks, properties = find_peaks(
            filtered,
            distance=min_distance,
            height=0.15 * sig_std,  # peaks must be at least 15% of std above zero
            prominence=0.1 * sig_std,
        )

        peak_count = len(peaks)
        if peak_count < 2:
            return peak_count, 0.0, 0.0

        # Compute inter-peak intervals
        intervals = np.diff(peaks)
        mean_interval = np.mean(intervals)

        if mean_interval < 1e-6:
            return peak_count, 0.0, 0.0

        # Regularity: coefficient of variation (CV) of intervals
        # Low CV = regular spacing = real heartbeat
        # High CV = irregular = noise
        interval_cv = np.std(intervals) / mean_interval
        # Convert CV to a 0-1 regularity score
        # Real heartbeats have natural HRV (CV ~0.1-0.25), so use a gentler curve
        # CV=0 → 1.0, CV~0.4 → 0.5, CV>0.8 → ~0
        regularity = max(0.0, 1.0 - 1.2 * interval_cv)

        # Average peak amplitude relative to signal std
        peak_amplitudes = filtered[peaks]
        avg_peak_amplitude = np.mean(np.abs(peak_amplitudes)) / sig_std

        return peak_count, regularity, avg_peak_amplitude

    def reset(self):
        """Clear the signal buffer (e.g. when face is lost)."""
        self.signal_buffer.clear()
