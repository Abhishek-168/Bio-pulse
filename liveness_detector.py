"""
LivenessDetector — decides REAL HUMAN vs SPOOF from pulse data.

Uses multiple criteria to prevent photo/video/deepfake spoofing:
1. BPM must be in a physiologically plausible range (45–180 BPM)
2. Signal-to-noise ratio (SNR) must exceed a threshold
3. Signal must show temporal consistency (not wild jumps)
4. Peak regularity — real heartbeats have evenly-spaced peaks;
   random noise from a photo does not
5. Multi-ROI cross-correlation — forehead and cheek signals must
   correlate (blood flows everywhere on a real face; sensor noise
   on a photo is uncorrelated between regions)

A photo, video replay, or deepfake will produce either:
  - A flat-line (no pulse → SNR ≈ 0)
  - Random noise that lacks peak regularity
  - Uncorrelated signals between face regions
  - BPM outside the human range
"""

import numpy as np
from collections import deque


class LivenessDetector:
    """
    Real-time liveness detector based on rPPG pulse quality.

    Parameters
    ----------
    bpm_window : int
        Number of recent BPM readings to keep for consistency check.
    snr_threshold : float
        Minimum SNR (dB) to consider the signal as containing a real pulse.
    bpm_range : tuple
        (min_bpm, max_bpm) for physiologically plausible heart rate.
    consistency_threshold : float
        Maximum allowed standard deviation of recent BPM readings.
    min_readings : int
        Minimum number of BPM readings before making a decision.
    regularity_threshold : float
        Minimum peak regularity score (0–1) to consider the signal real.
    correlation_threshold : float
        Minimum cross-correlation between forehead and cheek signals.
    """

    # Decision states
    PENDING = "SCANNING..."
    ALIVE = "ACCESS GRANTED"
    DENIED = "ACCESS DENIED"

    def __init__(
        self,
        bpm_window: int = 8,
        snr_threshold: float = 4.0,
        bpm_range: tuple = (50, 160),
        consistency_threshold: float = 15.0,
        min_readings: int = 5,
        regularity_threshold: float = 0.2,
        correlation_threshold: float = 0.4,
    ):
        self.bpm_history = deque(maxlen=bpm_window)
        self.snr_history = deque(maxlen=bpm_window)
        self.regularity_history = deque(maxlen=bpm_window)
        self.correlation_history = deque(maxlen=bpm_window)
        self.snr_threshold = snr_threshold
        self.bpm_min, self.bpm_max = bpm_range
        self.consistency_threshold = consistency_threshold
        self.min_readings = min_readings
        self.regularity_threshold = regularity_threshold
        self.correlation_threshold = correlation_threshold
        self.decision = self.PENDING
        self.confidence = 0.0
        self._failed_readings = 0  # consecutive bad readings counter
        self._check_details = {}  # for debugging

    def update(self, bpm: float, snr: float,
               regularity: float = 1.0, correlation: float = 1.0) -> str:
        """
        Feed a new BPM + SNR + regularity + correlation reading.

        Parameters
        ----------
        bpm : float
            Latest BPM estimate from PulseExtractor.
        snr : float
            Latest SNR value from PulseExtractor.
        regularity : float
            Peak regularity score (0–1) from PulseExtractor.
        correlation : float
            Cross-correlation between forehead and cheek signals.

        Returns
        -------
        decision : str
            One of PENDING, ALIVE, or DENIED.
        """
        if bpm <= 0 or snr <= 0:
            self._failed_readings += 1
            # After enough failed readings, this is a spoof/photo
            if self._failed_readings >= self.min_readings * 2:
                self.decision = self.DENIED
                self.confidence = 0.0
            else:
                self.decision = self.PENDING
                self.confidence = 0.0
            return self.decision

        # Got a valid reading — reset failure counter
        self._failed_readings = 0

        self.bpm_history.append(bpm)
        self.snr_history.append(snr)
        self.regularity_history.append(regularity)
        self.correlation_history.append(correlation)

        # Need enough readings for a stable decision
        if len(self.bpm_history) < self.min_readings:
            self.decision = self.PENDING
            self.confidence = len(self.bpm_history) / self.min_readings
            return self.decision

        # --- Check 1: BPM in human range ---
        avg_bpm = sum(self.bpm_history) / len(self.bpm_history)
        bpm_valid = self.bpm_min <= avg_bpm <= self.bpm_max

        # --- Check 2: SNR above threshold ---
        avg_snr = sum(self.snr_history) / len(self.snr_history)
        snr_valid = avg_snr >= self.snr_threshold

        # --- Check 3: BPM consistency (not wild jumps) ---
        if len(self.bpm_history) >= 3:
            bpm_list = list(self.bpm_history)
            bpm_std = self._std(bpm_list)
            consistent = bpm_std < self.consistency_threshold
        else:
            consistent = True

        # --- Check 4: Peak regularity (anti-photo) ---
        avg_regularity = sum(self.regularity_history) / len(self.regularity_history)
        regular = avg_regularity >= self.regularity_threshold

        # --- Check 5: Multi-ROI correlation (anti-photo) ---
        avg_correlation = sum(self.correlation_history) / len(self.correlation_history)
        correlated = avg_correlation >= self.correlation_threshold

        # Store check details for debugging
        self._check_details = {
            "bpm_valid": bpm_valid,
            "snr_valid": snr_valid,
            "consistent": consistent,
            "regular": regular,
            "correlated": correlated,
            "avg_bpm": avg_bpm,
            "avg_snr": avg_snr,
            "bpm_std": bpm_std if len(self.bpm_history) >= 3 else 0.0,
            "avg_regularity": avg_regularity,
            "avg_correlation": avg_correlation,
        }

        # --- Final decision ---
        # Must pass ALL checks to be considered alive
        if bpm_valid and snr_valid and consistent and regular and correlated:
            self.decision = self.ALIVE
            # Confidence: weighted combination of all metrics
            scores = [
                min(1.0, avg_snr / (self.snr_threshold * 2)),
                avg_regularity,
                avg_correlation,
            ]
            self.confidence = sum(scores) / len(scores)
        else:
            self.decision = self.DENIED
            self.confidence = 0.0

        return self.decision

    def get_status(self) -> dict:
        """
        Return a dict with all current metrics for display.
        """
        avg_bpm = 0.0
        avg_snr = 0.0
        bpm_std = 0.0
        avg_regularity = 0.0
        avg_correlation = 0.0

        if self.bpm_history:
            bpm_list = list(self.bpm_history)
            avg_bpm = sum(bpm_list) / len(bpm_list)
            if len(bpm_list) >= 2:
                bpm_std = self._std(bpm_list)

        if self.snr_history:
            avg_snr = sum(self.snr_history) / len(self.snr_history)

        if self.regularity_history:
            avg_regularity = sum(self.regularity_history) / len(self.regularity_history)

        if self.correlation_history:
            avg_correlation = sum(self.correlation_history) / len(self.correlation_history)

        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "avg_bpm": avg_bpm,
            "avg_snr": avg_snr,
            "bpm_std": bpm_std,
            "avg_regularity": avg_regularity,
            "avg_correlation": avg_correlation,
            "readings": len(self.bpm_history),
            "min_readings": self.min_readings,
            "check_details": self._check_details,
        }

    def reset(self):
        """Clear history (e.g. when face is lost)."""
        self.bpm_history.clear()
        self.snr_history.clear()
        self.regularity_history.clear()
        self.correlation_history.clear()
        self.decision = self.PENDING
        self.confidence = 0.0
        self._failed_readings = 0
        self._check_details = {}

    @staticmethod
    def _std(values: list) -> float:
        """Compute standard deviation without importing numpy."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        return variance ** 0.5
