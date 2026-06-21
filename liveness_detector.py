"""
LivenessDetector — decides REAL HUMAN vs SPOOF from pulse data.

Uses multiple criteria to prevent photo/video/deepfake spoofing:
1. BPM must be in a physiologically plausible range (45–180 BPM)
2. Signal-to-noise ratio (SNR) must exceed a threshold
3. Signal must show temporal consistency (not wild jumps)
4. Peak regularity — real heartbeats have evenly-spaced peaks; random noise
   from a photo does not. This is a LOWER bound (anti-noise).
5. HRV jitter band — real heartbeats also have *natural* beat-to-beat
   variability. A synthetic injected tone is unnaturally perfect. So we also
   require an UPPER bound on regularity via the HRV `jitter_ok` flag.

   Together, checks 4 and 5 close a contradiction that existed before: peak
   regularity alone REWARDS perfect periodicity (score → 1.0), so a synthetic
   perfectly-periodic "fake pulse" used to pass. `regular` rejects noise from
   below; `jitter_ok` rejects too-perfect synthesis from above.
6. Multi-ROI magnitude correlation — forehead and cheek signals must
   correlate (blood flows everywhere on a real face).
7. Multi-ROI phase coherence — the two regions must also be in-phase at the
   pulse frequency. A replay can inject the same tone in both ROIs (high
   magnitude correlation) but stable cross-region phase is much harder to fake.
8. Harmonic structure — a real PPG waveform has a 2nd harmonic; a pure
   synthetic sinusoid does not.

SILENT SCREEN VETO: a separate ScreenDetector supplies an `is_screen` score.
If it is high, the decision is forced to DENIED immediately and NOTHING about
it is exposed for display — an attacker holding up a phone/laptop gets no
feedback about why they were rejected.

A photo, video replay, or deepfake will trip at least one of these.
"""

import numpy as np
from collections import deque


class LivenessDetector:
    """
    Real-time liveness detector based on rPPG pulse quality.

    Parameters
    ----------
    bpm_window : int
        Number of recent readings to keep for averaging/consistency.
    snr_threshold : float
        Minimum SNR (dB) to consider the signal as containing a real pulse.
    bpm_range : tuple
        (min_bpm, max_bpm) for physiologically plausible heart rate.
    consistency_threshold : float
        Maximum allowed standard deviation of recent BPM readings.
    min_readings : int
        Minimum number of readings before making a decision.
    regularity_threshold : float
        Minimum peak regularity score (0–1) — anti-noise lower bound.
    correlation_threshold : float
        Minimum magnitude cross-correlation between forehead and cheek.
    harmonic_threshold : float
        Minimum 2nd-harmonic/fundamental energy ratio.
    phase_threshold : float
        Minimum cross-ROI phase coherence at the pulse frequency.
    screen_veto_threshold : float
        Averaged is_screen score at/above which the decision is silently DENIED.
    """

    # Decision states
    PENDING = "SCANNING..."
    ALIVE = "ACCESS GRANTED"
    DENIED = "ACCESS DENIED"

    def __init__(
        self,
        bpm_window: int = 8,
        snr_threshold: float = 1.5,
        bpm_range: tuple = (45, 180),
        consistency_threshold: float = 25.0,
        min_readings: int = 5,
        regularity_threshold: float = 0.15,
        correlation_threshold: float = 0.3,
        harmonic_threshold: float = 0.03,
        phase_threshold: float = 0.1,
        screen_veto_threshold: float = 0.5,
        quality_ratio_threshold: float = 0.5,
    ):
        self.bpm_history = deque(maxlen=bpm_window)
        self.snr_history = deque(maxlen=bpm_window)
        self.regularity_history = deque(maxlen=bpm_window)
        self.correlation_history = deque(maxlen=bpm_window)
        self.harmonic_history = deque(maxlen=bpm_window)
        self.phase_history = deque(maxlen=bpm_window)
        self.jitter_history = deque(maxlen=bpm_window)
        self.screen_history = deque(maxlen=bpm_window)

        self.snr_threshold = snr_threshold
        self.bpm_min, self.bpm_max = bpm_range
        self.consistency_threshold = consistency_threshold
        self.min_readings = min_readings
        self.regularity_threshold = regularity_threshold
        self.correlation_threshold = correlation_threshold
        self.harmonic_threshold = harmonic_threshold
        self.phase_threshold = phase_threshold
        self.screen_veto_threshold = screen_veto_threshold
        self.quality_ratio_threshold = quality_ratio_threshold

        self.decision = self.PENDING
        self.confidence = 0.0
        self._failed_readings = 0  # consecutive bad readings counter
        self._check_details = {}  # for debugging

    def update(self, bpm: float, snr: float,
               regularity: float = 1.0, correlation: float = 1.0,
               *, harmonic_ratio: float = None, jitter_ok: bool = None,
               phase_coherence: float = None, is_screen: float = 0.0) -> str:
        """
        Feed a new set of pulse measurements and return the current decision.

        The first four parameters are positional for backward compatibility
        (old callers used update(bpm, snr) or update(bpm, snr, reg, corr)).
        The anti-spoof cues are keyword-only and default to "not provided"
        (None), in which case their gate is skipped.

        Parameters
        ----------
        bpm, snr : float
            Latest BPM / SNR estimate from PulseExtractor.
        regularity : float
            Peak regularity score (0–1) — anti-noise lower bound.
        correlation : float
            Magnitude cross-correlation between forehead and cheek signals.
        harmonic_ratio : float, optional
            2nd-harmonic/fundamental energy ratio.
        jitter_ok : bool, optional
            Whether HRV inter-beat-interval variability is in the natural band.
        phase_coherence : float, optional
            Cross-ROI phase coherence at the pulse frequency.
        is_screen : float
            Screen-replay likelihood [0–1] from ScreenDetector (SILENT veto).

        Returns
        -------
        decision : str
            One of PENDING, ALIVE, or DENIED.
        """
        # --- SILENT screen veto (highest priority) ---
        # Averaged so a single noisy frame can't veto, but a sustained screen
        # signature does. No visible check entry is produced.
        self.screen_history.append(float(is_screen))
        avg_screen = sum(self.screen_history) / len(self.screen_history)
        if avg_screen >= self.screen_veto_threshold:
            self.decision = self.DENIED
            self.confidence = 0.0
            self._check_details["_screen_veto"] = True  # debug only, never drawn
            return self.decision
        self._check_details.pop("_screen_veto", None)

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
        if harmonic_ratio is not None:
            self.harmonic_history.append(harmonic_ratio)
        if phase_coherence is not None:
            self.phase_history.append(phase_coherence)
        if jitter_ok is not None:
            self.jitter_history.append(1.0 if jitter_ok else 0.0)

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
            bpm_std = 0.0
            consistent = True

        # --- Check 4: Peak regularity (anti-noise lower bound) ---
        avg_regularity = sum(self.regularity_history) / len(self.regularity_history)
        regular = avg_regularity >= self.regularity_threshold

        # --- Check 5: Multi-ROI magnitude correlation (anti-photo) ---
        avg_correlation = sum(self.correlation_history) / len(self.correlation_history)
        correlated = avg_correlation >= self.correlation_threshold

        # --- Check 6: Harmonic structure (anti synthetic single-tone) ---
        if self.harmonic_history:
            avg_harmonic = sum(self.harmonic_history) / len(self.harmonic_history)
            harmonic_ok = avg_harmonic >= self.harmonic_threshold
        else:
            avg_harmonic = None
            harmonic_ok = True  # cue not provided → skip gate

        # --- Check 7: HRV jitter band (anti synthetic too-perfect) ---
        if self.jitter_history:
            jitter_frac = sum(self.jitter_history) / len(self.jitter_history)
            jitter_pass = jitter_frac >= 0.5  # majority of recent frames natural
        else:
            jitter_frac = None
            jitter_pass = True  # cue not provided → skip gate

        # --- Check 8: Cross-ROI phase coherence (anti replay) ---
        if self.phase_history:
            avg_phase = sum(self.phase_history) / len(self.phase_history)
            phase_ok = avg_phase >= self.phase_threshold
        else:
            avg_phase = None
            phase_ok = True  # cue not provided → skip gate

        # --- Quality vote ---
        # Real webcam rPPG is noisy: any single quality metric fails on a
        # fraction of frames even for a genuine live face. Requiring ALL of
        # them simultaneously (a hard 8-way AND) made live humans almost never
        # pass. Instead we treat the noisy quality cues as VOTES and require a
        # majority, while keeping the few reliable signals as hard gates.
        #
        # Hard requirements:
        #   - bpm_valid: pulse frequency is physically plausible
        #   - (screen veto handled earlier as a silent hard DENY)
        # Quality votes (each individually noisy):
        #   snr, consistency, regularity, correlation, phase, harmonic, jitter
        quality_votes = {
            "snr_valid": snr_valid,
            "consistent": consistent,
            "regular": regular,
            "correlated": correlated,
            "phase_ok": phase_ok,
            "harmonic_ok": harmonic_ok,
            "jitter_pass": jitter_pass,
        }
        n_pass = sum(1 for v in quality_votes.values() if v)
        n_total = len(quality_votes)
        quality_ratio = n_pass / n_total

        # Cross-ROI agreement is the key discriminator between a real pulse
        # (present in BOTH forehead and cheek) and uncorrelated noise / a flat
        # photo. We require at least ONE of {magnitude correlation, phase
        # coherence} to pass when those cues are available — lenient enough not
        # to reject a real face whose cheek signal is a little weak, but enough
        # to block noise, which fails both.
        has_cross = bool(self.correlation_history) or bool(self.phase_history)
        cross_roi_ok = (not has_cross) or correlated or phase_ok

        # Store check details for debugging / dashboard
        self._check_details = {
            "bpm_valid": bpm_valid,
            "cross_roi_ok": cross_roi_ok,
            "quality_ratio": quality_ratio,
            "n_pass": n_pass,
            "n_total": n_total,
            **quality_votes,
            "avg_bpm": avg_bpm,
            "avg_snr": avg_snr,
            "bpm_std": bpm_std,
            "avg_regularity": avg_regularity,
            "avg_correlation": avg_correlation,
            "avg_harmonic": avg_harmonic,
            "avg_phase": avg_phase,
            "jitter_frac": jitter_frac,
        }

        # --- Final decision ---
        # ALIVE iff the pulse frequency is plausible AND a majority of the
        # quality cues agree. This tolerates a couple of noisy gates while
        # still requiring substantial, multi-signal evidence of a real pulse.
        if bpm_valid and cross_roi_ok and quality_ratio >= self.quality_ratio_threshold:
            self.decision = self.ALIVE
            scores = [
                min(1.0, avg_snr / (self.snr_threshold * 2)) if avg_snr > 0 else 0.0,
                avg_regularity,
                avg_correlation,
                quality_ratio,
            ]
            if avg_phase is not None:
                scores.append(max(0.0, avg_phase))
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
        avg_harmonic = 0.0
        avg_phase = 0.0

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

        if self.harmonic_history:
            avg_harmonic = sum(self.harmonic_history) / len(self.harmonic_history)

        if self.phase_history:
            avg_phase = sum(self.phase_history) / len(self.phase_history)

        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "avg_bpm": avg_bpm,
            "avg_snr": avg_snr,
            "bpm_std": bpm_std,
            "avg_regularity": avg_regularity,
            "avg_correlation": avg_correlation,
            "avg_harmonic": avg_harmonic,
            "avg_phase": avg_phase,
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
        self.harmonic_history.clear()
        self.phase_history.clear()
        self.jitter_history.clear()
        self.screen_history.clear()
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
