"""
ScreenDetector — SILENT screen-replay / display spoof detection.

Detects when the "face" in front of the camera is actually a phone or laptop
screen playing back a video (a replay attack). It produces a single score,
is_screen() ∈ [0, 1], that LivenessDetector uses as a SILENT veto: a high
score forces ACCESS DENIED, but NOTHING about this detector is ever drawn on
the camera feed. Keeping it invisible denies an attacker the feedback they'd
need to tune a bypass.

Three signal-processing cues, all of which a real face lacks but a screen
exhibits:

1. Moiré / pixel-grid (spatial 2D-FFT). A display's pixel matrix and the
   beat against the camera's sensor grid create regular high-frequency
   texture (moiré). Real skin is spatially smooth, so its 2D spectrum is
   concentrated near DC. We measure the fraction of spectral energy at high
   spatial frequencies.

2. Backlight flicker (temporal FFT, OUTSIDE the cardiac band). A screen
   refreshes at 50/60/120 Hz; aliasing against the camera frame rate folds
   this into a strong, narrow out-of-band peak in the ROI's mean luminance
   over time. A real face has only broadband out-of-band content.

3. Specular glare (optional, low weight). Flat glossy screens often produce
   large near-saturated highlight regions.

No third-party deps beyond numpy/opencv (both already required).
"""

import numpy as np
import cv2
from collections import deque


class ScreenDetector:
    """
    Parameters
    ----------
    fps : float
        Camera frame rate (for the temporal flicker FFT).
    lum_buffer_seconds : float
        Seconds of ROI luminance history kept for flicker analysis.
    moire_threshold, flicker_threshold, glare_threshold : float
        Per-cue normalization points (a cue at its threshold maps to ~0.5
        after squashing). Defaults tuned for typical webcams.
    """

    def __init__(self, fps: float = 30.0, lum_buffer_seconds: float = 6.0,
                 moire_threshold: float = 0.35,
                 flicker_threshold: float = 8.0,
                 glare_threshold: float = 0.15):
        self.fps = fps
        self.low_hz = 0.75   # cardiac band to EXCLUDE from flicker analysis
        self.high_hz = 3.0
        self._lum_buf = deque(maxlen=int(fps * lum_buffer_seconds))
        self.moire_threshold = moire_threshold
        self.flicker_threshold = flicker_threshold
        self.glare_threshold = glare_threshold

    # ──────────────────────────────────────────────
    def add_frame(self, roi_bgr: np.ndarray):
        """Record one ROI frame's mean luminance for the flicker analysis."""
        if roi_bgr is None or roi_bgr.size == 0:
            return
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        self._lum_buf.append(float(gray.mean()))

    # ──────────────────────────────────────────────
    def moire_score(self, roi_bgr: np.ndarray) -> float:
        """
        Spatial high-frequency energy ratio (0–1-ish) via windowed 2D-FFT.

        Real skin → energy concentrated near DC → low. Screen pixel grid /
        moiré → elevated high-radius energy.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape
        if h < 8 or w < 8:
            return 0.0

        # 2D Hanning window to suppress crop-edge spectral leakage
        win = np.outer(np.hanning(h), np.hanning(w))
        gray = (gray - gray.mean()) * win

        spec = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_r = np.sqrt(cy ** 2 + cx ** 2)

        dc_mask = radius < 2  # exclude DC neighborhood
        hi_mask = radius > 0.5 * max_r

        total = spec.sum() - spec[dc_mask].sum() + 1e-12
        hi = spec[hi_mask].sum()
        return float(hi / total)

    # ──────────────────────────────────────────────
    def flicker_score(self) -> float:
        """
        Out-of-band temporal flicker peak ratio.

        Returns peak_out_of_band / median_out_of_band of the luminance
        spectrum, excluding DC and the cardiac band. A display's refresh beat
        produces a sharp narrowband out-of-band peak → high ratio; a real
        face has only broadband out-of-band content → low ratio. Returns 0.0
        until the buffer is reasonably full.
        """
        n = len(self._lum_buf)
        if n < int(self.fps * 3):
            return 0.0

        sig = np.asarray(self._lum_buf, dtype=np.float64)
        # linear detrend
        x = np.arange(n)
        sig = sig - np.polyval(np.polyfit(x, sig, 1), x)

        mag = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)

        out_mask = (freqs > 0.05) & ~((freqs >= self.low_hz) & (freqs <= self.high_hz))
        out_mags = mag[out_mask]
        if out_mags.size < 3:
            return 0.0

        med = np.median(out_mags) + 1e-9
        return float(np.max(out_mags) / med)

    # ──────────────────────────────────────────────
    def glare_score(self, roi_bgr: np.ndarray) -> float:
        """Fraction of near-saturated (gray > 245) pixels — flat screen glare."""
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray > 245))

    # ──────────────────────────────────────────────
    @staticmethod
    def _squash(value: float, threshold: float) -> float:
        """Map a non-negative cue to [0,1) with 0.5 at `threshold`."""
        if threshold <= 0:
            return 0.0
        return float(value / (value + threshold))

    def is_screen(self, roi_bgr: np.ndarray = None) -> float:
        """
        Fused screen-likelihood score in [0, 1].

        Pass the current ROI to include spatial moiré/glare cues; flicker uses
        the accumulated luminance buffer regardless. Weighting favors flicker
        (the strongest, hardest-to-fake display signature).
        """
        flicker = self._squash(self.flicker_score(), self.flicker_threshold)

        moire = 0.0
        glare = 0.0
        if roi_bgr is not None and roi_bgr.size > 0:
            moire = self._squash(self.moire_score(roi_bgr), self.moire_threshold)
            glare = self._squash(self.glare_score(roi_bgr), self.glare_threshold)

        # A real display leaves BOTH a spatial signature (pixel-grid moiré) AND
        # a temporal one (backlight flicker). Combining them as a geometric mean
        # means BOTH must be elevated to score high. This is the key robustness
        # fix: ambient room lighting (fluorescent/LED) flickers too and aliases
        # into an out-of-band peak, but it produces NO moiré — so flicker alone
        # no longer falsely vetoes a real face. Glare is a small additive bonus.
        screen_core = np.sqrt(moire * flicker)
        return float(min(1.0, screen_core + 0.1 * glare))

    def reset(self):
        """Clear the luminance history (e.g. when the face is lost)."""
        self._lum_buf.clear()
