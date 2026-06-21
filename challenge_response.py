"""
ChallengeResponse — active liveness challenge (VISIBLE on the camera feed).

The pulse-based checks are all PASSIVE: a high-quality pre-recorded video of a
compliant person could, in principle, satisfy them. An active challenge closes
that gap by demanding a real-time, randomly-chosen action ("blink", "turn your
head left", "open your mouth") and verifying the response from MediaPipe
landmarks within a short timeout. A static replay cannot comply on demand.

This module issues a SERIES of challenges (default 3) drawn randomly from a
varied pool. Each step has its own time budget: if it is not completed in time
the step is SKIPPED and the next one is issued, but a skipped step is recorded
as a miss. The whole sequence only PASSES when EVERY step was completed in its
window — if even one step is missed the sequence FAILS (access denied), with no
retry. Forcing several distinct live actions makes a pre-recorded clip
dramatically harder to fake than a single prompt.

Unlike ScreenDetector (which is deliberately silent), this module SHOULD draw
on the feed — it's a prompt the genuine user needs to see.

Landmark indices assume the 468-point FaceLandmarker model with iris refine
OFF (matching bio_pulse_demo.py's configuration). If refine_landmarks is ever
enabled the indices below still exist, but double-check before relying on them.
"""

import time
import random
import numpy as np
import cv2


class ChallengeResponse:
    # ── Challenge kinds ───────────────────────────────────────────
    BLINK = "BLINK"
    TURN_LEFT = "TURN HEAD LEFT"
    TURN_RIGHT = "TURN HEAD RIGHT"
    OPEN_MOUTH = "OPEN YOUR MOUTH"
    SMILE = "SMILE"
    RAISE_EYEBROWS = "RAISE EYEBROWS"
    NOD = "NOD YOUR HEAD"

    # Full pool a random sequence is drawn from.
    ALL_KINDS = [BLINK, TURN_LEFT, TURN_RIGHT, OPEN_MOUTH,
                 SMILE, RAISE_EYEBROWS, NOD]

    # States (whole-sequence)
    PENDING = "PENDING"    # no sequence active yet
    WAITING = "WAITING"    # a step is issued, awaiting response
    PASSED = "PASSED"      # every step passed
    FAILED = "FAILED"      # a step timed out

    # ── 468-pt FaceMesh landmark indices ──────────────────────────
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # p1..p6
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    # Mouth
    MOUTH_TOP = 13          # upper inner lip
    MOUTH_BOTTOM = 14       # lower inner lip
    MOUTH_LEFT = 78         # inner left corner
    MOUTH_RIGHT = 308       # inner right corner
    MOUTH_CORNER_L = 61     # outer left corner (smile width)
    MOUTH_CORNER_R = 291    # outer right corner
    # Eyebrows + eye tops (eyebrow-raise)
    BROW_R = 105
    BROW_L = 334
    EYE_TOP_R = 159
    EYE_TOP_L = 386
    # Vertical face span (pitch / nod normalisation)
    FOREHEAD = 10
    CHIN = 152

    def __init__(self, fps: float = 30.0, response_timeout_s: float = 4.0,
                 num_challenges: int = 3,
                 ear_threshold: float = 0.20,
                 turn_ratio_threshold: float = 0.15,
                 mouth_open_threshold: float = 0.45,
                 smile_gain: float = 0.18,
                 brow_gain: float = 0.14,
                 nod_threshold: float = 0.06,
                 baseline_frames: int = 6):
        self.fps = fps
        self.response_timeout_s = response_timeout_s
        self.num_challenges = max(1, int(num_challenges))

        # Per-kind detection thresholds
        self.ear_threshold = ear_threshold
        # how far the yaw ratio must move from center (0.5) to count as a turn
        self.turn_ratio_threshold = turn_ratio_threshold
        # mouth aspect ratio above which the mouth counts as "open"
        self.mouth_open_threshold = mouth_open_threshold
        # relative widening / raising required (vs. captured neutral baseline)
        self.smile_gain = smile_gain
        self.brow_gain = brow_gain
        # normalised vertical nose travel required for a nod
        self.nod_threshold = nod_threshold
        self.baseline_frames = baseline_frames

        self.state = self.PENDING
        self.sequence = []
        self.step_index = 0
        self.kind = None
        self._deadline = 0.0
        self._results = []   # per-step outcome: True=completed, False=skipped
        self._init_step_state()

    # ── per-step transient state ──────────────────────────────────
    def _init_step_state(self):
        # blink debounce (need open→closed→open)
        self._eye_was_open = False
        self._saw_closed = False
        # nod debounce (need neutral→down→back)
        self._saw_down = False
        # baseline capture for relative challenges (smile / brows)
        self._baseline_samples = []
        self._baseline = None

    # ──────────────────────────────────────────────
    def _random_sequence(self, n: int):
        """Pick n kinds at random with no two identical in a row."""
        seq = []
        for _ in range(n):
            choices = [k for k in self.ALL_KINDS if not seq or k != seq[-1]]
            seq.append(random.choice(choices))
        return seq

    def start_challenge(self, kind: str = None, now: float = None):
        """
        Begin a challenge sequence and start the first step's timer.

        If `kind` is given, a single-step sequence with that kind is issued
        (backwards-compatible). Otherwise a fresh random sequence of
        `num_challenges` steps is generated.
        """
        now = time.time() if now is None else now
        if kind is not None:
            self.sequence = [kind]
        else:
            self.sequence = self._random_sequence(self.num_challenges)
        self.step_index = 0
        self.kind = self.sequence[0]
        self.state = self.WAITING
        self._deadline = now + self.response_timeout_s
        self._results = []
        self._init_step_state()

    def _resolve_step(self, completed: bool, now: float):
        """
        Record the current step's outcome and move on. After the final step,
        the sequence PASSES only if every step was completed; a single missed
        (skipped) step yields FAILED.
        """
        self._results.append(completed)
        self.step_index += 1
        if self.step_index >= len(self.sequence):
            self.state = self.PASSED if all(self._results) else self.FAILED
            return
        self.kind = self.sequence[self.step_index]
        self._deadline = now + self.response_timeout_s
        self._init_step_state()

    def is_active(self) -> bool:
        return self.state == self.WAITING

    def passed(self) -> bool:
        return self.state == self.PASSED

    # ── geometry helpers ──────────────────────────────────────────
    @staticmethod
    def eye_aspect_ratio(landmarks, idx, w, h) -> float:
        """EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)."""
        pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in idx]
        p1, p2, p3, p4, p5, p6 = pts
        vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horiz = 2.0 * np.linalg.norm(p1 - p4) + 1e-6
        return float(vert / horiz)

    def head_yaw_ratio(self, landmarks, w, h) -> float:
        """
        Yaw proxy in [0,1]: nose horizontal position between the two outer eye
        corners. ~0.5 when facing forward; shifts toward 0 or 1 on turn.
        """
        nose = landmarks[self.NOSE_TIP].x * w
        left = landmarks[self.LEFT_EYE_OUTER].x * w
        right = landmarks[self.RIGHT_EYE_OUTER].x * w
        span = (right - left)
        if abs(span) < 1e-6:
            return 0.5
        return float((nose - left) / span)

    def _eye_span(self, landmarks, w, h) -> float:
        """Pixel distance between the outer eye corners (scale reference)."""
        l = np.array([landmarks[self.LEFT_EYE_OUTER].x * w,
                      landmarks[self.LEFT_EYE_OUTER].y * h])
        r = np.array([landmarks[self.RIGHT_EYE_OUTER].x * w,
                      landmarks[self.RIGHT_EYE_OUTER].y * h])
        return float(np.linalg.norm(r - l)) + 1e-6

    def mouth_aspect_ratio(self, landmarks, w, h) -> float:
        """Vertical lip gap over mouth width — rises sharply when mouth opens."""
        top = np.array([landmarks[self.MOUTH_TOP].x * w, landmarks[self.MOUTH_TOP].y * h])
        bot = np.array([landmarks[self.MOUTH_BOTTOM].x * w, landmarks[self.MOUTH_BOTTOM].y * h])
        left = np.array([landmarks[self.MOUTH_LEFT].x * w, landmarks[self.MOUTH_LEFT].y * h])
        right = np.array([landmarks[self.MOUTH_RIGHT].x * w, landmarks[self.MOUTH_RIGHT].y * h])
        width = np.linalg.norm(right - left) + 1e-6
        return float(np.linalg.norm(top - bot) / width)

    def smile_ratio(self, landmarks, w, h) -> float:
        """Mouth corner-to-corner width normalised by eye span (widens on smile)."""
        l = np.array([landmarks[self.MOUTH_CORNER_L].x * w, landmarks[self.MOUTH_CORNER_L].y * h])
        r = np.array([landmarks[self.MOUTH_CORNER_R].x * w, landmarks[self.MOUTH_CORNER_R].y * h])
        return float(np.linalg.norm(r - l) / self._eye_span(landmarks, w, h))

    def brow_ratio(self, landmarks, w, h) -> float:
        """Average brow-to-eye-top gap normalised by eye span (rises when raised)."""
        gap_r = abs(landmarks[self.BROW_R].y - landmarks[self.EYE_TOP_R].y) * h
        gap_l = abs(landmarks[self.BROW_L].y - landmarks[self.EYE_TOP_L].y) * h
        return float((gap_r + gap_l) / 2.0 / self._eye_span(landmarks, w, h))

    def nose_pitch_ratio(self, landmarks, w, h) -> float:
        """Nose-tip vertical position between forehead and chin (0..1)."""
        nose = landmarks[self.NOSE_TIP].y
        top = landmarks[self.FOREHEAD].y
        bottom = landmarks[self.CHIN].y
        span = (bottom - top)
        if abs(span) < 1e-6:
            return 0.5
        return float((nose - top) / span)

    def _capture_baseline(self, value: float) -> bool:
        """Accumulate a neutral baseline; returns True once it's ready."""
        if self._baseline is None:
            self._baseline_samples.append(value)
            if len(self._baseline_samples) >= self.baseline_frames:
                self._baseline = float(np.median(self._baseline_samples))
            return self._baseline is not None
        return True

    # ──────────────────────────────────────────────
    def update(self, landmarks, w, h, now: float = None) -> str:
        """
        Evaluate the current frame against the active step. When a step is
        satisfied, automatically advance to the next; the sequence reaches
        PASSED only after the final step.

        Note the camera feed is mirrored (cv2.flip) in the demo, so a user
        turning their physical head LEFT moves their nose toward the RIGHT side
        of the mirrored frame; the yaw direction below accounts for that.
        """
        if self.state != self.WAITING:
            return self.state

        now = time.time() if now is None else now
        if now > self._deadline:
            # This step's time budget ran out: SKIP it (recorded as a miss)
            # and issue the next step. A miss makes the final verdict FAILED.
            self._resolve_step(False, now)
            return self.state

        if landmarks is None:
            return self.state

        step_ok = False

        if self.kind == self.BLINK:
            ear_r = self.eye_aspect_ratio(landmarks, self.RIGHT_EYE, w, h)
            ear_l = self.eye_aspect_ratio(landmarks, self.LEFT_EYE, w, h)
            ear = (ear_r + ear_l) / 2.0
            # Debounced blink: require eyes open, then closed, then open again
            if ear > self.ear_threshold:
                if self._saw_closed:
                    step_ok = True
                self._eye_was_open = True
            elif ear < self.ear_threshold and self._eye_was_open:
                self._saw_closed = True

        elif self.kind in (self.TURN_LEFT, self.TURN_RIGHT):
            yaw = self.head_yaw_ratio(landmarks, w, h)
            # Mirrored frame: physical LEFT turn => nose moves to higher ratio
            if self.kind == self.TURN_LEFT and yaw > 0.5 + self.turn_ratio_threshold:
                step_ok = True
            elif self.kind == self.TURN_RIGHT and yaw < 0.5 - self.turn_ratio_threshold:
                step_ok = True

        elif self.kind == self.OPEN_MOUTH:
            if self.mouth_aspect_ratio(landmarks, w, h) > self.mouth_open_threshold:
                step_ok = True

        elif self.kind == self.SMILE:
            value = self.smile_ratio(landmarks, w, h)
            if self._capture_baseline(value):
                if value > self._baseline * (1.0 + self.smile_gain):
                    step_ok = True

        elif self.kind == self.RAISE_EYEBROWS:
            value = self.brow_ratio(landmarks, w, h)
            if self._capture_baseline(value):
                if value > self._baseline * (1.0 + self.brow_gain):
                    step_ok = True

        elif self.kind == self.NOD:
            value = self.nose_pitch_ratio(landmarks, w, h)
            if self._capture_baseline(value):
                # Debounced nod: dip the head down past threshold, then return
                if value > self._baseline + self.nod_threshold:
                    self._saw_down = True
                elif self._saw_down and value < self._baseline + self.nod_threshold * 0.5:
                    step_ok = True

        if step_ok:
            self._resolve_step(True, now)

        return self.state

    # ──────────────────────────────────────────────
    def draw(self, frame, now: float = None):
        """Render the current step prompt, progress + countdown (VISIBLE)."""
        now = time.time() if now is None else now
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.state == self.WAITING:
            total = len(self.sequence)
            step = self.step_index + 1
            remaining = max(0.0, self._deadline - now)
            prompt = f"CHALLENGE {step}/{total}: {self.kind}"

            # Centered banner near the top
            (tw, _), _ = cv2.getTextSize(prompt, font, 0.9, 2)
            cx = (w - tw) // 2
            cv2.rectangle(frame, (cx - 15, 20), (cx + tw + 15, 80), (20, 20, 20), -1)
            cv2.putText(frame, prompt, (cx, 55), font, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

            # Countdown bar
            frac = remaining / self.response_timeout_s
            cv2.rectangle(frame, (cx - 15, 85),
                          (cx - 15 + int((tw + 30) * frac), 92),
                          (0, 220, 255), -1)

            # Progress dots: filled = completed steps, hollow = remaining
            dot_r = 7
            gap = 24
            start_x = w // 2 - (total - 1) * gap // 2
            for i in range(total):
                center = (start_x + i * gap, 110)
                if i < len(self._results):
                    color = (0, 255, 100) if self._results[i] else (0, 0, 255)
                    cv2.circle(frame, center, dot_r, color, -1)           # done / skipped
                elif i == self.step_index:
                    cv2.circle(frame, center, dot_r, (0, 220, 255), 2)    # current
                else:
                    cv2.circle(frame, center, dot_r, (120, 120, 120), 1)  # pending

        elif self.state == self.PASSED:
            cv2.putText(frame, "ALL CHALLENGES PASSED", (w // 2 - 160, 55),
                        font, 0.8, (0, 255, 100), 2, cv2.LINE_AA)
        elif self.state == self.FAILED:
            cv2.putText(frame, "CHALLENGE FAILED", (w // 2 - 120, 55),
                        font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    def reset(self):
        self.state = self.PENDING
        self.sequence = []
        self.step_index = 0
        self.kind = None
        self._deadline = 0.0
        self._results = []   # per-step outcome: True=completed, False=skipped
        self._init_step_state()
