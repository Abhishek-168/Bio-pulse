"""
ChallengeResponse — active liveness challenge (VISIBLE on the camera feed).

The pulse-based checks are all PASSIVE: a high-quality pre-recorded video of a
compliant person could, in principle, satisfy them. An active challenge closes
that gap by demanding a real-time, randomly-chosen action ("blink", "turn your
head left") and verifying the response from MediaPipe landmarks within a short
timeout. A static replay cannot comply on demand.

Unlike ScreenDetector (which is deliberately silent), this module SHOULD draw
on the feed — it's a prompt the genuine user needs to see.

Landmark indices assume the 468-point FaceLandmarker model with iris refine
OFF (matching bio_pulse_demo.py's configuration). If refine_landmarks is ever
enabled the eye indices below still exist, but double-check before relying on
them.
"""

import time
import random
import numpy as np
import cv2


class ChallengeResponse:
    # Challenge kinds
    BLINK = "BLINK"
    TURN_LEFT = "TURN HEAD LEFT"
    TURN_RIGHT = "TURN HEAD RIGHT"

    # States
    PENDING = "PENDING"    # no challenge active yet
    WAITING = "WAITING"    # challenge issued, awaiting response
    PASSED = "PASSED"
    FAILED = "FAILED"

    # 468-pt FaceMesh eye landmark indices (p1..p6 per eye)
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    def __init__(self, fps: float = 30.0, response_timeout_s: float = 4.0,
                 ear_threshold: float = 0.20,
                 turn_ratio_threshold: float = 0.15):
        self.fps = fps
        self.response_timeout_s = response_timeout_s
        self.ear_threshold = ear_threshold
        # how far the yaw ratio must move from center (0.5) to count as a turn
        self.turn_ratio_threshold = turn_ratio_threshold

        self.state = self.PENDING
        self.kind = None
        self._deadline = 0.0
        self._eye_was_open = False  # for blink debounce (need open→closed→open)
        self._saw_closed = False

    # ──────────────────────────────────────────────
    def start_challenge(self, kind: str = None, now: float = None):
        """Issue a (random unless specified) challenge and start the timer."""
        now = time.time() if now is None else now
        self.kind = kind or random.choice([self.BLINK, self.TURN_LEFT, self.TURN_RIGHT])
        self.state = self.WAITING
        self._deadline = now + self.response_timeout_s
        self._eye_was_open = False
        self._saw_closed = False

    def is_active(self) -> bool:
        return self.state == self.WAITING

    def passed(self) -> bool:
        return self.state == self.PASSED

    # ──────────────────────────────────────────────
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

    # ──────────────────────────────────────────────
    def update(self, landmarks, w, h, now: float = None) -> str:
        """
        Evaluate the current frame against the active challenge.

        Returns the current state. Note the camera feed is mirrored
        (cv2.flip) in the demo, so a user turning their physical head LEFT
        moves their nose toward the RIGHT side of the mirrored frame; the
        ratio direction below accounts for that.
        """
        if self.state != self.WAITING:
            return self.state

        now = time.time() if now is None else now
        if now > self._deadline:
            self.state = self.FAILED
            return self.state

        if landmarks is None:
            return self.state

        if self.kind == self.BLINK:
            ear_r = self.eye_aspect_ratio(landmarks, self.RIGHT_EYE, w, h)
            ear_l = self.eye_aspect_ratio(landmarks, self.LEFT_EYE, w, h)
            ear = (ear_r + ear_l) / 2.0
            # Debounced blink: require eyes open, then closed, then open again
            if ear > self.ear_threshold:
                if self._saw_closed:
                    self.state = self.PASSED
                self._eye_was_open = True
            elif ear < self.ear_threshold and self._eye_was_open:
                self._saw_closed = True

        elif self.kind in (self.TURN_LEFT, self.TURN_RIGHT):
            yaw = self.head_yaw_ratio(landmarks, w, h)
            # Mirrored frame: physical LEFT turn => nose moves to higher ratio
            if self.kind == self.TURN_LEFT and yaw > 0.5 + self.turn_ratio_threshold:
                self.state = self.PASSED
            elif self.kind == self.TURN_RIGHT and yaw < 0.5 - self.turn_ratio_threshold:
                self.state = self.PASSED

        return self.state

    # ──────────────────────────────────────────────
    def draw(self, frame, now: float = None):
        """Render the challenge prompt + countdown on the feed (VISIBLE)."""
        now = time.time() if now is None else now
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.state == self.WAITING:
            remaining = max(0.0, self._deadline - now)
            prompt = f"CHALLENGE: {self.kind}"
            # Centered banner near the top
            (tw, _), _ = cv2.getTextSize(prompt, font, 0.9, 2)
            cx = (w - tw) // 2
            cv2.rectangle(frame, (cx - 15, 20), (cx + tw + 15, 80), (20, 20, 20), -1)
            cv2.putText(frame, prompt, (cx, 55), font, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
            # Countdown bar
            frac = remaining / self.response_timeout_s
            cv2.rectangle(frame, (cx - 15, 85), (cx - 15 + int((tw + 30) * frac), 92),
                          (0, 220, 255), -1)
        elif self.state == self.PASSED:
            cv2.putText(frame, "CHALLENGE PASSED", (w // 2 - 120, 55),
                        font, 0.8, (0, 255, 100), 2, cv2.LINE_AA)
        elif self.state == self.FAILED:
            cv2.putText(frame, "CHALLENGE FAILED", (w // 2 - 120, 55),
                        font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    def reset(self):
        self.state = self.PENDING
        self.kind = None
        self._deadline = 0.0
        self._eye_was_open = False
        self._saw_closed = False
