"""
Bio-Pulse Authenticator — Full integrated demo (Hours 1–3).

Combines:
  - Webcam capture + MediaPipe Face Mesh (Hour 0)
  - PulseExtractor: bandpass filter + FFT BPM (Hour 1–2)
  - LivenessDetector: real human vs spoof decision (Hour 2–3)
  - Live waveform overlay + BPM dashboard (Hour 3)

Run:  python bio_pulse_demo.py
Quit: press 'q'
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from pulse_extractor import PulseExtractor
from liveness_detector import LivenessDetector

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MEASURED_FPS = 30.0  # From hour0_debug.py output
BUFFER_SECONDS = 10  # Rolling window for pulse analysis
CAMERA_INDEX = 0

# ──────────────────────────────────────────────
# Initialize components
# ──────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

pulse_extractor = PulseExtractor(fps=MEASURED_FPS, buffer_seconds=BUFFER_SECONDS)
liveness = LivenessDetector(
    bpm_window=8,
    snr_threshold=3.0,
    bpm_range=(45, 180),
    min_readings=5,
)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS measurement
frame_count = 0
start_time = time.time()
actual_fps = MEASURED_FPS

# Face-lost counter: reset signals if face missing for too long
face_lost_frames = 0
FACE_LOST_THRESHOLD = 30  # ~1 second at 30fps


def get_forehead_roi(landmarks, frame_w, frame_h):
    """Compute forehead ROI from face mesh landmarks."""
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    face_left, face_right = min(xs), max(xs)
    face_top, face_bottom = min(ys), max(ys)
    face_w = face_right - face_left
    face_h = face_bottom - face_top

    roi_x1 = int(face_left + 0.32 * face_w)
    roi_x2 = int(face_left + 0.68 * face_w)
    roi_y1 = int(face_top + 0.06 * face_h)
    roi_y2 = int(face_top + 0.20 * face_h)

    return roi_x1, roi_y1, roi_x2, roi_y2


def draw_waveform(frame, filtered_signal, x_start, y_center, width, height, color):
    """
    Draw the filtered pulse waveform on the frame.

    Draws the last `width` samples of the filtered signal,
    scaled to fit within the given bounding box.
    """
    if len(filtered_signal) < 2:
        return

    # Take the last `width` samples (or fewer if not enough data)
    sig = filtered_signal[-width:]
    n = len(sig)

    if n < 2:
        return

    # Normalize signal to fit in the box
    sig_min, sig_max = sig.min(), sig.max()
    sig_range = sig_max - sig_min
    if sig_range < 1e-6:
        sig_range = 1.0  # avoid division by zero (flat signal)

    # Map signal values to y-pixel coordinates
    points = []
    for i in range(n):
        x = x_start + int(i * width / n)
        # Invert y because pixel y increases downward
        y = int(y_center - ((sig[i] - sig_min) / sig_range - 0.5) * height)
        points.append((x, y))

    # Draw the waveform as connected lines
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, 2, cv2.LINE_AA)


def draw_dashboard(frame, bpm, snr, decision, confidence, actual_fps):
    """
    Draw a semi-transparent dashboard overlay on the frame.
    Shows BPM, SNR, liveness decision, and FPS.
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark panel on the right side
    panel_w = 250
    panel_x = w - panel_w
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Vertical line separator
    cv2.line(frame, (panel_x, 0), (panel_x, h), (60, 60, 60), 2)

    x = panel_x + 15
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(frame, "BIO-PULSE", (x, 35), font, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "AUTHENTICATOR", (x, 60), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (x, 75), (w - 15, 75), (60, 60, 60), 1)

    # BPM display
    cv2.putText(frame, "HEART RATE", (x, 100), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    if bpm > 0:
        bpm_color = (0, 255, 100) if 45 <= bpm <= 180 else (0, 0, 255)
        cv2.putText(frame, f"{bpm:.0f}", (x, 145), font, 1.5, bpm_color, 3, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (x + 90, 145), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "---", (x, 145), font, 1.5, (100, 100, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (x + 90, 145), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    # SNR display
    cv2.putText(frame, "SIGNAL QUALITY", (x, 175), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    if snr > 0:
        # SNR bar
        bar_w = panel_w - 30
        bar_fill = min(1.0, snr / 10.0)
        bar_color = (0, 255, 100) if snr >= 3.0 else (0, 165, 255) if snr >= 1.5 else (0, 0, 255)
        cv2.rectangle(frame, (x, 185), (x + bar_w, 200), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, 185), (x + int(bar_w * bar_fill), 200), bar_color, -1)
        cv2.putText(frame, f"{snr:.1f} dB", (x + bar_w - 65, 197), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (x, 215), (w - 15, 215), (60, 60, 60), 1)

    # Liveness decision — big, colored text
    cv2.putText(frame, "LIVENESS CHECK", (x, 240), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    if decision == LivenessDetector.ALIVE:
        dec_color = (0, 255, 100)
        # Draw a pulsing green circle
        pulse_radius = int(8 + 4 * np.sin(time.time() * 4))
        cv2.circle(frame, (x + 10, 270), pulse_radius, dec_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "ACCESS", (x + 25, 268), font, 0.6, dec_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "GRANTED", (x + 25, 290), font, 0.6, dec_color, 2, cv2.LINE_AA)
    elif decision == LivenessDetector.DENIED:
        dec_color = (0, 0, 255)
        cv2.circle(frame, (x + 10, 270), 10, dec_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "ACCESS", (x + 25, 268), font, 0.6, dec_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "DENIED", (x + 25, 290), font, 0.6, dec_color, 2, cv2.LINE_AA)
    else:
        dec_color = (0, 165, 255)
        # Scanning animation: rotating dots
        angle = time.time() * 3
        for i in range(3):
            a = angle + i * 2.094  # 120 degrees apart
            dx = int(8 * np.cos(a))
            dy = int(8 * np.sin(a))
            cv2.circle(frame, (x + 10 + dx, 275 + dy), 3, dec_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "SCANNING...", (x + 25, 280), font, 0.6, dec_color, 1, cv2.LINE_AA)

        # Progress bar for scanning
        if confidence > 0:
            bar_w = panel_w - 30
            cv2.rectangle(frame, (x, 295), (x + bar_w, 305), (50, 50, 50), -1)
            cv2.rectangle(frame, (x, 295), (x + int(bar_w * confidence), 305), dec_color, -1)

    # Divider
    cv2.line(frame, (x, 320), (w - 15, 320), (60, 60, 60), 1)

    # FPS and buffer status
    cv2.putText(frame, f"FPS: {actual_fps:.1f}", (x, 345), font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    buf_pct = min(100, int(pulse_extractor.get_signal_length() / pulse_extractor.buffer_size * 100))
    cv2.putText(frame, f"Buffer: {buf_pct}%", (x, 365), font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (x, h - 20), font, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press 'r' to reset", (x, h - 40), font, 0.4, (80, 80, 80), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
print("=" * 50)
print("  Bio-Pulse Authenticator — Live Demo")
print("  Press 'q' to quit, 'r' to reset")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    bpm, snr, filtered = 0.0, 0.0, np.array([])
    decision = liveness.decision
    confidence = liveness.confidence

    if results.multi_face_landmarks:
        face_lost_frames = 0
        landmarks = results.multi_face_landmarks[0].landmark
        x1, y1, x2, y2 = get_forehead_roi(landmarks, w, h)

        # Draw ROI rectangle (green if tracking, cyan for the region)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Extract green channel mean from ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            green_mean = roi[:, :, 1].mean()
            pulse_extractor.add_sample(green_mean)

            # Get BPM estimate
            bpm, snr, filtered = pulse_extractor.get_bpm()

            # Update liveness decision
            decision = liveness.update(bpm, snr)
            status = liveness.get_status()
            confidence = status["confidence"]

        # Draw waveform overlay on the video feed (bottom left)
        waveform_y = h - 80
        waveform_h = 100
        waveform_w = w - 280  # leave room for dashboard

        # Semi-transparent background for waveform area
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, waveform_y - waveform_h // 2 - 10),
                       (waveform_w + 20, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "PULSE WAVEFORM", (10, waveform_y - waveform_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        if len(filtered) > 0:
            # Green waveform for real pulse, red for flat/denied
            wave_color = (0, 255, 100) if decision == LivenessDetector.ALIVE else \
                         (0, 0, 255) if decision == LivenessDetector.DENIED else \
                         (0, 200, 255)
            draw_waveform(frame, filtered, 10, waveform_y, waveform_w, waveform_h, wave_color)
    else:
        # No face detected
        face_lost_frames += 1
        if face_lost_frames > FACE_LOST_THRESHOLD:
            pulse_extractor.reset()
            liveness.reset()
            decision = liveness.decision
            confidence = 0.0

        cv2.putText(frame, "No face detected — look at the camera",
                    (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)

    # Draw the dashboard panel
    draw_dashboard(frame, bpm, snr, decision, confidence, actual_fps)

    # Update FPS measurement
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 2:
        actual_fps = frame_count / elapsed
        # Reset counters periodically to get recent FPS
        if frame_count > 300:
            frame_count = 0
            start_time = time.time()

    cv2.imshow("Bio-Pulse Authenticator", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        pulse_extractor.reset()
        liveness.reset()
        print("[RESET] Signal buffer and liveness cleared.")

print(f"\nFinal FPS: {actual_fps:.1f}")
status = liveness.get_status()
print(f"Last decision: {status['decision']}")
print(f"Last BPM: {status['avg_bpm']:.1f}")
print(f"Last SNR: {status['avg_snr']:.1f} dB")

cap.release()
cv2.destroyAllWindows()
