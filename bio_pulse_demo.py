"""
Bio-Pulse Authenticator — Full integrated demo (Hours 1–3).

Combines:
  - Webcam capture + MediaPipe Face Mesh (Hour 0)
  - PulseExtractor: bandpass filter + FFT BPM (Hour 1–2)
  - LivenessDetector: multi-check anti-spoof decision (Hour 2–3)
  - Dual-ROI cross-correlation (forehead + cheek) for photo detection
  - Live waveform overlay + BPM dashboard (Hour 3)

Run:  python bio_pulse_demo.py
Quit: press 'q'
Reset: press 'r'
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

# Two pulse extractors: forehead (primary) and cheek (anti-spoof)
pulse_forehead = PulseExtractor(fps=MEASURED_FPS, buffer_seconds=BUFFER_SECONDS)
pulse_cheek = PulseExtractor(fps=MEASURED_FPS, buffer_seconds=BUFFER_SECONDS)

liveness = LivenessDetector(
    bpm_window=8,
    snr_threshold=4.0,
    bpm_range=(50, 160),
    min_readings=5,
    regularity_threshold=0.35,
    correlation_threshold=0.4,
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


def get_cheek_roi(landmarks, frame_w, frame_h):
    """
    Compute left cheek ROI from face mesh landmarks.

    The cheek is a second skin region for cross-correlation.
    A real pulse appears in BOTH forehead and cheek (blood flows
    everywhere). A photo's noise is random per-region and won't
    correlate between the two ROIs.
    """
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    face_left, face_right = min(xs), max(xs)
    face_top, face_bottom = min(ys), max(ys)
    face_w = face_right - face_left
    face_h = face_bottom - face_top

    # Left cheek: lower-left area of the face
    roi_x1 = int(face_left + 0.10 * face_w)
    roi_x2 = int(face_left + 0.35 * face_w)
    roi_y1 = int(face_top + 0.50 * face_h)
    roi_y2 = int(face_top + 0.70 * face_h)

    return roi_x1, roi_y1, roi_x2, roi_y2


def compute_cross_correlation(signal_a, signal_b):
    """
    Compute normalized cross-correlation between two signals.

    Returns a value between -1 and 1. A real pulse on a living
    face will show correlation > 0.4 between forehead and cheek.
    A photo will show near-zero correlation (uncorrelated noise).
    """
    if len(signal_a) < 30 or len(signal_b) < 30:
        return 0.0

    # Use the shorter length
    n = min(len(signal_a), len(signal_b))
    a = signal_a[-n:]
    b = signal_b[-n:]

    # Normalize
    a = a - np.mean(a)
    b = b - np.mean(b)

    std_a = np.std(a)
    std_b = np.std(b)

    if std_a < 1e-8 or std_b < 1e-8:
        return 0.0

    # Pearson correlation
    correlation = np.sum(a * b) / (n * std_a * std_b)
    return float(np.clip(correlation, -1.0, 1.0))


def draw_waveform(frame, filtered_signal, x_start, y_center, width, height, color):
    """Draw the filtered pulse waveform on the frame."""
    if len(filtered_signal) < 2:
        return

    sig = filtered_signal[-width:]
    n = len(sig)
    if n < 2:
        return

    sig_min, sig_max = sig.min(), sig.max()
    sig_range = sig_max - sig_min
    if sig_range < 1e-6:
        sig_range = 1.0

    points = []
    for i in range(n):
        x = x_start + int(i * width / n)
        y = int(y_center - ((sig[i] - sig_min) / sig_range - 0.5) * height)
        points.append((x, y))

    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, 2, cv2.LINE_AA)


def draw_dashboard(frame, bpm, snr, decision, confidence, actual_fps,
                   regularity, correlation, check_details):
    """
    Draw a semi-transparent dashboard overlay on the frame.
    Shows BPM, SNR, liveness decision, anti-spoof checks, and FPS.
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark panel on the right side
    panel_w = 260
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
        bpm_color = (0, 255, 100) if 50 <= bpm <= 160 else (0, 0, 255)
        cv2.putText(frame, f"{bpm:.0f}", (x, 145), font, 1.5, bpm_color, 3, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (x + 90, 145), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "---", (x, 145), font, 1.5, (100, 100, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (x + 90, 145), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    # SNR display
    cv2.putText(frame, "SIGNAL QUALITY", (x, 175), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    if snr > 0:
        bar_w = panel_w - 30
        bar_fill = min(1.0, snr / 12.0)
        bar_color = (0, 255, 100) if snr >= 4.0 else (0, 165, 255) if snr >= 2.0 else (0, 0, 255)
        cv2.rectangle(frame, (x, 185), (x + bar_w, 200), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, 185), (x + int(bar_w * bar_fill), 200), bar_color, -1)
        cv2.putText(frame, f"{snr:.1f} dB", (x + bar_w - 65, 197), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (x, 210), (w - 15, 210), (60, 60, 60), 1)

    # Anti-spoof checks
    cv2.putText(frame, "ANTI-SPOOF CHECKS", (x, 230), font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    check_y = 250
    checks = [
        ("Peak Regularity", regularity, 0.35),
        ("ROI Correlation", correlation, 0.4),
        ("SNR Threshold", snr / 12.0 if snr > 0 else 0, 4.0 / 12.0),
    ]
    for label, value, threshold in checks:
        passed = value >= threshold
        icon_color = (0, 255, 100) if passed else (0, 0, 255)
        icon = "+" if passed else "x"
        cv2.putText(frame, icon, (x, check_y), font, 0.4, icon_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{label}: {value:.2f}", (x + 18, check_y), font, 0.35,
                    (200, 200, 200), 1, cv2.LINE_AA)
        check_y += 20

    # Divider
    cv2.line(frame, (x, check_y + 5), (w - 15, check_y + 5), (60, 60, 60), 1)

    # Liveness decision — big, colored text
    dec_y = check_y + 25
    cv2.putText(frame, "LIVENESS CHECK", (x, dec_y), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    if decision == LivenessDetector.ALIVE:
        dec_color = (0, 255, 100)
        pulse_radius = int(8 + 4 * np.sin(time.time() * 4))
        cv2.circle(frame, (x + 10, dec_y + 30), pulse_radius, dec_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "ACCESS", (x + 25, dec_y + 28), font, 0.6, dec_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "GRANTED", (x + 25, dec_y + 50), font, 0.6, dec_color, 2, cv2.LINE_AA)
    elif decision == LivenessDetector.DENIED:
        dec_color = (0, 0, 255)
        cv2.circle(frame, (x + 10, dec_y + 30), 10, dec_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "ACCESS", (x + 25, dec_y + 28), font, 0.6, dec_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "DENIED", (x + 25, dec_y + 50), font, 0.6, dec_color, 2, cv2.LINE_AA)
    else:
        dec_color = (0, 165, 255)
        angle = time.time() * 3
        for i in range(3):
            a = angle + i * 2.094
            dx = int(8 * np.cos(a))
            dy = int(8 * np.sin(a))
            cv2.circle(frame, (x + 10 + dx, dec_y + 35 + dy), 3, dec_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "SCANNING...", (x + 25, dec_y + 40), font, 0.6, dec_color, 1, cv2.LINE_AA)

        if confidence > 0:
            bar_w = panel_w - 30
            cv2.rectangle(frame, (x, dec_y + 55), (x + bar_w, dec_y + 65), (50, 50, 50), -1)
            cv2.rectangle(frame, (x, dec_y + 55), (x + int(bar_w * confidence), dec_y + 65), dec_color, -1)

    # FPS and buffer status at bottom
    cv2.putText(frame, f"FPS: {actual_fps:.1f}", (x, h - 40), font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    buf_pct = min(100, int(pulse_forehead.get_signal_length() / pulse_forehead.buffer_size * 100))
    cv2.putText(frame, f"Buffer: {buf_pct}%", (x + 100, h - 40), font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "'q' quit  'r' reset", (x, h - 15), font, 0.35, (80, 80, 80), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
print("=" * 50)
print("  Bio-Pulse Authenticator — Live Demo")
print("  Press 'q' to quit, 'r' to reset")
print("  Anti-spoof: dual-ROI + peak regularity")
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
    regularity_score = 0.0
    correlation_score = 0.0
    decision = liveness.decision
    confidence = liveness.confidence
    check_details = liveness._check_details

    if results.multi_face_landmarks:
        face_lost_frames = 0
        landmarks = results.multi_face_landmarks[0].landmark

        # --- Forehead ROI (primary signal) ---
        fx1, fy1, fx2, fy2 = get_forehead_roi(landmarks, w, h)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
        cv2.putText(frame, "Forehead", (fx1, fy1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Cheek ROI (anti-spoof cross-correlation) ---
        cx1, cy1, cx2, cy2 = get_cheek_roi(landmarks, w, h)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 200, 0), 2)
        cv2.putText(frame, "Cheek", (cx1, cy1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 200, 0), 1, cv2.LINE_AA)

        # Extract green channel means from both ROIs
        forehead_roi = frame[fy1:fy2, fx1:fx2]
        cheek_roi = frame[cy1:cy2, cx1:cx2]

        if forehead_roi.size > 0 and cheek_roi.size > 0:
            forehead_green = forehead_roi[:, :, 1].mean()
            cheek_green = cheek_roi[:, :, 1].mean()

            pulse_forehead.add_sample(forehead_green)
            pulse_cheek.add_sample(cheek_green)

            # Get BPM estimate from forehead (primary)
            bpm, snr, filtered = pulse_forehead.get_bpm()

            # Get peak regularity from forehead signal
            peak_count, regularity_score, peak_amplitude = pulse_forehead.get_peak_regularity()

            # Cross-correlation between forehead and cheek filtered signals
            forehead_filtered = pulse_forehead.get_filtered_signal()
            cheek_filtered = pulse_cheek.get_filtered_signal()
            correlation_score = compute_cross_correlation(forehead_filtered, cheek_filtered)

            # Update liveness decision with all metrics
            decision = liveness.update(bpm, snr, regularity_score, correlation_score)
            status = liveness.get_status()
            confidence = status["confidence"]
            check_details = status.get("check_details", {})

        # Draw waveform overlay on the video feed (bottom left)
        waveform_y = h - 80
        waveform_h = 100
        waveform_w = w - 290

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, waveform_y - waveform_h // 2 - 10),
                       (waveform_w + 20, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "PULSE WAVEFORM", (10, waveform_y - waveform_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        if len(filtered) > 0:
            wave_color = (0, 255, 100) if decision == LivenessDetector.ALIVE else \
                         (0, 0, 255) if decision == LivenessDetector.DENIED else \
                         (0, 200, 255)
            draw_waveform(frame, filtered, 10, waveform_y, waveform_w, waveform_h, wave_color)
    else:
        # No face detected
        face_lost_frames += 1
        if face_lost_frames > FACE_LOST_THRESHOLD:
            pulse_forehead.reset()
            pulse_cheek.reset()
            liveness.reset()
            decision = liveness.decision
            confidence = 0.0
            check_details = {}

        cv2.putText(frame, "No face detected — look at the camera",
                    (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)

    # Draw the dashboard panel
    draw_dashboard(frame, bpm, snr, decision, confidence, actual_fps,
                   regularity_score, correlation_score, check_details)

    # Update FPS measurement
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 2:
        actual_fps = frame_count / elapsed
        if frame_count > 300:
            frame_count = 0
            start_time = time.time()

    cv2.imshow("Bio-Pulse Authenticator", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        pulse_forehead.reset()
        pulse_cheek.reset()
        liveness.reset()
        print("[RESET] Signal buffer and liveness cleared.")

print(f"\nFinal FPS: {actual_fps:.1f}")
status = liveness.get_status()
print(f"Last decision: {status['decision']}")
print(f"Last BPM: {status['avg_bpm']:.1f}")
print(f"Last SNR: {status['avg_snr']:.1f} dB")
print(f"Last Regularity: {status['avg_regularity']:.2f}")
print(f"Last Correlation: {status['avg_correlation']:.2f}")

cap.release()
cv2.destroyAllWindows()
