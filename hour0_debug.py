import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

# Measure actual FPS — don't assume 30. You need the real number
# for the bandpass filter in the next stage.
frame_count = 0
start_time = time.time()
measured_fps = 30  # fallback


def get_forehead_roi(frame, landmarks, frame_w, frame_h):
    """
    Compute face bounding box from all landmarks, then carve out
    a forehead rectangle as a percentage of that box. This avoids
    depending on exact MediaPipe landmark index numbers, which is
    the easiest thing to get subtly wrong and waste an hour on.
    """
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    face_left, face_right = min(xs), max(xs)
    face_top, face_bottom = min(ys), max(ys)
    face_w = face_right - face_left
    face_h = face_bottom - face_top

    # Forehead: horizontally centered, upper portion of the face box.
    # Tune these percentages once you can see the box on your own face.
    roi_x1 = int(face_left + 0.32 * face_w)
    roi_x2 = int(face_left + 0.68 * face_w)
    roi_y1 = int(face_top + 0.06 * face_h)
    roi_y2 = int(face_top + 0.20 * face_h)

    return roi_x1, roi_y1, roi_x2, roi_y2


while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    frame = cv2.flip(frame, 1)  # mirror for natural display
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        x1, y1, x2, y2 = get_forehead_roi(frame, landmarks, w, h)

        # Draw the ROI box so you can visually confirm tracking
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # This crop is what Hour 1-2.5's PulseExtractor will consume
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            green_mean = roi[:, :, 1].mean()
            cv2.putText(frame, f"Green mean: {green_mean:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 2:
        measured_fps = frame_count / elapsed
        cv2.putText(frame, f"FPS: {measured_fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Bio-Pulse Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Final measured FPS: {measured_fps:.1f}")  # write this down, you need it next hour
cap.release()
cv2.destroyAllWindows()
