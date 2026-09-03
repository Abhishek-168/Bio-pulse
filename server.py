"""
Bio-Pulse Authenticator — FastAPI + WebSocket Streaming Server.

Connects the Bio-Pulse Python rPPG engine (pulse_extractor, liveness_detector,
screen_detector, challenge_response) directly to the web client via WebSockets.

Run:
    python server.py
    (Or: uvicorn server:app --host 0.0.0.0 --port 8000)
"""

import asyncio
import base64
import json
import os
import time
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import existing Bio-Pulse core components
from pulse_extractor import PulseExtractor, phase_coherence
from liveness_detector import LivenessDetector
from screen_detector import ScreenDetector
from challenge_response import ChallengeResponse

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

app = FastAPI(title="Bio-Pulse Telemetry API")

# Enable CORS for local dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Initialize MediaPipe Face Landmarker
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = FaceLandmarker.create_from_options(_options)

# MediaPipe face mesh landmark indices for ROIs
FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
CHEEK_INDICES = [205, 207, 214, 212, 138, 215, 177, 137, 227, 34, 143, 35, 124, 46]

@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "Bio-Pulse Authenticator Engine v1.0", "timestamp": time.time()}

class ConnectionSession:
    def __init__(self, fps=30.0, buffer_seconds=10):
        self.fps = fps
        self.pulse_forehead = PulseExtractor(fps=fps, buffer_seconds=buffer_seconds, channel_mode="pos")
        self.pulse_cheek = PulseExtractor(fps=fps, buffer_seconds=buffer_seconds, channel_mode="pos")
        self.liveness = LivenessDetector(
            bpm_window=8,
            snr_threshold=1.5,
            bpm_range=(45, 180),
            min_readings=5,
            consistency_threshold=25.0,
            regularity_threshold=0.15,
            correlation_threshold=0.3,
            harmonic_threshold=0.03,
            phase_threshold=0.1,
            screen_veto_threshold=0.5,
            quality_ratio_threshold=0.5,
        )
        self.screen_det = ScreenDetector(fps=fps)
        self.challenge = ChallengeResponse(fps=fps, response_timeout_s=5.0, num_challenges=3)
        self.session_start = time.time()
        self.session_duration = 10.0  # 10 second window
        self.alive_frames = 0
        self.denied_frames = 0
        self.verdict_locked = False
        self.final_verdict = "WARMUP"
        self.challenge_started = False

    def process_frame(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = landmarker.detect(mp_image)

        face_found = False
        landmarks_data = []

        if res.face_landmarks and len(res.face_landmarks) > 0:
            face_found = True
            face_lms = res.face_landmarks[0]
            landmarks_data = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in face_lms]

            # Extract Forehead ROI mean
            fh_pts = np.array([[int(face_lms[i].x * w), int(face_lms[i].y * h)] for i in FOREHEAD_INDICES if i < len(face_lms)])
            chk_pts = np.array([[int(face_lms[i].x * w), int(face_lms[i].y * h)] for i in CHEEK_INDICES if i < len(face_lms)])

            def get_roi_rgb(pts):
                if len(pts) < 3:
                    return (128.0, 128.0, 128.0)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
                means = cv2.mean(frame_rgb, mask=mask)
                return (means[0], means[1], means[2])

            fh_rgb = get_roi_rgb(fh_pts)
            chk_rgb = get_roi_rgb(chk_pts)

            self.pulse_forehead.add_sample(fh_rgb)
            self.pulse_cheek.add_sample(chk_rgb)

            screen_res = self.screen_det.update(frame_bgr, face_lms)
            is_screen = screen_res["is_screen"]
        else:
            is_screen = False

        # Pulse analysis
        bpm_fh, snr_fh, sig_fh = self.pulse_forehead.get_bpm()
        bpm_chk, snr_chk, sig_chk = self.pulse_cheek.get_bpm()

        corr_val = 0.0
        phase_val = 0.0
        harmonic_val = 0.0
        jitter_val = 0.0
        regularity_val = 0.0

        if len(sig_fh) > 30 and len(sig_chk) > 30:
            min_len = min(len(sig_fh), len(sig_chk))
            sf = sig_fh[-min_len:]
            sc = sig_chk[-min_len:]
            if np.std(sf) > 1e-6 and np.std(sc) > 1e-6:
                corr_val = float(np.corrcoef(sf, sc)[0, 1])
                phase_val = float(phase_coherence(sf, sc, self.fps, bpm_fh / 60.0))

            harmonic_val = float(self.pulse_forehead.get_harmonic_ratio())
            hrv = self.pulse_forehead.get_hrv_metrics()
            regularity_val = float(hrv.get("regularity", 0.0))
            jitter_val = float(hrv.get("jitter_ok", 0.0))

        # Update liveness detector
        raw_verdict, details = self.liveness.update(
            bpm=bpm_fh,
            snr=snr_fh,
            regularity=regularity_val,
            correlation=corr_val,
            harmonic=harmonic_val,
            jitter=jitter_val,
            phase=phase_val,
            is_screen=is_screen,
        )

        # Active challenge status
        challenge_prompt = "Align Face in Viewport"
        challenge_progress = 0
        if face_found:
            if not self.challenge_started and len(sig_fh) >= self.fps * 2:
                self.challenge.start()
                self.challenge_started = True

            if self.challenge_started:
                c_status, c_prompt, c_idx = self.challenge.update(res.face_landmarks[0], h, w)
                challenge_prompt = c_prompt
                challenge_progress = int((c_idx / max(1, self.challenge.num_challenges)) * 100)

        # Session Verdict Logic
        elapsed = time.time() - self.session_start
        time_remaining = max(0.0, self.session_duration - elapsed)

        if not self.verdict_locked:
            if raw_verdict == "ALIVE":
                self.alive_frames += 1
            elif raw_verdict == "DENIED":
                self.denied_frames += 1

            # Early grant if 2s worth of ALIVE frames (60 frames at 30fps)
            if self.alive_frames >= 45 and (not self.challenge_started or self.challenge.is_complete()):
                self.verdict_locked = True
                self.final_verdict = "ACCESS GRANTED"
            elif elapsed >= self.session_duration:
                self.verdict_locked = True
                if self.alive_frames > self.denied_frames and self.alive_frames >= 20:
                    self.final_verdict = "ACCESS GRANTED"
                else:
                    self.final_verdict = "ACCESS DENIED"
            else:
                self.final_verdict = raw_verdict

        return {
            "timestamp": time.time(),
            "elapsed": round(elapsed, 2),
            "time_remaining": round(time_remaining, 2),
            "face_found": face_found,
            "bpm": round(float(bpm_fh), 1),
            "kalman_bpm": round(float(details.get("avg_bpm", bpm_fh)), 1),
            "snr": round(float(snr_fh), 2),
            "regularity": round(float(regularity_val), 3),
            "correlation": round(float(corr_val), 3),
            "phase_coherence": round(float(phase_val), 3),
            "harmonic_ratio": round(float(harmonic_val), 3),
            "is_screen": is_screen,
            "raw_verdict": raw_verdict,
            "final_verdict": self.final_verdict,
            "verdict_locked": self.verdict_locked,
            "quality_votes": details.get("quality_votes", 0),
            "total_votes": details.get("total_votes", 7),
            "quality_ratio": round(details.get("quality_ratio", 0.0), 2),
            "cues": details.get("cues", {}),
            "challenge_prompt": challenge_prompt,
            "challenge_progress": challenge_progress,
            "signal_fh": [round(float(v), 4) for v in sig_fh[-100:]],
            "signal_chk": [round(float(v), 4) for v in sig_chk[-100:]],
            "landmarks": landmarks_data[::5] if landmarks_data else [] # Subsample landmarks for payload efficiency
        }

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    session = ConnectionSession()
    print("[WebSocket] Client connected to Bio-Pulse stream")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "reset":
                session = ConnectionSession()
                await websocket.send_json({"type": "reset_ack", "status": "session_reset"})
                continue

            if message.get("type") == "frame":
                img_bytes = base64.b64decode(message["data"].split(",")[-1])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    telemetry = session.process_frame(frame)
                    await websocket.send_json({"type": "telemetry", "data": telemetry})

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Bio-Pulse Python WebSocket Telemetry Engine on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
