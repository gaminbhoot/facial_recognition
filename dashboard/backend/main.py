import cv2
import sys
import os
import threading
import time
import signal
from collections import deque

# Add project root to sys.path to allow src imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import numpy as np
import joblib
from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from mtcnn import MTCNN

PROJECT_ROOT = project_root

from src.models.embedding_model import EmbeddingExtractor
from src.pipeline.preprocess import FacePreprocessor
from src.utils.profiler import PerformanceMonitor
from src.utils.database import FaceDatabase
from src.utils.liveness import detect_liveness, LivenessTracker
import asyncio

app = FastAPI()

# Enable CORS for frontend communication with restricted origins
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Custom middleware to inject OWASP security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "font-src 'self' https://fonts.gstatic.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'"
    )
    # Add Cache-Control header for static frontend files to prevent browser caching during updates
    path = request.url.path
    if path.endswith((".css", ".js", ".html")) or path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

class RecognitionEngine:
    def __init__(self):
        self.detector = MTCNN()
        self.extractor = EmbeddingExtractor()
        self.preprocessor = FacePreprocessor()
        self.perf = PerformanceMonitor()
        self.db = FaceDatabase()
        self.liveness_tracker = LivenessTracker(max_len=5)
        
        self.threshold = 0.60  # Cosine similarity matching threshold
        self.liveness_threshold = 35.0  # Liveness texture sharpness threshold
        self.scale_factor = 0.5
        self.frame_skip = 2
        self.privacy_mode = False
        self.frame_count = 0
        self.last_results = []
        
        # Load the SVM classifier model
        self.load_svm_model()

    def load_svm_model(self):
        model_path = os.path.join(PROJECT_ROOT, 'models', 'face_classifier.pkl')
        encoder_path = os.path.join(PROJECT_ROOT, 'models', 'label_encoder.pkl')
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            try:
                self.svm_model = joblib.load(model_path)
                self.svm_encoder = joblib.load(encoder_path)
                print("Loaded SVM face classifier and label encoder successfully.")
            except Exception as e:
                print(f"Error loading SVM model: {e}")
                self.svm_model = None
                self.svm_encoder = None
        else:
            self.svm_model = None
            self.svm_encoder = None
            print("Warning: SVM classifier files not found. Falling back to Cosine Similarity.")

    def process_frame(self, frame):
        self.perf.start('total_frame')
        
        if self.frame_count % self.frame_skip == 0:
            h, w, _ = frame.shape
            # Dynamic scale factor targeting ~320px width
            self.scale_factor = max(0.15, min(0.5, 320.0 / w))
            
            self.perf.start('detection')
            small_frame = cv2.resize(frame, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_NEAREST)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb_small)
            self.perf.stop('detection')
            
            current_results = []
            face_crops = []
            valid_faces_metadata = []
            
            for face_data in faces:
                x, y, fw, fh = [int(v / self.scale_factor) for v in face_data['box']]
                x, y = max(0, x), max(0, y)
                
                face_pixels = frame[y:y+fh, x:x+fw]
                if face_pixels.size == 0:
                    continue
                
                # Optimize: resize first, then convert color
                crop_resized = cv2.resize(face_pixels, (160, 160), interpolation=cv2.INTER_LINEAR)
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                face_crops.append(crop_rgb)
                valid_faces_metadata.append((crop_rgb, x, y, fw, fh))
            
            if face_crops:
                self.perf.start('extraction')
                embeddings = self.extractor.get_embeddings(face_crops)
                self.perf.stop('extraction')
                
                self.perf.start('classification')
                for i, embedding in enumerate(embeddings):
                    crop_rgb, x, y, fw, fh = valid_faces_metadata[i]
                    # Compute liveness on the raw/original resolution crop (BGR to RGB)
                    raw_face_pixels = frame[y:y+fh, x:x+fw]
                    raw_face_rgb = cv2.cvtColor(raw_face_pixels, cv2.COLOR_BGR2RGB)
                    _, raw_liveness_score = detect_liveness(raw_face_rgb, threshold=self.liveness_threshold)
                    
                    # Smooth liveness score across consecutive frames
                    is_live, smoothed_liveness_score = self.liveness_tracker.get_smoothed_liveness(
                        (x, y, fw, fh), raw_liveness_score, threshold=self.liveness_threshold
                    )
                    
                    if is_live:
                        # Use SVM model if available, fallback to Cosine matching
                        if self.svm_model is not None and self.svm_encoder is not None:
                            try:
                                probs = self.svm_model.predict_proba([embedding])
                                max_idx = np.argmax(probs[0])
                                svm_conf = float(probs[0][max_idx])
                                if svm_conf >= self.threshold:
                                    name = self.svm_encoder.inverse_transform([max_idx])[0]
                                    confidence = svm_conf
                                else:
                                    name = "Unknown"
                                    confidence = svm_conf
                            except Exception as e:
                                print(f"SVM prediction error, falling back: {e}")
                                name, confidence = self.db.match_face(embedding, threshold=self.threshold)
                        else:
                            name, confidence = self.db.match_face(embedding, threshold=self.threshold)
                    else:
                        name = "Spoof Attack"
                        confidence = smoothed_liveness_score
                    
                    current_results.append({'box': (x, y, fw, fh), 'name': name, 'conf': float(confidence)})
                self.perf.stop('classification')
            
            self.last_results = current_results

        # Draw overlays
        for res in self.last_results:
            x, y, w, h = res['box']
            name, conf = res['name'], res['conf']
            
            if name in ["Unknown", "Spoof Attack"]:
                color = (0, 0, 255)
                # Blur both Unknown and Spoof Attack if privacy_mode is active (or always blur spoofs for security)
                if self.privacy_mode or name == "Spoof Attack":
                    face_region = frame[y:y+h, x:x+w]
                    if face_region.size > 0:
                        frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_region, (99, 99), 30)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{name} ({conf:.2f})" if name != "Spoof Attack" else f"{name} (Sharpness: {conf:.1f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self.perf.stop('total_frame')
        self.frame_count += 1
        return frame

class CameraStreamer:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.camera = None
        self.latest_frame = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        print("CameraStreamer: Initializing video capture...")
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            print("CRITICAL ERROR: Could not open camera. Device might be in use.")
            self.is_running = False
            return

        print("CameraStreamer: Video capture thread started successfully.")
        while self.is_running:
            success, frame = self.camera.read()
            if success:
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.01) # Yield CPU

        print("CameraStreamer: Releasing camera...")
        self.camera.release()
        self.camera = None

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)

engine = RecognitionEngine()
streamer = CameraStreamer(0)

# Start camera streamer immediately
streamer.start()

@app.on_event("shutdown")
def shutdown_event():
    print("Application shutdown: Stopping camera streamer...")
    streamer.stop()

def gen_frames():
    print("Video stream request received. Starting frame generation...")
    # Ensure streamer is running
    streamer.start()
    
    while True:
        success, frame = streamer.read()
        if not success:
            time.sleep(0.1) # Wait for the camera to initialize and yield a frame
            continue
            
        try:
            frame = engine.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03) # Limit output to ~30 FPS to save CPU and bandwidth
        except Exception as e:
            print(f"Error in frame processing: {e}")
            break

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/status')
def get_status():
    report = engine.perf.get_report()
    return {
        "fps": report.get('fps', 0),
        "latency": report,
        "last_seen": engine.last_results,
        "privacy_mode": engine.privacy_mode
    }

@app.post('/toggle_privacy')
def toggle_privacy():
    engine.privacy_mode = not engine.privacy_mode
    return {"status": "success", "privacy_mode": engine.privacy_mode}

# REST APIs for Biometric DB Management & Settings
class RegisterRequest(BaseModel):
    name: str

class DeleteUserRequest(BaseModel):
    name: str

class SettingsRequest(BaseModel):
    recognition_threshold: float
    liveness_threshold: float

@app.post('/register_user')
def register_user(req: RegisterRequest):
    name = req.name.strip()
    if not name:
        return {"status": "error", "message": "Name cannot be empty."}
    
    # Read a frame from the camera streamer
    success, frame = streamer.read()
    if not success or frame is None:
        return {"status": "error", "message": "Could not read from camera. Make sure the webcam is active."}

    # Perform detection
    h, w, _ = frame.shape
    scale_factor = max(0.15, min(0.5, 320.0 / w))
    small_frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    faces = engine.detector.detect_faces(rgb_small)
    
    if not faces:
        return {"status": "error", "message": "No face detected. Please face the camera and try again."}
    if len(faces) > 1:
        return {"status": "error", "message": "Multiple faces detected. Please ensure only one person is in frame."}
        
    face_data = faces[0]
    x, y, fw, fh = [int(v / scale_factor) for v in face_data['box']]
    x, y = max(0, x), max(0, y)
    
    face_pixels = frame[y:y+fh, x:x+fw]
    if face_pixels.size == 0:
        return {"status": "error", "message": "Invalid face crop size."}
        
    # Run liveness check on original resolution crop
    face_pixels_rgb = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2RGB)
    is_live, liveness_score = detect_liveness(face_pixels_rgb, threshold=engine.liveness_threshold)
    
    if not is_live:
        return {"status": "error", "message": f"Biometric verification failed (suspected spoof attack, score: {liveness_score:.1f})."}
        
    # Extract embedding
    crop_resized = cv2.resize(face_pixels, (160, 160), interpolation=cv2.INTER_LINEAR)
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    embedding = engine.extractor.get_embedding(crop_rgb)
    if embedding is None:
        return {"status": "error", "message": "Failed to extract face embedding."}
        
    # Insert user and embedding into database
    user_id = engine.db.add_user(name)
    engine.db.add_embedding(user_id, "webcam_capture.jpg", embedding)
    
    # Retrain classifier in background thread so the SVM stays up to date
    def retrain():
        try:
            from src.models.train_classifier import train_classifier
            embeddings_path = os.path.join(PROJECT_ROOT, 'data', 'embeddings', 'face_embeddings.npz')
            model_path = os.path.join(PROJECT_ROOT, 'models', 'face_classifier.pkl')
            encoder_path = os.path.join(PROJECT_ROOT, 'models', 'label_encoder.pkl')
            train_classifier(embeddings_path, model_path, encoder_path)
            engine.load_svm_model()
        except Exception as e:
            print(f"Error in background retraining: {e}")
            
    threading.Thread(target=retrain, daemon=True).start()
    
    return {"status": "success", "message": f"Successfully enrolled '{name}' in database!"}

@app.get('/users')
def list_users():
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.name, COUNT(e.id), MAX(e.created_at)
            FROM users u
            LEFT JOIN face_embeddings e ON e.user_id = u.id
            GROUP BY u.id
        """)
        rows = cursor.fetchall()
        users_list = []
        for name, count, last_seen in rows:
            users_list.append({
                "name": name,
                "embedding_count": count,
                "last_seen": last_seen
            })
        return users_list

@app.post('/delete_user')
def delete_user(req: DeleteUserRequest):
    name = req.name.strip()
    if not name:
        return {"status": "error", "message": "Name cannot be empty."}
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE name = ?", (name,))
        conn.commit()
        
    # Retrain classifier in background thread so the SVM stays up to date
    def retrain():
        try:
            from src.models.train_classifier import train_classifier
            embeddings_path = os.path.join(PROJECT_ROOT, 'data', 'embeddings', 'face_embeddings.npz')
            model_path = os.path.join(PROJECT_ROOT, 'models', 'face_classifier.pkl')
            encoder_path = os.path.join(PROJECT_ROOT, 'models', 'label_encoder.pkl')
            train_classifier(embeddings_path, model_path, encoder_path)
            engine.load_svm_model()
        except Exception as e:
            print(f"Error in background retraining: {e}")
            
    threading.Thread(target=retrain, daemon=True).start()
    
    return {"status": "success", "message": f"Successfully deleted user '{name}'."}

@app.get('/settings')
def get_settings():
    return {
        "recognition_threshold": engine.threshold,
        "liveness_threshold": engine.liveness_threshold
    }

@app.post('/update_settings')
def update_settings(req: SettingsRequest):
    engine.threshold = req.recognition_threshold
    engine.liveness_threshold = req.liveness_threshold
    return {"status": "success", "message": "Settings updated successfully."}

@app.post('/shutdown')
def shutdown_server():
    def terminate():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=terminate, daemon=True).start()
    return {"status": "success", "message": "Biometric server shutting down..."}

# Mount frontend files
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'dashboard', 'frontend')

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
