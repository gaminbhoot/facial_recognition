import cv2
import os
import numpy as np
import joblib
from mtcnn import MTCNN
from src.models.embedding_model import EmbeddingExtractor
from src.pipeline.preprocess import FacePreprocessor
from src.utils.profiler import PerformanceMonitor
from src.utils.database import FaceDatabase
from src.utils.liveness import detect_liveness, LivenessTracker

class RealTimeRecognizer:
    def __init__(self, model_path=None, encoder_path=None, threshold=0.60, scale_factor=0.5, frame_skip=2, privacy_mode=False):
        """
        Initializes the real-time recognition system.
        :param threshold: Cosine similarity threshold for recognition
        :param scale_factor: Factor to scale down frames for detection (improves speed)
        :param frame_skip: Only perform full detection/recognition every N frames
        :param privacy_mode: If True, blurs faces that are not recognized
        """
        self.detector = MTCNN()
        self.extractor = EmbeddingExtractor()
        self.preprocessor = FacePreprocessor()
        self.perf = PerformanceMonitor()
        self.db = FaceDatabase()
        self.liveness_tracker = LivenessTracker(max_len=5)
        
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.frame_skip = frame_skip
        self.privacy_mode = privacy_mode
        self.frame_count = 0
        self.last_results = []
        
        # Load the SVM classifier model
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, 'models', 'face_classifier.pkl')
        encoder_path = os.path.join(project_root, 'models', 'label_encoder.pkl')
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
            
        print(f"System Initialized. Privacy Mode: {'ON' if self.privacy_mode else 'OFF'}")

    def start_stream(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Controls: 'q' - Quit | 'p' - Toggle Performance | 'v' - Toggle Privacy Mode")
        show_perf = False

        while True:
            self.perf.start('total_frame')
            ret, frame = cap.read()
            if not ret:
                break

            if self.frame_count % self.frame_skip == 0:
                # 1. Pre-processing & Downscaling for detection
                h, w, _ = frame.shape
                # Dynamic scale factor targeting ~640px width to improve detection accuracy
                self.scale_factor = max(0.25, min(1.0, 640.0 / w))
                
                self.perf.start('detection')
                small_frame = cv2.resize(frame, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_AREA)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detection on smaller frame
                faces = self.detector.detect_faces(rgb_small)
                self.perf.stop('detection')
                
                current_results = []
                face_crops = []
                valid_faces_metadata = []
                
                for face_data in faces:
                    # Scale coordinates back up
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
                    # Extraction (keras-facenet handles normalization)
                    self.perf.start('extraction')
                    embeddings = self.extractor.get_embeddings(face_crops)
                    self.perf.stop('extraction')
                    
                    self.perf.start('classification')
                    for i, embedding in enumerate(embeddings):
                        crop_rgb, x, y, fw, fh = valid_faces_metadata[i]
                        # Compute liveness on the raw/original resolution crop (BGR to RGB)
                        raw_face_pixels = frame[y:y+fh, x:x+fw]
                        raw_face_rgb = cv2.cvtColor(raw_face_pixels, cv2.COLOR_BGR2RGB)
                        _, raw_liveness_score = detect_liveness(raw_face_rgb, threshold=35.0)
                        
                        # Smooth liveness score across consecutive frames
                        is_live, smoothed_liveness_score, is_spoof_confirmed = self.liveness_tracker.get_smoothed_liveness(
                            (x, y, fw, fh), raw_liveness_score, threshold=35.0
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
                        elif is_spoof_confirmed:
                            name = "Spoof Attack"
                            confidence = smoothed_liveness_score
                        else:
                            name = "Unknown"
                            confidence = smoothed_liveness_score
                        
                        current_results.append({'box': (x, y, fw, fh), 'name': name, 'conf': float(confidence)})
                    self.perf.stop('classification')
                
                self.last_results = current_results

            # Visualization (use last results for skipped frames)
            for res in self.last_results:
                x, y, w, h = res['box']
                name, conf = res['name'], res['conf']
                
                if name in ["Unknown", "Spoof Attack"]:
                    color = (0, 0, 255)
                    label = f"Unknown ({conf:.2f})" if name == "Unknown" else f"{name} (Sharpness: {conf:.1f})"
                    # Privacy Hardening: Blur unknown faces or spoof attacks
                    if self.privacy_mode or name == "Spoof Attack":
                        face_region = frame[y:y+h, x:x+w]
                        if face_region.size > 0:
                            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                            frame[y:y+h, x:x+w] = blurred_face
                else:
                    color = (0, 255, 0)
                    label = f"{name} ({conf:.2f})"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Performance Overlay
            self.perf.stop('total_frame')
            if show_perf and self.frame_count % 30 == 0:
                self.perf.log_report()

            cv2.imshow('Facial Recognition System (Secured)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                show_perf = not show_perf
                print(f"Performance logging: {'ON' if show_perf else 'OFF'}")
            elif key == ord('v'):
                self.privacy_mode = not self.privacy_mode
                print(f"Privacy Mode (Unknown Blurring): {'ON' if self.privacy_mode else 'OFF'}")
            
            self.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Suggested default cosine similarity threshold
    CONFIDENCE_THRESHOLD = 0.60 
    
    try:
        recognizer = RealTimeRecognizer(threshold=CONFIDENCE_THRESHOLD)
        recognizer.start_stream()
    except Exception as e:
        print(f"Error starting system: {e}")
