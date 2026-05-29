# Facial Recognition System (VisionOS)

A high-performance, statistically robust, and privacy-hardened facial recognition system built with Python, FastAPI, TensorFlow, and OpenCV.

## Features
- **Real-time Performance:** 30+ FPS achieved through frame-skipping, detection downscaling, and batched inference.
- **Robust Detection:** Uses MTCNN for accurate face localization and alignment.
- **Deep Learning Embeddings:** Leverages pre-trained FaceNet models for 512-dimensional feature extraction.
- **SQLite Database with AES Encryption:** Secure storage of user identities and feature vectors. Embeddings are encrypted at rest using AES-128 (Fernet) to prevent theft or unauthorized access.
- **Biometric Liveness Check:** Prevents 2D spoofing attacks (such as printouts or screen displays) via Laplacian variance texture analysis and a sliding-window temporal liveness tracker (`LivenessTracker`).
- **Interactive Web Dashboard:** A premium glassmorphic multi-tab user interface with:
  - Real-time video stream with bounding boxes and status labels.
  - Active detection logs feed with auto-rate-limiting.
  - User enrollment (live webcam snapshot registration) and deletion.
  - Algorithmic control sliders for matching and liveness thresholds.
  - Performance telemetry (latency breakdown of MTCNN, FaceNet extraction, and similarity classification).
- **Privacy Obfuscation Mode:** Blurs any "Unknown" or spoofed faces in the stream.

## Project Structure
- `dashboard/`:
  - `backend/`: FastAPI server (`main.py`) serving endpoints and camera stream.
  - `frontend/`: Premium glassmorphic static files (`index.html`, `style.css`, `main.js`).
- `data/`: SQLite database storage (`face_recognition.db`) and secret key (`secret.key`) [both git-ignored for security].
- `models/`: Location for TensorFlow/Keras pre-trained weights.
- `src/`:
  - `models/`: Embedding extractor wrapper (`embedding_model.py`), SVM trainer (`train_classifier.py`), and evaluation scripts (`evaluate_model.py`).
  - `pipeline/`: Face preprocessing (`preprocess.py`) and command-line recognition stream (`realtime_recognition.py`).
  - `utils/`: SQLite Database interface (`database.py`), liveness algorithms (`liveness.py`), and latency profiler (`profiler.py`).
- `tests/`: Unit and integration test suite (`test_liveness_tracker.py`, `test_db_pipeline.py`, `test_security_hardening.py`).

## Installation

1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Pre-trained Model:**
   Download the FaceNet model weights (loaded via `keras-facenet`) or let the library fetch them automatically on first run.

## Usage Guide

### 1. Web Dashboard (Recommended UI)
Start the FastAPI server:
```bash
python dashboard/backend/main.py
```
Open your browser and navigate to `http://localhost:8000/`. From here, you can watch the stream, enroll new users, adjust thresholds, and track system latency.

### 2. Preprocessing & Embedding Generation (CLI)
For batch-processing offline folders of raw images:
```bash
# Detect, crop, and align faces from data/raw/ into data/processed/
python src/pipeline/preprocess.py

# Extract embeddings and register users in the SQLite database
python src/pipeline/generate_embeddings.py
```

### 3. Command-Line Stream
To run the live webcam stream in a simple OpenCV window:
```bash
python src/pipeline/realtime_recognition.py
```
**Controls:**
- `q`: Quit.
- `p`: Toggle performance reporting in console.
- `v`: Toggle Privacy Mode.

### 4. Running the Test Suite
Run the automated verification tests:
```bash
# Run database and liveness pipeline tests
python tests/test_db_pipeline.py

# Run liveness tracker temporal smoothing tests
python tests/test_liveness_tracker.py

# Run security and encryption checks
python tests/test_security_hardening.py
```

## Security & Privacy Compliance (GDPR)
- **Data Protection:** No raw face images are stored. Only 512-dimensional vector embeddings are saved in the database.
- **Encryption at Rest:** All vector embeddings are encrypted using Fernet (AES-128 symmetric encryption) on write and decrypted on the fly.
- **Privacy Mode:** Blurs faces of unknown users to protect their privacy in public deployments.
