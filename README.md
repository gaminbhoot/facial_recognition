# Facial Recognition System

A high-performance, statistically robust, and privacy-hardened facial recognition system built with Python, TensorFlow, and OpenCV.

## Features
- **Real-time Performance:** 30+ FPS achieved through frame-skipping and detection downscaling.
- **Robust Detection:** Uses MTCNN for accurate face localization and alignment.
- **Deep Learning Embeddings:** Leverages pre-trained Facenet models for high-dimensional feature extraction.
- **Statistical Integrity:** Dynamic confidence thresholding based on training data distribution.
- **Privacy Hardened:** Optional 'Privacy Mode' to blur unknown faces in live streams.
- **Performance Monitoring:** Integrated profiler to monitor latency across all pipeline stages.

## Installation

1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Pre-trained Model:**
   Download `facenet_keras.h5` and place it in the `models/` directory.

## Project Structure
- `data/`: Raw and processed facial images.
- `models/`: Trained classifiers and deep learning weights.
- `src/pipeline/`: Core logic for preprocessing, embedding generation, and real-time recognition.
- `src/models/`: Training and evaluation scripts.
- `src/utils/`: Performance profiling and shared utilities.

## Usage Guide

### 1. Data Collection
Place images of people you want to recognize in `data/raw/<name>/`. For best results, use at least 5-10 varied images per person.

### 2. Preprocessing & Embedding Generation
```bash
# Detect and align faces
python src/pipeline/preprocess.py

# Generate the embeddings database
python src/pipeline/generate_embeddings.py
```

### 3. Training & Evaluation
```bash
# Train the SVM classifier
python src/models/train_classifier.py

# Run statistical evaluation to find the optimal threshold
python src/models/evaluate_model.py
```

### 4. Real-time Recognition
```bash
python src/pipeline/realtime_recognition.py
```
**Live Controls:**
- `q`: Quit application.
- `p`: Toggle performance logs in console.
- `v`: Toggle Privacy Mode (Blur unknown faces).

## Security & Privacy Best Practices
- **Biometric Data:** This system processes facial biometric data. Ensure you have consent before enrolling users.
- **Vector Storage:** The system stores embeddings (mathematical vectors), not raw images, by default in the database for better security.
- **Privacy Mode:** Enable Privacy Mode (`v`) when using the system in public spaces to protect the identity of non-enrolled individuals.
