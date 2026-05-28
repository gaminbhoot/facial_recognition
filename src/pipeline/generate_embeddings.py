import sys
import os

# Add project root to sys.path to allow src imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import numpy as np
import cv2
from src.pipeline.preprocess import FacePreprocessor
from src.models.embedding_model import EmbeddingExtractor
from src.utils.database import FaceDatabase

def generate_embeddings_db(processed_dir, output_path):
    """
    Reads processed face images, extracts embeddings, and saves them with labels.
    Now also stores them in the SQLite database.
    """
    preprocessor = FacePreprocessor()
    extractor = EmbeddingExtractor()
    db = FaceDatabase()
    
    if extractor.model is None:
        print("Aborting: Embedding model not loaded.")
        return

    X, y = [], []

    for person_name in os.listdir(processed_dir):
        person_path = os.path.join(processed_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        
        print(f"Extracting embeddings for: {person_name}")
        user_id = db.add_user(person_name)
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            
            # Load processed image
            face_pixels = cv2.imread(image_path)
            face_pixels = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2RGB)
            
            # Extract embedding (keras-facenet handles normalization internally)
            embedding = extractor.get_embedding(face_pixels)
            
            if embedding is not None:
                X.append(embedding)
                y.append(person_name)
                db.add_embedding(user_id, image_name, embedding)

    # Convert to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # Save to disk as backup
    np.savez_compressed(output_path, embeddings=X, labels=y)
    print(f"Saved {len(X)} embeddings to {output_path} and SQLite database.")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'embeddings', 'face_embeddings.npz')
    
    generate_embeddings_db(PROCESSED_DIR, OUTPUT_FILE)
