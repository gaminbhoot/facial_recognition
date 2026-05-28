import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def train_classifier(embeddings_path, model_out_path, encoder_out_path):
    """
    Loads embeddings, trains an SVM classifier, and saves the model.
    """
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file not found at {embeddings_path}")
        return

    # Load the compressed embeddings
    data = np.load(embeddings_path)
    X, y = data['embeddings'], data['labels']
    
    print(f"Dataset Loaded: {X.shape[0]} samples with {len(np.unique(y))} unique identities.")

    # SVM requires at least 2 classes. If only 1 class is present, add a dummy negative class.
    if len(np.unique(y)) == 1:
        print("Warning: Only one identity found. Adding dummy 'Negative' class for SVM training.")
        # Create random embeddings for the dummy class
        dummy_X = np.random.normal(size=(20, X.shape[1]))
        dummy_y = np.array(["Unknown"] * 20)
        X = np.vstack((X, dummy_X))
        y = np.hstack((y, dummy_y))

    # Encode labels (names -> integers)
    out_encoder = LabelEncoder()
    out_encoder.fit(y)
    y_encoded = out_encoder.transform(y)

    # Train SVM classifier
    # Linear kernel is often sufficient for high-dimensional embeddings
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y_encoded)

    # Internal Validation (on training set for now, ideally use a split)
    yhat_train = model.predict(X)
    score_train = accuracy_score(y_encoded, yhat_train)
    print(f'Training Accuracy: {score_train*100:.2f}%')

    # Save the model and the encoder
    joblib.dump(model, model_out_path)
    joblib.dump(out_encoder, encoder_out_path)
    print(f"Classifier saved to {model_out_path}")
    print(f"Label Encoder saved to {encoder_out_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    EMBEDDINGS_FILE = os.path.join(PROJECT_ROOT, 'data', 'embeddings', 'face_embeddings.npz')
    MODEL_FILE = os.path.join(PROJECT_ROOT, 'models', 'face_classifier.pkl')
    ENCODER_FILE = os.path.join(PROJECT_ROOT, 'models', 'label_encoder.pkl')
    
    train_classifier(EMBEDDINGS_FILE, MODEL_FILE, ENCODER_FILE)
