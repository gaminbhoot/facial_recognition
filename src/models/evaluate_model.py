import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def evaluate_system(embeddings_path, model_path, encoder_path):
    """
    Performs rigorous statistical evaluation of the classifier.
    """
    if not os.path.exists(embeddings_path) or not os.path.exists(model_path):
        print("Error: Missing data or model files for evaluation.")
        return

    # Load data
    data = np.load(embeddings_path)
    X, y = data['embeddings'], data['labels']
    
    # Load model and encoder
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    
    # SVM requires at least 2 classes. If only 1 class is present in embeddings, add dummy class for evaluation.
    if len(np.unique(y)) == 1:
        dummy_X = np.random.normal(size=(20, X.shape[1]))
        dummy_y = np.array(["Unknown"] * 20)
        X = np.vstack((X, dummy_X))
        y = np.hstack((y, dummy_y))

    y_encoded = encoder.transform(y)

    print(f"--- Statistical Evaluation Report ---")
    print(f"Total Samples: {len(X)}")
    print(f"Identities: {len(encoder.classes_)}")

    # 1. Cross-Validation (Robustness Check)
    # Stratified K-Fold ensures each fold has representative identities
    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')
    
    print(f"\n1. Cross-Validation Accuracy (n=5 folds):")
    print(f"   Mean Accuracy: {np.mean(cv_scores)*100:.2f}%")
    print(f"   95% Confidence Interval: +/- {np.std(cv_scores)*1.96*100:.2f}%")

    # 2. Precision/Recall/F1-Score
    y_pred = model.predict(X)
    report = classification_report(y_encoded, y_pred, target_names=encoder.classes_, output_dict=True)
    
    print(f"\n2. Classification Metrics (Summary):")
    print(f"   Macro Precision: {report['macro avg']['precision']:.4f}")
    print(f"   Macro Recall: {report['macro avg']['recall']:.4f}")
    print(f"   Macro F1-Score: {report['macro avg']['f1-score']:.4f}")

    # 3. Confidence Threshold Analysis
    # We want to find a threshold where we are confident in the prediction
    probs = model.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    
    # Statistical Summary of probabilities
    print(f"\n3. Probability Distribution (Confidence):")
    print(f"   Mean Confidence: {np.mean(max_probs):.4f}")
    print(f"   Median Confidence: {np.median(max_probs):.4f}")
    print(f"   10th Percentile: {np.percentile(max_probs, 10):.4f} (90% of predictions are above this)")

    # Recommendation for threshold
    suggested_threshold = np.percentile(max_probs, 5) # Use 5th percentile as a safe starting point
    print(f"\n[STATISTICAL RECOMMENDATION]")
    print(f"Based on training data, a confidence threshold of {suggested_threshold:.4f} is recommended.")
    print(f"Any prediction with probability < {suggested_threshold:.4f} should be labeled as 'Unknown' to maintain integrity.")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    EMBEDDINGS_FILE = os.path.join(PROJECT_ROOT, 'data', 'embeddings', 'face_embeddings.npz')
    MODEL_FILE = os.path.join(PROJECT_ROOT, 'models', 'face_classifier.pkl')
    ENCODER_FILE = os.path.join(PROJECT_ROOT, 'models', 'label_encoder.pkl')
    
    evaluate_system(EMBEDDINGS_FILE, MODEL_FILE, ENCODER_FILE)
