import sys
import os
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_database_and_matching():
    print("--- Testing Database Loading & Matching Flow ---")
    try:
        from src.utils.database import FaceDatabase
        db = FaceDatabase()
        
        # Add a temporary test user
        test_user = "TempVerificationUser"
        user_id = db.add_user(test_user)
        
        # Generate a random mock normalized embedding vector
        mock_emb = np.random.normal(size=(512,)).astype(np.float32)
        mock_emb = mock_emb / (np.linalg.norm(mock_emb) + 1e-10)
        
        db.add_embedding(user_id, "test_crop.jpg", mock_emb)
        
        # Load and verify it's there
        embs, labels = db.load_all_embeddings()
        print(f"Database contains {len(labels)} embeddings.")
        assert test_user in labels, f"Expected to find '{test_user}' in database labels"
        
        # Perform matching check
        match_name, confidence = db.match_face(mock_emb, threshold=0.60)
        print(f"Matching exact match: Name={match_name}, Confidence={confidence:.4f}")
        assert match_name == test_user, f"Expected match '{test_user}', got '{match_name}'"
        assert confidence > 0.99, f"Expected confidence close to 1.0, got {confidence}"
        
        # Mismatch check with a orthogonal/random vector
        random_vec = np.random.normal(size=(512,)).astype(np.float32)
        random_vec = random_vec / (np.linalg.norm(random_vec) + 1e-10)
        
        mismatch_name, mismatch_conf = db.match_face(random_vec, threshold=0.90)
        print(f"Matching random noise: Name={mismatch_name}, Confidence={mismatch_conf:.4f}")
        assert mismatch_name == "Unknown" or mismatch_name != test_user, "Random noise should not match our target user with high threshold"
        
        # Cleanup
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE name = ?", (test_user,))
            conn.commit()
            
        print("Database matching pipeline verification passed!")
    except Exception as e:
        print(f"Database verification failed: {e}")
        raise e

def test_liveness_detection():
    print("\n--- Testing Liveness Detection ---")
    try:
        from src.utils.liveness import detect_liveness
        
        # Test 1: Flat/uniform image (mocking a completely blank screen/printout crop)
        flat_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        is_live, score = detect_liveness(flat_crop, threshold=35.0)
        print(f"Flat crop liveness test: is_live={is_live}, score={score:.4f}")
        assert not is_live, "Flat crop should fail liveness check"
        assert score == 0.0, f"Expected 0.0 score, got {score}"
        
        # Test 2: Textured image (noise crop)
        textured_crop = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        is_live, score = detect_liveness(textured_crop, threshold=35.0)
        print(f"Textured crop liveness test: is_live={is_live}, score={score:.4f}")
        assert is_live, "Textured crop should pass liveness check"
        assert score > 1000.0, f"Expected high score for noise, got {score}"
        
        print("Liveness detection verification passed!")
    except Exception as e:
        print(f"Liveness verification failed: {e}")
        raise e

if __name__ == "__main__":
    test_database_and_matching()
    test_liveness_detection()
    print("\nAll pipeline verification tests passed successfully!")
