import os
import sys
import sqlite3
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.database import FaceDatabase

def test_db_encryption():
    print("--- Verifying Database Encryption at Rest ---")
    db = FaceDatabase()
    
    # Add a mock embedding to verify encryption
    test_user = "TempSecurityUser"
    user_id = db.add_user(test_user)
    mock_emb = np.random.normal(size=(512,)).astype(np.float32)
    db.add_embedding(user_id, "sec_test.jpg", mock_emb)
    
    # Query database directly to see if the raw bytes are encrypted
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM face_embeddings WHERE user_id = ?", (user_id,))
        embedding_bytes = cursor.fetchone()[0]
        
    # Clean up test user
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE name = ?", (test_user,))
        conn.commit()
        
    print(f"Stored embedding BLOB size: {len(embedding_bytes)} bytes")
    
    # Decryption check: a raw float32 array of 512 elements is exactly 2048 bytes.
    # Fernet encrypted text is always larger (typically 2828 bytes for 2048-byte payload).
    assert len(embedding_bytes) != 2048, "VULNERABILITY: Stored embedding is 2048 bytes (Plaintext!)"
    assert embedding_bytes.startswith(b"gAAAA"), "Stored embedding does not start with Fernet token prefix!"
    print("Database encryption at rest verification passed!")

def test_backend_headers():
    print("\n--- Verifying FastAPI Security Hardening & Headers ---")
    try:
        from fastapi.testclient import TestClient
        from dashboard.backend.main import app
        
        client = TestClient(app)
        
        # 1. Test security headers
        response = client.get("/")
        print(f"Homepage headers: {dict(response.headers)}")
        assert response.headers.get("x-content-type-options") == "nosniff", "Missing X-Content-Type-Options"
        assert response.headers.get("x-frame-options") == "DENY", "Missing X-Frame-Options"
        assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin", "Missing Referrer-Policy"
        assert "content-security-policy" in response.headers, "Missing Content-Security-Policy"
        print("OWASP Security Headers verification passed!")

        # 2. Test CORS restriction
        # A request with an unauthorized origin should not receive CORS access headers (or receive block)
        unauth_response = client.get("/", headers={"Origin": "http://evil-attacker.com"})
        assert "access-control-allow-origin" not in unauth_response.headers, "VULNERABILITY: Allowed untrusted CORS origin!"
        
        # A request with an authorized origin should receive CORS headers
        auth_response = client.get("/", headers={"Origin": "http://localhost:8000"})
        # Note: FastAPI CORSMiddleware only attaches access-control-allow-origin if it's in the allowed list
        assert auth_response.headers.get("access-control-allow-origin") == "http://localhost:8000", "Failed to allow authorized CORS origin!"
        print("CORS Origin restriction verification passed!")
        
    except ImportError:
        print("fastapi.testclient or dependencies not installed, skipping HTTP verification.")
    except Exception as e:
        print(f"HTTP verification failed: {e}")
        raise e

if __name__ == "__main__":
    test_db_encryption()
    test_backend_headers()
    print("\nAll security hardening verification tests passed successfully!")
