import sqlite3
import numpy as np
import os
from cryptography.fernet import Fernet

class FaceDatabase:
    def __init__(self, db_path=None):
        if db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(project_root, 'data', 'face_recognition.db')
        
        self.db_path = db_path
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize encryption key
        self.key_path = os.path.join(os.path.dirname(self.db_path), 'secret.key')
        self._init_key()
        
        self.initialize_tables()

    def _init_key(self):
        """Loads the encryption key from disk or generates a new one if not present."""
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(self.key_path, 'wb') as f:
                f.write(self.key)
        self.fernet = Fernet(self.key)

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def initialize_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table for users
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table for face embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    image_name TEXT,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def add_user(self, name):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # User already exists, fetch their ID
                cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
                return cursor.fetchone()[0]

    def add_embedding(self, user_id, image_name, embedding):
        """Encrypts and adds a face embedding to the database."""
        # Serialize numpy array to bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()
        # Encrypt embedding at rest
        encrypted_bytes = self.fernet.encrypt(embedding_bytes)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO face_embeddings (user_id, image_name, embedding) VALUES (?, ?, ?)",
                (user_id, image_name, encrypted_bytes)
            )
            conn.commit()

    def load_all_embeddings(self):
        """
        Loads all face embeddings from the database, decrypting them on the fly.
        Automatically migrates legacy plaintext embeddings to encrypted-at-rest.
        Returns a tuple of (embeddings_array, labels_list).
        """
        embeddings = []
        labels = []
        updates = []  # Keep track of legacy records to migrate
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.id, u.name, e.embedding 
                FROM face_embeddings e 
                JOIN users u ON e.user_id = u.id
            """)
            rows = cursor.fetchall()
            for row in rows:
                row_id, name, db_bytes = row
                
                # Dynamic check: raw float32 of 512 elements is exactly 2048 bytes
                if len(db_bytes) == 2048:
                    embedding = np.frombuffer(db_bytes, dtype=np.float32)
                    # Queue for encryption migration
                    encrypted_bytes = self.fernet.encrypt(db_bytes)
                    updates.append((encrypted_bytes, row_id))
                else:
                    # Decrypt encrypted embedding
                    try:
                        decrypted_bytes = self.fernet.decrypt(db_bytes)
                        embedding = np.frombuffer(decrypted_bytes, dtype=np.float32)
                    except Exception as e:
                        print(f"Error decrypting embedding for user {name} (Row ID: {row_id}): {e}")
                        continue
                
                embeddings.append(embedding)
                labels.append(name)
            
            # Apply dynamic migrations if legacy records were loaded
            if updates:
                print(f"Migrating {len(updates)} legacy plaintext embeddings to encrypted-at-rest...")
                cursor.executemany(
                    "UPDATE face_embeddings SET embedding = ? WHERE id = ?",
                    updates
                )
                conn.commit()
                print("Migration complete!")
        
        if not embeddings:
            return np.empty((0, 512)), []
            
        return np.array(embeddings), labels

    def match_face(self, query_embedding, threshold=0.60):
        """
        Compares query_embedding against all embeddings in the database using Cosine Similarity.
        Returns (matched_name, confidence_score)
        """
        db_embeddings, db_labels = self.load_all_embeddings()
        if len(db_embeddings) == 0:
            return "Unknown", 0.0

        # L2-normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # L2-normalize database embeddings
        norms = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
        db_embeddings_norm = db_embeddings / (norms + 1e-10)

        # Compute cosine similarities (dot products)
        similarities = np.dot(db_embeddings_norm, query_norm)
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        if best_similarity >= threshold:
            return db_labels[best_idx], float(best_similarity)
        
        return "Unknown", float(best_similarity)
