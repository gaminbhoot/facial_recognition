import numpy as np
import os
from keras_facenet import FaceNet

class EmbeddingExtractor:
    def __init__(self, model_path=None):
        """
        Initializes the embedding model using the keras-facenet wrapper.
        This handles legacy H5 model loading issues automatically.
        """
        print("Initializing EmbeddingExtractor...")
        try:
            # FaceNet() automatically loads the pre-trained model
            print("Loading FaceNet pre-trained weights (this may take a minute)...")
            self.model = FaceNet()
            print("Successfully initialized FaceNet embedding model.")
        except Exception as e:
            print(f"ERROR: Failed to initialize FaceNet: {e}")
            self.model = None

    def get_embedding(self, face_pixels):
        """
        Generates a 512-d embedding for a single face.
        FaceNet() expects (160, 160, 3) and handles normalization internally.
        """
        if self.model is None:
            return None
        
        # keras-facenet.embeddings handles expansion and normalization
        # Note: it returns a list of embeddings if multiple are passed, 
        # so we pass a list of one and take the first result.
        embeddings = self.model.embeddings([face_pixels])
        return embeddings[0]

    def get_embeddings(self, list_of_face_pixels):
        """
        Generates 512-d embeddings for a batch of faces in a single forward pass.
        :param list_of_face_pixels: List of face crops of shape (160, 160, 3)
        :return: List of 512-d numpy arrays
        """
        if self.model is None or not list_of_face_pixels:
            return []
        
        return self.model.embeddings(list_of_face_pixels)

if __name__ == "__main__":
    # Test initialization
    extractor = EmbeddingExtractor()
