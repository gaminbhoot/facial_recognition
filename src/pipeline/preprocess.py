import cv2
import os
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF opener to handle .heic and .heif files
register_heif_opener()

class FacePreprocessor:
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size
        self.detector = MTCNN()

    def load_image(self, image_path):
        """
        Loads an image from path using Pillow to support multiple formats (HEIF, PNG, JPEG).
        Returns the image as an RGB numpy array.
        """
        try:
            pil_img = Image.open(image_path).convert('RGB')
            return np.array(pil_img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def detect_and_crop(self, image_path):
        """
        Detects a face in the image and crops it.
        Returns the cropped and resized face array.
        """
        img_rgb = self.load_image(image_path)
        if img_rgb is None:
            return None
        
        results = self.detector.detect_faces(img_rgb)
        
        if not results:
            print(f"Warning: No face detected in {image_path}")
            return None
        
        # Take the first detected face
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0:
            return None
            
        face_resized = cv2.resize(face, self.target_size)
        return face_resized

    def normalize(self, face_pixels):
        """
        Normalizes pixel values (e.g., standardizing for Facenet).
        """
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        return face_pixels

    def process_directory(self, raw_dir, processed_dir):
        """
        Processes all images in raw_dir and saves them to processed_dir.
        Maintains sub-directory structure (labeling by folder name).
        """
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        for person_name in os.listdir(raw_dir):
            person_raw_path = os.path.join(raw_dir, person_name)
            if not os.path.isdir(person_raw_path):
                continue
            
            person_processed_path = os.path.join(processed_dir, person_name)
            if not os.path.exists(person_processed_path):
                os.makedirs(person_processed_path)

            for image_name in os.listdir(person_raw_path):
                image_path = os.path.join(person_raw_path, image_name)
                
                # Change output extension to .jpg for uniform saving
                base_name = os.path.splitext(image_name)[0]
                processed_image_path = os.path.join(person_processed_path, base_name + ".jpg")
                
                face = self.detect_and_crop(image_path)
                if face is not None:
                    # Save the cropped face for inspection (optional, but good for debugging)
                    # Note: cv2.imwrite expects BGR
                    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(processed_image_path, face_bgr)
                    print(f"Processed: {image_path} -> {processed_image_path}")

if __name__ == "__main__":
    # Get the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    
    preprocessor = FacePreprocessor()
    preprocessor.process_directory(RAW_DATA_DIR, PROCESSED_DATA_DIR)
