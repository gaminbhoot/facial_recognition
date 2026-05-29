import os
import sys
import cv2
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from mtcnn import MTCNN
from src.utils.liveness import detect_liveness

def audit_liveness():
    detector = MTCNN()
    jay_dir = os.path.join(project_root, "data", "raw", "Jay")
    if not os.path.exists(jay_dir):
        print(f"Directory {jay_dir} not found. Skipping local audit.")
        return
        
    image_files = [f for f in os.listdir(jay_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    print("Auditing images detail...")
    for filename in image_files:
        img_path = os.path.join(jay_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        scale = max(0.15, min(0.5, 320.0 / w))
        small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        faces = detector.detect_faces(rgb_small)
        for i, face_data in enumerate(faces):
            x, y, fw, fh = [int(v / scale) for v in face_data['box']]
            x, y = max(0, x), max(0, y)
            crop = img[y:y+fh, x:x+fw]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_LINEAR)
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            
            is_live, score = detect_liveness(crop_rgb, threshold=35.0)
            if not is_live:
                print(f"SUSPECTED SPOOF: {filename} (Face index {i}, Score: {score:.2f}, Image resolution: {w}x{h})")

if __name__ == "__main__":
    audit_liveness()
