import cv2
import numpy as np
from collections import deque
import time

def detect_liveness(face_crop, threshold=35.0):
    """
    Checks if a face crop is real (liveness) or a spoof attempt (e.g., photo print, screen capture).
    Uses Laplacian variance to measure texture sharpness and detail.
    Returns:
        is_live (bool): True if live, False if suspected spoof
        variance (float): The calculated Laplacian variance
    """
    if face_crop is None or face_crop.size == 0:
        return False, 0.0

    # Convert to grayscale
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    
    # Calculate Laplacian variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    is_live = variance >= threshold
    return is_live, float(variance)

class LivenessTracker:
    def __init__(self, max_len=5, grace_period=3.0):
        self.max_len = max_len
        self.grace_period = grace_period
        self.history = {} # maps "cx,cy" center coordinates to deque of scores
        self.spoof_timers = {} # maps "cx,cy" to float (timestamp of first spoof detection)

    def get_smoothed_liveness(self, box, score, threshold=35.0):
        """
        Smooths liveness scores across consecutive frames to prevent motion blur false positives.
        :param box: bounding box (x, y, w, h) of the face
        :param score: current frame raw liveness score
        :param threshold: liveness detection threshold
        :return: (is_live, smoothed_score, is_spoof_confirmed)
        """
        x, y, w, h = box
        cx, cy = x + w // 2, y + h // 2
        
        # Find closest match in history (within 1.0x box width distance)
        matched_key = None
        min_dist = float(w) # Max movement allowed is face width
        
        for key in list(self.history.keys()):
            try:
                kx, ky = map(int, key.split(','))
                dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    matched_key = key
            except ValueError:
                continue
                
        now = time.time()
        
        if matched_key:
            # Update position tracking
            scores = self.history.pop(matched_key)
            scores.append(score)
            
            # Update spoof timer tracking
            spoof_start = self.spoof_timers.pop(matched_key, None)
            
            new_key = f"{cx},{cy}"
            self.history[new_key] = scores
            if spoof_start is not None:
                self.spoof_timers[new_key] = spoof_start
        else:
            new_key = f"{cx},{cy}"
            scores = deque([score], maxlen=self.max_len)
            self.history[new_key] = scores
            spoof_start = None
            
        # Clean up stale history items
        if len(self.history) > 10:
            keys = list(self.history.keys())
            for k in keys[:-10]:
                self.history.pop(k, None)
                self.spoof_timers.pop(k, None)
                
        # Take the maximum score in the window (if any frame in the window was sharp, it is live)
        smoothed_score = max(scores)
        raw_is_live = smoothed_score >= threshold
        
        # Liveness logic with timer
        if raw_is_live:
            # Reset spoof timer for this face
            self.spoof_timers.pop(new_key, None)
            is_live = True
            is_spoof_confirmed = False
        else:
            # Start or check spoof timer
            if new_key not in self.spoof_timers:
                self.spoof_timers[new_key] = now
                is_live = False
                is_spoof_confirmed = False
            else:
                elapsed = now - self.spoof_timers[new_key]
                if elapsed >= self.grace_period:
                    is_live = False
                    is_spoof_confirmed = True
                else:
                    is_live = False
                    is_spoof_confirmed = False
                    
        return is_live, smoothed_score, is_spoof_confirmed
