import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.liveness import LivenessTracker

def test_tracker():
    print("--- Testing LivenessTracker Temporal Smoothing ---")
    tracker = LivenessTracker(max_len=5)
    
    box = (100, 100, 50, 50) # x, y, w, h
    threshold = 35.0
    
    # 1. Test case: All low scores (static flat spoof)
    print("Testing 5 consecutive low scores (flat spoof):")
    for i in range(5):
        is_live, smoothed = tracker.get_smoothed_liveness(box, score=10.0, threshold=threshold)
        print(f"  Frame {i+1}: raw=10.0, smoothed={smoothed:.2f}, is_live={is_live}")
    assert not is_live, "All low scores should be classified as spoof!"
    assert smoothed == 10.0, f"Expected smoothed to be 10.0, got {smoothed}"
    
    # 2. Test case: Transient motion blur (4 low frames, 1 high frame)
    box2 = (110, 105, 50, 50)
    print("\nTesting 4 low frames (motion blur) + 1 high frame (sharp):")
    
    # Frame 1: high score (sharp)
    is_live, smoothed = tracker.get_smoothed_liveness(box2, score=50.0, threshold=threshold)
    print(f"  Frame 1: raw=50.0, smoothed={smoothed:.2f}, is_live={is_live}")
    assert is_live, "High score frame should pass"
    
    # Frame 2-5: low scores (moving, blurry)
    for i in range(2, 6):
        box_moving = (110 + i*2, 105 + i, 50, 50) # slight movement
        is_live, smoothed = tracker.get_smoothed_liveness(box_moving, score=15.0, threshold=threshold)
        print(f"  Frame {i}: raw=15.0, smoothed={smoothed:.2f}, is_live={is_live}")
        assert is_live, f"Frame {i} should remain classified as LIVE due to temporal smoothing (max score in window is 50.0)!"
        assert smoothed == 50.0, f"Expected smoothed to remain 50.0, got {smoothed}"
        
    # Frame 6: After 5 frames since the last high score, it should drop to low
    box_moving = (130, 115, 50, 50)
    is_live, smoothed = tracker.get_smoothed_liveness(box_moving, score=15.0, threshold=threshold)
    print(f"  Frame 6 (5 frames after sharp): raw=15.0, smoothed={smoothed:.2f}, is_live={is_live}")
    assert not is_live, "Should drop back to spoof if no high frames occur in the last 5 frames!"
    
    print("\nLivenessTracker temporal smoothing tests passed successfully!")

if __name__ == "__main__":
    test_tracker()
