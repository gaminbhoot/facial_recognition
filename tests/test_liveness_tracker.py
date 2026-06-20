import sys
import os
import time

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.liveness import LivenessTracker

def test_tracker():
    print("--- Testing LivenessTracker Temporal Smoothing & Grace Period ---")
    tracker = LivenessTracker(max_len=5, grace_period=1.0) # 1.0 second grace period for fast testing
    
    box = (100, 100, 50, 50) # x, y, w, h
    threshold = 35.0
    
    # 1. Test case: All low scores (static flat spoof)
    print("Testing consecutive low scores (flat spoof):")
    # First frame of spoof: should NOT be confirmed yet
    is_live, smoothed, is_spoof_confirmed = tracker.get_smoothed_liveness(box, score=10.0, threshold=threshold)
    print(f"  Frame 1: raw=10.0, smoothed={smoothed:.2f}, is_live={is_live}, is_spoof_confirmed={is_spoof_confirmed}")
    assert not is_live, "Should not be live"
    assert not is_spoof_confirmed, "Spoof should not be confirmed immediately during grace period"
    
    # Wait for 1.1 seconds (exceeding 1.0s grace period)
    print("  Waiting 1.1s to exceed grace period...")
    time.sleep(1.1)
    
    # Next frame of spoof: should now be confirmed!
    is_live, smoothed, is_spoof_confirmed = tracker.get_smoothed_liveness(box, score=10.0, threshold=threshold)
    print(f"  Frame 2: raw=10.0, smoothed={smoothed:.2f}, is_live={is_live}, is_spoof_confirmed={is_spoof_confirmed}")
    assert not is_live, "Should not be live"
    assert is_spoof_confirmed, "Spoof should be confirmed after grace period expires!"
    
    # 2. Test case: Transient motion blur (4 low frames, 1 high frame)
    tracker = LivenessTracker(max_len=5, grace_period=2.0)
    box2 = (110, 105, 50, 50)
    print("\nTesting 4 low frames (motion blur) + 1 high frame (sharp):")
    
    # Frame 1: high score (sharp)
    is_live, smoothed, is_spoof_confirmed = tracker.get_smoothed_liveness(box2, score=50.0, threshold=threshold)
    print(f"  Frame 1: raw=50.0, smoothed={smoothed:.2f}, is_live={is_live}")
    assert is_live, "High score frame should pass"
    
    # Frame 2-5: low scores (moving, blurry)
    for i in range(2, 6):
        box_moving = (110 + i*2, 105 + i, 50, 50) # slight movement
        is_live, smoothed, is_spoof_confirmed = tracker.get_smoothed_liveness(box_moving, score=15.0, threshold=threshold)
        print(f"  Frame {i}: raw=15.0, smoothed={smoothed:.2f}, is_live={is_live}")
        assert is_live, f"Frame {i} should remain classified as LIVE due to temporal smoothing (max score in window is 50.0)!"
        assert smoothed == 50.0, f"Expected smoothed to remain 50.0, got {smoothed}"
        
    # Frame 6: After 5 frames since the last high score, it should drop to low
    box_moving = (130, 115, 50, 50)
    is_live, smoothed, is_spoof_confirmed = tracker.get_smoothed_liveness(box_moving, score=15.0, threshold=threshold)
    print(f"  Frame 6 (5 frames after sharp): raw=15.0, smoothed={smoothed:.2f}, is_live={is_live}, is_spoof_confirmed={is_spoof_confirmed}")
    assert not is_live, "Should drop back to spoof if no high frames occur in the last 5 frames!"
    assert not is_spoof_confirmed, "Spoof should not be confirmed immediately on first frame after dropping"
    
    print("\nLivenessTracker temporal smoothing & grace period tests passed successfully!")

if __name__ == "__main__":
    test_tracker()
