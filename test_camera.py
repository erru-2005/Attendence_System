#!/usr/bin/env python3
"""
Simple camera test script to check camera availability.
"""
import cv2
import sys

def test_cameras():
    """Test available cameras."""
    print("Testing camera availability...")
    
    # Test camera indices 0-5
    for i in range(6):
        print(f"\nTesting camera {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"✓ Camera {i} is available")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {i} can read frames")
                print(f"  Frame size: {frame.shape}")
            else:
                print(f"✗ Camera {i} cannot read frames")
            
            cap.release()
        else:
            print(f"✗ Camera {i} is not available")
    
    print("\nCamera test completed!")

if __name__ == "__main__":
    test_cameras()
