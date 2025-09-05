#!/usr/bin/env python3
"""
Test script to run the attendance system and test face change detection.
"""
import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from attendance_system.recognition.face_matcher import FaceMatcher
from attendance_system.registration.embedding_extraction import FaceEmbeddingExtractor
from attendance_system.recognition.embedding_updater import EmbeddingUpdater

def test_attendance_with_face_change():
    """Test the attendance system with face change detection."""
    print("🎯 Testing Attendance System with Face Change Detection")
    print("=" * 60)
    
    # Initialize components
    face_matcher = FaceMatcher(recognition_threshold=0.5)
    embedding_extractor = FaceEmbeddingExtractor()
    
    print("📊 System initialized successfully!")
    print("🔄 Loading embeddings from database...")
    
    # Create a test face image
    test_face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Create a test embedding that will give around 53% confidence
    # We'll use a real embedding from the database and modify it slightly
    test_student_id = "TEST001"
    embedding_file = f"database/embeddings/{test_student_id}.json"
    
    if os.path.exists(embedding_file):
        import json
        with open(embedding_file, 'r') as f:
            data = json.load(f)
        
        if data.get("embeddings"):
            # Get the first embedding as reference
            ref_embedding = np.array(data["embeddings"][0]["vector"])
            
            # Create a modified embedding that will give ~53% confidence
            # Add some noise to reduce similarity
            test_embedding = ref_embedding + np.random.normal(0, 0.15, ref_embedding.shape)
            
            print(f"🎯 Testing with modified embedding for {test_student_id}")
            print(f"📊 Expected confidence: ~53%")
            
            # Test the face matcher
            print("\n" + "=" * 40)
            print("🔄 Running face recognition...")
            print("=" * 40)
            
            is_match, roll_number, confidence = face_matcher.match_face(test_embedding, test_face_image)
            
            print(f"\n📊 Results:")
            print(f"   Match: {is_match}")
            print(f"   Student: {roll_number}")
            print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            
            # Check if face change should be detected
            if 0.5 <= confidence <= 0.7:
                print(f"\n🔄 FACE CHANGE DETECTED!")
                print(f"   Confidence {confidence*100:.1f}% is in face change range (50-70%)")
                print(f"   System should append new embedding...")
                
                # Test the embedding updater
                embedding_updater = EmbeddingUpdater()
                success = embedding_updater.update_face_change_embedding(
                    test_student_id, test_face_image, confidence, "insightface"
                )
                
                if success:
                    print(f"✅ Face change embedding successfully appended!")
                    
                    # Check the updated count
                    count = embedding_updater.get_student_embedding_count(test_student_id)
                    print(f"📈 Total embeddings for {test_student_id}: {count}")
                else:
                    print(f"❌ Failed to append face change embedding!")
            else:
                print(f"\n❌ Face change NOT detected")
                print(f"   Confidence {confidence*100:.1f}% is outside face change range (50-70%)")
            
        else:
            print("❌ No embeddings found in database")
    else:
        print(f"❌ Embedding file not found: {embedding_file}")

def test_with_real_camera():
    """Test with real camera input."""
    print("\n" + "=" * 60)
    print("📸 Testing with Real Camera Input")
    print("=" * 60)
    
    print("🎥 Starting camera...")
    print("📋 Instructions:")
    print("   - Show your face to the camera")
    print("   - Look for recognition percentage in terminal")
    print("   - If confidence is 50-70%, face change embedding will be appended")
    print("   - Press 'q' to quit")
    print("\n🔄 Starting camera stream...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Initialize face recognition components
    face_matcher = FaceMatcher(recognition_threshold=0.5)
    embedding_extractor = FaceEmbeddingExtractor()
    
    print("✅ Camera started successfully!")
    print("👤 Show your face to test recognition...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Display the frame
            cv2.imshow('Attendance Test - Press Q to quit', frame)
            
            # Process every 10th frame for performance
            if int(time.time() * 10) % 10 == 0:
                # Extract face embedding
                try:
                    embedding = embedding_extractor.extract_embedding(frame)
                    if embedding is not None:
                        # Test recognition
                        is_match, roll_number, confidence = face_matcher.match_face(embedding, frame)
                        
                        # Clear previous lines and print current status
                        print(f"\r🎯 Recognition: {confidence*100:.1f}% - Student: {roll_number or 'Unknown'}", end="")
                        
                        # Check for face change
                        if 0.5 <= confidence <= 0.7 and is_match:
                            print(f"\n🔄 FACE CHANGE DETECTED! Confidence: {confidence*100:.1f}%")
                            print("   📝 Appending new embedding...")
                            
                            # Update embedding
                            embedding_updater = EmbeddingUpdater()
                            success = embedding_updater.update_face_change_embedding(
                                roll_number, frame, confidence, "insightface"
                            )
                            
                            if success:
                                print("✅ Face change embedding appended successfully!")
                            else:
                                print("❌ Failed to append face change embedding!")
                    
                except Exception as e:
                    print(f"\r❌ Error: {e}", end="")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Stopping camera...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera stopped")

if __name__ == "__main__":
    print("🚀 Starting Attendance System Test...")
    
    # Test 1: Simulated face change detection
    test_attendance_with_face_change()
    
    # Test 2: Real camera test
    test_with_real_camera()
    
    print("\n�� Test complete!") 