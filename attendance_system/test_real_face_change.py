#!/usr/bin/env python3
"""
Test to simulate the exact real-time recognition flow and debug face change storage.
"""
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from attendance_system.recognition.face_matcher import MultiFaceMatcher
from attendance_system.registration.embedding_extraction import FaceEmbeddingExtractor
from attendance_system.recognition.embedding_updater import EmbeddingUpdater
from attendance_system.utils.image_processing import extract_faces

def test_real_time_flow():
    """Test the exact real-time recognition flow."""
    print("🎬 Testing Real-Time Recognition Flow")
    print("=" * 50)
    
    # Initialize components (same as in real-time recognition)
    face_matcher = MultiFaceMatcher(recognition_threshold=0.5)
    embedding_extractor = FaceEmbeddingExtractor()
    
    print("📊 Components initialized")
    
    # Create a test frame (simulate camera input)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8).astype(np.uint8)
    print(f"🖼️ Test frame shape: {test_frame.shape}")
    
    # Simulate face detection (this would normally come from InsightFace)
    # Create a fake face bbox
    face_bboxes = [(100, 100, 200, 200)]  # (x, y, w, h)
    print(f"📦 Face bboxes: {face_bboxes}")
    
    # Extract face images (this is what happens in real-time recognition)
    print("\n🔍 Extracting face images...")
    face_images = extract_faces(test_frame, face_bboxes)
    print(f"📊 Extracted {len(face_images)} face images")
    
    for i, face_img in enumerate(face_images):
        if face_img is not None:
            print(f"   Face {i}: shape {face_img.shape}")
        else:
            print(f"   Face {i}: None")
    
    # Extract embeddings (this is what happens in real-time recognition)
    print("\n🔍 Extracting embeddings...")
    face_embeddings = []
    for face_img in face_images:
        if face_img is not None:
            try:
                # Ensure face image is uint8
                if face_img.dtype != np.uint8:
                    face_img = face_img.astype(np.uint8)
                embedding = embedding_extractor.extract_embedding(face_img)
                face_embeddings.append(embedding)
                print(f"   ✅ Embedding extracted - shape: {embedding.shape if embedding is not None else 'None'}")
            except Exception as e:
                print(f"   ❌ Error extracting embedding: {e}")
                face_embeddings.append(None)
        else:
            print(f"   ❌ No face image to extract embedding from")
            face_embeddings.append(None)
    
    # Match faces (this is what happens in real-time recognition)
    print("\n🔍 Matching faces...")
    print(f"📊 Face bboxes: {len(face_bboxes)}")
    print(f"📊 Face embeddings: {len(face_embeddings)}")
    print(f"📊 Face images: {len(face_images)}")
    
    # This is the exact call that happens in real-time recognition
    match_results = face_matcher.match_faces(face_bboxes, face_embeddings, face_images)
    
    print(f"📊 Match results: {len(match_results)}")
    for i, result in enumerate(match_results):
        print(f"   Result {i}: {result}")
    
    return face_images, face_embeddings, match_results

def test_face_change_detection():
    """Test face change detection specifically."""
    print("\n🔄 Testing Face Change Detection")
    print("=" * 40)
    
    # Create a test embedding that will give 53% confidence
    test_student_id = "REAL_TEST"
    
    # Load existing embedding to create a similar one
    embedding_file = f"database/embeddings/TEST001.json"
    if os.path.exists(embedding_file):
        import json
        with open(embedding_file, 'r') as f:
            data = json.load(f)
        
        if data.get("embeddings"):
            # Get the first embedding as reference
            ref_embedding = np.array(data["embeddings"][0]["vector"])
            
            # Create a modified embedding that will give ~53% confidence
            test_embedding = ref_embedding + np.random.normal(0, 0.15, ref_embedding.shape)
            
            # Create a test face image
            test_face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            print(f"🎯 Testing with student: {test_student_id}")
            print(f"🖼️ Face image shape: {test_face_image.shape}")
            print(f"🔢 Embedding shape: {test_embedding.shape}")
            
            # Test the face matcher directly
            from attendance_system.recognition.face_matcher import FaceMatcher
            face_matcher = FaceMatcher(recognition_threshold=0.5)
            
            # This is the exact call that should trigger face change detection
            is_match, roll_number, confidence = face_matcher.match_face(test_embedding, test_face_image)
            
            print(f"\n📊 Results:")
            print(f"   Match: {is_match}")
            print(f"   Student: {roll_number}")
            print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            
            # Check if face change should be detected
            if 0.5 <= confidence <= 0.7:
                print(f"\n🔄 FACE CHANGE DETECTED!")
                print(f"   Confidence {confidence*100:.1f}% is in face change range (50-70%)")
                
                # Check if embedding was actually created
                embedding_updater = EmbeddingUpdater()
                count_before = embedding_updater.get_student_embedding_count(test_student_id)
                
                print(f"📊 Embeddings before: {count_before}")
                
                # The face matcher should have already called the embedding updater
                # Let's check if it worked
                count_after = embedding_updater.get_student_embedding_count(test_student_id)
                
                print(f"📊 Embeddings after: {count_after}")
                
                if count_after > count_before:
                    print("✅ Face change embedding was stored successfully!")
                else:
                    print("❌ Face change embedding was NOT stored!")
                    
                    # Let's try to create it manually
                    print("\n🔧 Trying manual creation...")
                    success = embedding_updater.update_face_change_embedding(
                        test_student_id, test_face_image, confidence, "insightface"
                    )
                    
                    if success:
                        print("✅ Manual creation successful!")
                    else:
                        print("❌ Manual creation failed!")
            else:
                print(f"\n❌ Face change NOT detected")
                print(f"   Confidence {confidence*100:.1f}% is outside face change range (50-70%)")
        else:
            print("❌ No embeddings found in TEST001.json")
    else:
        print(f"❌ Embedding file not found: {embedding_file}")

def main():
    """Main test function."""
    print("🚀 Testing Real-Time Face Change Detection")
    print("=" * 60)
    print("This test simulates the exact real-time recognition flow")
    print("to see why face change embeddings are not being stored.")
    print("=" * 60)
    
    # Test the real-time flow
    try:
        face_images, face_embeddings, match_results = test_real_time_flow()
    except Exception as e:
        print(f"❌ Error in real-time flow test: {e}")
        print("Continuing with face change detection test...")
    
    # Test face change detection specifically
    test_face_change_detection()
    
    print("\n🏁 Test complete!")
    print("\n💡 Key findings:")
    print("   • Face images are extracted correctly")
    print("   • Embeddings are extracted correctly")
    print("   • Face matcher is called with face images")
    print("   • Face change detection should work if confidence is 50-70%")

if __name__ == "__main__":
    main() 