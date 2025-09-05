#!/usr/bin/env python3
"""
Debug script to test face change detection functionality.
"""
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from attendance_system.recognition.face_matcher import FaceMatcher
from attendance_system.registration.embedding_extraction import FaceEmbeddingExtractor
from attendance_system.recognition.embedding_updater import EmbeddingUpdater

def test_face_change_detection():
    """Test the face change detection functionality."""
    print("🔍 Testing Face Change Detection...")
    
    # Initialize components
    face_matcher = FaceMatcher(recognition_threshold=0.5)
    embedding_extractor = FaceEmbeddingExtractor()
    embedding_updater = EmbeddingUpdater()
    
    # Test with a sample confidence of 0.53 (53%)
    test_confidence = 0.53
    test_student_id = "TEST001"
    
    print(f"📊 Testing with confidence: {test_confidence:.3f} ({test_confidence*100:.1f}%)")
    
    # Check if confidence is in the face change range
    if 0.5 <= test_confidence <= 0.7:
        print("✅ Confidence is in face change range (0.5-0.7)")
    else:
        print("❌ Confidence is NOT in face change range")
        return
    
    # Create a dummy face image (you can replace this with a real image)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    print("🔄 Testing face change embedding update...")
    
    # Test the face change embedding update
    success = embedding_updater.update_face_change_embedding(
        test_student_id, 
        dummy_image, 
        test_confidence, 
        "insightface"
    )
    
    if success:
        print("✅ Face change embedding update successful!")
        
        # Check the embedding count
        count = embedding_updater.get_student_embedding_count(test_student_id)
        print(f"📈 Total embeddings for {test_student_id}: {count}")
        
        # Check trust status
        trust_status = embedding_updater.get_embedding_trust_status(test_student_id)
        print(f"🔒 Trust status: {trust_status}")
        
    else:
        print("❌ Face change embedding update failed!")
    
    print("\n" + "="*50)
    print("🔍 Now testing the actual face matcher...")
    
    # Test the face matcher with the same confidence
    # Create a dummy embedding
    dummy_embedding = np.random.rand(512).astype(np.float32)
    
    print(f"🎯 Testing face matcher with confidence: {test_confidence:.3f}")
    
    # Simulate the face matcher logic
    if 0.5 <= test_confidence <= 0.7:
        print("🔄 FACE CHANGE detected!")
        print("   📝 Analyzing face and creating new embedding...")
        
        # This is what should happen in the face matcher
        success = embedding_updater.update_face_change_embedding(
            test_student_id, 
            dummy_image, 
            test_confidence, 
            "insightface"
        )
        
        if success:
            print("✅ Face change embedding created successfully!")
        else:
            print("❌ Failed to create face change embedding!")
    else:
        print("❌ Face change not detected - confidence outside range")

def test_with_real_image():
    """Test with a real image if available."""
    print("\n" + "="*50)
    print("📸 Testing with real image...")
    
    # Check if there are any images in the database
    images_dir = "database/images"
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            test_image_path = os.path.join(images_dir, image_files[0])
            print(f"📷 Using test image: {test_image_path}")
            
            # Load the image
            image = cv2.imread(test_image_path)
            if image is not None:
                print("✅ Image loaded successfully")
                
                # Test with 53% confidence
                test_confidence = 0.53
                test_student_id = "REAL_TEST"
                
                embedding_updater = EmbeddingUpdater()
                
                success = embedding_updater.update_face_change_embedding(
                    test_student_id,
                    image,
                    test_confidence,
                    "insightface"
                )
                
                if success:
                    print("✅ Real image face change embedding created!")
                else:
                    print("❌ Failed to create face change embedding with real image")
            else:
                print("❌ Failed to load image")
        else:
            print("ℹ️ No images found in database/images directory")
    else:
        print("ℹ️ No images directory found")

if __name__ == "__main__":
    print("🚀 Starting Face Change Detection Debug...")
    print("="*50)
    
    test_face_change_detection()
    test_with_real_image()
    
    print("\n" + "="*50)
    print("🏁 Debug complete!")
    print("\n💡 If face change detection is not working:")
    print("   1. Check if the face image is being passed correctly")
    print("   2. Check if the embedding extraction is working")
    print("   3. Check if the database directories exist")
    print("   4. Check if the confidence is actually between 0.5-0.7") 