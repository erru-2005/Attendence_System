#!/usr/bin/env python3
"""
Test script to simulate the real-time recognition scenario and debug face change detection.
"""
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from attendance_system.recognition.face_matcher import FaceMatcher, MultiFaceMatcher
from attendance_system.registration.embedding_extraction import FaceEmbeddingExtractor
from attendance_system.recognition.embedding_updater import EmbeddingUpdater

def simulate_real_time_recognition():
    """Simulate the exact real-time recognition scenario."""
    print("🎬 Simulating Real-Time Recognition Scenario...")
    print("="*60)
    
    # Initialize components (same as in real-time recognition)
    face_matcher = MultiFaceMatcher(recognition_threshold=0.5)
    embedding_extractor = FaceEmbeddingExtractor()
    
    print("📊 Testing with 53% confidence scenario...")
    
    # Create a realistic face image (224x224, typical face size)
    face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Create a realistic embedding (512 dimensions, typical for InsightFace)
    face_embedding = np.random.rand(512).astype(np.float32)
    
    # Simulate the exact scenario: 53% confidence
    test_confidence = 0.53
    test_student_id = "TEST001"
    
    print(f"🎯 Confidence: {test_confidence:.3f} ({test_confidence*100:.1f}%)")
    print(f"👤 Student ID: {test_student_id}")
    print(f"🖼️ Face image shape: {face_image.shape}")
    print(f"🔢 Embedding shape: {face_embedding.shape}")
    
    # Test 1: Direct face matcher call (what happens in the real system)
    print("\n🔍 Test 1: Direct FaceMatcher.match_face() call")
    print("-" * 40)
    
    # This is what happens in the real system
    is_match, roll_number, confidence = face_matcher.face_matcher.match_face(
        face_embedding, face_image
    )
    
    print(f"Match result: {is_match}")
    print(f"Roll number: {roll_number}")
    print(f"Confidence: {confidence:.3f}")
    
    # Test 2: Simulate the exact confidence scenario
    print("\n🔍 Test 2: Simulating 53% confidence scenario")
    print("-" * 40)
    
    # Manually set the confidence to 0.53 and test the face change logic
    if 0.5 <= test_confidence <= 0.7:
        print("🔄 FACE CHANGE detected!")
        print("   📝 Analyzing face and creating new embedding...")
        
        embedding_updater = EmbeddingUpdater()
        success = embedding_updater.update_face_change_embedding(
            test_student_id, 
            face_image, 
            test_confidence, 
            "insightface"
        )
        
        if success:
            print("✅ Face change embedding created successfully!")
            
            # Check the results
            count = embedding_updater.get_student_embedding_count(test_student_id)
            trust_status = embedding_updater.get_embedding_trust_status(test_student_id)
            
            print(f"📈 Total embeddings: {count}")
            print(f"🔒 Trust status: {trust_status}")
        else:
            print("❌ Failed to create face change embedding!")
    else:
        print("❌ Face change not detected - confidence outside range")
    
    # Test 3: Check what happens in MultiFaceMatcher
    print("\n🔍 Test 3: MultiFaceMatcher.match_faces() call")
    print("-" * 40)
    
    # Simulate the multi-face matcher call
    face_bboxes = [(100, 100, 200, 200)]  # Single face bbox
    face_embeddings = [face_embedding]
    face_images = [face_image]
    
    results = face_matcher.match_faces(face_bboxes, face_embeddings, face_images)
    
    print(f"Number of results: {len(results)}")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Face ID: {result['face_id']}")
        print(f"  Roll number: {result['roll_number']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Is match: {result['is_match']}")

def check_database_state():
    """Check the current state of the database."""
    print("\n" + "="*60)
    print("📊 Database State Check")
    print("="*60)
    
    # Check embeddings directory
    embeddings_dir = "database/embeddings"
    if os.path.exists(embeddings_dir):
        embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.json')]
        print(f"📁 Embedding files found: {len(embedding_files)}")
        for file in embedding_files:
            print(f"   📄 {file}")
    else:
        print("❌ Embeddings directory not found")
    
    # Check images directory
    images_dir = "database/images"
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"🖼️ Image files found: {len(image_files)}")
        for file in image_files[:5]:  # Show first 5
            print(f"   🖼️ {file}")
    else:
        print("ℹ️ Images directory not found")
    
    # Check students database
    students_file = "database/students.json"
    if os.path.exists(students_file):
        print("✅ Students database exists")
    else:
        print("❌ Students database not found")

def test_with_real_embedding():
    """Test with a real embedding from the database."""
    print("\n" + "="*60)
    print("🔍 Testing with Real Database Embedding")
    print("="*60)
    
    # Load a real embedding from the database
    embedding_file = "database/embeddings/TEST001.json"
    if os.path.exists(embedding_file):
        import json
        with open(embedding_file, 'r') as f:
            data = json.load(f)
        
        embeddings = data.get("embeddings", [])
        print(f"📊 Found {len(embeddings)} embeddings for TEST001")
        
        if embeddings:
            # Use the first embedding as reference
            ref_embedding = np.array(embeddings[0]["vector"])
            print(f"🔢 Reference embedding shape: {ref_embedding.shape}")
            
            # Create a test embedding with 53% similarity
            test_embedding = ref_embedding + np.random.normal(0, 0.1, ref_embedding.shape)
            
            # Calculate similarity
            similarity = np.dot(ref_embedding, test_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(test_embedding))
            print(f"📊 Calculated similarity: {similarity:.3f}")
            
            # Test face change detection
            if 0.5 <= similarity <= 0.7:
                print("🔄 Face change would be detected!")
            else:
                print("❌ Face change would NOT be detected")
        else:
            print("❌ No embeddings found in TEST001.json")
    else:
        print("❌ TEST001.json not found")

if __name__ == "__main__":
    print("🚀 Starting Real-Time Recognition Debug...")
    
    check_database_state()
    simulate_real_time_recognition()
    test_with_real_embedding()
    
    print("\n" + "="*60)
    print("🏁 Debug Complete!")
    print("\n💡 Possible issues:")
    print("   1. Face image is None in real-time recognition")
    print("   2. Embedding extraction is failing")
    print("   3. Confidence calculation is different in real system")
    print("   4. Face detection is failing on your image")
    print("   5. The face matcher is not being called with face images") 