#!/usr/bin/env python3
"""
Debug script to test why face change embeddings are not being stored.
"""
import cv2
import numpy as np
import os
import json
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from attendance_system.recognition.embedding_updater import EmbeddingUpdater
from attendance_system.registration.embedding_extraction import FaceEmbeddingExtractor

def test_embedding_storage():
    """Test embedding storage functionality."""
    print("🔍 Debugging Embedding Storage")
    print("=" * 50)
    
    # Initialize components
    embedding_updater = EmbeddingUpdater()
    embedding_extractor = FaceEmbeddingExtractor()
    
    # Test student
    test_student_id = "DEBUG_TEST"
    
    print(f"📊 Testing with student: {test_student_id}")
    
    # Create a test face image
    test_face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    print(f"🖼️ Test face image shape: {test_face_image.shape}")
    
    # Test confidence
    test_confidence = 0.53
    print(f"🎯 Test confidence: {test_confidence:.3f} ({test_confidence*100:.1f}%)")
    
    # Check if confidence is in range
    if 0.5 <= test_confidence <= 0.7:
        print("✅ Confidence is in face change range")
    else:
        print("❌ Confidence is NOT in face change range")
        return
    
    # Test embedding extraction
    print("\n🔍 Testing embedding extraction...")
    try:
        extracted_embedding = embedding_extractor.extract_embedding(test_face_image)
        if extracted_embedding is not None:
            print(f"✅ Embedding extracted successfully - shape: {extracted_embedding.shape}")
        else:
            print("❌ Embedding extraction failed")
            return
    except Exception as e:
        print(f"❌ Error extracting embedding: {e}")
        return
    
    # Test face change embedding creation
    print("\n🔄 Testing face change embedding creation...")
    try:
        success = embedding_updater.update_face_change_embedding(
            test_student_id, 
            test_face_image, 
            test_confidence, 
            "insightface"
        )
        
        if success:
            print("✅ Face change embedding created successfully!")
            
            # Check if file was created
            embedding_file = f"database/embeddings/{test_student_id}.json"
            if os.path.exists(embedding_file):
                print(f"✅ Embedding file created: {embedding_file}")
                
                # Read and verify the file
                with open(embedding_file, 'r') as f:
                    data = json.load(f)
                
                print(f"📄 File structure:")
                print(f"   Student ID: {data.get('student_id')}")
                print(f"   Total embeddings: {data.get('total_embeddings')}")
                print(f"   Embeddings count: {len(data.get('embeddings', []))}")
                
                # Check the last embedding
                embeddings = data.get("embeddings", [])
                if embeddings:
                    last_embedding = embeddings[-1]
                    print(f"📊 Last embedding:")
                    print(f"   Source: {last_embedding.get('source')}")
                    print(f"   Trusted: {last_embedding.get('trusted')}")
                    print(f"   Confidence: {last_embedding.get('confidence')}")
                    print(f"   Face change detected: {last_embedding.get('face_change_detected')}")
                    print(f"   Vector length: {len(last_embedding.get('vector', []))}")
                else:
                    print("❌ No embeddings found in file")
            else:
                print(f"❌ Embedding file not created: {embedding_file}")
        else:
            print("❌ Face change embedding creation failed")
            
    except Exception as e:
        print(f"❌ Error creating face change embedding: {e}")
        import traceback
        traceback.print_exc()

def test_with_real_embedding():
    """Test with a real embedding from existing data."""
    print("\n" + "=" * 50)
    print("🔍 Testing with Real Embedding Data")
    print("=" * 50)
    
    # Check existing embeddings
    embeddings_dir = "database/embeddings"
    if os.path.exists(embeddings_dir):
        files = [f for f in os.listdir(embeddings_dir) if f.endswith('.json')]
        print(f"📁 Found {len(files)} embedding files:")
        for file in files:
            print(f"   📄 {file}")
            
            # Read file info
            file_path = os.path.join(embeddings_dir, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                student_id = data.get('student_id', 'Unknown')
                total_embeddings = data.get('total_embeddings', 0)
                embeddings_count = len(data.get('embeddings', []))
                
                print(f"      Student: {student_id}")
                print(f"      Total embeddings: {total_embeddings}")
                print(f"      Actual embeddings: {embeddings_count}")
                
                # Check for face change embeddings
                face_change_count = 0
                for emb in data.get('embeddings', []):
                    if emb.get('face_change_detected', False):
                        face_change_count += 1
                
                print(f"      Face change embeddings: {face_change_count}")
                
            except Exception as e:
                print(f"      ❌ Error reading file: {e}")
    else:
        print("❌ Embeddings directory not found")

def test_embedding_updater_initialization():
    """Test embedding updater initialization."""
    print("\n" + "=" * 50)
    print("🔍 Testing Embedding Updater Initialization")
    print("=" * 50)
    
    try:
        embedding_updater = EmbeddingUpdater()
        print("✅ Embedding updater initialized successfully")
        
        # Check directories
        embeddings_dir = "database/embeddings"
        images_dir = "database/images"
        
        print(f"📁 Embeddings directory exists: {os.path.exists(embeddings_dir)}")
        print(f"📁 Images directory exists: {os.path.exists(images_dir)}")
        
        # Check stats
        stats = embedding_updater.get_update_stats()
        print(f"📊 Update stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error initializing embedding updater: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function."""
    print("🚀 Debugging Face Change Embedding Storage")
    print("=" * 60)
    print("This will test why embeddings are not being stored properly.")
    print("=" * 60)
    
    # Test embedding updater initialization
    test_embedding_updater_initialization()
    
    # Test with real embedding data
    test_with_real_embedding()
    
    # Test embedding storage
    test_embedding_storage()
    
    print("\n🏁 Debug complete!")
    print("\n💡 If embeddings are not being stored:")
    print("   1. Check if face image is None")
    print("   2. Check if embedding extraction is failing")
    print("   3. Check if file permissions are correct")
    print("   4. Check if directories exist")

if __name__ == "__main__":
    main() 