"""
Demo script for testing trusted embedding functionality.
This script demonstrates how the system handles trusted vs untrusted embeddings.
"""

import os
import json
import numpy as np
from datetime import datetime
from attendance_system.registration.embedding_extraction import FaceEmbeddingExtractor
from attendance_system.recognition.embedding_updater import EmbeddingUpdater

def create_sample_embedding_data():
    """Create sample embedding data for testing."""
    sample_data = {
        "student_id": "TEST001",
        "embeddings": [
            {
                "type": "insightface",
                "vector": np.random.rand(512).tolist(),  # Random embedding
                "confidence": 0.85,
                "capture_date": datetime.now().isoformat(),
                "source": "registration",
                "trusted": False  # Initial embedding is untrusted
            }
        ],
        "created_date": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_embeddings": 1
    }
    
    # Save to file
    os.makedirs("database/embeddings", exist_ok=True)
    with open("database/embeddings/TEST001.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Created sample embedding data for TEST001")

def test_trusted_embedding_logic():
    """Test the trusted embedding logic."""
    print("\n🧪 Testing Trusted Embedding Logic")
    print("=" * 50)
    
    # Create sample data
    create_sample_embedding_data()
    
    # Initialize embedding updater
    updater = EmbeddingUpdater()
    
    # Test 1: Check initial trust status
    print("\n📊 Test 1: Initial Trust Status")
    trust_status = updater.get_embedding_trust_status("TEST001")
    print(f"Trusted embeddings: {trust_status['trusted']}")
    print(f"Untrusted embeddings: {trust_status['untrusted']}")
    
    # Test 2: Create a trusted embedding (simulating recognition match)
    print("\n🔒 Test 2: Creating Trusted Embedding")
    # Create a dummy face image (random array)
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    success = updater.update_trusted_embedding_on_match(
        "TEST001", 
        dummy_face, 
        0.85, 
        "insightface"
    )
    
    if success:
        print("✅ Successfully created trusted embedding")
    else:
        print("❌ Failed to create trusted embedding")
    
    # Test 3: Check updated trust status
    print("\n📊 Test 3: Updated Trust Status")
    trust_status = updater.get_embedding_trust_status("TEST001")
    print(f"Trusted embeddings: {trust_status['trusted']}")
    print(f"Untrusted embeddings: {trust_status['untrusted']}")
    
    # Test 4: Show embedding file structure
    print("\n📄 Test 4: Embedding File Structure")
    try:
        with open("database/embeddings/TEST001.json", 'r') as f:
            data = json.load(f)
        
        print("Embedding entries:")
        for i, emb in enumerate(data["embeddings"]):
            print(f"  {i}: {emb['source']} - Trusted: {emb['trusted']} - Confidence: {emb['confidence']}")
    
    except Exception as e:
        print(f"Error reading embedding file: {e}")

def test_embedding_creation():
    """Test embedding creation with trusted flag."""
    print("\n🎯 Test: Embedding Creation with Trusted Flag")
    print("=" * 50)
    
    updater = EmbeddingUpdater()
    
    # Create dummy face image
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Test 1: Create untrusted embedding (normal registration)
    print("\n📝 Creating untrusted embedding (registration)...")
    success1 = updater.update_embedding(
        "TEST002", 
        dummy_face, 
        0.8, 
        "insightface", 
        trusted=False
    )
    
    if success1:
        print("✅ Created untrusted embedding")
    else:
        print("❌ Failed to create untrusted embedding")
    
    # Test 2: Create trusted embedding (recognition match)
    print("\n🔒 Creating trusted embedding (recognition match)...")
    success2 = updater.update_embedding(
        "TEST002", 
        dummy_face, 
        0.9, 
        "insightface", 
        trusted=True
    )
    
    if success2:
        print("✅ Created trusted embedding")
    else:
        print("❌ Failed to create trusted embedding")
    
    # Check final status
    trust_status = updater.get_embedding_trust_status("TEST002")
    print(f"\n📊 Final status - Trusted: {trust_status['trusted']}, Untrusted: {trust_status['untrusted']}")

def main():
    """Main demo function."""
    print("🚀 Trusted Embedding System Demo")
    print("=" * 60)
    print("This demo shows how the system handles trusted vs untrusted embeddings.")
    print("1. New embeddings are created as 'trusted: false'")
    print("2. When an untrusted embedding matches during recognition,")
    print("   a new 'trusted: true' embedding is created")
    print("3. This helps adapt to face changes over time")
    print("=" * 60)
    
    try:
        # Test basic functionality
        test_trusted_embedding_logic()
        
        # Test embedding creation
        test_embedding_creation()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📁 Check the 'database/embeddings/' directory for generated files")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 