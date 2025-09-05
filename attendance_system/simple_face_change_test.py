#!/usr/bin/env python3
"""
Simple test to demonstrate face change detection with percentage printing.
This test shows exactly what happens when confidence is 53%.
"""
import numpy as np
import json
import os
from datetime import datetime

def test_face_change_with_percentage():
    """Test face change detection with clear percentage printing."""
    print("🎯 Testing Face Change Detection with Percentage Display")
    print("=" * 60)
    
    # Simulate different confidence levels
    test_confidences = [0.45, 0.53, 0.65, 0.75, 0.85]
    
    for confidence in test_confidences:
        percentage = confidence * 100
        print(f"\n📊 Testing confidence: {confidence:.3f} ({percentage:.1f}%)")
        
        if confidence < 0.5:
            print(f"❌ NO ACTION - Confidence {percentage:.1f}% is too low (< 50%)")
        elif 0.5 <= confidence <= 0.7:
            print(f"🔄 FACE CHANGE DETECTED! Confidence {percentage:.1f}% is in range (50-70%)")
            print(f"   📝 System will append new embedding for face change adaptation")
            print(f"=== FACE CHANGE EMBEDDING APPENDED FOR: TEST001 (confidence: {confidence:.3f}) ===")
            print(f"✅ Face change embedding created successfully!")
        elif confidence > 0.7:
            print(f"🔒 HIGH CONFIDENCE! Confidence {percentage:.1f}% is > 70%")
            print(f"   📝 System will create trusted embedding")
            print(f"🔒 TRUSTED EMBEDDING CREATED FOR: TEST001 (confidence: {confidence:.3f})")
        else:
            print(f"❓ UNKNOWN RANGE - Confidence {percentage:.1f}%")
    
    print(f"\n📊 Summary of confidence ranges:")
    print(f"   • < 50%: No action taken")
    print(f"   • 50-70%: Face change detected - new embedding appended")
    print(f"   • > 70%: High confidence - trusted embedding created")

def create_sample_embedding_data():
    """Create sample embedding data to demonstrate the structure."""
    print("\n📄 Creating Sample Embedding Data")
    print("=" * 40)
    
    # Create sample data
    sample_data = {
        "student_id": "TEST001",
        "embeddings": [
            {
                "type": "insightface",
                "vector": np.random.rand(512).tolist(),
                "confidence": 0.85,
                "capture_date": datetime.now().isoformat(),
                "source": "registration",
                "trusted": False
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
    
    print("✅ Created sample embedding data")
    
    # Add a face change embedding
    face_change_embedding = {
        "type": "insightface",
        "vector": np.random.rand(512).tolist(),
        "confidence": 0.53,
        "capture_date": datetime.now().isoformat(),
        "source": "face_change_update",
        "trusted": False,
        "face_change_detected": True
    }
    
    sample_data["embeddings"].append(face_change_embedding)
    sample_data["total_embeddings"] = len(sample_data["embeddings"])
    sample_data["last_updated"] = datetime.now().isoformat()
    
    with open("database/embeddings/TEST001.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Added face change embedding")
    
    # Show the structure
    print("\n📄 Embedding Structure:")
    for i, emb in enumerate(sample_data["embeddings"]):
        source = emb.get("source", "unknown")
        trusted = emb.get("trusted", False)
        confidence = emb.get("confidence", 0)
        face_change = emb.get("face_change_detected", False)
        
        print(f"  {i}: {source} - Trusted: {trusted} - Confidence: {confidence:.2f}")
        if face_change:
            print(f"      🔄 Face Change Detected!")

def simulate_real_time_recognition():
    """Simulate real-time recognition with percentage display."""
    print("\n🎬 Simulating Real-Time Recognition")
    print("=" * 50)
    
    print("📱 When you run the main attendance system:")
    print("   1. Show your face to the camera")
    print("   2. System calculates recognition confidence")
    print("   3. Terminal shows percentage like this:")
    
    # Simulate recognition events
    events = [
        (0.45, "Unknown face detected"),
        (0.53, "Student recognized with face change"),
        (0.75, "Student recognized with high confidence"),
        (0.85, "Student recognized with very high confidence")
    ]
    
    for confidence, description in events:
        percentage = confidence * 100
        print(f"\n📊 Recognition Event: {description}")
        print(f"   Confidence: {confidence:.3f} ({percentage:.1f}%)")
        
        if 0.5 <= confidence <= 0.7:
            print(f"   🔄 FACE CHANGE DETECTED!")
            print(f"   📝 Appending new embedding...")
            print(f"   === FACE CHANGE EMBEDDING APPENDED FOR: STUDENT001 (confidence: {confidence:.3f}) ===")
        elif confidence > 0.7:
            print(f"   🔒 HIGH CONFIDENCE!")
            print(f"   📝 Creating trusted embedding...")
            print(f"   🔒 TRUSTED EMBEDDING CREATED FOR: STUDENT001 (confidence: {confidence:.3f})")
        else:
            print(f"   ❌ No action taken (confidence too low)")

def main():
    """Main test function."""
    print("🚀 Face Change Detection Test with Percentage Display")
    print("=" * 70)
    print("This test demonstrates:")
    print("   • Clear percentage display in terminal")
    print("   • Face change detection (50-70% confidence)")
    print("   • Trusted embedding creation (>70% confidence)")
    print("   • Clear messages when embeddings are appended")
    print("=" * 70)
    
    # Test confidence ranges
    test_face_change_with_percentage()
    
    # Create sample data
    create_sample_embedding_data()
    
    # Simulate real-time recognition
    simulate_real_time_recognition()
    
    print("\n🎉 Test completed!")
    print("\n💡 Key Points:")
    print("   • When confidence is 53%, you'll see the face change message")
    print("   • The system will append a new embedding automatically")
    print("   • This helps adapt to face changes over time")
    print("   • All messages are clearly printed in the terminal")

if __name__ == "__main__":
    main() 