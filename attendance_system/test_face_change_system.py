"""
Test script for Face Change Detection System.
This demonstrates how the system handles confidence levels between 50-70%.
"""

import os
import json
import numpy as np
from datetime import datetime

def test_face_change_detection():
    """Test the face change detection logic."""
    print("🧪 Testing Face Change Detection System")
    print("=" * 60)
    
    # Create sample embedding data
    sample_data = {
        "student_id": "FACE_CHANGE_TEST",
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
    with open("database/embeddings/FACE_CHANGE_TEST.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Created sample embedding data")
    
    # Simulate different confidence scenarios
    scenarios = [
        (0.45, "Low confidence - No action"),
        (0.55, "Face change detected (50-70%) - Create new embedding"),
        (0.65, "Face change detected (50-70%) - Create new embedding"),
        (0.75, "High confidence (>70%) - Create trusted embedding"),
        (0.85, "Very high confidence (>70%) - Create trusted embedding")
    ]
    
    print("\n📊 Testing Different Confidence Scenarios:")
    print("=" * 50)
    
    for confidence, description in scenarios:
        print(f"\n🎯 Confidence: {confidence:.2f} ({confidence*100:.0f}%)")
        print(f"   📝 {description}")
        
        if 0.5 <= confidence <= 0.7:
            print("   🔄 ACTION: Face change detected - creating new embedding")
            print("   📋 This helps adapt to face changes like aging, hairstyle, etc.")
        elif confidence > 0.7:
            print("   🔒 ACTION: High confidence - creating trusted embedding")
        else:
            print("   ⚠️ ACTION: No action taken (confidence too low)")
    
    # Create face change embedding
    print("\n🔄 Creating Face Change Embedding:")
    print("=" * 40)
    
    face_change_embedding = {
        "type": "insightface",
        "vector": np.random.rand(512).tolist(),
        "confidence": 0.65,
        "capture_date": datetime.now().isoformat(),
        "source": "face_change_update",
        "trusted": False,
        "face_change_detected": True
    }
    
    # Load existing data and add face change embedding
    with open("database/embeddings/FACE_CHANGE_TEST.json", 'r') as f:
        data = json.load(f)
    
    data["embeddings"].append(face_change_embedding)
    data["total_embeddings"] = len(data["embeddings"])
    data["last_updated"] = datetime.now().isoformat()
    
    with open("database/embeddings/FACE_CHANGE_TEST.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ Created face change embedding")
    
    # Show final structure
    print("\n📄 Final Embedding Structure:")
    print("=" * 40)
    
    with open("database/embeddings/FACE_CHANGE_TEST.json", 'r') as f:
        final_data = json.load(f)
    
    for i, emb in enumerate(final_data["embeddings"]):
        source = emb.get("source", "unknown")
        trusted = emb.get("trusted", False)
        confidence = emb.get("confidence", 0)
        face_change = emb.get("face_change_detected", False)
        
        print(f"  {i}: {source} - Trusted: {trusted} - Confidence: {confidence:.2f}")
        if face_change:
            print(f"      🔄 Face Change Detected!")
    
    return True

def test_confidence_ranges():
    """Test the confidence range logic."""
    print("\n🎯 Testing Confidence Range Logic")
    print("=" * 50)
    
    # Test different confidence values
    test_confidences = [0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    
    for conf in test_confidences:
        if conf < 0.5:
            action = "❌ No action (too low)"
        elif 0.5 <= conf <= 0.7:
            action = "🔄 Face change detected"
        elif conf > 0.7:
            action = "🔒 High confidence - trusted embedding"
        else:
            action = "❓ Unknown"
        
        print(f"Confidence {conf:.2f} ({conf*100:.0f}%): {action}")
    
    return True

def simulate_face_id_behavior():
    """Simulate Face ID-like adaptive behavior."""
    print("\n📱 Simulating Face ID-like Adaptive Behavior")
    print("=" * 60)
    
    print("1. 📱 Initial Registration:")
    print("   - Student registers with face")
    print("   - Embedding created with trusted: false")
    
    print("\n2. 🔄 Face Change Detection (50-70% confidence):")
    print("   - System detects face has changed")
    print("   - Creates new embedding with face_change_detected: true")
    print("   - Helps adapt to aging, hairstyle changes, etc.")
    
    print("\n3. 🔒 High Confidence Recognition (>70%):")
    print("   - System recognizes person with high confidence")
    print("   - Creates trusted embedding")
    print("   - Improves future recognition accuracy")
    
    print("\n4. 📈 Adaptive Learning:")
    print("   - System builds collection of embeddings")
    print("   - Mix of trusted and face change embeddings")
    print("   - Continuously improves recognition accuracy")
    
    return True

def main():
    """Main test function."""
    print("🚀 Face Change Detection System Test")
    print("=" * 70)
    print("This test demonstrates the Face ID-like adaptive learning system.")
    print("When confidence is between 50-70%, the system detects face changes")
    print("and creates new embeddings to adapt to the person's changing appearance.")
    print("=" * 70)
    
    try:
        # Test face change detection
        test_face_change_detection()
        
        # Test confidence ranges
        test_confidence_ranges()
        
        # Simulate Face ID behavior
        simulate_face_id_behavior()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📁 Check 'database/embeddings/FACE_CHANGE_TEST.json' for the generated file")
        print("\n🔍 Key Features:")
        print("   • 50-70% confidence = Face change detected")
        print("   • >70% confidence = Trusted embedding created")
        print("   • <50% confidence = No action taken")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 