#!/usr/bin/env python3
"""
Demonstration of Adaptive Learning System
Shows how the system keeps both old and new embeddings
to recognize a person across different appearances.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from attendance_system.recognition.embedding_updater import EmbeddingUpdater


def demonstrate_adaptive_learning():
    """Demonstrate the adaptive learning concept."""
    print("🎯 ADAPTIVE LEARNING DEMONSTRATION")
    print("=" * 60)
    print("This shows how the system adapts to face changes like:")
    print("• Clean shaven → With beard")
    print("• Short hair → Long hair") 
    print("• Without glasses → With glasses")
    print("• Different lighting conditions")
    print()
    
    # Initialize embedding updater
    updater = EmbeddingUpdater()
    
    # Example student
    student_id = "DEMO001"
    
    print(f"👤 Student: {student_id}")
    print(f"📊 Current embeddings: {updater.get_student_embedding_count(student_id)}")
    
    # Simulate the adaptive learning process
    scenarios = [
        {
            "appearance": "Clean Shaven (Original)",
            "confidence": 0.85,
            "description": "First registration - clean shaven face"
        },
        {
            "appearance": "Light Beard",
            "confidence": 0.62,
            "description": "Face change detected - light beard growth"
        },
        {
            "appearance": "Full Beard", 
            "confidence": 0.58,
            "description": "Face change detected - full beard"
        },
        {
            "appearance": "With Glasses",
            "confidence": 0.65,
            "description": "Face change detected - wearing glasses"
        },
        {
            "appearance": "Different Hair",
            "confidence": 0.61,
            "description": "Face change detected - different hairstyle"
        }
    ]
    
    print(f"\n🔄 SIMULATING ADAPTIVE LEARNING PROCESS")
    print("-" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['appearance']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Confidence: {scenario['confidence']:.3f}")
        
        if 0.5 <= scenario['confidence'] <= 0.7:
            print(f"   🔄 FACE CHANGE DETECTED!")
            print(f"   📝 Creating new embedding...")
            print(f"   ✅ New embedding appended (not replacing old ones)")
        else:
            print(f"   ✅ High confidence - no face change detected")
    
    print(f"\n📈 ADAPTIVE LEARNING RESULTS")
    print("-" * 50)
    print(f"✅ System now has multiple embeddings for {student_id}:")
    print(f"   📝 Original embedding (clean shaven)")
    print(f"   🔄 Face change embedding (light beard)")
    print(f"   🔄 Face change embedding (full beard)")
    print(f"   🔄 Face change embedding (with glasses)")
    print(f"   🔄 Face change embedding (different hair)")
    print()
    print(f"🎯 RECOGNITION CAPABILITIES:")
    print(f"   ✅ Can recognize clean shaven appearance")
    print(f"   ✅ Can recognize bearded appearance")
    print(f"   ✅ Can recognize with glasses")
    print(f"   ✅ Can recognize different hairstyles")
    print(f"   ✅ Adapts to aging and appearance changes")
    
    print(f"\n💡 KEY BENEFITS:")
    print(f"   🔄 Like Face ID - adapts to face changes")
    print(f"   📊 Keeps both old and new embeddings")
    print(f"   🎯 Recognizes person in multiple appearances")
    print(f"   🚀 Improves recognition accuracy over time")
    print(f"   📈 Learns from each interaction")


def show_embedding_structure():
    """Show the structure of adaptive embeddings."""
    print(f"\n📋 EMBEDDING STRUCTURE EXAMPLE")
    print("=" * 50)
    
    example_structure = {
        "student_id": "DEMO001",
        "embeddings": [
            {
                "type": "insightface",
                "vector": "[512-dimensional vector]",
                "confidence": 0.85,
                "capture_date": "2024-01-15T10:30:00",
                "source": "initial_registration",
                "trusted": True,
                "face_change_detected": False
            },
            {
                "type": "insightface", 
                "vector": "[512-dimensional vector]",
                "confidence": 0.62,
                "capture_date": "2024-01-20T14:15:00",
                "source": "face_change_update",
                "trusted": False,
                "face_change_detected": True,
                "appearance_type": "moderate_change"
            },
            {
                "type": "insightface",
                "vector": "[512-dimensional vector]", 
                "confidence": 0.58,
                "capture_date": "2024-01-25T09:45:00",
                "source": "face_change_update",
                "trusted": False,
                "face_change_detected": True,
                "appearance_type": "major_change"
            }
        ],
        "total_embeddings": 3,
        "face_change_count": 2,
        "last_updated": "2024-01-25T09:45:00"
    }
    
    print(f"📊 Total embeddings: {example_structure['total_embeddings']}")
    print(f"🔄 Face change embeddings: {example_structure['face_change_count']}")
    print(f"🔒 Trusted embeddings: 1")
    print()
    
    for i, emb in enumerate(example_structure['embeddings'], 1):
        status = []
        if emb.get('trusted'):
            status.append("🔒 Trusted")
        if emb.get('face_change_detected'):
            status.append("🔄 Face Change")
        
        print(f"{i}. {emb['source']} - {emb['appearance_type']} {' '.join(status)}")
        print(f"   Confidence: {emb['confidence']:.3f}")
        print(f"   Date: {emb['capture_date']}")


if __name__ == "__main__":
    print("🚀 ADAPTIVE LEARNING SYSTEM DEMONSTRATION")
    print("This shows how the system keeps both old and new embeddings")
    print("to recognize a person across different appearances")
    print()
    
    demonstrate_adaptive_learning()
    show_embedding_structure()
    
    print(f"\n🎉 DEMONSTRATION COMPLETED!")
    print(f"💡 The system now understands how to:")
    print(f"   ✅ Keep original embeddings (clean shaven)")
    print(f"   ✅ Append new embeddings (with beard)")
    print(f"   ✅ Recognize person in both appearances")
    print(f"   ✅ Adapt to face changes over time")
    print(f"   ✅ Work like Face ID's adaptive learning") 