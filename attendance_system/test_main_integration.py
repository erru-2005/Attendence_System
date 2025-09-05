#!/usr/bin/env python3
"""
Integration test for main.py with face change detection.
This test demonstrates the face change detection working with the main attendance system.
"""
import cv2
import numpy as np
import os
import json
import time
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import the main system
from main import DeepFaceAttendance

class EnhancedAttendanceSystem(DeepFaceAttendance):
    """Enhanced attendance system with face change detection."""
    
    def __init__(self):
        super().__init__()
        self.face_change_count = 0
        self.trusted_embedding_count = 0
        
    def recognize_face_with_face_change(self, face_embedding, face_image=None):
        """Enhanced face recognition with face change detection."""
        if face_embedding is None:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for student_id, student_data in self.embeddings.items():
            stored_embedding = student_data.get('embedding')
            if stored_embedding is not None:
                # Convert to numpy if needed
                if isinstance(stored_embedding, list):
                    stored_embedding = np.array(stored_embedding)
                
                similarity = self.calculate_similarity(face_embedding, stored_embedding)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = student_id
        
        # Print recognition percentage clearly
        percentage = best_score * 100
        if best_match and best_score >= self.recognition_threshold:
            print(f"✅ RECOGNIZED: {best_match} - {percentage:.1f}% confidence")
            
            # Handle face change detection (50-70% confidence)
            if 0.5 <= best_score <= 0.7:
                print(f"🔄 FACE CHANGE DETECTED! Confidence: {percentage:.1f}%")
                print(f"   📝 Analyzing face and creating new embedding...")
                
                if face_image is not None:
                    self.create_face_change_embedding(best_match, face_image, best_score)
                    self.face_change_count += 1
                    
            # Handle trusted embedding creation (>70% confidence)
            elif best_score > 0.7:
                print(f"🔒 HIGH CONFIDENCE: {percentage:.1f}% - Creating trusted embedding")
                if face_image is not None:
                    self.create_trusted_embedding(best_match, face_image, best_score)
                    self.trusted_embedding_count += 1
        else:
            print(f"❌ NO MATCH - Best score: {percentage:.1f}%")
        
        return best_match, best_score
    
    def create_face_change_embedding(self, student_id, face_image, confidence):
        """Create a face change embedding."""
        try:
            # Extract new embedding
            new_embedding = self.extract_embedding(face_image)
            if new_embedding is None:
                print(f"❌ Failed to extract face change embedding")
                return False
            
            # Create embedding entry
            embedding_entry = {
                "type": "insightface",
                "vector": new_embedding.tolist(),
                "confidence": confidence,
                "capture_date": datetime.now().isoformat(),
                "source": "face_change_update",
                "trusted": False,
                "face_change_detected": True
            }
            
            # Add to student's embeddings
            if student_id not in self.embeddings:
                self.embeddings[student_id] = {
                    "name": self.students.get(student_id, {}).get("name", student_id),
                    "embeddings": []
                }
            
            if "embeddings" not in self.embeddings[student_id]:
                self.embeddings[student_id]["embeddings"] = []
            
            self.embeddings[student_id]["embeddings"].append(embedding_entry)
            
            # Save updated data
            self.save_data()
            
            print(f"=== FACE CHANGE EMBEDDING APPENDED FOR: {student_id} (confidence: {confidence:.3f}) ===")
            print(f"✅ Face change embedding created successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating face change embedding: {e}")
            return False
    
    def create_trusted_embedding(self, student_id, face_image, confidence):
        """Create a trusted embedding."""
        try:
            # Extract new embedding
            new_embedding = self.extract_embedding(face_image)
            if new_embedding is None:
                print(f"❌ Failed to extract trusted embedding")
                return False
            
            # Create embedding entry
            embedding_entry = {
                "type": "insightface",
                "vector": new_embedding.tolist(),
                "confidence": confidence,
                "capture_date": datetime.now().isoformat(),
                "source": "trusted_update",
                "trusted": True
            }
            
            # Add to student's embeddings
            if student_id not in self.embeddings:
                self.embeddings[student_id] = {
                    "name": self.students.get(student_id, {}).get("name", student_id),
                    "embeddings": []
                }
            
            if "embeddings" not in self.embeddings[student_id]:
                self.embeddings[student_id]["embeddings"] = []
            
            self.embeddings[student_id]["embeddings"].append(embedding_entry)
            
            # Save updated data
            self.save_data()
            
            print(f"🔒 TRUSTED EMBEDDING CREATED FOR: {student_id} (confidence: {confidence:.3f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating trusted embedding: {e}")
            return False
    
    def take_attendance_enhanced(self):
        """Enhanced attendance with face change detection."""
        print("\n📊 TAKING ATTENDANCE WITH FACE CHANGE DETECTION")
        print("-" * 50)
        print("Press ESC to stop")
        print("🔄 Face change detection: 50-70% confidence")
        print("🔒 Trusted embedding: >70% confidence")
        
        if not self.students:
            print("❌ No students registered")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not found")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        recognized_students = set()
        frame_count = 0
        start_time = time.time()
        last_recognition_time = 0
        recognition_interval = 0.5  # Recognize every 0.5 seconds
        
        print("\n🎯 Starting face recognition...")
        print("📊 Recognition percentages will be shown in terminal")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            current_time = time.time()
            
            # Detect faces every frame
            faces = self.detect_faces(frame)
            
            # Process recognition at intervals
            if current_time - last_recognition_time >= recognition_interval:
                for face in faces:
                    bbox = face['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Get embedding for recognition
                    face_embedding = face['embedding']
                    if face_embedding is None:
                        # Extract embedding from face region
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size > 0:
                            face_embedding = self.extract_embedding(face_img)
                    
                    # Enhanced recognition with face change detection
                    student_id, confidence = self.recognize_face_with_face_change(face_embedding, frame)
                    
                    if student_id and confidence >= self.recognition_threshold:
                        name = self.students[student_id]['name']
                        recognized_students.add(student_id)
                        
                        # Log attendance
                        self.log_attendance(student_id, name)
                        
                        # Draw recognized face box
                        self.draw_face_box_with_name(frame, bbox, name, confidence, True)
                    else:
                        # Draw unknown face box
                        self.draw_face_box_with_name(frame, bbox, "Unknown", confidence, False)
                
                last_recognition_time = current_time
            else:
                # Draw face boxes without recognition (for continuous tracking)
                for face in faces:
                    bbox = face['bbox']
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Draw statistics
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Recognized: {len(recognized_students)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Face Changes: {self.face_change_count}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Trusted: {self.trusted_embedding_count}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press ESC to stop", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Enhanced Attendance System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Attendance Summary:")
        print(f"   Students recognized: {len(recognized_students)}")
        print(f"   Face change embeddings created: {self.face_change_count}")
        print(f"   Trusted embeddings created: {self.trusted_embedding_count}")
        print("✅ Enhanced attendance completed!")

def test_face_change_detection():
    """Test the face change detection functionality."""
    print("🧪 Testing Face Change Detection with Main System")
    print("=" * 60)
    
    # Create enhanced system
    system = EnhancedAttendanceSystem()
    
    # Add a test student if none exist
    if not system.students:
        print("📝 Adding test student...")
        system.students["TEST001"] = {
            "name": "Test Student",
            "registration_date": datetime.now().isoformat()
        }
        
        # Create a test embedding
        test_embedding = np.random.rand(512).tolist()
        system.embeddings["TEST001"] = {
            "name": "Test Student",
            "embedding": test_embedding,
            "registration_date": datetime.now().isoformat()
        }
        
        system.save_data()
        print("✅ Test student added")
    
    print(f"📊 Loaded {len(system.students)} students")
    print("🎯 Ready to test face change detection!")
    
    return system

def main():
    """Main test function."""
    print("🚀 Enhanced Attendance System with Face Change Detection")
    print("=" * 70)
    print("This system will:")
    print("   • Show recognition percentages in terminal")
    print("   • Detect face changes (50-70% confidence)")
    print("   • Create trusted embeddings (>70% confidence)")
    print("   • Print clear messages when embeddings are appended")
    print("=" * 70)
    
    # Test the system
    system = test_face_change_detection()
    
    # Run enhanced attendance
    system.take_attendance_enhanced()
    
    print("\n🏁 Test complete!")

if __name__ == "__main__":
    main() 