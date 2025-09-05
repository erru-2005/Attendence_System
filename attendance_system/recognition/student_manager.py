"""
Student Management Module
Handles student data, embeddings, and database operations
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime

class StudentManager:
    def __init__(self):
        self.data_file = "database/students_ai_data.json"
        self.images_dir = "database/images"
        self.embeddings_dir = "database/embeddings"
        
        # Ensure directories exist
        os.makedirs("database", exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize data structures
        self.students = {}
        self.embeddings = []
        self.student_ids = []
        self.embedding_types = []
        
        self.load_student_data()
        print("📚 Student Manager initialized!")

    def load_student_data(self):
        """Load all student data and embeddings (ALL models)"""
        # Load student information
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.students = data.get("students", {})
            except Exception as e:
                print(f"Error loading student data: {e}")
        
        # Load ALL types of embeddings
        self.embeddings = []
        self.student_ids = []
        self.embedding_types = []
        if os.path.exists(self.embeddings_dir):
            for filename in os.listdir(self.embeddings_dir):
                if filename.endswith('.json'):
                    student_id = filename.replace('.json', '')
                    try:
                        with open(os.path.join(self.embeddings_dir, filename), 'r') as f:
                            embedding_data = json.load(f)
                            if "embeddings" in embedding_data:
                                embeddings_list = embedding_data["embeddings"]
                                if isinstance(embeddings_list, list):
                                    for emb in embeddings_list:
                                        if (
                                            isinstance(emb, dict)
                                            and "vector" in emb
                                            and emb.get("type") in ["insightface", "opencv_dnn", "opencv_cascade"]
                                        ):
                                            self.embeddings.append(np.array(emb["vector"]))
                                            self.student_ids.append(student_id)
                                            self.embedding_types.append(emb.get("type", "unknown"))
                    except Exception as e:
                        print(f"Error loading embeddings for {student_id}: {e}")
                        continue
        
        # Count embeddings by type
        insightface_count = sum(1 for t in self.embedding_types if t == "insightface")
        opencv_dnn_count = sum(1 for t in self.embedding_types if t == "opencv_dnn")
        opencv_cascade_count = sum(1 for t in self.embedding_types if t == "opencv_cascade")
        
        print(f"✅ Loaded {len(self.embeddings)} embeddings:")
        print(f"   • InsightFace: {insightface_count}")
        print(f"   • OpenCV DNN: {opencv_dnn_count}")
        print(f"   • OpenCV Cascade: {opencv_cascade_count}")
        print(f"   • Total Students: {len(set(self.student_ids))}")

    def save_student_data(self):
        """Save student data to file"""
        data = {
            "students": self.students,
            "last_updated": datetime.now().isoformat()
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_student(self, name, student_id, face_recognition_system):
        """Add a new student with face capture (ALL models)"""
        if not name or not student_id:
            print("❌ Name and ID are required!")
            return False
        print(f"\n📸 Face Capture for {name}")
        print("Please look directly at the camera and maintain good lighting")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        captured_images = []
        captured_embeddings = []
        image_count = 0
        max_images = 7
        last_capture = time.time()
        capture_interval = 1.2
        while image_count < max_images:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            current_time = time.time()
            faces = face_recognition_system.detect_faces_fast(frame)
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Captured: {image_count}/{max_images}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Look at camera", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if len(faces) > 0 and current_time - last_capture >= capture_interval:
                x, y, w, h, detection_method = faces[0]
                face_img = frame[y:y+h, x:x+w]
                img_path = f"{self.images_dir}/{student_id}_{image_count+1}.jpg"
                cv2.imwrite(img_path, face_img)
                captured_images.append(img_path)
                
                # Extract embeddings using ALL available models
                print(f"🔄 Extracting embeddings for image {image_count+1}...")
                
                # InsightFace embedding
                if face_recognition_system.use_insightface:
                    insight_embeddings = face_recognition_system.extract_embeddings_optimized(face_img)
                    if "insightface" in insight_embeddings:
                        captured_embeddings.append({
                            "type": "insightface",
                            "vector": insight_embeddings["insightface"].tolist()
                        })
                        print(f"   ✅ InsightFace embedding extracted")
                    else:
                        print(f"   ❌ InsightFace embedding failed")
                
                # OpenCV DNN embedding (simplified features)
                if face_recognition_system.use_opencv_dnn:
                    try:
                        # Use OpenCV DNN features as embedding
                        face_resized = cv2.resize(face_img, (64, 64))
                        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                        normalized = gray.flatten().astype(np.float32) / 255.0
                        captured_embeddings.append({
                            "type": "opencv_dnn",
                            "vector": normalized.tolist()
                        })
                        print(f"   ✅ OpenCV DNN embedding extracted")
                    except Exception as e:
                        print(f"   ❌ OpenCV DNN embedding failed: {e}")
                
                # OpenCV Cascade embedding (histogram features)
                try:
                    face_resized = cv2.resize(face_img, (64, 64))
                    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                    hist_normalized = hist.flatten() / np.sum(hist)
                    captured_embeddings.append({
                        "type": "opencv_cascade",
                        "vector": hist_normalized.tolist()
                    })
                    print(f"   ✅ OpenCV Cascade embedding extracted")
                except Exception as e:
                    print(f"   ❌ OpenCV Cascade embedding failed: {e}")
                
                image_count += 1
                last_capture = current_time
                print(f"✅ Captured image {image_count}/{max_images} with {len(captured_embeddings)} embeddings")
            for face_data in faces:
                x, y, w, h, detection_method = face_data
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Face Capture', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        if captured_images and captured_embeddings:
            self.students[student_id] = {
                "name": name,
                "student_id": student_id,
                "images": captured_images,
                "registration_date": datetime.now().isoformat(),
                "embeddings_count": len(captured_embeddings),
                "quality_score": 0.95
            }
            embedding_data = {
                "student_id": student_id,
                "name": name,
                "embeddings": captured_embeddings,
                "capture_date": datetime.now().isoformat()
            }
            
            # Debug: Show what we're saving
            print(f"💾 Saving {len(captured_embeddings)} embeddings:")
            for i, emb in enumerate(captured_embeddings):
                print(f"   Embedding {i+1}: type={emb.get('type', 'unknown')}, vector_length={len(emb.get('vector', []))}")
            
            with open(f"{self.embeddings_dir}/{student_id}.json", 'w') as f:
                json.dump(embedding_data, f, indent=2)
            self.save_student_data()
            print(f"✅ Successfully added {name} with {len(captured_images)} images!")
            
            # Reload and show debug info
            self.load_student_data()
            print(f"🔄 After reload: {len(self.embeddings)} embeddings loaded")
            return True
        else:
            print("❌ No suitable images captured. Please try again.")
            return False

    def re_register_student(self, student_id, face_recognition_system):
        """Re-register a student (update images and embeddings)"""
        if student_id not in self.students:
            print("❌ Student not found!")
            return False
        name = self.students[student_id]["name"]
        print(f"\n🔄 Re-registering {name} (ID: {student_id})")
        return self.add_student(name, student_id, face_recognition_system)

    def delete_student(self, student_id):
        """Delete a student from the system"""
        if student_id not in self.students:
            print("❌ Student not found!")
            return False
        
        name = self.students[student_id]["name"]
        confirm = input(f"Are you sure you want to delete {name}? (y/N): ").strip().lower()
        
        if confirm == 'y':
            # Remove from memory
            del self.students[student_id]
            
            # Remove embeddings and related lists in sync using a mask
            mask = [sid != student_id for sid in self.student_ids]
            self.embeddings = [emb for emb, keep in zip(self.embeddings, mask) if keep]
            self.embedding_types = [t for t, keep in zip(self.embedding_types, mask) if keep]
            self.student_ids = [sid for sid in self.student_ids if sid != student_id]
            
            # Remove files
            try:
                os.remove(f"{self.embeddings_dir}/{student_id}.json")
            except:
                pass
            
            for filename in os.listdir(self.images_dir):
                if filename.startswith(f"{student_id}_"):
                    try:
                        os.remove(os.path.join(self.images_dir, filename))
                    except:
                        pass
            
            self.save_student_data()
            print(f"✅ Successfully deleted {name}")
            return True
        else:
            print("❌ Deletion cancelled")
            return False

    def list_students(self):
        """List all registered students"""
        if not self.students:
            print("❌ No students registered!")
            return
        
        print("\n📋 Registered Students:")
        for student_id, data in self.students.items():
            print(f"   • {data['name']} (ID: {student_id})")

    def get_student_data(self):
        """Get all student data for recognition"""
        return {
            'students': self.students,
            'embeddings': self.embeddings,
            'student_ids': self.student_ids,
            'embedding_types': self.embedding_types
        }

    def log_attendance(self, student_id, name):
        """Log attendance for a student"""
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"database/attendance_{today}.json"
        attendance_data = {}
        
        if os.path.exists(attendance_file):
            try:
                with open(attendance_file, 'r') as f:
                    attendance_data = json.load(f)
            except:
                pass
        
        if student_id not in attendance_data:
            current_time = datetime.now().strftime("%H:%M:%S")
            attendance_data[student_id] = {
                "name": name,
                "time": current_time,
                "status": "present"
            }
            with open(attendance_file, 'w') as f:
                json.dump(attendance_data, f, indent=2)
            print(f"✅ Attendance marked for {name}")
            return True
        return False 