#!/usr/bin/env python3
"""
Standalone headless face recognition script.
This bypasses GUI issues and works with the available camera.
"""
import sys
import os
import cv2
import numpy as np
import time
import argparse
import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attendance_system.config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FACE_CONFIDENCE_THRESHOLD,
    RECOGNITION_THRESHOLD, MIN_RECOGNITION_INTERVAL
)
from attendance_system.utils.camera_utils import CameraStream
from attendance_system.utils.image_processing import (
    preprocess_face, detect_blur, crop_face, is_face_centered
)
from attendance_system.registration.embedding_extraction import FaceDetector, FaceEmbeddingExtractor
from attendance_system.registration.database_manager import DatabaseManager


class SimpleHeadlessRecognition:
    """
    Simple headless face recognition system.
    """
    
    def __init__(self, camera_index=0, resolution=(640, 480), threshold=0.5):
        """
        Initialize the recognition system.
        
        Args:
            camera_index (int): Camera index
            resolution (tuple): Camera resolution
            threshold (float): Recognition threshold
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.threshold = threshold
        
        # Initialize components
        print("Initializing recognition components...")
        self.db_manager = DatabaseManager()
        self.face_detector = FaceDetector()
        self.embedding_extractor = FaceEmbeddingExtractor()
        print("Components initialized successfully!")
        
        # Load registered students
        self.students = self.db_manager.get_all_students()
        print(f"Loaded {len(self.students)} registered students")
        
        # Recognition tracking
        self.last_recognition_time = {}  # Track last recognition time per student
        self.recognition_count = 0
        
    def start(self):
        """Start the headless face recognition system."""
        print("Starting headless face recognition system...")
        print(f"Recognition threshold: {self.threshold}")
        print(f"Camera index: {self.camera_index}")
        print(f"Resolution: {self.resolution}")
        print(f"Registered students: {len(self.students)}")
        
        if len(self.students) == 0:
            print("⚠️  No students registered. Please register students first using:")
            print("   python -m attendance_system register --camera 0")
            return
        
        # Open camera stream
        try:
            with CameraStream(self.camera_index, self.resolution[0], self.resolution[1]) as camera:
                print("Camera stream started successfully")
                print("Press Ctrl+C to stop recognition")
                
                # Main loop
                running = True
                frame_count = 0
                
                while running:
                    # Get frame from camera
                    success, frame = camera.read()
                    if not success or frame is None:
                        print("Error: Could not read from camera")
                        break
                    
                    frame_count += 1
                    
                    # Process every 10th frame for performance
                    if frame_count % 10 != 0:
                        continue
                    
                    # Detect faces in the frame
                    faces = self.face_detector.detect_faces(frame, FACE_CONFIDENCE_THRESHOLD)
                    
                    # Process each detected face
                    for face in faces:
                        self._process_face(frame, face)
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nRecognition stopped by user")
        except Exception as e:
            print(f"Error with camera stream: {e}")
    
    def _process_face(self, frame, face):
        """Process a detected face for recognition."""
        try:
            # Crop face
            face_img = crop_face(frame, face["bbox"])
            
            # Check if face is blurry
            is_blurry, blur_value = detect_blur(face_img)
            if is_blurry:
                print(f"Face too blurry (blur_value={blur_value:.2f})")
                return
            
            # Extract embedding
            embedding = self.embedding_extractor.extract_embedding(face_img)
            if embedding is None:
                print("Failed to extract embedding")
                return
            
            # Find best match
            best_match = self._find_best_match(embedding)
            
            if best_match:
                student, similarity = best_match
                current_time = time.time()
                
                # Check if enough time has passed since last recognition
                if self._can_recognize(student["roll_number"], current_time):
                    # Log attendance
                    self._log_attendance(student, similarity, current_time)
                    
                    # Update last recognition time
                    self.last_recognition_time[student["roll_number"]] = current_time
                    self.recognition_count += 1
                    
                    print(f"✅ Recognized: {student['name']} (Roll: {student['roll_number']})")
                    print(f"   Similarity: {similarity:.3f}, Confidence: {face['confidence']:.3f}")
                    print(f"   Total recognitions: {self.recognition_count}")
                else:
                    print(f"⏳ Already recognized {student['name']} recently")
            else:
                print("❌ Unknown person detected")
                
        except Exception as e:
            print(f"Error processing face: {e}")
    
    def _find_best_match(self, embedding):
        """Find the best matching student for the given embedding."""
        best_match = None
        best_similarity = 0
        
        for student in self.students:
            if "embeddings" not in student or not student["embeddings"]:
                continue
                
            # Calculate similarity with each stored embedding
            for stored_embedding in student["embeddings"]:
                similarity = self._calculate_similarity(embedding, np.array(stored_embedding))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (student, similarity)
        
        # Return match only if similarity is above threshold
        if best_match and best_match[1] >= self.threshold:
            return best_match
        
        return None
    
    def _calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return similarity
    
    def _can_recognize(self, roll_number, current_time):
        """Check if enough time has passed since last recognition."""
        if roll_number not in self.last_recognition_time:
            return True
        
        time_diff = current_time - self.last_recognition_time[roll_number]
        return time_diff >= MIN_RECOGNITION_INTERVAL
    
    def _log_attendance(self, student, similarity, timestamp):
        """Log attendance for the recognized student."""
        try:
            # Create session based on time
            hour = datetime.datetime.fromtimestamp(timestamp).hour
            if 8 <= hour < 12:
                session = "morning"
            elif 12 <= hour < 17:
                session = "afternoon"
            else:
                session = "evening"
            
            # Log attendance
            self.db_manager.log_attendance(
                roll_number=student["roll_number"],
                confidence=similarity,
                session=session
            )
            
            print(f"📝 Attendance logged for {student['name']} ({session} session)")
            
        except Exception as e:
            print(f"Error logging attendance: {e}")


def main():
    """Main function for headless face recognition."""
    parser = argparse.ArgumentParser(description="Headless Face Recognition System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Recognition threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    print("Headless Face Recognition System")
    print("="*40)
    
    # Create and start the recognition system
    recognition_system = SimpleHeadlessRecognition(
        camera_index=args.camera,
        resolution=(args.width, args.height),
        threshold=args.threshold
    )
    
    try:
        recognition_system.start()
    except KeyboardInterrupt:
        print("\nRecognition interrupted by user")
    except Exception as e:
        print(f"Error during recognition: {e}")


if __name__ == "__main__":
    main()

