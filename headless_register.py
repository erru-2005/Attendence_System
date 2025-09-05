#!/usr/bin/env python3
"""
Standalone headless face registration script.
This bypasses GUI issues and works with the available camera.
"""
import sys
import os
import cv2
import numpy as np
import time
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attendance_system.config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FACE_CONFIDENCE_THRESHOLD,
    REQUIRED_FACE_ANGLES, MIN_PHOTOS_PER_ANGLE, MAX_PHOTOS_PER_ANGLE
)
from attendance_system.utils.camera_utils import CameraStream
from attendance_system.utils.image_processing import (
    preprocess_face, detect_blur, crop_face, is_face_centered
)
from attendance_system.registration.embedding_extraction import FaceDetector, FaceEmbeddingExtractor
from attendance_system.registration.database_manager import DatabaseManager


class SimpleHeadlessRegistration:
    """
    Simple headless face registration system.
    """
    
    def __init__(self, camera_index=0, resolution=(640, 480)):
        """
        Initialize the registration system.
        
        Args:
            camera_index (int): Camera index
            resolution (tuple): Camera resolution
        """
        self.camera_index = camera_index
        self.resolution = resolution
        
        # Initialize components
        print("Initializing components...")
        self.db_manager = DatabaseManager()
        self.face_detector = FaceDetector()
        self.embedding_extractor = FaceEmbeddingExtractor()
        print("Components initialized successfully!")
        
        # Variables for face capture
        self.captured_faces = {angle: [] for angle in REQUIRED_FACE_ANGLES}
        self.current_angle = REQUIRED_FACE_ANGLES[0]
        self.auto_capture = True
        self.auto_capture_interval = 3.0  # seconds
        self.last_auto_capture = 0
        
        # Status tracking
        self.total_captured = 0
        self.total_required = len(REQUIRED_FACE_ANGLES) * MIN_PHOTOS_PER_ANGLE
    
    def start(self):
        """Start the headless face registration system."""
        print("Starting headless face registration system...")
        print(f"Required angles: {REQUIRED_FACE_ANGLES}")
        print(f"Photos per angle: {MIN_PHOTOS_PER_ANGLE}-{MAX_PHOTOS_PER_ANGLE}")
        print(f"Total required: {self.total_required}")
        print(f"Camera index: {self.camera_index}")
        print(f"Resolution: {self.resolution}")
        
        # Open camera stream
        try:
            with CameraStream(self.camera_index, self.resolution[0], self.resolution[1]) as camera:
                print("Camera stream started successfully")
                
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
                    
                    # Process every 15th frame for performance
                    if frame_count % 15 != 0:
                        continue
                    
                    # Detect faces in the frame
                    faces = self.face_detector.detect_faces(frame, FACE_CONFIDENCE_THRESHOLD)
                    
                    # Process the largest face
                    if faces:
                        # Sort faces by size (largest first)
                        faces = sorted(faces, key=lambda x: x["bbox"][2] * x["bbox"][3], reverse=True)
                        face = faces[0]
                        
                        bbox = face["bbox"]
                        confidence = face["confidence"]
                        
                        print(f"Face detected: confidence={confidence:.2f}, bbox={bbox}")
                        
                        # Process face if centered
                        if is_face_centered(bbox, frame.shape):
                            print("Face is centered")
                            
                            # Auto-capture if enabled and time elapsed
                            if self.auto_capture:
                                current_time = time.time()
                                if current_time - self.last_auto_capture >= self.auto_capture_interval:
                                    # Crop and process face
                                    self._process_face(frame, face)
                                    self.last_auto_capture = current_time
                        else:
                            print("Face not centered - please center your face")
                    else:
                        print("No face detected")
                    
                    # Check if registration is complete
                    if self._is_registration_complete():
                        print("Registration complete!")
                        self._register_student()
                        running = False
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.2)
                    
        except Exception as e:
            print(f"Error with camera stream: {e}")
    
    def _process_face(self, frame, face):
        """Process a detected face."""
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
            
            # Store face data
            face_data = {
                "image": face_img,
                "embedding": embedding,
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "blur_value": blur_value,
                "timestamp": time.time()
            }
            
            # Add to captured faces
            if len(self.captured_faces[self.current_angle]) < MAX_PHOTOS_PER_ANGLE:
                self.captured_faces[self.current_angle].append(face_data)
                self.total_captured += 1
                
                print(f"✓ Captured face for angle '{self.current_angle}' ({len(self.captured_faces[self.current_angle])}/{MAX_PHOTOS_PER_ANGLE})")
                print(f"Total captured: {self.total_captured}/{self.total_required}")
                
                # Move to next angle if current angle is complete
                if len(self.captured_faces[self.current_angle]) >= MIN_PHOTOS_PER_ANGLE:
                    self._next_angle()
            else:
                print(f"Maximum photos reached for angle '{self.current_angle}'")
                
        except Exception as e:
            print(f"Error processing face: {e}")
    
    def _next_angle(self):
        """Move to the next angle."""
        current_index = REQUIRED_FACE_ANGLES.index(self.current_angle)
        next_index = (current_index + 1) % len(REQUIRED_FACE_ANGLES)
        self.current_angle = REQUIRED_FACE_ANGLES[next_index]
        print(f"Moving to next angle: {self.current_angle}")
    
    def _is_registration_complete(self):
        """Check if registration is complete."""
        for angle in REQUIRED_FACE_ANGLES:
            if len(self.captured_faces[angle]) < MIN_PHOTOS_PER_ANGLE:
                return False
        return True
    
    def _register_student(self):
        """Register the student with captured data."""
        print("\n" + "="*50)
        print("REGISTRATION COMPLETE!")
        print("="*50)
        
        # Get student information
        roll_number = input("Enter roll number: ")
        name = input("Enter student name: ")
        class_name = input("Enter class (optional): ")
        section = input("Enter section (optional): ")
        
        # Combine all embeddings
        all_embeddings = []
        for angle in REQUIRED_FACE_ANGLES:
            for face_data in self.captured_faces[angle]:
                all_embeddings.append(face_data["embedding"])
        
        if not all_embeddings:
            print("No embeddings available for registration")
            return
        
        # Average the embeddings
        avg_embedding = np.mean(all_embeddings, axis=0)
        
        # Add to database
        try:
            self.db_manager.add_student(
                roll_number=roll_number,
                name=name,
                embeddings=[avg_embedding.tolist()],
                class_name=class_name if class_name else None,
                section=section if section else None
            )
            print(f"\n✅ Student {name} (Roll: {roll_number}) registered successfully!")
            print(f"Total faces captured: {self.total_captured}")
            print(f"Embeddings stored: {len(all_embeddings)}")
        except Exception as e:
            print(f"Error registering student: {e}")


def main():
    """Main function for headless face registration."""
    parser = argparse.ArgumentParser(description="Headless Face Registration System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    
    args = parser.parse_args()
    
    print("Headless Face Registration System")
    print("="*40)
    
    # Create and start the registration system
    registration_system = SimpleHeadlessRegistration(
        camera_index=args.camera,
        resolution=(args.width, args.height)
    )
    
    try:
        registration_system.start()
    except KeyboardInterrupt:
        print("\nRegistration interrupted by user")
    except Exception as e:
        print(f"Error during registration: {e}")


if __name__ == "__main__":
    main()

