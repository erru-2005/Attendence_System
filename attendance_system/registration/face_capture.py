"""
Face capture script for student registration.
Captures multiple face angles from the camera and stores embeddings.
"""
import os
import cv2
import numpy as np
import time
import argparse
from typing import List, Dict, Tuple, Any
import threading

from ..config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FACE_CONFIDENCE_THRESHOLD,
    REQUIRED_FACE_ANGLES, MIN_PHOTOS_PER_ANGLE, MAX_PHOTOS_PER_ANGLE,
    FACE_DETECTION_SIZE, FACE_RECOGNITION_SIZE
)
from ..utils.camera_utils import CameraStream, draw_face_rectangle
from ..utils.image_processing import (
    preprocess_face, detect_blur, crop_face, is_face_centered
)
from .embedding_extraction import FaceDetector, FaceEmbeddingExtractor
from .database_manager import DatabaseManager


class FaceRegistrationSystem:
    """
    System for registering new students by capturing face images and extracting embeddings.
    """
    
    def __init__(self, camera_index=CAMERA_INDEX, resolution=(FRAME_WIDTH, FRAME_HEIGHT)):
        """
        Initialize the face registration system.
        
        Args:
            camera_index (int): Camera index
            resolution (tuple): Camera resolution
        """
        self.camera_index = camera_index
        self.resolution = resolution
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.face_detector = FaceDetector()
        self.embedding_extractor = FaceEmbeddingExtractor()
        
        # Variables for face capture
        self.captured_faces = {angle: [] for angle in REQUIRED_FACE_ANGLES}
        self.current_angle = REQUIRED_FACE_ANGLES[0]
        self.capture_in_progress = False
        self.auto_capture = False
        self.auto_capture_interval = 1.0  # seconds
        self.last_auto_capture = 0
        
        # Status messages
        self.status_message = ""
        self.status_color = (0, 255, 0)  # Green
        
        # Thread lock for face capture
        self.capture_lock = threading.Lock()
    
    def start(self):
        """Start the face registration system."""
        print("Starting face registration system...")
        
        # Open camera stream
        with CameraStream(self.camera_index, self.resolution[0], self.resolution[1]) as camera:
            # Main loop
            running = True
            while running:
                # Get frame from camera
                success, frame = camera.read()
                if not success or frame is None:
                    print("Error: Could not read from camera")
                    break
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Detect faces in the frame
                faces = self.face_detector.detect_faces(frame, FACE_CONFIDENCE_THRESHOLD)
                
                # Process the largest face
                if faces:
                    # Sort faces by size (largest first)
                    faces = sorted(faces, key=lambda x: x["bbox"][2] * x["bbox"][3], reverse=True)
                    face = faces[0]
                    
                    # Draw rectangle around face
                    bbox = face["bbox"]
                    confidence = face["confidence"]
                    display_frame = draw_face_rectangle(
                        display_frame, bbox, 
                        f"Face: {confidence:.2f}", confidence,
                        color=(0, 255, 0)
                    )
                    
                    # Process face if centered
                    if is_face_centered(bbox, frame.shape):
                        # Auto-capture if enabled and time elapsed
                        if self.auto_capture:
                            current_time = time.time()
                            if current_time - self.last_auto_capture >= self.auto_capture_interval:
                                # Crop and process face
                                self._process_face(frame, face)
                                self.last_auto_capture = current_time
                    else:
                        self.status_message = "Center your face"
                        self.status_color = (0, 165, 255)  # Orange
                else:
                    self.status_message = "No face detected"
                    self.status_color = (0, 0, 255)  # Red
                
                # Draw current angle instructions
                angle_text = f"Current Angle: {self.current_angle}"
                cv2.putText(
                    display_frame, angle_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                # Draw capture progress
                total_required = len(REQUIRED_FACE_ANGLES) * MIN_PHOTOS_PER_ANGLE
                total_captured = sum(len(faces) for faces in self.captured_faces.values())
                progress_text = f"Progress: {total_captured}/{total_required}"
                cv2.putText(
                    display_frame, progress_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                # Draw current angle progress
                angle_progress = f"{self.current_angle}: {len(self.captured_faces[self.current_angle])}/{MAX_PHOTOS_PER_ANGLE}"
                cv2.putText(
                    display_frame, angle_progress, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                # Draw status message
                cv2.putText(
                    display_frame, self.status_message, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.status_color, 2
                )
                
                # Draw instructions
                instructions = [
                    "SPACE: Capture face",
                    "A: Toggle auto-capture",
                    "N: Next angle",
                    "S: Save and register",
                    "R: Reset",
                    "Q: Quit"
                ]
                
                for i, instruction in enumerate(instructions):
                    cv2.putText(
                        display_frame, instruction, (20, 170 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                
                # Show frame
                cv2.imshow("Face Registration", display_frame)
                
                # Process key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit
                    running = False
                elif key == ord(' '):
                    # Capture face
                    if faces:
                        self._process_face(frame, faces[0])
                elif key == ord('a'):
                    # Toggle auto-capture
                    self.auto_capture = not self.auto_capture
                    self.status_message = f"Auto-capture: {'ON' if self.auto_capture else 'OFF'}"
                elif key == ord('n'):
                    # Next angle
                    self._next_angle()
                elif key == ord('s'):
                    # Save and register
                    if self._is_registration_complete():
                        self._register_student()
                        running = False
                    else:
                        self.status_message = "Registration not complete"
                        self.status_color = (0, 0, 255)  # Red
                elif key == ord('r'):
                    # Reset
                    self._reset_capture()
        
        # Clean up
        cv2.destroyAllWindows()
    
    def _process_face(self, frame, face):
        """
        Process a detected face for the current angle.
        
        Args:
            frame (numpy.ndarray): Current camera frame
            face (dict): Detected face information
        """
        with self.capture_lock:
            # Check if we already have enough photos for this angle
            if len(self.captured_faces[self.current_angle]) >= MAX_PHOTOS_PER_ANGLE:
                self.status_message = f"Max photos for {self.current_angle} reached"
                self.status_color = (0, 165, 255)  # Orange
                return
            
            # Crop the face
            bbox = face["bbox"]
            face_img = crop_face(frame, bbox)
            
            # Check if the face is blurry
            is_blurry, blur_value = detect_blur(face_img)
            if is_blurry:
                self.status_message = f"Image too blurry ({blur_value:.1f})"
                self.status_color = (0, 0, 255)  # Red
                return
            
            # Add the face to the captured faces
            self.captured_faces[self.current_angle].append(face_img)
            
            # Update status
            self.status_message = f"Captured {self.current_angle} ({len(self.captured_faces[self.current_angle])})"
            self.status_color = (0, 255, 0)  # Green
            
            # Check if we have enough photos for this angle
            if len(self.captured_faces[self.current_angle]) >= MIN_PHOTOS_PER_ANGLE:
                if self.current_angle == REQUIRED_FACE_ANGLES[-1]:
                    self.status_message = "All angles captured, press S to save"
                    self.status_color = (0, 255, 0)  # Green
                else:
                    self.status_message = f"{self.current_angle} complete, press N for next angle"
                    self.status_color = (0, 255, 0)  # Green
    
    def _next_angle(self):
        """Switch to the next face angle for capture."""
        current_idx = REQUIRED_FACE_ANGLES.index(self.current_angle)
        if current_idx < len(REQUIRED_FACE_ANGLES) - 1:
            next_idx = current_idx + 1
            self.current_angle = REQUIRED_FACE_ANGLES[next_idx]
            self.status_message = f"Now capturing {self.current_angle} angle"
            self.status_color = (255, 165, 0)  # Blue
        else:
            self.status_message = "Last angle reached"
            self.status_color = (0, 165, 255)  # Orange
    
    def _reset_capture(self):
        """Reset all captured faces."""
        with self.capture_lock:
            self.captured_faces = {angle: [] for angle in REQUIRED_FACE_ANGLES}
            self.current_angle = REQUIRED_FACE_ANGLES[0]
            self.status_message = "Capture reset"
            self.status_color = (0, 165, 255)  # Orange
    
    def _is_registration_complete(self):
        """
        Check if registration is complete (all angles have minimum required photos).
        
        Returns:
            bool: True if registration is complete, False otherwise
        """
        for angle in REQUIRED_FACE_ANGLES:
            if len(self.captured_faces[angle]) < MIN_PHOTOS_PER_ANGLE:
                return False
        return True
    
    def _register_student(self):
        """Register student with captured face images."""
        # Collect all face images
        all_faces = []
        for angle in REQUIRED_FACE_ANGLES:
            all_faces.extend(self.captured_faces[angle])
        
        # Get student information
        print("\nStudent Registration")
        print("===================")
        name = input("Enter student name: ")
        roll_number = input("Enter roll number: ")
        class_name = input("Enter class: ")
        section = input("Enter section: ")
        
        print("\nExtracting face embeddings...")
        
        # Extract embeddings from all faces
        embeddings = []
        for face_img in all_faces:
            try:
                # Preprocess the face
                preprocessed = preprocess_face(face_img, FACE_RECOGNITION_SIZE)
                
                # Extract embedding
                embedding = self.embedding_extractor.extract_embedding(preprocessed)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error extracting embedding: {e}")
        
        if not embeddings:
            print("Error: Could not extract any embeddings")
            return
        
        print(f"Extracted {len(embeddings)} embeddings")
        
        # Add student to database
        success, message = self.db_manager.add_student(
            roll_number, name, embeddings, class_name, section
        )
        
        print(message)
        
        if success:
            print(f"Student {name} ({roll_number}) registered successfully")
        else:
            print(f"Registration failed: {message}")


def main():
    """Main function for the face registration system."""
    parser = argparse.ArgumentParser(description="Face Registration System")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, 
                        help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH, 
                        help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT, 
                        help="Frame height (default: 480)")
    
    args = parser.parse_args()
    
    # Create and start the face registration system
    system = FaceRegistrationSystem(
        camera_index=args.camera,
        resolution=(args.width, args.height)
    )
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("Registration system stopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
