"""
Enhanced Real-time Face Recognition with Ensemble Models
Uses multiple AI models for improved accuracy and reliability.
"""
import os
import cv2
import numpy as np
import time
import argparse
import datetime
import threading
from typing import List, Dict, Tuple, Optional, Any

from ..config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FRAME_SKIP, 
    FACE_CONFIDENCE_THRESHOLD, RECOGNITION_THRESHOLD, DISPLAY_WINDOW
)
from ..utils.camera_utils import CameraStream, draw_face_rectangle, enhance_frame
from ..utils.image_processing import (
    preprocess_face, detect_blur, crop_face, detect_liveness, extract_faces
)
from ..registration.embedding_extraction import FaceDetector
from .ensemble_matcher import MultiFaceEnsembleMatcher
from .attendance_logger import AttendanceLogger


class EnhancedRealTimeRecognition:
    """
    Enhanced real-time face recognition using ensemble models for improved accuracy.
    """
    
    def __init__(self, camera_index=CAMERA_INDEX, resolution=(FRAME_WIDTH, FRAME_HEIGHT),
                 display_window=DISPLAY_WINDOW, recognition_threshold=RECOGNITION_THRESHOLD):
        """
        Initialize the enhanced real-time recognition system.
        
        Args:
            camera_index (int): Camera index
            resolution (tuple): Camera resolution (width, height)
            display_window (bool): Whether to display the video window
            recognition_threshold (float): Threshold for face recognition
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.display_window = display_window
        self.recognition_threshold = recognition_threshold
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.ensemble_matcher = MultiFaceEnsembleMatcher(recognition_threshold)
        self.attendance_logger = AttendanceLogger()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        # Processing control
        self.processing_enabled = True
        self.liveness_detection_enabled = True
        self.frame_skip = FRAME_SKIP
        self.process_this_frame = True
        
        # Status message
        self.status_message = ""
        self.status_color = (0, 255, 0)  # Green
        
        # Ensemble model display
        self.show_model_scores = True
        self.ensemble_stats = {}
        
        print("Enhanced Real-time Recognition System initialized")
        print("Using ensemble models: ArcFace + FaceNet + MobileFaceNet")
    
    def start(self):
        """Start the enhanced real-time recognition system."""
        print("Starting enhanced real-time face recognition...")
        print("Controls:")
        print("   • ESC/Q: Quit")
        print("   • P: Pause/Resume processing")
        print("   • L: Toggle liveness detection")
        print("   • M: Toggle model scores display")
        print("   • S: Show ensemble statistics")
        print("   • R: Refresh database")
        print("   • +/-: Adjust frame skip")
        print("   • T: Save attendance report")
        
        # Open camera stream
        with CameraStream(self.camera_index, self.resolution) as camera:
            # Main loop
            running = True
            while running:
                # Get frame from camera
                frame = camera.read()
                if frame is None:
                    print("Error: Could not read from camera")
                    break
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Calculate FPS
                self._update_fps()
                
                # Process every nth frame for better performance
                self.process_this_frame = self.frame_count % self.frame_skip == 0
                
                if self.process_this_frame and self.processing_enabled:
                    # Detect faces in the frame
                    start_time = time.time()
                    faces = self.face_detector.detect_faces(frame, FACE_CONFIDENCE_THRESHOLD)
                    detection_time = time.time() - start_time
                    
                    # Process detected faces
                    if faces:
                        # Extract face images and preprocess them
                        face_bboxes = [face["bbox"] for face in faces]
                        face_images = extract_faces(frame, face_bboxes)
                        
                        # Check liveness if enabled
                        if self.liveness_detection_enabled:
                            for i, face_img in enumerate(face_images):
                                if face_img is None:
                                    continue
                                    
                                is_live, live_confidence = detect_liveness(face_img)
                                if not is_live:
                                    # Mark as spoof attempt
                                    self.status_message = "Spoof attempt detected!"
                                    self.status_color = (0, 0, 255)  # Red
                                    continue
                        
                        # Match faces using ensemble approach
                        match_results = self.ensemble_matcher.match_faces_ensemble(face_bboxes, face_images)
                        
                        # Process match results
                        for result in match_results:
                            face_id = result["face_id"]
                            bbox = result["bbox"]
                            roll_number = result["roll_number"]
                            confidence = result["confidence"]
                            is_match = result["is_match"]
                            model_scores = result.get("model_scores", {})
                            
                            # Draw rectangle around face
                            label = roll_number if is_match else "Unknown"
                            color = (0, 255, 0) if is_match else (0, 0, 255)
                            
                            display_frame = draw_face_rectangle(
                                display_frame, bbox, label, confidence, color
                            )
                            
                            # Draw model scores if enabled
                            if self.show_model_scores and model_scores:
                                self._draw_model_scores(display_frame, bbox, model_scores)
                            
                            # Log attendance for matched faces
                            if is_match and roll_number:
                                success, message = self.attendance_logger.log_attendance(roll_number, confidence)
                                if not success:
                                    # Already logged recently, just update display
                                    pass
                    else:
                        self.status_message = "No faces detected"
                
                # Draw success messages
                display_frame = self.attendance_logger.draw_success_messages(display_frame)
                
                # Draw FPS and system info
                self._draw_system_info(display_frame)
                
                # Show frame if display is enabled
                if self.display_window:
                    cv2.imshow("Enhanced Attendance System", display_frame)
                
                # Process key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC or Q
                    # Quit
                    running = False
                elif key == ord('p'):
                    # Toggle processing
                    self.processing_enabled = not self.processing_enabled
                    self.status_message = f"Processing {'enabled' if self.processing_enabled else 'paused'}"
                    self.status_color = (0, 255, 0) if self.processing_enabled else (0, 165, 255)
                elif key == ord('l'):
                    # Toggle liveness detection
                    self.liveness_detection_enabled = not self.liveness_detection_enabled
                    self.status_message = f"Liveness detection {'enabled' if self.liveness_detection_enabled else 'disabled'}"
                    self.status_color = (0, 255, 0) if self.liveness_detection_enabled else (0, 165, 255)
                elif key == ord('m'):
                    # Toggle model scores display
                    self.show_model_scores = not self.show_model_scores
                    self.status_message = f"Model scores {'enabled' if self.show_model_scores else 'disabled'}"
                    self.status_color = (0, 255, 0) if self.show_model_scores else (0, 165, 255)
                elif key == ord('s'):
                    # Show ensemble statistics
                    self._show_ensemble_stats()
                elif key == ord('+') or key == ord('='):
                    # Increase frame skip (process fewer frames)
                    self.frame_skip = min(10, self.frame_skip + 1)
                    self.status_message = f"Frame skip: {self.frame_skip}"
                elif key == ord('-'):
                    # Decrease frame skip (process more frames)
                    self.frame_skip = max(1, self.frame_skip - 1)
                    self.status_message = f"Frame skip: {self.frame_skip}"
                elif key == ord('r'):
                    # Refresh database
                    self.ensemble_matcher.ensemble_matcher.refresh_embeddings()
                    self.status_message = "Database refreshed"
                elif key == ord('t'):
                    # Save attendance report
                    self._save_attendance_report()
                
                # Increment frame counter
                self.frame_count += 1
        
        # Clean up
        cv2.destroyAllWindows()
    
    def _draw_model_scores(self, frame, bbox, model_scores):
        """
        Draw individual model scores on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            bbox (tuple): Face bounding box
            model_scores (dict): Model name -> score mapping
        """
        if not model_scores:
            return
        
        x, y, w, h = bbox
        y_offset = 0
        
        for model_name, score in model_scores.items():
            if score > 0:
                # Color based on score
                if score >= 0.8:
                    color = (0, 255, 0)  # Green
                elif score >= 0.6:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 165, 255)  # Orange
                
                # Draw model score
                text = f"{model_name}: {score:.2f}"
                cv2.putText(frame, text, (x, y + h + 20 + y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 15
    
    def _draw_system_info(self, frame):
        """
        Draw system information on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
        """
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw frame skip
        cv2.putText(frame, f"Frame Skip: {self.frame_skip}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw ensemble status
        cv2.putText(frame, "Ensemble AI: Active", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw status message
        cv2.putText(frame, self.status_message, (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.status_color, 2)
        
        # Draw processing status
        status = "Processing: ON" if self.processing_enabled else "Processing: PAUSED"
        status_color = (0, 255, 0) if self.processing_enabled else (0, 165, 255)
        cv2.putText(frame, status, (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw liveness detection status
        liveness_status = "Liveness: ON" if self.liveness_detection_enabled else "Liveness: OFF"
        liveness_color = (0, 255, 0) if self.liveness_detection_enabled else (0, 165, 255)
        cv2.putText(frame, liveness_status, (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, liveness_color, 2)
        
        # Draw model scores status
        scores_status = "Model Scores: ON" if self.show_model_scores else "Model Scores: OFF"
        scores_color = (0, 255, 0) if self.show_model_scores else (0, 165, 255)
        cv2.putText(frame, scores_status, (20, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, scores_color, 2)
    
    def _show_ensemble_stats(self):
        """Display ensemble recognition statistics."""
        stats = self.ensemble_matcher.get_ensemble_stats()
        
        print("\n" + "="*60)
        print("📊 ENSEMBLE RECOGNITION STATISTICS")
        print("="*60)
        print(f"🎯 Total Recognitions: {stats['total_recognitions']}")
        print(f"✅ Successful Recognitions: {stats['successful_recognitions']}")
        print(f"❌ Failed Recognitions: {stats['failed_recognitions']}")
        print(f"📈 Overall Accuracy: {stats['overall_accuracy']:.2%}")
        
        print("\n🤖 MODEL PERFORMANCE:")
        for model_name, model_stats in stats['model_performance'].items():
            if model_stats['total'] > 0:
                accuracy = model_stats['accuracy']
                success = model_stats['success']
                total = model_stats['total']
                print(f"   • {model_name.upper()}: {accuracy:.2%} ({success}/{total})")
        
        print("="*60)
        
        # Update status message
        self.status_message = f"Accuracy: {stats['overall_accuracy']:.1%}"
        self.status_color = (0, 255, 0) if stats['overall_accuracy'] > 0.8 else (0, 165, 255)
    
    def _update_fps(self):
        """Update the FPS counter."""
        current_time = time.time()
        elapsed = current_time - self.last_fps_update
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def _save_attendance_report(self):
        """Save attendance report to Excel file."""
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_attendance_report_{timestamp}.xlsx"
        
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Full path to report
        report_path = os.path.join(reports_dir, filename)
        
        # Export attendance data
        success, message = self.attendance_logger.export_attendance_report(report_path)
        
        if success:
            self.status_message = f"Report saved: {filename}"
            self.status_color = (0, 255, 0)
        else:
            self.status_message = f"Error saving report: {message}"
            self.status_color = (0, 0, 255)
        
        print(message)


def main():
    """Main function for the enhanced real-time recognition system."""
    parser = argparse.ArgumentParser(description="Enhanced Real-time Face Recognition System")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, 
                        help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH, 
                        help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT, 
                        help="Frame height (default: 480)")
    parser.add_argument("--threshold", type=float, default=RECOGNITION_THRESHOLD, 
                        help="Recognition threshold (default: 0.5)")
    parser.add_argument("--no-display", action="store_true", 
                        help="Disable display window")
    
    args = parser.parse_args()
    
    # Create and start the enhanced recognition system
    system = EnhancedRealTimeRecognition(
        camera_index=args.camera,
        resolution=(args.width, args.height),
        display_window=not args.no_display,
        recognition_threshold=args.threshold
    )
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("Enhanced recognition system stopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 