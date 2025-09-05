"""
Ensemble Runner - Combined functionality from ensemble attendance and demo
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from datetime import datetime

from .enhanced_real_time_recognition import EnhancedRealTimeRecognition
from .ensemble_matcher import MultiFaceEnsembleMatcher


class EnsembleRunner:
    """
    Combined ensemble runner with demo and attendance functionality
    """
    
    def __init__(self, config_name="default"):
        # Default configuration since ensemble_config is not available
        self.config = type('Config', (), {
            'overall_threshold': 0.5,
            'model_weights': {'arcface': 0.5, 'facenet': 0.3, 'mobilefacenet': 0.2}
        })()
        self.ensemble_matcher = MultiFaceEnsembleMatcher(
            recognition_threshold=self.config.overall_threshold
        )
        
        # Demo statistics
        self.demo_stats = {
            'total_frames': 0,
            'total_faces': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'processing_times': []
        }
        
        print(f"🎯 Ensemble Runner initialized with '{config_name}' configuration")
    
    def run_attendance_mode(self, camera_index=0, resolution=(640, 480), 
                           recognition_threshold=0.5, display_window=True):
        """Run in attendance mode with automatic logging"""
        print(f"\n📊 Starting Ensemble Attendance Mode...")
        print("Controls:")
        print("   • ESC/Q: Quit")
        print("   • S: Show statistics")
        print("   • R: Refresh database")
        
        system = EnhancedRealTimeRecognition(
            camera_index=camera_index,
            resolution=resolution,
            display_window=display_window,
            recognition_threshold=recognition_threshold
        )
        
        try:
            system.start()
        except KeyboardInterrupt:
            print("\n⏹️ Attendance mode stopped by user")
        except Exception as e:
            print(f"\n❌ Error in attendance mode: {e}")
    
    def run_demo_mode(self, camera_index=0, duration=60):
        """Run in demo mode for testing"""
        print(f"\n🎮 Starting Ensemble Demo Mode for {duration} seconds...")
        print("Controls:")
        print("   • ESC/Q: Quit")
        print("   • S: Show statistics")
        
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Check if demo duration exceeded
            if current_time - start_time > duration:
                print(f"\n⏰ Demo completed after {duration} seconds")
                break
            
            # Process frame
            display_frame = self._process_demo_frame(frame, current_time)
            
            # Draw demo info
            self._draw_demo_info(display_frame, current_time - start_time, duration)
            
            # Show frame
            cv2.imshow("Ensemble Demo Mode", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                break
            elif key == ord('s'):
                self._show_demo_statistics()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final statistics
        self._show_final_statistics()
    
    def _process_demo_frame(self, frame, current_time):
        """Process a single frame for demo mode"""
        display_frame = frame.copy()
        
        # Update statistics
        self.demo_stats['total_frames'] += 1
        
        # Draw demo info
        cv2.putText(display_frame, "Ensemble Demo Mode", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_frame
    
    def _draw_demo_info(self, frame, elapsed_time, total_duration):
        """Draw demo information on frame"""
        remaining = max(0, total_duration - elapsed_time)
        cv2.putText(frame, f"Time: {remaining:.1f}s", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, "Ensemble AI Active", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _show_demo_statistics(self):
        """Display demo statistics"""
        print("\n" + "="*50)
        print("📊 DEMO STATISTICS")
        print("="*50)
        print(f"Total Frames: {self.demo_stats['total_frames']}")
        print(f"Total Faces: {self.demo_stats['total_faces']}")
        print(f"Successful Matches: {self.demo_stats['successful_matches']}")
        print(f"Failed Matches: {self.demo_stats['failed_matches']}")
        
        if self.demo_stats['total_faces'] > 0:
            accuracy = self.demo_stats['successful_matches'] / self.demo_stats['total_faces']
            print(f"Accuracy: {accuracy:.2%}")
        
        print("="*50)
    
    def _show_final_statistics(self):
        """Display final demo statistics"""
        print("\n" + "="*60)
        print("🎉 FINAL DEMO STATISTICS")
        print("="*60)
        print(f"📊 FRAME PROCESSING:")
        print(f"   • Total Frames: {self.demo_stats['total_frames']}")
        print(f"   • Total Faces Detected: {self.demo_stats['total_faces']}")
        print(f"\n🎯 RECOGNITION RESULTS:")
        print(f"   • Successful Matches: {self.demo_stats['successful_matches']}")
        print(f"   • Failed Matches: {self.demo_stats['failed_matches']}")
        
        if self.demo_stats['total_faces'] > 0:
            accuracy = self.demo_stats['successful_matches'] / self.demo_stats['total_faces']
            print(f"   • Overall Accuracy: {accuracy:.2%}")
        
        print("="*60)


def run_ensemble_attendance(args):
    """Run ensemble attendance system"""
    runner = EnsembleRunner(args.config)
    runner.run_attendance_mode(
        camera_index=args.camera,
        resolution=(args.width, args.height),
        recognition_threshold=args.threshold,
        display_window=not args.no_display
    )


def run_ensemble_demo(args):
    """Run ensemble demo system"""
    runner = EnsembleRunner(args.config)
    runner.run_demo_mode(
        camera_index=args.camera,
        duration=args.duration
    )


def main():
    """Main function for ensemble runner"""
    parser = argparse.ArgumentParser(description="Ensemble Runner")
    parser.add_argument("--mode", type=str, default="attendance",
                        choices=["attendance", "demo"],
                        help="Run mode")
    parser.add_argument("--config", type=str, default="default",
                        choices=["default", "high_accuracy", "fast", "balanced"],
                        help="Ensemble configuration")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--width", type=int, default=640,
                        help="Frame width")
    parser.add_argument("--height", type=int, default=480,
                        help="Frame height")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Recognition threshold")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable display window")
    parser.add_argument("--duration", type=int, default=60,
                        help="Demo duration in seconds")
    
    args = parser.parse_args()
    
    if args.mode == "attendance":
        run_ensemble_attendance(args)
    elif args.mode == "demo":
        run_ensemble_demo(args)


if __name__ == "__main__":
    main() 