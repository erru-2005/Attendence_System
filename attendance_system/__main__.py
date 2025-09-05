"""
Main entry point for the face recognition attendance system.
"""
import os
import sys
import argparse
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from attendance_system.registration.face_capture import main as run_registration
from attendance_system.recognition.real_time_recognition import main as run_recognition
from attendance_system.download_models import main as download_models


def main():
    """Main entry point for the attendance system."""
    parser = argparse.ArgumentParser(
        description="Face Recognition Attendance System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Registration command
    register_parser = subparsers.add_parser("register", help="Run student registration system")
    register_parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    register_parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    register_parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    
    # Recognition command
    recognize_parser = subparsers.add_parser("recognize", help="Run attendance recognition system")
    recognize_parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    recognize_parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    recognize_parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    recognize_parser.add_argument("--threshold", type=float, default=0.5, help="Recognition threshold (default: 0.5)")
    recognize_parser.add_argument("--no-display", action="store_true", help="Disable display window")
    
    # Download models command
    download_parser = subparsers.add_parser("download", help="Download required models")
    download_parser.add_argument("--insightface", action="store_true", help="Download only InsightFace models")
    download_parser.add_argument("--opencv", action="store_true", help="Download only OpenCV DNN models")
    download_parser.add_argument("--facenet", action="store_true", help="Download only FaceNet model")
    
    args = parser.parse_args()
    
    if args.command == "register":
        # Set sys.argv for the registration script
        sys.argv = [
            "face_capture.py",
            "--camera", str(args.camera),
            "--width", str(args.width),
            "--height", str(args.height)
        ]
        run_registration()
    
    elif args.command == "recognize":
        # Set sys.argv for the recognition script
        sys.argv = [
            "real_time_recognition.py",
            "--camera", str(args.camera),
            "--width", str(args.width),
            "--height", str(args.height),
            "--threshold", str(args.threshold)
        ]
        
        if args.no_display:
            sys.argv.append("--no-display")
        
        run_recognition()
    
    elif args.command == "download":
        # Set sys.argv for the download script
        sys.argv = ["download_models.py"]
        
        if args.insightface:
            sys.argv.append("--insightface")
        elif args.opencv:
            sys.argv.append("--opencv")
        elif args.facenet:
            sys.argv.append("--facenet")
        
        download_models()
    
    else:
        # If no command is provided, show help
        parser.print_help()


if __name__ == "__main__":
    main() 