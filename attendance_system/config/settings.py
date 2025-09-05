"""
Global settings for the Face Recognition Attendance System.
"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATABASE_DIR = os.path.join(DATA_DIR, "database")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, DATABASE_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Database files
STUDENT_DB_FILE = os.path.join(DATABASE_DIR, "students.json")
ATTENDANCE_DB_FILE = os.path.join(DATABASE_DIR, "attendance.json")
BACKUP_DIR = os.path.join(DATABASE_DIR, "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
FRAME_SKIP = 3  # Process every 3rd frame for performance

# Face detection and recognition settings
FACE_CONFIDENCE_THRESHOLD = 0.6
RECOGNITION_THRESHOLD = 0.5
FACE_DETECTION_SIZE = (320, 320)  # Resize for detection
FACE_RECOGNITION_SIZE = (112, 112)  # Standard size for many models

# Registration settings
REQUIRED_FACE_ANGLES = ["front", "slight_left", "slight_right"]
MIN_PHOTOS_PER_ANGLE = 1
MAX_PHOTOS_PER_ANGLE = 3
TOTAL_REQUIRED_PHOTOS = len(REQUIRED_FACE_ANGLES) * MIN_PHOTOS_PER_ANGLE

# Recognition performance settings
USE_GPU = False  # Set to True if GPU is available
BATCH_SIZE = 4  # Number of faces to process in batch
FACE_EMBEDDING_SIZE = 512
IN_MEMORY_DB_LIMIT = 1000  # Max number of students to keep in memory

# Attendance settings
ATTENDANCE_SESSIONS = {
    "morning": {"start": "08:00", "end": "12:00"},
    "afternoon": {"start": "12:01", "end": "17:00"},
    "evening": {"start": "17:01", "end": "21:00"}
}
MIN_RECOGNITION_INTERVAL = 30  # Seconds between attendance records for same person

# UI and feedback settings
DISPLAY_WINDOW = True  # Show camera feed window
AUDIO_FEEDBACK = True  # Play sound on successful recognition
SUCCESS_MESSAGE_DURATION = 2  # Seconds to show success message

# Security settings
ENABLE_LIVENESS_DETECTION = True  # Check if face is real (not a photo)
ENABLE_BLINK_DETECTION = True  # Check for blinking
SUSPICIOUS_ACTIVITY_THRESHOLD = 5  # Number of failed attempts before logging

# Embedding update settings
ENABLE_AUTO_EMBEDDING_UPDATE = False  # Automatically append new embeddings when person is recognized
EMBEDDING_UPDATE_CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence to update embeddings
MAX_EMBEDDINGS_PER_STUDENT = 10  # Maximum number of embeddings to store per student
EMBEDDING_UPDATE_INTERVAL = 300  # Seconds between embedding updates for same person (5 minutes)

# Logging settings
LOG_LEVEL = "INFO"
ENABLE_FILE_LOGGING = True

# Model settings
DEFAULT_FACE_DETECTION_MODEL = "retinaface"  # Options: "retinaface", "opencv_dnn"
DEFAULT_FACE_RECOGNITION_MODEL = "arcface"   # Options: "arcface", "facenet"
DEFAULT_ANTI_SPOOFING_MODEL = "silent"       # Options: "silent", "depth"

# Backup settings
BACKUP_FREQUENCY = 24  # Hours between automatic backups
MAX_BACKUPS = 7  # Maximum number of backup files to keep
