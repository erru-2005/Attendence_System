"""
Utility modules for the attendance system.
Contains optimization, image processing, and camera utilities.
"""

from .optimization import MemoryCache, BatchProcessor
from .image_processing import (
    extract_faces, preprocess_face, normalize_face,
    detect_faces, align_face, enhance_face_quality,
    detect_blur, crop_face, is_face_centered, detect_liveness
)
from .camera_utils import (
    CameraStream, draw_face_rectangle, enhance_frame
)

__all__ = [
    'MemoryCache', 'BatchProcessor',
    'extract_faces', 'preprocess_face', 'normalize_face',
    'detect_faces', 'align_face', 'enhance_face_quality',
    'detect_blur', 'crop_face', 'is_face_centered', 'detect_liveness',
    'CameraStream', 'draw_face_rectangle', 'enhance_frame'
]
