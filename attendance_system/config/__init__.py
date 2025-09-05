"""
Configuration module for attendance system
"""

from .settings import *
from .model_config import *

__all__ = [
    'RECOGNITION_THRESHOLD',
    'CAMERA_INDEX',
    'FRAME_WIDTH', 
    'FRAME_HEIGHT'
]
