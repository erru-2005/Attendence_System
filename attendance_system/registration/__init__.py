"""
Registration module for attendance system
"""

from .database_manager import DatabaseManager
from .embedding_extraction import FaceEmbeddingExtractor, FaceDetector

__all__ = [
    'DatabaseManager',
    'FaceEmbeddingExtractor',
    'FaceDetector'
]
