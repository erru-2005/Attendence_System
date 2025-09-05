"""
Recognition module - AI-powered face recognition functions
"""

# Import main recognition classes
from .enhanced_real_time_recognition import EnhancedRealTimeRecognition
from .ensemble_matcher import EnsembleFaceMatcher, MultiFaceEnsembleMatcher
from .real_time_recognition import RealTimeRecognition
from .attendance_logger import AttendanceLogger
from .face_matcher import FaceMatcher
from .ensemble_runner import EnsembleRunner, run_ensemble_attendance, run_ensemble_demo
from .advanced_face_recognition import AdvancedFaceRecognition
from .optimized_face_recognition import OptimizedFaceRecognition
from .student_manager import StudentManager
from .deepface_recognition import DeepFaceRecognition

# Import utility functions
from .enhanced_real_time_recognition import main as run_enhanced_attendance

__all__ = [
    'EnhancedRealTimeRecognition',
    'EnsembleFaceMatcher', 
    'MultiFaceEnsembleMatcher',
    'RealTimeRecognition',
    'AttendanceLogger',
    'FaceMatcher',
    'EnsembleRunner',
    'AdvancedFaceRecognition',
    'OptimizedFaceRecognition',
    'StudentManager',
    'DeepFaceRecognition',
    'run_enhanced_attendance',
    'run_ensemble_attendance',
    'run_ensemble_demo'
]
