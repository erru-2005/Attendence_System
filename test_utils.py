#!/usr/bin/env python3
"""
Test script to verify utils module imports work correctly.
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_utils_imports():
    """Test that all utils modules can be imported."""
    try:
        print("Testing utils module imports...")
        
        # Test optimization module
        from attendance_system.utils.optimization import MemoryCache, BatchProcessor
        print("✓ MemoryCache and BatchProcessor imported successfully")
        
        # Test image processing module
        from attendance_system.utils.image_processing import (
            extract_faces, preprocess_face, normalize_face,
            detect_faces, align_face, enhance_face_quality,
            detect_blur, crop_face, is_face_centered, detect_liveness
        )
        print("✓ Image processing functions imported successfully")
        
        # Test camera utils module
        from attendance_system.utils.camera_utils import (
            CameraStream, draw_face_rectangle, enhance_frame
        )
        print("✓ Camera utils imported successfully")
        
        # Test utils package
        from attendance_system.utils import (
            MemoryCache, BatchProcessor,
            extract_faces, preprocess_face, normalize_face,
            detect_faces, align_face, enhance_face_quality,
            detect_blur, crop_face, is_face_centered, detect_liveness,
            CameraStream, draw_face_rectangle, enhance_frame
        )
        print("✓ Utils package imported successfully")
        
        print("\nAll utils module imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_database_manager_import():
    """Test that database_manager can import utils."""
    try:
        print("\nTesting database_manager import...")
        from attendance_system.registration.database_manager import DatabaseManager
        print("✓ DatabaseManager imported successfully")
        return True
    except ImportError as e:
        print(f"✗ DatabaseManager import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Testing attendance system utils module...")
    
    success1 = test_utils_imports()
    success2 = test_database_manager_import()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The utils module is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
