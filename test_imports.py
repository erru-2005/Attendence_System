#!/usr/bin/env python3
"""
Test script to verify that import issues are resolved.
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all critical imports work."""
    try:
        print("Testing critical imports...")
        
        # Test config imports
        from attendance_system.config.settings import MODEL_DIR
        print("✓ Config settings imported successfully")
        
        from attendance_system.config.model_config import MODELS
        print("✓ Config model_config imported successfully")
        
        # Test utils imports
        from attendance_system.utils.optimization import MemoryCache
        print("✓ Utils optimization imported successfully")
        
        from attendance_system.utils.image_processing import detect_faces
        print("✓ Utils image_processing imported successfully")
        
        from attendance_system.utils.camera_utils import CameraStream
        print("✓ Utils camera_utils imported successfully")
        
        # Test registration imports
        from attendance_system.registration.database_manager import DatabaseManager
        print("✓ Registration database_manager imported successfully")
        
        # Test download_models import
        from attendance_system.download_models import download_all_models
        print("✓ Download models imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Testing attendance system imports...")
    
    if test_imports():
        print("\n✅ All imports are working correctly!")
    else:
        print("\n❌ Some imports failed. Please check the errors above.")
        sys.exit(1)
