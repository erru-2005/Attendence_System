"""
Camera utilities for the attendance system.
Provides camera stream handling and frame processing functions.
"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Callable
from queue import Queue


class CameraStream:
    """
    Thread-safe camera stream handler.
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera stream.
        
        Args:
            camera_id (int): Camera device ID
            width (int): Frame width
            height (int): Frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.stopped = False
        self.frame_queue = Queue(maxsize=1)
        self.lock = threading.Lock()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        
    def start(self) -> bool:
        """
        Start the camera stream.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Start reading thread
            self.stopped = False
            thread = threading.Thread(target=self._update, daemon=True)
            thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting camera stream: {e}")
            return False
    
    def _update(self):
        """Update frame in background thread."""
        while not self.stopped:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame from camera")
                break
            
            # Update frame queue (replace old frame)
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            self.frame_queue.put(frame)
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest frame.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.stopped or self.cap is None:
            return False, None
        
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except:
            return False, None
    
    def stop(self):
        """Stop the camera stream."""
        self.stopped = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def is_running(self) -> bool:
        """Check if camera stream is running."""
        return not self.stopped and self.cap is not None and self.cap.isOpened()
    
    def get_frame_size(self) -> Tuple[int, int]:
        """Get current frame size."""
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return self.width, self.height


def draw_face_rectangle(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                       text: str = "", confidence: float = 0.0,
                       color: Tuple[int, int, int] = (0, 255, 0), 
                       thickness: int = 2) -> np.ndarray:
    """
    Draw rectangle around detected face with optional text.
    
    Args:
        frame (np.ndarray): Input frame
        bbox (Tuple[int, int, int, int]): Face bounding box (x, y, w, h)
        text (str): Text to display
        confidence (float): Confidence score
        color (Tuple[int, int, int]): BGR color for rectangle
        thickness (int): Line thickness
        
    Returns:
        Frame with drawn rectangle
    """
    frame_copy = frame.copy()
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
    
    # Draw text if provided
    if text:
        # Position text above rectangle
        text_y = max(y - 10, 20)
        cv2.putText(frame_copy, text, (x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame_copy


def enhance_frame(frame: np.ndarray, 
                 brightness: float = 1.0, 
                 contrast: float = 1.0,
                 blur_reduction: bool = True) -> np.ndarray:
    """
    Enhance frame quality for better face detection.
    
    Args:
        frame (np.ndarray): Input frame
        brightness (float): Brightness adjustment factor
        contrast (float): Contrast adjustment factor
        blur_reduction (bool): Whether to apply blur reduction
        
    Returns:
        Enhanced frame
    """
    enhanced = frame.copy()
    
    # Apply brightness and contrast
    enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness * 50)
    
    # Apply blur reduction if requested
    if blur_reduction:
        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize frame to target size.
    
    Args:
        frame (np.ndarray): Input frame
        target_size (Tuple[int, int]): Target (width, height)
        
    Returns:
        Resized frame
    """
    return cv2.resize(frame, target_size)


def flip_frame(frame: np.ndarray, flip_code: int = 1) -> np.ndarray:
    """
    Flip frame horizontally or vertically.
    
    Args:
        frame (np.ndarray): Input frame
        flip_code (int): Flip code (1 for horizontal, 0 for vertical)
        
    Returns:
        Flipped frame
    """
    return cv2.flip(frame, flip_code)


def apply_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Apply region of interest to frame.
    
    Args:
        frame (np.ndarray): Input frame
        roi (Tuple[int, int, int, int]): Region of interest (x, y, w, h)
        
    Returns:
        Frame cropped to ROI
    """
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def get_frame_info(frame: np.ndarray) -> dict:
    """
    Get information about the frame.
    
    Args:
        frame (np.ndarray): Input frame
        
    Returns:
        Dictionary with frame information
    """
    if frame is None:
        return {}
    
    height, width = frame.shape[:2]
    channels = frame.shape[2] if len(frame.shape) > 2 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'dtype': str(frame.dtype),
        'size': frame.size
    }
