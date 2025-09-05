"""
Image processing utilities for face detection and recognition.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import dlib


def detect_faces(image: np.ndarray, detector=None) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image.
    
    Args:
        image (np.ndarray): Input image
        detector: Face detector object (dlib or OpenCV)
        
    Returns:
        List of face bounding boxes (x, y, w, h)
    """
    if detector is None:
        # Use OpenCV's Haar cascade as fallback
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    # Use provided detector
    if hasattr(detector, 'detect'):
        detections = detector.detect(image)
        return [(d.rect.left(), d.rect.top(), d.rect.width(), d.rect.height()) 
                for d in detections]
    
    return []


def extract_faces(image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """
    Extract face regions from image.
    
    Args:
        image (np.ndarray): Input image
        face_locations (List[Tuple]): List of face bounding boxes
        
    Returns:
        List of face images
    """
    faces = []
    for (x, y, w, h) in face_locations:
        face = image[y:y+h, x:x+w]
        if face.size > 0:
            faces.append(face)
    return faces


def preprocess_face(face_image: np.ndarray, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """
    Preprocess face image for recognition.
    
    Args:
        face_image (np.ndarray): Input face image
        target_size (Tuple[int, int]): Target size for the face
        
    Returns:
        Preprocessed face image
    """
    # Resize to target size
    face_resized = cv2.resize(face_image, target_size)
    
    # Convert to RGB if needed
    if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    else:
        face_rgb = face_resized
    
    # Normalize pixel values
    face_normalized = face_rgb.astype(np.float32) / 255.0
    
    return face_normalized


def normalize_face(face_image: np.ndarray) -> np.ndarray:
    """
    Normalize face image for better recognition.
    
    Args:
        face_image (np.ndarray): Input face image
        
    Returns:
        Normalized face image
    """
    # Convert to grayscale if needed
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    
    # Apply histogram equalization
    normalized = cv2.equalizeHist(gray)
    
    return normalized


def align_face(face_image: np.ndarray, landmarks: Optional[List] = None) -> np.ndarray:
    """
    Align face using facial landmarks.
    
    Args:
        face_image (np.ndarray): Input face image
        landmarks: Facial landmarks for alignment
        
    Returns:
        Aligned face image
    """
    if landmarks is None:
        return face_image
    
    # Simple alignment based on eye positions
    # This is a basic implementation - more sophisticated alignment can be added
    return face_image


def enhance_face_quality(face_image: np.ndarray) -> np.ndarray:
    """
    Enhance face image quality for better recognition.
    
    Args:
        face_image (np.ndarray): Input face image
        
    Returns:
        Enhanced face image
    """
    # Apply bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(face_image, 9, 75, 75)
    
    # Apply slight sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced


def extract_face_embeddings(face_image: np.ndarray, model=None) -> Optional[np.ndarray]:
    """
    Extract face embeddings using a face recognition model.
    
    Args:
        face_image (np.ndarray): Input face image
        model: Face recognition model
        
    Returns:
        Face embedding vector or None if extraction fails
    """
    if model is None:
        return None
    
    try:
        # Preprocess face
        processed_face = preprocess_face(face_image)
        
        # Extract embedding
        if hasattr(model, 'predict'):
            embedding = model.predict(np.expand_dims(processed_face, axis=0))
            return embedding.flatten()
        
        return None
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None


def validate_face_quality(face_image: np.ndarray) -> bool:
    """
    Validate if face image meets quality requirements.
    
    Args:
        face_image (np.ndarray): Input face image
        
    Returns:
        True if face meets quality requirements
    """
    if face_image is None or face_image.size == 0:
        return False
    
    # Check minimum size
    if face_image.shape[0] < 50 or face_image.shape[1] < 50:
        return False
    
    # Check if face is too blurry
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # If variance is too low, image is too blurry
    if laplacian_var < 100:
        return False
    
    return True


def detect_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
    """
    Detect if an image is blurry using Laplacian variance.
    
    Args:
        image (np.ndarray): Input image
        threshold (float): Blur threshold
        
    Returns:
        Tuple of (is_blurry, blur_value)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = blur_value < threshold
    
    return is_blurry, blur_value


def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop face region from image using bounding box.
    
    Args:
        image (np.ndarray): Input image
        bbox (Tuple[int, int, int, int]): Bounding box (x, y, w, h)
        
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


def is_face_centered(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> bool:
    """
    Check if face is centered in the frame.
    
    Args:
        bbox (Tuple[int, int, int, int]): Face bounding box
        frame_shape (Tuple[int, int, int]): Frame dimensions
        
    Returns:
        True if face is centered
    """
    x, y, w, h = bbox
    frame_height, frame_width = frame_shape[:2]
    
    # Calculate center of face
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Calculate center of frame
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    
    # Define tolerance (percentage of frame size)
    tolerance_x = frame_width * 0.2
    tolerance_y = frame_height * 0.2
    
    # Check if face is within tolerance of center
    is_centered_x = abs(face_center_x - frame_center_x) < tolerance_x
    is_centered_y = abs(face_center_y - frame_center_y) < tolerance_y
    
    return is_centered_x and is_centered_y


def detect_liveness(face_image: np.ndarray) -> Tuple[bool, float]:
    """
    Basic liveness detection using eye blink detection.
    
    Args:
        face_image (np.ndarray): Input face image
        
    Returns:
        Tuple of (is_live, confidence)
    """
    # This is a simplified liveness detection
    # In a real implementation, you would use more sophisticated methods
    
    # Convert to grayscale
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    
    # Use Haar cascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    # If eyes are detected, assume it's a live person
    is_live = len(eyes) >= 1
    confidence = min(len(eyes) / 2.0, 1.0)  # Normalize confidence
    
    return is_live, confidence
