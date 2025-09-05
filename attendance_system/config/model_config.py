"""
Model configurations for face detection, recognition, and anti-spoofing.
"""
import os
from .settings import MODEL_DIR

# Model file paths
MODELS = {
    # Face Detection Models
    "retinaface": {
        "model_file": os.path.join(MODEL_DIR, "retinaface", "retinaface_model.onnx"),
        "config_file": os.path.join(MODEL_DIR, "retinaface", "retinaface_config.yaml"),
        "size": (640, 640),
        "threshold": 0.8,
    },
    "opencv_dnn": {
        "model_file": os.path.join(MODEL_DIR, "opencv_dnn", "opencv_face_detector.caffemodel"),
        "config_file": os.path.join(MODEL_DIR, "opencv_dnn", "opencv_face_detector.prototxt"),
        "size": (300, 300),
        "threshold": 0.6,
    },
    
    # Face Recognition Models
    "arcface": {
        "model_file": os.path.join(MODEL_DIR, "arcface", "arcface_model.onnx"),
        "size": (112, 112),
        "embedding_size": 512,
    },
    "facenet": {
        "model_file": os.path.join(MODEL_DIR, "facenet", "facenet_model.pb"),
        "size": (160, 160),
        "embedding_size": 512,
    },
    "mobilefacenet": {
        "model_file": os.path.join(MODEL_DIR, "mobilefacenet", "mobilefacenet.onnx"),
        "size": (112, 112),
        "embedding_size": 128,
    },
    
    # Anti-Spoofing Models
    "silent": {
        "model_file": os.path.join(MODEL_DIR, "silent", "silent_face_anti_spoofing.onnx"),
        "size": (80, 80),
        "threshold": 0.8,
    },
    "depth": {
        "model_file": os.path.join(MODEL_DIR, "depth", "depth_estimation_model.onnx"),
        "size": (224, 224),
        "threshold": 0.7,
    }
}

# Model download URLs
MODEL_URLS = {
    "retinaface": "https://github.com/example/retinaface/releases/download/v1.0/retinaface_model.onnx",
    "arcface": "https://github.com/example/arcface/releases/download/v1.0/arcface_model.onnx",
    "silent": "https://github.com/example/silent/releases/download/v1.0/silent_face_anti_spoofing.onnx",
    # Add more URLs for other models
}

# Model-specific parameters
FACE_DETECTION_PARAMS = {
    "retinaface": {
        "nms_threshold": 0.4,
        "min_face_size": 20,
    },
    "opencv_dnn": {
        "scale_factor": 1.0,
        "mean_values": (104.0, 177.0, 123.0),
    },
}

FACE_RECOGNITION_PARAMS = {
    "arcface": {
        "normalization": True,
        "normalization_mean": 127.5,
        "normalization_std": 128.0,
    },
    "facenet": {
        "normalization": True,
        "normalization_mean": 127.5,
        "normalization_std": 128.0,
    },
    "mobilefacenet": {
        "normalization": True,
        "normalization_mean": 127.5,
        "normalization_std": 128.0,
    },
}

ANTI_SPOOFING_PARAMS = {
    "silent": {
        "normalization_mean": [0.5, 0.5, 0.5],
        "normalization_std": [0.5, 0.5, 0.5],
    },
    "depth": {
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    },
}

# Function to check if models exist, create directories if they don't
def ensure_model_dirs():
    """Create model directories if they don't exist."""
    for model_type in MODELS:
        model_path = os.path.dirname(MODELS[model_type]["model_file"])
        os.makedirs(model_path, exist_ok=True)

# Create model directories
ensure_model_dirs()
