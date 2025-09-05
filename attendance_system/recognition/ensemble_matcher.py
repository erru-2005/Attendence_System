"""
Ensemble Face Recognition Module
Combines multiple AI models for improved accuracy using score-level fusion.
"""
import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..config.settings import RECOGNITION_THRESHOLD
from ..registration.database_manager import DatabaseManager
from ..registration.embedding_extraction import FaceEmbeddingExtractor
from ..utils.optimization import MemoryCache
from .embedding_updater import EmbeddingUpdater


class ModelType(Enum):
    """Supported face recognition models."""
    ARCFACE = "arcface"
    FACENET = "facenet"
    MOBILEFACENET = "mobilefacenet"
    DLIB = "dlib"


@dataclass
class ModelConfig:
    """Configuration for a face recognition model."""
    name: str
    weight: float
    threshold: float
    embedding_size: int
    enabled: bool = True


class EnsembleFaceMatcher:
    """
    Ensemble face recognition system that combines multiple AI models
    using score-level fusion for improved accuracy.
    """
    
    def __init__(self, recognition_threshold=RECOGNITION_THRESHOLD):
        """
        Initialize the ensemble face matcher.
        
        Args:
            recognition_threshold (float): Overall recognition threshold
        """
        self.recognition_threshold = recognition_threshold
        self.db_manager = DatabaseManager()
        
        # Initialize model configurations with recommended weights
        self.model_configs = {
            ModelType.ARCFACE: ModelConfig(
                name="arcface",
                weight=0.5,  # Best accuracy
                threshold=0.6,
                embedding_size=512,
                enabled=True
            ),
            ModelType.FACENET: ModelConfig(
                name="facenet", 
                weight=0.3,  # Stable and fast
                threshold=0.7,
                embedding_size=512,
                enabled=True
            ),
            ModelType.MOBILEFACENET: ModelConfig(
                name="mobilefacenet",
                weight=0.2,  # Lightweight fallback
                threshold=0.65,
                embedding_size=128,
                enabled=True
            )
        }
        
        # Initialize embedding extractors for each model
        self.embedding_extractors = {}
        self._initialize_models()
        
        # Cache for embeddings to improve performance
        self.embedding_cache = MemoryCache(max_size=1000)
        
        # Preload embeddings into memory
        self._preload_embeddings()
        
        # Tracking for recent matches (to avoid duplicate detections)
        self.recent_matches = {}  # roll_number -> timestamp
        self.match_cooldown = 2.0  # seconds
        
        # Initialize embedding updater
        self.embedding_updater = EmbeddingUpdater()
        
        # Performance tracking
        self.recognition_stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'model_performance': {model.value: {'success': 0, 'total': 0} for model in ModelType}
        }
        
        logging.info("Ensemble Face Matcher initialized successfully")
    
    def _initialize_models(self):
        """Initialize embedding extractors for each enabled model."""
        for model_type, config in self.model_configs.items():
            if not config.enabled:
                continue
                
            try:
                extractor = FaceEmbeddingExtractor(model_name=config.name)
                self.embedding_extractors[model_type] = extractor
                logging.info(f"Initialized {config.name} model (weight: {config.weight})")
            except Exception as e:
                logging.error(f"Failed to initialize {config.name} model: {e}")
                config.enabled = False
    
    def _preload_embeddings(self):
        """Preload student embeddings into memory for faster matching."""
        logging.info("Preloading student embeddings for ensemble matching...")
        
        # Get all embeddings from database
        all_embeddings, all_roll_numbers = self.db_manager.get_all_embeddings()
        
        # Store in cache grouped by roll number
        for i, roll_number in enumerate(all_roll_numbers):
            embedding = all_embeddings[i]
            
            if roll_number not in self.embedding_cache.cache:
                self.embedding_cache.set(roll_number, [embedding])
            else:
                current_embeddings = self.embedding_cache.get(roll_number)
                current_embeddings.append(embedding)
                self.embedding_cache.set(roll_number, current_embeddings)
        
        logging.info(f"Loaded embeddings for {len(self.embedding_cache.cache)} students")
    
    def extract_ensemble_embeddings(self, face_image: np.ndarray) -> Dict[ModelType, np.ndarray]:
        """
        Extract embeddings from all enabled models.
        
        Args:
            face_image (numpy.ndarray): Input face image
            
        Returns:
            dict: Model type -> embedding mapping
        """
        embeddings = {}
        
        for model_type, extractor in self.embedding_extractors.items():
            if not self.model_configs[model_type].enabled:
                continue
                
            try:
                embedding = extractor.extract_embedding(face_image)
                if embedding is not None and embedding.size > 0:
                    embeddings[model_type] = embedding
            except Exception as e:
                logging.warning(f"Failed to extract embedding from {model_type.value}: {e}")
        
        return embeddings
    
    def calculate_model_similarity(self, query_embedding: np.ndarray, 
                                 reference_embedding: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings using cosine similarity.
        
        Args:
            query_embedding (numpy.ndarray): Query embedding
            reference_embedding (numpy.ndarray): Reference embedding
            
        Returns:
            float: Similarity score (0-1)
        """
        if query_embedding is None or reference_embedding is None:
            return 0.0
        
        # Normalize embeddings
        query_norm = np.linalg.norm(query_embedding)
        ref_norm = np.linalg.norm(reference_embedding)
        
        if query_norm == 0 or ref_norm == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, reference_embedding) / (query_norm * ref_norm)
        return max(0.0, similarity)
    
    def ensemble_fusion(self, model_scores: Dict[ModelType, float]) -> Tuple[float, Dict[str, float]]:
        """
        Perform score-level fusion of multiple model predictions.
        
        Args:
            model_scores (dict): Model type -> similarity score mapping
            
        Returns:
            tuple: (final_score, detailed_scores)
        """
        if not model_scores:
            return 0.0, {}
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        detailed_scores = {}
        
        for model_type, score in model_scores.items():
            if not self.model_configs[model_type].enabled:
                continue
                
            weight = self.model_configs[model_type].weight
            weighted_sum += score * weight
            total_weight += weight
            detailed_scores[model_type.value] = score
        
        if total_weight == 0:
            return 0.0, detailed_scores
        
        final_score = weighted_sum / total_weight
        return final_score, detailed_scores
    
    def match_face_ensemble(self, face_image: np.ndarray) -> Tuple[bool, Optional[str], float, Dict]:
        """
        Match a face using ensemble of multiple models.
        
        Args:
            face_image (numpy.ndarray): Face image to match
            
        Returns:
            tuple: (is_match, student_id, confidence, detailed_results)
        """
        if face_image is None:
            return False, None, 0.0, {}
        
        start_time = time.time()
        
        # Extract embeddings from all models
        model_embeddings = self.extract_ensemble_embeddings(face_image)
        
        if not model_embeddings:
            logging.warning("No embeddings extracted from any model")
            return False, None, 0.0, {}
        
        best_match = None
        best_score = -1
        best_roll_number = None
        best_model_scores = {}
        
        # Compare with all stored embeddings
        for roll_number, embeddings in self.embedding_cache.cache.items():
            # Check if this roll number was matched recently
            if roll_number in self.recent_matches:
                last_match_time = self.recent_matches[roll_number]
                if time.time() - last_match_time < self.match_cooldown:
                    continue
            
            # Calculate similarity scores for each model
            model_scores = {}
            
            for model_type, query_embedding in model_embeddings.items():
                # Use the first embedding for this roll number (can be extended to use multiple)
                ref_embedding = embeddings[0] if embeddings else None
                
                if ref_embedding is not None:
                    similarity = self.calculate_model_similarity(query_embedding, ref_embedding)
                    model_scores[model_type] = similarity
            
            # Perform ensemble fusion
            if model_scores:
                ensemble_score, detailed_scores = self.ensemble_fusion(model_scores)
                
                # Update best match if better
                if ensemble_score > best_score:
                    best_score = ensemble_score
                    best_roll_number = roll_number
                    best_model_scores = detailed_scores
        
        match_time = time.time() - start_time
        
        # Check if the best match exceeds the threshold
        is_match = best_score >= self.recognition_threshold
        
        # Record match time for rate limiting
        if is_match:
            self.recent_matches[best_roll_number] = time.time()
            
            # Check if we should update embeddings
            if self.embedding_updater.should_update_embedding(best_roll_number, best_score):
                self.embedding_updater.update_embedding(
                    best_roll_number, face_image, best_score, "ensemble"
                )
        
        # Update statistics
        self.recognition_stats['total_recognitions'] += 1
        if is_match:
            self.recognition_stats['successful_recognitions'] += 1
        else:
            self.recognition_stats['failed_recognitions'] += 1
        
        # Update model performance statistics
        for model_type, score in best_model_scores.items():
            if model_type in self.recognition_stats['model_performance']:
                self.recognition_stats['model_performance'][model_type]['total'] += 1
                if score >= self.model_configs[ModelType(model_type)].threshold:
                    self.recognition_stats['model_performance'][model_type]['success'] += 1
        
        detailed_results = {
            'model_scores': best_model_scores,
            'match_time': match_time,
            'models_used': list(model_embeddings.keys()),
            'ensemble_threshold': self.recognition_threshold
        }
        
        logging.info(f"Ensemble match: {is_match}, Roll: {best_roll_number}, "
                    f"Score: {best_score:.4f}, Time: {match_time*1000:.1f}ms")
        
        return is_match, best_roll_number, best_score, detailed_results
    
    def match_faces_batch(self, face_images: List[np.ndarray]) -> List[Tuple[bool, Optional[str], float, Dict]]:
        """
        Match multiple faces using ensemble approach.
        
        Args:
            face_images (List[numpy.ndarray]): List of face images to match
            
        Returns:
            List[tuple]: List of (is_match, student_id, confidence, detailed_results) tuples
        """
        results = []
        for face_image in face_images:
            result = self.match_face_ensemble(face_image)
            results.append(result)
        return results
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for all models.
        
        Returns:
            dict: Performance statistics
        """
        stats = self.recognition_stats.copy()
        
        # Calculate accuracy rates
        total = stats['total_recognitions']
        if total > 0:
            stats['overall_accuracy'] = stats['successful_recognitions'] / total
        else:
            stats['overall_accuracy'] = 0.0
        
        # Calculate model-specific accuracies
        for model_name, model_stats in stats['model_performance'].items():
            if model_stats['total'] > 0:
                model_stats['accuracy'] = model_stats['success'] / model_stats['total']
            else:
                model_stats['accuracy'] = 0.0
        
        return stats
    
    def update_model_weights(self, new_weights: Dict[str, float]):
        """
        Update model weights based on performance or user preference.
        
        Args:
            new_weights (dict): Model name -> new weight mapping
        """
        for model_name, weight in new_weights.items():
            for model_type in ModelType:
                if model_type.value == model_name:
                    self.model_configs[model_type].weight = weight
                    logging.info(f"Updated {model_name} weight to {weight}")
    
    def enable_model(self, model_name: str, enabled: bool = True):
        """
        Enable or disable a specific model.
        
        Args:
            model_name (str): Name of the model to enable/disable
            enabled (bool): Whether to enable the model
        """
        for model_type in ModelType:
            if model_type.value == model_name:
                self.model_configs[model_type].enabled = enabled
                logging.info(f"{'Enabled' if enabled else 'Disabled'} {model_name} model")
                break
    
    def refresh_embeddings(self):
        """Refresh the embedding cache from the database."""
        self.embedding_cache.clear()
        self._preload_embeddings()
        logging.info("Embedding cache refreshed")
    
    def reset_recent_matches(self):
        """Reset the recent matches tracking."""
        self.recent_matches = {}
    
    def set_match_cooldown(self, seconds: float):
        """
        Set the cooldown period for rate limiting repeated matches.
        
        Args:
            seconds (float): Cooldown period in seconds
        """
        self.match_cooldown = seconds
    
    def set_recognition_threshold(self, threshold: float):
        """
        Set the overall recognition threshold.
        
        Args:
            threshold (float): New recognition threshold
        """
        self.recognition_threshold = threshold
        logging.info(f"Updated recognition threshold to {threshold}")


class MultiFaceEnsembleMatcher:
    """
    Extended ensemble matcher that can handle multiple faces in a single frame
    with temporal consistency and ensemble fusion.
    """
    
    def __init__(self, recognition_threshold=RECOGNITION_THRESHOLD,
                 temporal_consistency_frames=5):
        """
        Initialize the multi-face ensemble matcher.
        
        Args:
            recognition_threshold (float): Threshold for face recognition
            temporal_consistency_frames (int): Number of frames for temporal consistency
        """
        self.ensemble_matcher = EnsembleFaceMatcher(recognition_threshold)
        self.temporal_consistency_frames = temporal_consistency_frames
        
        # Temporal consistency tracking
        self.temporal_matches = {}  # face_id -> {roll_number: count}
        self.max_faces = 10  # Maximum number of faces to track
        
        # Track face positions to maintain identity between frames
        self.tracked_faces = {}  # face_id -> (bbox, roll_number, last_seen)
        self.next_face_id = 0
    
    def match_faces_ensemble(self, face_bboxes: List[Tuple[int, int, int, int]],
                           face_images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Match multiple faces in a frame using ensemble approach with temporal consistency.
        
        Args:
            face_bboxes (List[tuple]): List of face bounding boxes (x, y, w, h)
            face_images (List[numpy.ndarray]): List of face images
            
        Returns:
            List[dict]: List of match results with face_id, bbox, roll_number, confidence, is_match, model_scores
        """
        if not face_bboxes or not face_images or len(face_bboxes) != len(face_images):
            return []
        
        current_time = time.time()
        results = []
        
        # Associate detections with tracked faces
        assigned_detections = set()
        assigned_tracks = set()
        
        # First, try to associate based on IOU (Intersection over Union)
        for i, bbox in enumerate(face_bboxes):
            best_iou = 0.4  # Minimum IOU threshold
            best_face_id = None
            
            for face_id, (tracked_bbox, _, last_seen) in self.tracked_faces.items():
                # Skip if this track was already assigned
                if face_id in assigned_tracks:
                    continue
                
                # Skip if the track is too old (5 seconds)
                if current_time - last_seen > 5.0:
                    continue
                
                # Calculate IOU
                iou = self._calculate_iou(bbox, tracked_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_face_id = face_id
            
            if best_face_id is not None:
                # Update the tracked face with new bbox
                roll_number = self.tracked_faces[best_face_id][1]
                self.tracked_faces[best_face_id] = (bbox, roll_number, current_time)
                
                # Mark as assigned
                assigned_detections.add(i)
                assigned_tracks.add(best_face_id)
                
                # Add to results
                results.append({
                    "face_id": best_face_id,
                    "bbox": bbox,
                    "roll_number": roll_number,
                    "confidence": 1.0,  # Using tracked identity, high confidence
                    "is_match": True,
                    "model_scores": {"tracked": 1.0}
                })
        
        # For unassigned detections, perform ensemble face recognition
        for i in range(len(face_bboxes)):
            if i in assigned_detections:
                continue
                
            bbox = face_bboxes[i]
            face_image = face_images[i]
            
            # Match face using ensemble approach
            is_match, roll_number, confidence, detailed_results = self.ensemble_matcher.match_face_ensemble(face_image)
            
            if is_match:
                # Create new face ID for tracking
                face_id = self.next_face_id
                self.next_face_id += 1
                
                # Add to tracked faces
                self.tracked_faces[face_id] = (bbox, roll_number, current_time)
                
                # Initialize temporal consistency
                self.temporal_matches[face_id] = {roll_number: 1}
                
                # Add to results
                results.append({
                    "face_id": face_id,
                    "bbox": bbox,
                    "roll_number": roll_number,
                    "confidence": confidence,
                    "is_match": True,
                    "model_scores": detailed_results.get("model_scores", {})
                })
            else:
                # No match found
                # Create new face ID for tracking unknown face
                face_id = self.next_face_id
                self.next_face_id += 1
                
                results.append({
                    "face_id": face_id,
                    "bbox": bbox,
                    "roll_number": None,
                    "confidence": confidence,
                    "is_match": False,
                    "model_scores": detailed_results.get("model_scores", {})
                })
        
        # Clean up old tracks
        self._clean_old_tracks(current_time)
        
        return results
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1 (tuple): First bounding box (x, y, w, h)
            bbox2 (tuple): Second bounding box (x, y, w, h)
            
        Returns:
            float: IOU score
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate coordinates of intersection
        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1 + w1, x2 + w2)
        yy2 = min(y1 + h1, y2 + h2)
        
        # Check if there is an intersection
        if xx2 < xx1 or yy2 < yy1:
            return 0.0
        
        # Area of intersection
        intersection = (xx2 - xx1) * (yy2 - yy1)
        
        # Areas of each bbox
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Area of union
        union = area1 + area2 - intersection
        
        # IOU
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def _clean_old_tracks(self, current_time, max_age=5.0):
        """
        Remove old tracks that haven't been seen for a while.
        
        Args:
            current_time (float): Current timestamp
            max_age (float): Maximum age in seconds before removing a track
        """
        # Identify old tracks
        old_track_ids = []
        for face_id, (_, _, last_seen) in self.tracked_faces.items():
            if current_time - last_seen > max_age:
                old_track_ids.append(face_id)
        
        # Remove old tracks
        for face_id in old_track_ids:
            if face_id in self.tracked_faces:
                del self.tracked_faces[face_id]
            
            if face_id in self.temporal_matches:
                del self.temporal_matches[face_id]
        
        # If we have too many tracks, remove the oldest ones
        if len(self.tracked_faces) > self.max_faces:
            # Sort by last seen time
            sorted_tracks = sorted(
                self.tracked_faces.items(),
                key=lambda x: x[1][2]  # Sort by last_seen
            )
            
            # Remove the oldest ones
            tracks_to_remove = sorted_tracks[:len(sorted_tracks) - self.max_faces]
            for face_id, _ in tracks_to_remove:
                del self.tracked_faces[face_id]
                if face_id in self.temporal_matches:
                    del self.temporal_matches[face_id]
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """
        Get ensemble recognition statistics.
        
        Returns:
            dict: Ensemble statistics
        """
        return self.ensemble_matcher.get_model_performance_stats()
    
    def reset(self):
        """Reset the matcher state."""
        self.temporal_matches = {}
        self.tracked_faces = {}
        self.ensemble_matcher.reset_recent_matches()
        self.next_face_id = 0 