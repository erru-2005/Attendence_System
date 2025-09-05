"""
Face matching module for attendance recognition system.
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any

from ..config.settings import RECOGNITION_THRESHOLD, IN_MEMORY_DB_LIMIT
from ..registration.database_manager import DatabaseManager
from ..registration.embedding_extraction import FaceEmbeddingExtractor
from ..utils.optimization import BatchProcessor, MemoryCache
from .embedding_updater import EmbeddingUpdater


class FaceMatcher:
    """
    Matches detected faces against registered faces in the database.
    """
    
    def __init__(self, recognition_threshold=RECOGNITION_THRESHOLD):
        """
        Initialize the face matcher.
        
        Args:
            recognition_threshold (float): Threshold for face recognition
        """
        self.recognition_threshold = recognition_threshold
        self.db_manager = DatabaseManager()
        self.embedding_extractor = FaceEmbeddingExtractor()
        
        # Cache for embeddings to improve performance
        self.embedding_cache = MemoryCache(max_size=IN_MEMORY_DB_LIMIT)
        
        # Preload embeddings into memory
        self._preload_embeddings()
        
        # Tracking for recent matches (to avoid duplicate detections)
        self.recent_matches = {}  # roll_number -> timestamp
        self.match_cooldown = 2.0  # seconds
        
        # Initialize embedding updater
        self.embedding_updater = EmbeddingUpdater()
    
    def _preload_embeddings(self):
        """Preload student embeddings into memory for faster matching."""
        print("Preloading student embeddings...")
        
        # Get all embeddings from database
        all_embeddings, all_roll_numbers = self.db_manager.get_all_embeddings()
        
        # Store in cache
        for i, roll_number in enumerate(all_roll_numbers):
            embedding = all_embeddings[i]
            
            # Group embeddings by roll number
            if roll_number not in self.embedding_cache.cache:
                self.embedding_cache.set(roll_number, [embedding])
            else:
                # Append to existing embeddings
                current_embeddings = self.embedding_cache.get(roll_number)
                current_embeddings.append(embedding)
                self.embedding_cache.set(roll_number, current_embeddings)
        
        print(f"Loaded embeddings for {len(self.embedding_cache.cache)} students")
    
    def match_face(self, face_embedding: np.ndarray, face_image: np.ndarray = None) -> Tuple[bool, Optional[str], float]:
        """
        Match a face embedding against registered faces.
        
        Args:
            face_embedding (numpy.ndarray): Face embedding to match
            face_image (numpy.ndarray): Original face image for potential embedding update
            
        Returns:
            tuple: (is_match, student_id, confidence)
        """
        if face_embedding is None:
            return False, None, 0.0
        
        start_time = time.time()
        
        best_match = None
        best_score = -1
        best_roll_number = None
        best_embedding_trusted = False
        
        # Compare with all stored embeddings
        for roll_number, embeddings in self.embedding_cache.cache.items():
            # Check if this roll number was matched recently
            if roll_number in self.recent_matches:
                last_match_time = self.recent_matches[roll_number]
                if time.time() - last_match_time < self.match_cooldown:
                    # Skip this roll number if it was matched recently
                    continue
            
            for ref_embedding in embeddings:
                # Calculate similarity score
                score = self.embedding_extractor.calculate_similarity(face_embedding, ref_embedding)
                
                # Update best match if better
                if score > best_score:
                    best_score = score
                    best_roll_number = roll_number
                    # Check if this embedding is trusted (for now, assume all are untrusted)
                    # In a real implementation, you'd check the embedding's trusted status
                    best_embedding_trusted = False  # Default to untrusted
        
        match_time = time.time() - start_time
        
        # Check if the best match exceeds the threshold
        is_match = best_score >= self.recognition_threshold
        
        # Record match time for rate limiting
        if is_match:
            self.recent_matches[best_roll_number] = time.time()
            
            # Handle different confidence ranges
            if face_image is not None:
                embedding_created = False
                
                if 0.5 <= best_score <= 0.7:
                    # Face change detected (like Face ID)
                    print(f"🔄 FACE CHANGE detected for student {best_roll_number} (score: {best_score:.3f})")
                    print(f"   📝 Analyzing face and creating new embedding...")
                    embedding_created = self.embedding_updater.update_face_change_embedding(
                        best_roll_number, face_image, best_score, "insightface"
                    )
                elif best_score > 0.7 and not best_embedding_trusted:
                    # High confidence untrusted match - create trusted embedding
                    print(f"🔒 Creating trusted embedding for student {best_roll_number} (score: {best_score:.3f})")
                    embedding_created = self.embedding_updater.update_trusted_embedding_on_match(
                        best_roll_number, face_image, best_score, "insightface"
                    )
                elif self.embedding_updater.should_update_embedding(best_roll_number, best_score):
                    # Regular embedding update (if enabled)
                    embedding_created = self.embedding_updater.update_embedding(
                        best_roll_number, face_image, best_score, "insightface", trusted=False
                    )
                
                # Refresh embeddings cache if new embedding was created
                if embedding_created:
                    print(f"🔄 Refreshing embeddings cache for student {best_roll_number}...")
                    self.refresh_embeddings()
        
        # Print recognition percentage clearly
        percentage = best_score * 100
        if is_match:
            print(f"✅ RECOGNIZED: {best_roll_number} - {percentage:.1f}% confidence")
        else:
            print(f"❌ NO MATCH - Best score: {percentage:.1f}%")
        
        print(f"Match result: {is_match}, Roll: {best_roll_number}, Score: {best_score:.4f}, Time: {match_time*1000:.1f}ms")
        
        return is_match, best_roll_number, best_score
    
    def match_face_batch(self, face_embeddings: List[np.ndarray]) -> List[Tuple[bool, Optional[str], float]]:
        """
        Match multiple face embeddings against registered faces.
        
        Args:
            face_embeddings (List[numpy.ndarray]): List of face embeddings to match
            
        Returns:
            List[tuple]: List of (is_match, student_id, confidence) tuples
        """
        results = []
        for embedding in face_embeddings:
            result = self.match_face(embedding)
            results.append(result)
        return results
    
    def refresh_embeddings(self):
        """Refresh the embedding cache from the database."""
        # Clear existing cache
        self.embedding_cache.clear()
        
        # Reload embeddings
        self._preload_embeddings()
    
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


class MultiFaceMatcher:
    """
    Extended face matcher that can handle multiple faces in a single frame.
    Implements additional strategies for robust recognition.
    """
    
    def __init__(self, recognition_threshold=RECOGNITION_THRESHOLD,
                 temporal_consistency_frames=5):
        """
        Initialize the multi-face matcher.
        
        Args:
            recognition_threshold (float): Threshold for face recognition
            temporal_consistency_frames (int): Number of frames for temporal consistency
        """
        self.face_matcher = FaceMatcher(recognition_threshold)
        self.temporal_consistency_frames = temporal_consistency_frames
        
        # Temporal consistency tracking
        self.temporal_matches = {}  # face_id -> {roll_number: count}
        self.max_faces = 10  # Maximum number of faces to track
        
        # Track face positions to maintain identity between frames
        self.tracked_faces = {}  # face_id -> (bbox, roll_number, last_seen)
        self.next_face_id = 0
    
    def match_faces(self, face_bboxes: List[Tuple[int, int, int, int]],
                    face_embeddings: List[np.ndarray], face_images: List[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Match multiple faces in a frame with temporal consistency.
        
        Args:
            face_bboxes (List[tuple]): List of face bounding boxes (x, y, w, h)
            face_embeddings (List[numpy.ndarray]): List of face embeddings
            
        Returns:
            List[dict]: List of match results with face_id, bbox, roll_number, confidence, is_match
        """
        if not face_bboxes or not face_embeddings or len(face_bboxes) != len(face_embeddings):
            return []
        
        # Ensure face_images list exists
        if face_images is None:
            face_images = [None] * len(face_embeddings)
        
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
                    "is_match": True
                })
        
        # For unassigned detections, perform face recognition
        for i in range(len(face_bboxes)):
            if i in assigned_detections:
                continue
                
            bbox = face_bboxes[i]
            embedding = face_embeddings[i]
            
            # Match face against database
            face_image = face_images[i] if i < len(face_images) else None
            is_match, roll_number, confidence = self.face_matcher.match_face(embedding, face_image)
            
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
                    "is_match": True
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
                    "is_match": False
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
    
    def reset(self):
        """Reset the matcher state."""
        self.temporal_matches = {}
        self.tracked_faces = {}
        self.face_matcher.reset_recent_matches()
        self.next_face_id = 0
