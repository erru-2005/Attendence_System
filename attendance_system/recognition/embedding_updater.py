"""
Embedding Update Module
Handles automatic embedding updates when a person is recognized
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from ..config.settings import (
    ENABLE_AUTO_EMBEDDING_UPDATE,
    EMBEDDING_UPDATE_CONFIDENCE_THRESHOLD,
    MAX_EMBEDDINGS_PER_STUDENT,
    EMBEDDING_UPDATE_INTERVAL
)
from ..registration.embedding_extraction import FaceEmbeddingExtractor


class EmbeddingUpdater:
    """
    Handles automatic embedding updates when a person is recognized.
    """
    
    def __init__(self):
        self.embedding_extractor = FaceEmbeddingExtractor()
        self.embeddings_dir = "database/embeddings"
        self.images_dir = "database/images"
        
        # Ensure directories exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Track last update time for each student to avoid too frequent updates
        self.last_update_times = {}  # student_id -> timestamp
        
        # Track update statistics
        self.update_stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "last_update": None
        }
        
        print("🔄 Embedding Updater initialized!")
    
    def should_update_embedding(self, student_id: str, confidence: float) -> bool:
        """
        Check if embedding should be updated for this recognition.
        
        Args:
            student_id (str): Student ID
            confidence (float): Recognition confidence score
            
        Returns:
            bool: True if embedding should be updated
        """
        if not ENABLE_AUTO_EMBEDDING_UPDATE:
            return False
        
        # Check confidence threshold
        if confidence < EMBEDDING_UPDATE_CONFIDENCE_THRESHOLD:
            return False
        
        # Check update interval
        current_time = time.time()
        last_update = self.last_update_times.get(student_id, 0)
        if current_time - last_update < EMBEDDING_UPDATE_INTERVAL:
            return False
        
        return True
    
    def update_embedding(self, student_id: str, face_image: np.ndarray, 
                        confidence: float, recognition_type: str = "insightface", trusted: bool = False) -> bool:
        """
        Update embedding for a recognized student.
        
        Args:
            student_id (str): Student ID
            face_image (np.ndarray): Face image
            confidence (float): Recognition confidence
            recognition_type (str): Type of recognition model used
            trusted (bool): Whether this embedding is trusted (default: False)
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Extract new embedding
            new_embedding = self.embedding_extractor.extract_embedding(face_image)
            if new_embedding is None:
                print(f"❌ Failed to extract embedding for student {student_id}")
                self.update_stats["failed_updates"] += 1
                return False
            
            # Load existing embeddings
            embedding_file = os.path.join(self.embeddings_dir, f"{student_id}.json")
            embedding_data = self._load_embedding_data(embedding_file, student_id)
            
            # Create new embedding entry
            new_embedding_entry = {
                "type": recognition_type,
                "vector": new_embedding.tolist(),
                "confidence": confidence,
                "capture_date": datetime.now().isoformat(),
                "source": "auto_update",
                "trusted": trusted
            }
            
            # Add to embeddings list
            if "embeddings" not in embedding_data:
                embedding_data["embeddings"] = []
            
            embedding_data["embeddings"].append(new_embedding_entry)
            
            # Limit number of embeddings per student
            if len(embedding_data["embeddings"]) > MAX_EMBEDDINGS_PER_STUDENT:
                # Remove oldest embeddings (keep the most recent ones)
                embedding_data["embeddings"] = embedding_data["embeddings"][-MAX_EMBEDDINGS_PER_STUDENT:]
            
            # Update metadata
            embedding_data["last_updated"] = datetime.now().isoformat()
            embedding_data["total_embeddings"] = len(embedding_data["embeddings"])
            
            # Save updated embeddings
            with open(embedding_file, 'w') as f:
                json.dump(embedding_data, f, indent=2)
            
            # Update tracking
            self.last_update_times[student_id] = time.time()
            self.update_stats["successful_updates"] += 1
            self.update_stats["total_updates"] += 1
            self.update_stats["last_update"] = datetime.now().isoformat()
            
            print(f"✅ Updated embedding for student {student_id} (confidence: {confidence:.3f}, trusted: {trusted})")
            return True
            
        except Exception as e:
            print(f"❌ Error updating embedding for student {student_id}: {e}")
            self.update_stats["failed_updates"] += 1
            self.update_stats["total_updates"] += 1
            return False
    
    def update_trusted_embedding_on_match(self, student_id: str, face_image: np.ndarray, 
                                         confidence: float, recognition_type: str = "insightface") -> bool:
        """
        Update embedding with trusted=True when an untrusted embedding matches during recognition.
        This helps adapt to face changes over time.
        
        Args:
            student_id (str): Student ID
            face_image (np.ndarray): Face image
            confidence (float): Recognition confidence
            recognition_type (str): Type of recognition model used
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Extract new embedding
            new_embedding = self.embedding_extractor.extract_embedding(face_image)
            if new_embedding is None:
                print(f"❌ Failed to extract trusted embedding for student {student_id}")
                return False
            
            # Load existing embeddings
            embedding_file = os.path.join(self.embeddings_dir, f"{student_id}.json")
            embedding_data = self._load_embedding_data(embedding_file, student_id)
            
            # Create new trusted embedding entry
            new_embedding_entry = {
                "type": recognition_type,
                "vector": new_embedding.tolist(),
                "confidence": confidence,
                "capture_date": datetime.now().isoformat(),
                "source": "trusted_update",
                "trusted": True
            }
            
            # Add to embeddings list
            if "embeddings" not in embedding_data:
                embedding_data["embeddings"] = []
            
            embedding_data["embeddings"].append(new_embedding_entry)
            
            # Limit number of embeddings per student
            if len(embedding_data["embeddings"]) > MAX_EMBEDDINGS_PER_STUDENT:
                # Remove oldest embeddings (keep the most recent ones)
                embedding_data["embeddings"] = embedding_data["embeddings"][-MAX_EMBEDDINGS_PER_STUDENT:]
            
            # Update metadata
            embedding_data["last_updated"] = datetime.now().isoformat()
            embedding_data["total_embeddings"] = len(embedding_data["embeddings"])
            
            # Save updated embeddings
            with open(embedding_file, 'w') as f:
                json.dump(embedding_data, f, indent=2)
            
            print(f"🔒 Created TRUSTED embedding for student {student_id} (confidence: {confidence:.3f})")
            return True
            
        except Exception as e:
            print(f"❌ Error creating trusted embedding for student {student_id}: {e}")
            return False

    def update_face_change_embedding(self, student_id: str, face_image: np.ndarray, 
                                   confidence: float, recognition_type: str = "insightface") -> bool:
        """
        Update embedding when face change is detected (confidence 50-70%).
        This is similar to Face ID's adaptive learning when face changes are detected.
        The system keeps both old and new embeddings to recognize the person in different appearances.
        
        Args:
            student_id (str): Student ID
            face_image (np.ndarray): Face image
            confidence (float): Recognition confidence (between 0.5-0.7)
            recognition_type (str): Type of recognition model used
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Validate confidence range
            if not (0.5 <= confidence <= 0.7):
                print(f"⚠️ Confidence {confidence:.3f} is not in face change range (0.5-0.7)")
                return False
            
            # Extract new embedding
            new_embedding = self.embedding_extractor.extract_embedding(face_image)
            if new_embedding is None:
                print(f"❌ Failed to extract face change embedding for student {student_id}")
                return False
            
            # Load existing embeddings
            embedding_file = os.path.join(self.embeddings_dir, f"{student_id}.json")
            embedding_data = self._load_embedding_data(embedding_file, student_id)
            
            # Create new face change embedding entry
            new_embedding_entry = {
                "type": recognition_type,
                "vector": new_embedding.tolist(),
                "confidence": confidence,
                "capture_date": datetime.now().isoformat(),
                "source": "face_change_update",
                "trusted": False,  # Face change embeddings start as untrusted
                "face_change_detected": True,
                "appearance_type": self._analyze_appearance_change(embedding_data, new_embedding)
            }
            
            # Add to embeddings list (APPEND, don't replace)
            if "embeddings" not in embedding_data:
                embedding_data["embeddings"] = []
            
            embedding_data["embeddings"].append(new_embedding_entry)
            
            # Limit number of embeddings per student (keep both old and new)
            if len(embedding_data["embeddings"]) > MAX_EMBEDDINGS_PER_STUDENT:
                # Keep a mix of old and new embeddings for better recognition
                old_embeddings = [emb for emb in embedding_data["embeddings"] if not emb.get("face_change_detected", False)]
                new_embeddings = [emb for emb in embedding_data["embeddings"] if emb.get("face_change_detected", False)]
                
                # Keep at least 2 old embeddings and recent new embeddings
                keep_old = max(2, len(old_embeddings) // 2)
                keep_new = MAX_EMBEDDINGS_PER_STUDENT - keep_old
                
                old_embeddings = old_embeddings[-keep_old:] if len(old_embeddings) > keep_old else old_embeddings
                new_embeddings = new_embeddings[-keep_new:] if len(new_embeddings) > keep_new else new_embeddings
                
                embedding_data["embeddings"] = old_embeddings + new_embeddings
            
            # Update metadata
            embedding_data["last_updated"] = datetime.now().isoformat()
            embedding_data["total_embeddings"] = len(embedding_data["embeddings"])
            embedding_data["face_change_count"] = len([emb for emb in embedding_data["embeddings"] 
                                                     if emb.get("face_change_detected", False)])
            
            # Save updated embeddings
            with open(embedding_file, 'w') as f:
                json.dump(embedding_data, f, indent=2)
            
            print(f"\n=== FACE CHANGE EMBEDDING APPENDED FOR: {student_id} (confidence: {confidence:.3f}) ===\n")
            print(f"🔄 Created FACE CHANGE embedding for student {student_id} (confidence: {confidence:.3f})")
            print(f"   📝 This helps adapt to face changes like aging, hairstyle, beard, etc.")
            print(f"   📊 Total embeddings: {embedding_data['total_embeddings']} (face changes: {embedding_data['face_change_count']})")
            return True
            
        except Exception as e:
            print(f"❌ Error creating face change embedding for student {student_id}: {e}")
            return False
    
    def _analyze_appearance_change(self, embedding_data: Dict, new_embedding: np.ndarray) -> str:
        """
        Analyze what type of appearance change this might be.
        
        Args:
            embedding_data (Dict): Existing embedding data
            new_embedding (np.ndarray): New embedding to analyze
            
        Returns:
            str: Type of appearance change
        """
        try:
            # Get recent embeddings for comparison
            recent_embeddings = embedding_data.get("embeddings", [])[-3:]  # Last 3 embeddings
            
            if not recent_embeddings:
                return "initial"
            
            # Calculate average similarity with recent embeddings
            similarities = []
            for emb in recent_embeddings:
                if "vector" in emb:
                    old_embedding = np.array(emb["vector"])
                    similarity = self.embedding_extractor.calculate_similarity(new_embedding, old_embedding)
                    similarities.append(similarity)
            
            if not similarities:
                return "unknown"
            
            avg_similarity = np.mean(similarities)
            
            # Classify based on similarity
            if avg_similarity > 0.8:
                return "minor_change"  # Small changes like lighting, expression
            elif avg_similarity > 0.6:
                return "moderate_change"  # Medium changes like hairstyle, glasses
            else:
                return "major_change"  # Major changes like beard, significant aging
                
        except Exception as e:
            print(f"Warning: Could not analyze appearance change: {e}")
            return "unknown"

    def get_embedding_trust_status(self, student_id: str) -> Dict[str, int]:
        """
        Get the trust status of embeddings for a student.
        
        Args:
            student_id (str): Student ID
            
        Returns:
            Dict: Count of trusted and untrusted embeddings
        """
        embedding_file = os.path.join(self.embeddings_dir, f"{student_id}.json")
        if not os.path.exists(embedding_file):
            return {"trusted": 0, "untrusted": 0}
        
        try:
            with open(embedding_file, 'r') as f:
                data = json.load(f)
            
            trusted_count = 0
            untrusted_count = 0
            
            for embedding in data.get("embeddings", []):
                if embedding.get("trusted", False):
                    trusted_count += 1
                else:
                    untrusted_count += 1
            
            return {"trusted": trusted_count, "untrusted": untrusted_count}
            
        except Exception as e:
            print(f"Error reading trust status for {student_id}: {e}")
            return {"trusted": 0, "untrusted": 0}

    def mark_embedding_as_trusted(self, student_id: str, embedding_index: int) -> bool:
        """
        Mark a specific embedding as trusted.
        
        Args:
            student_id (str): Student ID
            embedding_index (int): Index of embedding to mark as trusted
            
        Returns:
            bool: True if successful
        """
        embedding_file = os.path.join(self.embeddings_dir, f"{student_id}.json")
        if not os.path.exists(embedding_file):
            return False
        
        try:
            with open(embedding_file, 'r') as f:
                data = json.load(f)
            
            embeddings = data.get("embeddings", [])
            if 0 <= embedding_index < len(embeddings):
                embeddings[embedding_index]["trusted"] = True
                embeddings[embedding_index]["source"] = "manual_trust"
                
                with open(embedding_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✅ Marked embedding {embedding_index} as trusted for student {student_id}")
                return True
            else:
                print(f"❌ Invalid embedding index {embedding_index} for student {student_id}")
                return False
                
        except Exception as e:
            print(f"Error marking embedding as trusted: {e}")
            return False
    
    def _load_embedding_data(self, embedding_file: str, student_id: str) -> Dict:
        """
        Load existing embedding data or create new structure.
        
        Args:
            embedding_file (str): Path to embedding file
            student_id (str): Student ID
            
        Returns:
            Dict: Embedding data structure
        """
        if os.path.exists(embedding_file):
            try:
                with open(embedding_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Error loading embedding file for {student_id}: {e}")
        
        # Create new embedding data structure
        return {
            "student_id": student_id,
            "embeddings": [],
            "created_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_embeddings": 0
        }
    
    def get_update_stats(self) -> Dict:
        """
        Get embedding update statistics.
        
        Returns:
            Dict: Update statistics
        """
        return self.update_stats.copy()
    
    def reset_stats(self):
        """Reset update statistics."""
        self.update_stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "last_update": None
        }
    
    def get_student_embedding_count(self, student_id: str) -> int:
        """
        Get number of embeddings for a student.
        
        Args:
            student_id (str): Student ID
            
        Returns:
            int: Number of embeddings
        """
        embedding_file = os.path.join(self.embeddings_dir, f"{student_id}.json")
        if os.path.exists(embedding_file):
            try:
                with open(embedding_file, 'r') as f:
                    data = json.load(f)
                    return len(data.get("embeddings", []))
            except:
                pass
        return 0
    
    def cleanup_old_embeddings(self, max_age_days: int = 30):
        """
        Remove old embeddings to save space.
        
        Args:
            max_age_days (int): Maximum age of embeddings in days
        """
        current_time = datetime.now()
        cleaned_count = 0
        
        for filename in os.listdir(self.embeddings_dir):
            if not filename.endswith('.json'):
                continue
                
            embedding_file = os.path.join(self.embeddings_dir, filename)
            try:
                with open(embedding_file, 'r') as f:
                    data = json.load(f)
                
                # Filter out old embeddings
                original_count = len(data.get("embeddings", []))
                if "embeddings" in data:
                    data["embeddings"] = [
                        emb for emb in data["embeddings"]
                        if self._is_embedding_recent(emb, current_time, max_age_days)
                    ]
                
                # Save if changes were made
                if len(data["embeddings"]) < original_count:
                    with open(embedding_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    cleaned_count += original_count - len(data["embeddings"])
                    
            except Exception as e:
                print(f"Error cleaning embeddings in {filename}: {e}")
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned {cleaned_count} old embeddings")
    
    def _is_embedding_recent(self, embedding: Dict, current_time: datetime, max_age_days: int) -> bool:
        """
        Check if embedding is recent enough to keep.
        
        Args:
            embedding (Dict): Embedding data
            current_time (datetime): Current time
            max_age_days (int): Maximum age in days
            
        Returns:
            bool: True if embedding is recent enough
        """
        try:
            capture_date = datetime.fromisoformat(embedding.get("capture_date", ""))
            age_delta = current_time - capture_date
            return age_delta.days < max_age_days
        except:
            # If we can't parse the date, keep the embedding
            return True 

    def get_face_change_embeddings_count(self, student_id: str) -> int:
        """
        Get the number of face change embeddings for a student.
        
        Args:
            student_id (str): Student ID
            
        Returns:
            int: Number of face change embeddings
        """
        embedding_file = os.path.join(self.embeddings_dir, f"{student_id}.json")
        if not os.path.exists(embedding_file):
            return 0
        
        try:
            with open(embedding_file, 'r') as f:
                data = json.load(f)
            
            face_change_count = 0
            for embedding in data.get("embeddings", []):
                if embedding.get("face_change_detected", False):
                    face_change_count += 1
            
            return face_change_count
            
        except Exception as e:
            print(f"Error reading face change embeddings for {student_id}: {e}")
            return 0

    def analyze_face_changes(self, student_id: str) -> Dict[str, Any]:
        """
        Analyze face changes for a student.
        
        Args:
            student_id (str): Student ID
            
        Returns:
            Dict: Analysis results
        """
        embedding_file = os.path.join(self.embeddings_dir, f"{student_id}.json")
        if not os.path.exists(embedding_file):
            return {"error": "Student not found"}
        
        try:
            with open(embedding_file, 'r') as f:
                data = json.load(f)
            
            embeddings = data.get("embeddings", [])
            
            # Analyze embedding types
            analysis = {
                "total_embeddings": len(embeddings),
                "trusted_embeddings": 0,
                "untrusted_embeddings": 0,
                "face_change_embeddings": 0,
                "registration_embeddings": 0,
                "trusted_updates": 0,
                "face_change_updates": 0,
                "latest_embedding_date": None,
                "face_change_frequency": "Low"
            }
            
            for embedding in embeddings:
                # Count by trust status
                if embedding.get("trusted", False):
                    analysis["trusted_embeddings"] += 1
                else:
                    analysis["untrusted_embeddings"] += 1
                
                # Count by source
                source = embedding.get("source", "unknown")
                if source == "registration":
                    analysis["registration_embeddings"] += 1
                elif source == "trusted_update":
                    analysis["trusted_updates"] += 1
                elif source == "face_change_update":
                    analysis["face_change_updates"] += 1
                
                # Count face change embeddings
                if embedding.get("face_change_detected", False):
                    analysis["face_change_embeddings"] += 1
                
                # Track latest date
                capture_date = embedding.get("capture_date")
                if capture_date:
                    if analysis["latest_embedding_date"] is None or capture_date > analysis["latest_embedding_date"]:
                        analysis["latest_embedding_date"] = capture_date
            
            # Determine face change frequency
            if analysis["face_change_embeddings"] >= 5:
                analysis["face_change_frequency"] = "High"
            elif analysis["face_change_embeddings"] >= 2:
                analysis["face_change_frequency"] = "Medium"
            else:
                analysis["face_change_frequency"] = "Low"
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"} 