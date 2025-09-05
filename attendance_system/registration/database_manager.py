"""
Database manager for storing and retrieving student data and attendance records.
"""
import os
import json
import datetime
import shutil
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from ..config.settings import (
    STUDENT_DB_FILE, ATTENDANCE_DB_FILE, BACKUP_DIR,
    BACKUP_FREQUENCY, MAX_BACKUPS
)
from ..utils.optimization import MemoryCache

class DatabaseManager:
    """
    Manages the database operations for student records and attendance data.
    Implements thread-safe operations and backup functionality.
    """
    
    def __init__(self, student_db_file=STUDENT_DB_FILE,
                 attendance_db_file=ATTENDANCE_DB_FILE, backup_dir=BACKUP_DIR):
        """
        Initialize the database manager.
        
        Args:
            student_db_file (str): Path to student database file
            attendance_db_file (str): Path to attendance database file
            backup_dir (str): Path to backup directory
        """
        self.student_db_file = student_db_file
        self.attendance_db_file = attendance_db_file
        self.backup_dir = backup_dir
        
        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(student_db_file), exist_ok=True)
        os.makedirs(os.path.dirname(attendance_db_file), exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        
        # Initialize database files if they don't exist
        self._init_student_db()
        self._init_attendance_db()
        
        # Thread locks for database access
        self.student_lock = threading.RLock()
        self.attendance_lock = threading.RLock()
        
        # In-memory cache for frequently accessed data
        self.student_cache = MemoryCache()
        self.last_backup_time = datetime.datetime.now()
    
    def _init_student_db(self):
        """Initialize the student database file if it doesn't exist."""
        if not os.path.exists(self.student_db_file):
            with open(self.student_db_file, 'w') as f:
                json.dump({"students": []}, f, indent=2)
    
    def _init_attendance_db(self):
        """Initialize the attendance database file if it doesn't exist."""
        if not os.path.exists(self.attendance_db_file):
            with open(self.attendance_db_file, 'w') as f:
                json.dump({"attendance": []}, f, indent=2)
    
    def _load_student_db(self):
        """
        Load the student database from file.
        
        Returns:
            dict: The student database
        """
        try:
            with open(self.student_db_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Handle corrupted or missing file
            print(f"Error loading student database. Creating new database.")
            return {"students": []}
    
    def _save_student_db(self, db):
        """
        Save the student database to file.
        
        Args:
            db (dict): The student database
        """
        with open(self.student_db_file, 'w') as f:
            json.dump(db, f, indent=2)
        
        # Check if backup is needed
        self._check_and_backup()
    
    def _load_attendance_db(self):
        """
        Load the attendance database from file.
        
        Returns:
            dict: The attendance database
        """
        try:
            with open(self.attendance_db_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Handle corrupted or missing file
            print(f"Error loading attendance database. Creating new database.")
            return {"attendance": []}
    
    def _save_attendance_db(self, db):
        """
        Save the attendance database to file.
        
        Args:
            db (dict): The attendance database
        """
        with open(self.attendance_db_file, 'w') as f:
            json.dump(db, f, indent=2)
        
        # Check if backup is needed
        self._check_and_backup()
    
    def _check_and_backup(self):
        """Check if a backup is needed and perform it if necessary."""
        now = datetime.datetime.now()
        hours_since_last_backup = (now - self.last_backup_time).total_seconds() / 3600
        
        if hours_since_last_backup >= BACKUP_FREQUENCY:
            self._create_backup()
            self.last_backup_time = now
    
    def _create_backup(self):
        """Create a backup of the database files."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup student database
        student_backup_file = os.path.join(
            self.backup_dir, f"students_{timestamp}.json"
        )
        shutil.copy2(self.student_db_file, student_backup_file)
        
        # Backup attendance database
        attendance_backup_file = os.path.join(
            self.backup_dir, f"attendance_{timestamp}.json"
        )
        shutil.copy2(self.attendance_db_file, attendance_backup_file)
        
        # Remove old backups if too many
        self._clean_old_backups()
        
        print(f"Database backup created at {timestamp}")
    
    def _clean_old_backups(self):
        """Remove old backup files if there are too many."""
        # Get student backup files
        student_backups = sorted([
            f for f in os.listdir(self.backup_dir) 
            if f.startswith("students_") and f.endswith(".json")
        ])
        
        # Get attendance backup files
        attendance_backups = sorted([
            f for f in os.listdir(self.backup_dir) 
            if f.startswith("attendance_") and f.endswith(".json")
        ])
        
        # Remove old student backups
        if len(student_backups) > MAX_BACKUPS:
            for old_backup in student_backups[:-MAX_BACKUPS]:
                try:
                    os.remove(os.path.join(self.backup_dir, old_backup))
                except Exception as e:
                    print(f"Error removing old backup {old_backup}: {e}")
        
        # Remove old attendance backups
        if len(attendance_backups) > MAX_BACKUPS:
            for old_backup in attendance_backups[:-MAX_BACKUPS]:
                try:
                    os.remove(os.path.join(self.backup_dir, old_backup))
                except Exception as e:
                    print(f"Error removing old backup {old_backup}: {e}")
    
    def add_student(self, roll_number, name, embeddings, class_name=None, section=None):
        """
        Add a new student to the database.
        
        Args:
            roll_number (str): Unique roll number of the student
            name (str): Name of the student
            embeddings (list): List of face embeddings
            class_name (str): Class of the student
            section (str): Section of the student
            
        Returns:
            tuple: (success (bool), message (str))
        """
        with self.student_lock:
            db = self._load_student_db()
            
            # Check if student already exists
            for student in db["students"]:
                if student["roll_number"] == roll_number:
                    return False, f"Student with roll number {roll_number} already exists"
            
            # Convert numpy arrays to lists for JSON serialization
            embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                             for emb in embeddings]
            
            # Create new student record
            new_student = {
                "roll_number": roll_number,
                "name": name,
                "embeddings": embeddings_list,
                "class": class_name,
                "section": section,
                "metadata": {
                    "registration_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "last_updated": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "photo_count": len(embeddings)
                }
            }
            
            # Add to database
            db["students"].append(new_student)
            self._save_student_db(db)
            
            # Update cache
            self.student_cache.set(roll_number, new_student)
            
            return True, f"Student {name} added successfully"
    
    def update_student(self, roll_number, **kwargs):
        """
        Update an existing student's information.
        
        Args:
            roll_number (str): Unique roll number of the student
            **kwargs: Fields to update
            
        Returns:
            tuple: (success (bool), message (str))
        """
        with self.student_lock:
            db = self._load_student_db()
            
            # Find student to update
            for i, student in enumerate(db["students"]):
                if student["roll_number"] == roll_number:
                    # Update fields
                    for key, value in kwargs.items():
                        if key == "embeddings" and isinstance(value, list):
                            # Convert numpy arrays to lists for JSON serialization
                            value = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                                   for emb in value]
                        
                        # Update nested metadata fields
                        if key == "metadata" and isinstance(value, dict):
                            if "metadata" not in student:
                                student["metadata"] = {}
                            student["metadata"].update(value)
                        else:
                            student[key] = value
                    
                    # Update last_updated timestamp
                    if "metadata" not in student:
                        student["metadata"] = {}
                    student["metadata"]["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d")
                    
                    # Save changes
                    db["students"][i] = student
                    self._save_student_db(db)
                    
                    # Update cache
                    self.student_cache.set(roll_number, student)
                    
                    return True, f"Student {roll_number} updated successfully"
            
            return False, f"Student with roll number {roll_number} not found"
    
    def delete_student(self, roll_number):
        """
        Delete a student from the database.
        
        Args:
            roll_number (str): Unique roll number of the student
            
        Returns:
            tuple: (success (bool), message (str))
        """
        with self.student_lock:
            db = self._load_student_db()
            
            # Find and remove student
            for i, student in enumerate(db["students"]):
                if student["roll_number"] == roll_number:
                    # Remove from database
                    del db["students"][i]
                    self._save_student_db(db)
                    
                    # Remove from cache
                    self.student_cache.remove(roll_number)
                    
                    return True, f"Student {roll_number} deleted successfully"
            
            return False, f"Student with roll number {roll_number} not found"
    
    def get_student(self, roll_number):
        """
        Get a student's information by roll number.
        
        Args:
            roll_number (str): Unique roll number of the student
            
        Returns:
            dict: Student information or None if not found
        """
        # Try cache first
        cached_student = self.student_cache.get(roll_number)
        if cached_student:
            return cached_student
        
        # Not in cache, load from database
        with self.student_lock:
            db = self._load_student_db()
            
            for student in db["students"]:
                if student["roll_number"] == roll_number:
                    # Update cache and return
                    self.student_cache.set(roll_number, student)
                    return student
            
            return None
    
    def get_all_students(self):
        """
        Get all students from the database.
        
        Returns:
            list: List of all student records
        """
        with self.student_lock:
            db = self._load_student_db()
            return db["students"]
    
    def get_student_count(self):
        """
        Get the total number of students in the database.
        
        Returns:
            int: Number of students
        """
        with self.student_lock:
            db = self._load_student_db()
            return len(db["students"])
    
    def get_all_embeddings(self):
        """
        Get all face embeddings with associated student IDs.
        
        Returns:
            tuple: (embeddings, roll_numbers)
                embeddings: List of face embeddings
                roll_numbers: Corresponding roll numbers for each embedding
        """
        embeddings_dir = "database/embeddings"
        all_embeddings = []
        all_roll_numbers = []
        
        if not os.path.exists(embeddings_dir):
            print(f"Embeddings directory not found: {embeddings_dir}")
            return all_embeddings, all_roll_numbers
        
        # Load embeddings from individual JSON files
        for filename in os.listdir(embeddings_dir):
            if filename.endswith('.json'):
                student_id = filename.replace('.json', '')
                embedding_file = os.path.join(embeddings_dir, filename)
                
                try:
                    with open(embedding_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract embeddings from the file
                    for embedding_entry in data.get("embeddings", []):
                        if "vector" in embedding_entry:
                            # Convert to numpy array
                            embedding_vector = np.array(embedding_entry["vector"])
                            all_embeddings.append(embedding_vector)
                            all_roll_numbers.append(student_id)
                            
                except Exception as e:
                    print(f"Error loading embeddings from {filename}: {e}")
                    continue
        
        print(f"Loaded {len(all_embeddings)} embeddings from {len(set(all_roll_numbers))} students")
        return all_embeddings, all_roll_numbers
    
    def log_attendance(self, roll_number, confidence=None, session="unknown"):
        """
        Log attendance for a student.
        
        Args:
            roll_number (str): Unique roll number of the student
            confidence (float): Recognition confidence score
            session (str): Session name (morning, afternoon, evening)
            
        Returns:
            tuple: (success (bool), message (str))
        """
        with self.attendance_lock:
            # First check if student exists
            student = self.get_student(roll_number)
            if not student:
                return False, f"Student with roll number {roll_number} not found"
            
            db = self._load_attendance_db()
            
            # Create attendance record
            attendance_record = {
                "roll_number": roll_number,
                "name": student["name"],
                "timestamp": datetime.datetime.now().isoformat(),
                "date": datetime.date.today().isoformat(),
                "session": session,
                "confidence": confidence if confidence is not None else 1.0
            }
            
            # Add to database
            db["attendance"].append(attendance_record)
            self._save_attendance_db(db)
            
            return True, f"Attendance logged for {student['name']}"
    
    def get_attendance(self, date=None, roll_number=None, session=None):
        """
        Get attendance records with optional filters.
        
        Args:
            date (str): Date to filter by (YYYY-MM-DD)
            roll_number (str): Roll number to filter by
            session (str): Session to filter by
            
        Returns:
            list: Filtered attendance records
        """
        with self.attendance_lock:
            db = self._load_attendance_db()
            records = db["attendance"]
            
            # Apply filters
            if date:
                records = [r for r in records if r.get("date") == date]
            
            if roll_number:
                records = [r for r in records if r["roll_number"] == roll_number]
            
            if session:
                records = [r for r in records if r.get("session") == session]
            
            return records
    
    def export_attendance_to_excel(self, output_file, start_date=None, end_date=None):
        """
        Export attendance data to an Excel file.
        
        Args:
            output_file (str): Path to the output Excel file
            start_date (str): Start date for filtering (YYYY-MM-DD)
            end_date (str): End date for filtering (YYYY-MM-DD)
            
        Returns:
            tuple: (success (bool), message (str))
        """
        with self.attendance_lock:
            db = self._load_attendance_db()
            records = db["attendance"]
            
            # Apply date filters
            if start_date:
                records = [r for r in records if r.get("date", "") >= start_date]
            
            if end_date:
                records = [r for r in records if r.get("date", "") <= end_date]
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save to Excel
            try:
                df.to_excel(output_file, index=False)
                return True, f"Attendance data exported to {output_file}"
            except Exception as e:
                return False, f"Error exporting attendance data: {e}"
    
    def get_attendance_summary(self, start_date=None, end_date=None):
        """
        Get a summary of attendance data.
        
        Args:
            start_date (str): Start date for filtering (YYYY-MM-DD)
            end_date (str): End date for filtering (YYYY-MM-DD)
            
        Returns:
            dict: Attendance summary statistics
        """
        with self.attendance_lock:
            # Get filtered attendance records
            records = self.get_attendance()
            
            # Apply date filters
            if start_date:
                records = [r for r in records if r.get("date", "") >= start_date]
            
            if end_date:
                records = [r for r in records if r.get("date", "") <= end_date]
            
            if not records:
                return {
                    "total_records": 0,
                    "students_present": 0,
                    "dates": [],
                    "sessions": {}
                }
            
            # Calculate summary statistics
            unique_students = set(r["roll_number"] for r in records)
            unique_dates = sorted(set(r.get("date", "") for r in records if r.get("date")))
            
            # Count by session
            session_counts = {}
            for record in records:
                session = record.get("session", "unknown")
                if session not in session_counts:
                    session_counts[session] = 0
                session_counts[session] += 1
            
            return {
                "total_records": len(records),
                "students_present": len(unique_students),
                "dates": unique_dates,
                "sessions": session_counts
            }
    
    def restore_from_backup(self, timestamp=None):
        """
        Restore database from a backup.
        
        Args:
            timestamp (str): Timestamp of backup to restore (YYYYMMDD_HHMMSS)
                            If None, restores the most recent backup
            
        Returns:
            tuple: (success (bool), message (str))
        """
        # Find available backups
        student_backups = sorted([
            f for f in os.listdir(self.backup_dir) 
            if f.startswith("students_") and f.endswith(".json")
        ])
        
        attendance_backups = sorted([
            f for f in os.listdir(self.backup_dir) 
            if f.startswith("attendance_") and f.endswith(".json")
        ])
        
        if not student_backups or not attendance_backups:
            return False, "No backups found"
        
        # Determine which backup to restore
        if timestamp:
            student_backup = f"students_{timestamp}.json"
            attendance_backup = f"attendance_{timestamp}.json"
            
            if student_backup not in student_backups or attendance_backup not in attendance_backups:
                return False, f"Backup with timestamp {timestamp} not found"
        else:
            # Use the most recent backup
            student_backup = student_backups[-1]
            attendance_backup = attendance_backups[-1]
        
        try:
            # Restore student database
            with self.student_lock:
                shutil.copy2(
                    os.path.join(self.backup_dir, student_backup),
                    self.student_db_file
                )
            
            # Restore attendance database
            with self.attendance_lock:
                shutil.copy2(
                    os.path.join(self.backup_dir, attendance_backup),
                    self.attendance_db_file
                )
            
            # Clear cache
            self.student_cache.clear()
            
            timestamp = student_backup.replace("students_", "").replace(".json", "")
            return True, f"Database restored from backup {timestamp}"
        except Exception as e:
            return False, f"Error restoring database: {e}"
