"""
Attendance logger for the face recognition attendance system.
"""
import os
import cv2
import datetime
import time
import threading
import pyttsx3
from typing import Dict, List, Optional, Tuple

from ..config.settings import (
    ATTENDANCE_SESSIONS, MIN_RECOGNITION_INTERVAL, AUDIO_FEEDBACK,
    SUCCESS_MESSAGE_DURATION
)
from ..registration.database_manager import DatabaseManager


class AttendanceLogger:
    """
    Logs attendance when students are recognized.
    Provides audio and visual feedback.
    """
    
    def __init__(self, min_interval=MIN_RECOGNITION_INTERVAL, audio_feedback=AUDIO_FEEDBACK):
        """
        Initialize the attendance logger.
        
        Args:
            min_interval (float): Minimum interval between attendance logs for same student
            audio_feedback (bool): Whether to provide audio feedback
        """
        self.db_manager = DatabaseManager()
        self.min_interval = min_interval
        self.audio_feedback = audio_feedback
        
        # Last attendance time for each student
        self.last_attendance = {}  # roll_number -> timestamp
        
        # Success message display
        self.success_messages = {}  # roll_number -> (message, end_time)
        
        # Initialize text-to-speech engine
        if self.audio_feedback:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_queue = []
                self.tts_lock = threading.Lock()
                self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
                self.tts_thread.start()
                print("Text-to-speech engine initialized")
            except Exception as e:
                print(f"Error initializing text-to-speech: {e}")
                self.audio_feedback = False
    
    def log_attendance(self, roll_number: str, confidence: float) -> Tuple[bool, str]:
        """
        Log attendance for a student.
        
        Args:
            roll_number (str): Student roll number
            confidence (float): Recognition confidence score
            
        Returns:
            tuple: (success, message)
        """
        # Check if this student was recently logged
        current_time = time.time()
        if roll_number in self.last_attendance:
            last_time = self.last_attendance[roll_number]
            elapsed = current_time - last_time
            
            if elapsed < self.min_interval:
                # Too soon to log again
                return False, f"Already logged attendance for {roll_number} ({int(elapsed)} seconds ago)"
        
        # Determine session based on current time
        session = self._determine_session()
        
        # Log attendance in database
        success, message = self.db_manager.log_attendance(roll_number, confidence, session)
        
        if success:
            # Update last attendance time
            self.last_attendance[roll_number] = current_time
            
            # Get student name
            student = self.db_manager.get_student(roll_number)
            student_name = student["name"] if student else roll_number
            
            # Set success message
            success_msg = f"Attendance recorded: {student_name}"
            self.success_messages[roll_number] = (
                success_msg,
                current_time + SUCCESS_MESSAGE_DURATION
            )
            
            # Provide audio feedback
            if self.audio_feedback:
                self._speak(f"Attendance recorded for {student_name}")
        
        return success, message
    
    def _determine_session(self) -> str:
        """
        Determine the current session based on time of day.
        
        Returns:
            str: Session name
        """
        now = datetime.datetime.now().time()
        now_str = now.strftime("%H:%M")
        
        for session_name, times in ATTENDANCE_SESSIONS.items():
            start = datetime.datetime.strptime(times["start"], "%H:%M").time()
            end = datetime.datetime.strptime(times["end"], "%H:%M").time()
            
            if start <= now <= end:
                return session_name
        
        return "unknown"
    
    def _speak(self, text: str):
        """
        Queue text to be spoken.
        
        Args:
            text (str): Text to speak
        """
        with self.tts_lock:
            self.tts_queue.append(text)
    
    def _tts_worker(self):
        """Worker thread for text-to-speech."""
        while True:
            text_to_speak = None
            
            # Get the next item from the queue
            with self.tts_lock:
                if self.tts_queue:
                    text_to_speak = self.tts_queue.pop(0)
            
            # Speak if we have something
            if text_to_speak:
                try:
                    self.tts_engine.say(text_to_speak)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"Error in text-to-speech: {e}")
            
            # Sleep to avoid busy waiting
            time.sleep(0.1)
    
    def get_success_messages(self) -> Dict[str, Tuple[str, float]]:
        """
        Get success messages to display.
        Also cleans up expired messages.
        
        Returns:
            dict: Roll number -> (message, end_time)
        """
        # Clean up expired messages
        current_time = time.time()
        expired = []
        
        for roll_number, (message, end_time) in self.success_messages.items():
            if current_time > end_time:
                expired.append(roll_number)
        
        # Remove expired messages
        for roll_number in expired:
            del self.success_messages[roll_number]
        
        return self.success_messages
    
    def draw_success_messages(self, frame):
        """
        Draw success messages on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            
        Returns:
            numpy.ndarray: Frame with messages drawn
        """
        if frame is None:
            return frame
        
        current_time = time.time()
        
        # Get active messages
        active_messages = []
        for roll_number, (message, end_time) in self.success_messages.items():
            if current_time <= end_time:
                # Calculate remaining time as a percentage
                time_left = (end_time - current_time) / SUCCESS_MESSAGE_DURATION
                active_messages.append((message, time_left))
        
        # Draw messages at the bottom of the frame
        if active_messages:
            # Create overlay for messages
            h, w = frame.shape[:2]
            overlay = frame.copy()
            
            # Draw background
            msg_height = 40 * len(active_messages)
            cv2.rectangle(overlay, (0, h - msg_height), (w, h), (0, 0, 0), -1)
            
            # Apply transparency
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw messages
            for i, (message, time_left) in enumerate(active_messages):
                y_pos = h - (len(active_messages) - i) * 30 - 10
                cv2.putText(frame, message, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def export_attendance_report(self, output_path: str, start_date: str = None, end_date: str = None) -> Tuple[bool, str]:
        """
        Export attendance data to Excel.
        
        Args:
            output_path (str): Path to save the report
            start_date (str): Start date for report (YYYY-MM-DD)
            end_date (str): End date for report (YYYY-MM-DD)
            
        Returns:
            tuple: (success, message)
        """
        if not start_date:
            # Default to today
            start_date = datetime.date.today().isoformat()
        
        if not end_date:
            # Default to today
            end_date = datetime.date.today().isoformat()
        
        # Export using database manager
        return self.db_manager.export_attendance_to_excel(output_path, start_date, end_date)
    
    def get_attendance_summary(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        Get attendance summary statistics.
        
        Args:
            start_date (str): Start date for summary (YYYY-MM-DD)
            end_date (str): End date for summary (YYYY-MM-DD)
            
        Returns:
            dict: Summary statistics
        """
        return self.db_manager.get_attendance_summary(start_date, end_date)
    
    def reset(self):
        """Reset logger state."""
        self.last_attendance = {}
        self.success_messages = {}
