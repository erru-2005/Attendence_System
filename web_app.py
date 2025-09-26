import os
import sys
import time
import json
import glob
from io import BytesIO, TextIOWrapper
from datetime import datetime

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    VideoProcessorBase,
    webrtc_streamer,
)

# Ensure imports resolve relative to this directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from main import DeepFaceAttendance


st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="🤖",
    layout="centered",
)

# Keep data paths consistent with main.py (database/...)
os.chdir(CURRENT_DIR)


def inject_styles():
    """Inject subtle animations and active-state styling for buttons."""
    st.markdown(
        """
        <style>
        :root {
            --brand-primary: #0a66c2;
            --brand-dark: #0a2239;
            --text-muted: #5b6b7b;
            --surface: #ffffff;
        }

        /* Header */
        .app-header {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 10px 14px 18px 8px;
            border-bottom: 1px solid rgba(0,0,0,0.06);
            background: var(--surface);
            position: relative;
            isolation: isolate;
            animation: headerFade 400ms ease 1;
        }
        .app-header .brand-logo {
            width: 60px;
            height: 60px;
            border-radius: 12px;
            box-shadow: 0 6px 14px rgba(10,102,194,.18);
            transform: translateY(0);
            transition: transform 240ms ease;
        }
        .app-header:hover .brand-logo { transform: translateY(-2px); }
        .brand-text h1 {
            font-size: 22px;
            line-height: 1.25;
            margin: 0;
            color: #0f172a;
            letter-spacing: .2px;
        }
        .brand-text p {
            margin: 2px 0 0 0;
            color: var(--text-muted);
            font-size: 12px;
        }
        @keyframes headerFade { from {opacity: 0; transform: translateY(-6px);} to {opacity: 1; transform: translateY(0);} }

        /* Footer */
        .app-footer {
            width: 100%;
            margin-top: 28px;
            padding: 14px 10px;
            color: #334155;
            border-top: 1px solid rgba(0,0,0,0.06);
            background: linear-gradient(180deg, rgba(10,102,194,0.03), rgba(0,0,0,0));
            font-size: 12px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            position: relative;
        }
        .app-footer .dot { width:4px; height:4px; border-radius:50%; background:#94a3b8; opacity:.9; }
        .app-footer .brand { color: var(--brand-primary); font-weight: 600; }

        /* Smooth animation for all buttons */
        .stButton > button {
            transition: transform 120ms ease, box-shadow 180ms ease, background-color 180ms ease;
            will-change: transform;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 14px rgba(0,0,0,0.20);
        }
        .stButton > button:active {
            transform: translateY(0);
            box-shadow: 0 2px 6px rgba(0,0,0,0.15) inset;
        }
        /* Primary button pulse to indicate active mode */
        .stButton > button[kind="primary"] {
            animation: pulseGlow 1.8s ease-in-out infinite;
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 rgba(0, 148, 255, 0.0); }
            50% { box-shadow: 0 0 0 6px rgba(0, 148, 255, 0.15); }
            100% { box-shadow: 0 0 0 rgba(0, 148, 255, 0.0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _get_logo_base64() -> str:
    try:
        if "logo_b64" in st.session_state:
            return st.session_state.logo_b64
        logo_path = os.path.join(CURRENT_DIR, "static", "logo-removebg-preview.png")
        with open(logo_path, "rb") as f:
            import base64
            b64 = base64.b64encode(f.read()).decode("utf-8")
            st.session_state.logo_b64 = b64
            return b64
    except Exception:
        return ""

def render_header():
    logo_b64 = _get_logo_base64()
    html = f"""
    <div class="app-header">
        <img class="brand-logo" alt="Logo" src="data:image/png;base64,{logo_b64}" />
        <div class="brand-text">
            <h1>Dr. B. B. Hegde First Grade College, Kundapura</h1>
            <p>A Unit of Coondapur Education Society (R)</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <span>© 2025 <span class="brand">Graahi</span></span>
            <span class="dot"></span>
            <span>Developed by Incubation Center</span>
            <span class="dot"></span>
            <span>All Rights Reserved.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def ensure_backend():
    if "system" not in st.session_state:
        st.session_state.system = DeepFaceAttendance()


def navigate(page: str):
    st.session_state.page = page


def back_to_home_button(key: str):
    st.button("\u2190 Back", key=key, on_click=lambda: navigate("home"))


# -------------------- Admin/Sidebar Utilities -------------------- #

def _database_dir() -> str:
    return os.path.join(CURRENT_DIR, "database")


def _attendance_dir() -> str:
    return os.path.join(_database_dir(), "attendance")


def _credentials_path() -> str:
    return os.path.join(_database_dir(), "admin_credentials.json")


def ensure_admin_credentials() -> None:
    os.makedirs(_database_dir(), exist_ok=True)
    creds_path = _credentials_path()
    if not os.path.exists(creds_path):
        with open(creds_path, "w", encoding="utf-8") as f:
            json.dump({"username": "bbhcadmin", "password": "12345"}, f, indent=2)


def verify_admin_credentials(username: str, password: str) -> bool:
    try:
        with open(_credentials_path(), "r", encoding="utf-8") as f:
            creds = json.load(f)
        return username == creds.get("username") and password == creds.get("password")
    except Exception:
        return False


def render_sidebar():
    st.sidebar.title("Navigation")
    if st.sidebar.button("Home", key="sb-home", use_container_width=True):
        navigate("home")
        st.rerun()
    if st.sidebar.button("Admin", key="sb-admin", use_container_width=True):
        target = "admin_dashboard" if st.session_state.get("is_admin_authenticated") else "admin_login"
        navigate(target)
        st.rerun()


def create_tabular_bytes(records: list[dict]):
    """Return (data_bytes, mime, ext) for Excel if pandas available, else CSV.

    - records: list of {student_id, name, timestamp}
    """
    # Normalize order of keys for readability. Support both legacy and new schema
    normalized = []
    def _fmt_date(val: str) -> str:
        if not val:
            return ""
        try:
            # Handle possible 'Z' suffix
            iso_val = val.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_val)
            return dt.strftime("%d/%m/%Y %I:%M %p")
        except Exception:
            # If not ISO, try common formats; else return as-is
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
                try:
                    return datetime.strptime(val, fmt).strftime("%d/%m/%Y %I:%M %p")
                except Exception:
                    pass
            return val
    for r in records:
        checkin = r.get("checkin")
        checkout = r.get("checkout")
        timestamp = r.get("timestamp")
        if (checkin is None and checkout is None) and timestamp is not None:
            checkin = timestamp
            checkout = ""
        normalized.append({
            "student_id": r.get("student_id", ""),
            "name": r.get("name", ""),
            "checkin": _fmt_date(checkin or ""),
            "checkout": _fmt_date(checkout or ""),
        })

    try:
        import pandas as pd  # lazy import; optional dependency

        df = pd.DataFrame(normalized)
        # Rename columns for export with capitalized first letters
        df = df.rename(columns={
            "student_id": "Faculty_Id",
            "name": "Name",
            "checkin": "Checkin",
            "checkout": "Checkout",
        })
        buffer = BytesIO()
        # engine chosen automatically; openpyxl/xlsxwriter if present
        with pd.ExcelWriter(buffer) as writer:
            df.to_excel(writer, index=False, sheet_name="Attendance")
            # Auto-adjust column widths
            try:
                sheet_name = "Attendance"
                worksheet = writer.sheets.get(sheet_name)
                if worksheet is not None:
                    for idx, col in enumerate(df.columns, start=0):
                        # Compute max width per column (header vs values)
                        series = df[col].astype(str)
                        max_len = max([len(col)] + series.map(len).tolist())
                        # Add small padding
                        width = min(max_len + 2, 60)
                        # openpyxl/xlsxwriter both support set_column with (first_col, last_col, width)
                        try:
                            worksheet.set_column(idx, idx, width)
                        except Exception:
                            # Fallback for openpyxl: use column_dimensions
                            try:
                                # Convert 0-based index to Excel letters
                                import string
                                letters = []
                                n = idx
                                while True:
                                    n, r = divmod(n, 26)
                                    letters.append(string.ascii_uppercase[r])
                                    if n == 0:
                                        break
                                    n -= 1
                                col_letter = "".join(reversed(letters))
                                worksheet.column_dimensions[col_letter].width = width
                            except Exception:
                                pass
            except Exception:
                pass
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
    except Exception:
        # Fallback to CSV
        try:
            import csv
            buffer = BytesIO()
            # Write UTF-8 BOM so Excel opens it nicely
            buffer.write("\ufeff".encode("utf-8"))
            fieldnames = ["Faculty_Id", "Name", "Checkin", "Checkout"]
            writer = csv.DictWriter(TextIOWrapper(buffer, encoding="utf-8", write_through=True), fieldnames=fieldnames)
            writer.writeheader()
            for row in normalized:
                writer.writerow({
                    "Faculty_Id": row.get("student_id", ""),
                    "Name": row.get("name", ""),
                    "Checkin": row.get("checkin", ""),
                    "Checkout": row.get("checkout", ""),
                })
            return buffer.getvalue(), "text/csv", ".csv"
        except Exception:
            # As a last resort, dump JSON bytes
            data = json.dumps(normalized, ensure_ascii=False, indent=2).encode("utf-8")
            return data, "application/json", ".json"


def list_attendance_files() -> list[tuple[str, str]]:
    """Return list of (date_str, file_path) sorted by date desc."""
    directory = _attendance_dir()
    os.makedirs(directory, exist_ok=True)
    files = glob.glob(os.path.join(directory, "*.json"))
    items: list[tuple[str, str]] = []
    for fp in files:
        basename = os.path.basename(fp)
        date_str = os.path.splitext(basename)[0]
        items.append((date_str, fp))
    # Sort descending by date string (YYYY-MM-DD format sorts lexicographically)
    items.sort(key=lambda x: x[0], reverse=True)
    return items


def read_attendance(fp: str) -> list[dict]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def clear_attendance(fp: str) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        f.write("[]")


# -------------------- Check-In / Check-Out helpers -------------------- #

def _today_attendance_path() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(_attendance_dir(), f"{today}.json")


def _load_attendance_records() -> list[dict]:
    fp = _today_attendance_path()
    os.makedirs(_attendance_dir(), exist_ok=True)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def _save_attendance_records(records: list[dict]) -> None:
    fp = _today_attendance_path()
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def log_check_event(student_id: str, name: str, mode: str) -> tuple[bool, str]:
    """Log a check-in or check-out event into today's attendance file.

    - Keeps backward compatibility: if an old-style entry exists (with 'timestamp'),
      it will be migrated into the new schema (checkin populated, checkout empty).
    - New schema per record:
      { student_id, name, checkin: iso or "", checkout: iso or "" }
    """
    mode = (mode or "checkin").lower()
    now_iso = datetime.now().isoformat()
    records = _load_attendance_records()

    # Migrate any legacy rows
    for r in records:
        if "timestamp" in r and ("checkin" not in r and "checkout" not in r):
            r["checkin"] = r.get("timestamp", "")
            r["checkout"] = ""
            r.pop("timestamp", None)
        if "checkin" not in r:
            r["checkin"] = ""
        if "checkout" not in r:
            r["checkout"] = ""

    # Find open session (row with checkin set and checkout empty) for this student
    open_index = None
    for i in range(len(records) - 1, -1, -1):  # search from latest
        r = records[i]
        if r.get("student_id") == student_id:
            if r.get("checkin") and not r.get("checkout"):
                open_index = i
                break

    if mode == "checkin":
        if open_index is not None:
            # Already checked in; must checkout first
            return False, "Already checked in"
        # create new row
        records.append({
            "student_id": student_id,
            "name": name,
            "checkin": now_iso,
            "checkout": ""
        })
        _save_attendance_records(records)
        return True, "Check-in successfull"
    else:  # checkout
        if open_index is None:
            return False, "Currently in check-out mode"
        records[open_index]["checkout"] = now_iso
        _save_attendance_records(records)
        return True, "Check-out successfull"


RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
})


def home_page():
    render_header()

    st.button("Take Attendance", key="nav-attendance", use_container_width=True, on_click=lambda: navigate("attendance"))
    # Removed Add Faculty and Set Threshold from Home per request


def add_student_page():
    render_header()
    st.header("Add Faculty")
    back_to_home_button("back-add")
    ensure_backend()
    system = st.session_state.system

    with st.form("add_form", clear_on_submit=False):
        student_id = st.text_input("Faculty ID")
        name = st.text_input("Faculty Name")
        submitted = st.form_submit_button("Start Camera", type="primary", use_container_width=False)

    if submitted:
        if not student_id or not name:
            st.error("Both fields are required.")
        elif student_id in system.students:
            st.error("Student already exists.")
        else:
            st.session_state.add_student_id = student_id.strip()
            st.session_state.add_student_name = name.strip()
            st.session_state.add_started = True
            st.session_state.add_saved = False

    if st.session_state.get("add_started"):

        class AddStudentProcessor(VideoProcessorBase):
            def __init__(self):
                self.system = system
                self.embedding = None
                self.captured = False
                self.capture_attempts = 0
                self.max_attempts = 50
                self.error_count = 0
                self.last_successful_frame = None
                self.face_detected = False
                self.captured_bbox = None
                self.face_data = None

            def recv(self, frame):
                try:
                    # Try to convert frame to numpy array with better error handling
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        if img is None or img.size == 0:
                            raise ValueError("Empty frame received")
                    except Exception as frame_error:
                        self.error_count += 1
                        print(f"AddStudent frame conversion error #{self.error_count}: {frame_error}")
                        # If we have a last successful frame, use it
                        if self.last_successful_frame is not None and self.error_count < 3:
                            return self.last_successful_frame
                        # Create error frame
                        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2 = __import__("cv2")
                        cv2.putText(error_img, f"Frame Error ({self.error_count})", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        return av.VideoFrame.from_ndarray(error_img, format="bgr24")

                    cv2 = __import__("cv2")

                    # Reset error count on successful frame processing
                    if self.error_count > 0:
                        self.error_count = max(0, self.error_count - 1)

                    # Detect faces with error handling
                    faces = []
                    try:
                        faces = self.system.detect_faces(img)
                    except Exception as e:
                        print(f"Face detection error in AddStudent: {e}")
                        faces = []

                    # Update face detection status (without session state to avoid warnings)
                    current_face_detected = bool(faces) and not self.captured
                    self.face_detected = current_face_detected

                    # Display information on camera (without name)
                    student_id = st.session_state.get("add_student_id", "")

                    # Main instruction
                    if self.face_detected:
                        cv2.putText(
                            img,
                            "Face detected! Click Submit to save.",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.putText(
                            img,
                            "Position your face in the camera",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                    # Faculty ID only
                    cv2.putText(
                        img,
                        f"Faculty ID: {student_id}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    # Status information
                    cv2.putText(
                        img,
                        f"Faces Detected: {len(faces)}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    # Instructions
                    cv2.putText(
                        img,
                        "Click Submit button when ready",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                    # Draw face boxes and handle face detection
                    for face in faces:
                        try:
                            bbox = face.get("bbox")
                            if bbox is None:
                                continue
                            x1, y1, x2, y2 = [int(v) for v in bbox]

                            # Draw face box
                            if self.face_detected:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for detected
                                cv2.putText(
                                    img,
                                    "Face Ready",
                                    (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 0),
                                    2,
                                )
                            else:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow for detecting
                                cv2.putText(
                                    img,
                                    "Detecting...",
                                    (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 0),
                                    2,
                                )

                            # Always store face data when detected
                            if self.face_detected:
                                embedding = face.get("embedding")
                                if embedding is None:
                                    face_img = img[y1:y2, x1:x2]
                                    if face_img.size > 0:
                                        try:
                                            embedding = self.system.extract_embedding(face_img)
                                        except Exception as e:
                                            print(f"Embedding extraction error in AddStudent: {e}")
                                            embedding = None

                                # Store the face data in processor (avoid session state in video processor)
                                if embedding is not None:
                                    self.face_data = {
                                        'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                                        'bbox': bbox
                                    }

                        except Exception as face_error:
                            print(f"Face processing error in AddStudent: {face_error}")
                            continue

                    # Store successful frame for fallback
                    result_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                    self.last_successful_frame = result_frame
                    return result_frame

                except Exception as e:
                    self.error_count += 1
                    print(f"AddStudent processing error #{self.error_count}: {e}")

                    # Return last successful frame or simple error frame
                    if self.last_successful_frame is not None:
                        return self.last_successful_frame
                    else:
                        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2 = __import__("cv2")
                        cv2.putText(error_img, f"Processing Error ({self.error_count})", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        return av.VideoFrame.from_ndarray(error_img, format="bgr24")

        ctx = webrtc_streamer(
            key="add-student",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=AddStudentProcessor,
        )

        # Create placeholders for dynamic content
        button_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Function to render buttons based on current state
        def render_buttons():
            with button_placeholder.container():
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Submit button - always show when camera is running
                    if not st.session_state.get("add_saved", False):
                        if st.button("💾 Submit Faculty", key="add-submit", type="primary", use_container_width=True):
                            # Get the latest face data from processor if available
                            face_data = None
                            if ctx.video_processor and hasattr(ctx.video_processor, 'face_data'):
                                face_data = ctx.video_processor.face_data

                            sid = st.session_state.add_student_id
                            name = st.session_state.add_student_name

                            # Final validation before save
                            if not sid or not name:
                                st.error("Faculty ID and Name are required.")
                            elif sid in system.students:
                                st.error("Faculty already exists.")
                            elif face_data is None:
                                st.error("No face data captured. Please ensure your face is visible in the camera.")
                            else:
                                # Use face data from processor
                                embedding = face_data['embedding']

                                system.students[sid] = {
                                    "name": name,
                                    "registration_date": datetime.now().isoformat(),
                                }
                                embedding_entry = {
                                    "vector": embedding,
                                    "timestamp": datetime.now().isoformat(),
                                    "confidence": 1.0,
                                    "uniqueness": 1.0,
                                }
                                system.embeddings[sid] = {
                                    "name": name,
                                    "embeddings": [embedding_entry],
                                    "embedding": embedding,
                                    "registration_date": datetime.now().isoformat(),
                                    "diversity_score": 1 / system.max_embeddings,
                                }
                                system.save_data()
                                st.session_state.add_saved = True
                                st.success(f"✅ Faculty {name} registered successfully!")
                                st.rerun()
                
                # Removed 'Add Another' button per request to avoid duplicate keys and simplify UI
        
        # Function to render status messages
        def render_status():
            with status_placeholder.container():
                if st.session_state.get("add_saved", False):
                    st.success(f"🎉 Faculty {st.session_state.add_student_name} registered successfully!")
                elif ctx.video_processor and hasattr(ctx.video_processor, 'face_detected') and ctx.video_processor.face_detected:
                    st.info("👤 Face detected! Click Submit to save the faculty.")
                else:
                    st.info("📷 Position your face in the camera. Click Submit when ready.")
        
        # Initial render
        render_status()
        render_buttons()
        
        # Real-time update mechanism using a controlled approach
        if ctx.video_processor and ctx.state.playing:
            # Check if we need to update the UI based on state changes
            processor_face_detected = hasattr(ctx.video_processor, 'face_detected') and ctx.video_processor.face_detected
            current_state = {
                'face_detected': processor_face_detected,
                'saved': st.session_state.get("add_saved", False)
            }
            
            # Store previous state for comparison
            if "add_previous_state" not in st.session_state:
                st.session_state.add_previous_state = current_state
            
            # If state has changed, update the UI
            if st.session_state.add_previous_state != current_state:
                st.session_state.add_previous_state = current_state
                render_status()
                render_buttons()

        # Back to home button when saved
        if st.session_state.get("add_saved", False):
            if st.button("← Back to Home", key="back-add-done", use_container_width=True):
                # Clean up session state
                st.session_state.add_started = False
                st.session_state.add_captured = False
                st.session_state.add_face_detected = False
                st.session_state.add_saved = False
                if "add_face_data" in st.session_state:
                    del st.session_state.add_face_data
                if "add_captured_embedding" in st.session_state:
                    del st.session_state.add_captured_embedding
                navigate("home")
                st.rerun()


def attendance_page():
    render_header()
    st.header("Take Attendance")
    back_to_home_button("back-attendance")
    ensure_backend()
    system = st.session_state.system

    # Select check mode (Check-In by default)
    if "attendance_mode" not in st.session_state:
        st.session_state.attendance_mode = "checkin"

    col_m1, col_m2, col_m3 = st.columns([1, 1, 3])
    with col_m1:
        if st.button("Check-In", type=("primary" if st.session_state.attendance_mode == "checkin" else "secondary"), use_container_width=True, key="btn-checkin"):
            st.session_state.attendance_mode = "checkin"
            st.rerun()
    with col_m2:
        if st.button("Check-Out", type=("primary" if st.session_state.attendance_mode == "checkout" else "secondary"), use_container_width=True, key="btn-checkout"):
            st.session_state.attendance_mode = "checkout"
            st.rerun()

    st.caption(f"Mode: {st.session_state.attendance_mode.title()}")

    # Display system status
    st.info(f"Camera running... Click Stop to end. Faculty registered: {len(system.students)}")
    
    # Debug information
    if len(system.students) == 0:
        st.warning("⚠️ No faculty registered! Please add faculty first.")
    else:
        st.success(f"✅ {len(system.students)} faculty ready for recognition")
        st.caption(f"Recognition threshold: {system.recognition_threshold:.2f}")

    class AttendanceProcessor(VideoProcessorBase):
        def __init__(self):
            self.system = system
            self.recognized_student_ids = set()
            self.frame_count = 0
            self.start_time = time.time()
            self.camera_start_time = self.start_time
            self.last_recognition_time = 0.0
            self.recognition_interval = 0.8  # Increased to reduce processing load and improve smoothness
            self.last_embedding_update = {}
            self.notification_text = ""
            self.notification_color = (255, 255, 255)
            self.notification_end_time = 0.0
            self.error_count = 0
            self.max_errors = 5  # Reduced max errors for faster recovery
            self.consecutive_errors = 0
            self.last_successful_frame = None
            self.frame_skip_count = 0
            self.max_frame_skip = 3  # Skip processing every 3rd frame for better performance
            self.mode = st.session_state.get("attendance_mode", "checkin")

        def recv(self, frame):
            try:
                # Skip some frames for better performance
                self.frame_skip_count += 1
                if self.frame_skip_count % self.max_frame_skip != 0:
                    # Return the last successful frame or a simple frame
                    if self.last_successful_frame is not None:
                        return self.last_successful_frame
                    else:
                        # Create a simple placeholder frame
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2 = __import__("cv2")
                        cv2.putText(placeholder, "Initializing...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        return av.VideoFrame.from_ndarray(placeholder, format="bgr24")

                # Try to convert frame to numpy array with better error handling
                try:
                    img = frame.to_ndarray(format="bgr24")
                    if img is None or img.size == 0:
                        raise ValueError("Empty frame received")
                except Exception as frame_error:
                    self.consecutive_errors += 1
                    print(f"Frame conversion error #{self.consecutive_errors}: {frame_error}")
                    
                    # If we have a last successful frame, use it
                    if self.last_successful_frame is not None and self.consecutive_errors < 3:
                        return self.last_successful_frame
                    
                    # Create error frame
                    error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2 = __import__("cv2")
                    cv2.putText(error_img, f"Frame Error ({self.consecutive_errors})", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    return av.VideoFrame.from_ndarray(error_img, format="bgr24")

                cv2 = __import__("cv2")
                self.frame_count += 1
                current_time = time.time()
                
                # Reset consecutive errors on successful frame processing
                self.consecutive_errors = 0
                # Use mode provided by UI via outer thread (updated below)
                current_mode = getattr(self, "mode", "checkin")
                
                # Face detection with error handling
                faces = []
                try:
                    faces = self.system.detect_faces(img)
                    
                    # Reset error count on successful processing
                    if self.error_count > 0:
                        self.error_count = max(0, self.error_count - 1)
                        
                except Exception as e:
                    print(f"Face detection error: {e}")
                    faces = []

                # Process recognition at intervals for better performance
                if current_time - self.last_recognition_time >= self.recognition_interval:
                    for face in faces:
                        try:
                            bbox = face.get("bbox")
                            if bbox is None:
                                continue
                            x1, y1, x2, y2 = [int(v) for v in bbox]

                            # Ensure we have a valid face region
                            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                                continue

                            face_embedding = face.get("embedding")
                            if face_embedding is None:
                                face_img = img[y1:y2, x1:x2]
                                if face_img.size > 0:
                                    try:
                                        face_embedding = self.system.extract_embedding(face_img)
                                    except Exception as e:
                                        print(f"Embedding extraction error: {e}")
                                        face_embedding = None

                            if face_embedding is not None:
                                student_id, confidence = self.system.recognize_face(face_embedding)
                            else:
                                student_id, confidence = None, 0.0

                            if student_id and confidence >= self.system.recognition_threshold:
                                name = self.system.students[student_id]["name"]
                                if student_id not in self.recognized_student_ids:
                                    recognition_time = current_time - self.camera_start_time
                                    # Console output mirrors main.py
                                    print(f"{name:<40}{confidence*100:.2f}%{recognition_time:.2f} seconds")

                                self.recognized_student_ids.add(student_id)
                                # Log according to rules; show brief message if blocked
                                success, msg = log_check_event(student_id, name, current_mode)
                                self.notification_text = msg
                                self.notification_color = (0, 255, 0) if success else (0, 0, 255)
                                self.notification_end_time = current_time + 2
                                self.system.draw_face_box_with_name(img, (x1, y1, x2, y2), name, confidence, True)
                            elif student_id and 0.5 <= confidence < self.system.recognition_threshold:
                                name = self.system.students[student_id]["name"]
                                can_update = True
                                if student_id in self.last_embedding_update:
                                    if current_time - self.last_embedding_update[student_id] < self.system.min_embedding_update_interval:
                                        can_update = False

                                if can_update and face_embedding is not None:
                                    if self.system.update_student_embedding(student_id, face_embedding, confidence):
                                        self.notification_text = f"New appearance learned for {name}!"
                                        self.notification_color = (0, 255, 0)
                                        self.notification_end_time = current_time + 3
                                        self.last_embedding_update[student_id] = current_time

                                    if student_id not in self.recognized_student_ids:
                                        print(f"{'AUTO-CONFIRMED:':<40}{confidence*100:.2f}%{current_time - self.camera_start_time:.2f} seconds")

                                    self.recognized_student_ids.add(student_id)
                                    success, msg = log_check_event(student_id, name, current_mode)
                                    self.notification_text = msg
                                    self.notification_color = (0, 255, 0) if success else (0, 0, 255)
                                    self.notification_end_time = current_time + 2

                                self.system.draw_face_box_with_name(img, (x1, y1, x2, y2), name, confidence, True)
                                # If we didn't update due to cooldown, still log once for this mode
                                if not can_update:
                                    if student_id not in self.recognized_student_ids:
                                        self.recognized_student_ids.add(student_id)
                                    success, msg = log_check_event(student_id, name, current_mode)
                                    self.notification_text = msg
                                    self.notification_color = (0, 255, 0) if success else (0, 0, 255)
                                    self.notification_end_time = current_time + 2
                            else:
                                # Show confidence even for unknown faces
                                confidence_display = confidence if student_id else 0.0
                                self.system.draw_face_box_with_name(img, (x1, y1, x2, y2), "Unknown", confidence_display, False)
                        except Exception as face_error:
                            print(f"Face processing error: {face_error}")
                            continue

                    self.last_recognition_time = current_time
                else:
                    # Draw face boxes without recognition (for continuous tracking)
                    for face in faces:
                        try:
                            bbox = face.get("bbox")
                            if bbox is None:
                                continue
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            cv2.putText(img, "Tracking...", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        except Exception as tracking_error:
                            print(f"Tracking error: {tracking_error}")
                            continue

                # Draw statistics with better error handling
                try:
                    fps = self.frame_count / max(1e-6, (current_time - self.start_time))
                    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(img, f"Recognized: {len(self.recognized_student_ids)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(img, f"Threshold: {self.system.recognition_threshold:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(img, "Auto-Learning Active", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    # Show current mode on the video overlay
                    mode_text = f"Mode: {'Check-Out' if current_mode == 'checkout' else 'Check-In'}"
                    cv2.putText(img, mode_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                
                # Add error count display for debugging
                    if self.error_count > 0:
                        cv2.putText(img, f"Errors: {self.error_count}/{self.max_errors}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except Exception as stats_error:
                    print(f"Stats drawing error: {stats_error}")

                # Draw notification with error handling
                try:
                    if current_time < self.notification_end_time and self.notification_text:
                        (text_width, text_height), _ = cv2.getTextSize(self.notification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(
                            img,
                            (img.shape[1] // 2 - text_width // 2 - 10, img.shape[0] - 80),
                            (img.shape[1] // 2 + text_width // 2 + 10, img.shape[0] - 40),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            img,
                            self.notification_text,
                            (img.shape[1] // 2 - text_width // 2, img.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            self.notification_color,
                            2,
                        )
                except Exception as notification_error:
                    print(f"Notification drawing error: {notification_error}")

                # Store successful frame for fallback
                result_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                self.last_successful_frame = result_frame
                return result_frame
                
            except Exception as e:
                self.error_count += 1
                self.consecutive_errors += 1
                print(f"Video processing error #{self.error_count}: {e}")
                
                # If too many consecutive errors, show error screen
                if self.consecutive_errors >= self.max_errors:
                    error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2 = __import__("cv2")
                    cv2.putText(error_img, "Camera Error - Too Many Failures", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(error_img, "Please refresh the page", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(error_img, f"Errors: {self.error_count}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    return av.VideoFrame.from_ndarray(error_img, format="bgr24")
                else:
                    # Return last successful frame or simple error frame
                    if self.last_successful_frame is not None:
                        return self.last_successful_frame
                    else:
                        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2 = __import__("cv2")
                    cv2.putText(error_img, f"Processing Error ({self.error_count}/{self.max_errors})", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    return av.VideoFrame.from_ndarray(error_img, format="bgr24")

    ctx = webrtc_streamer(
        key="attendance",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=AttendanceProcessor,
    )

    # Inline mode controls placed right below the live video
    controls_container = st.container()
    with controls_container:
        col_i1, col_i2, _ = st.columns([1, 1, 3])
        with col_i1:
            click_in = st.button(
                "Check-In",
                key="btn-checkin-inline",
                type=("primary" if st.session_state.attendance_mode == "checkin" else "secondary"),
                use_container_width=True,
            )
            if click_in and st.session_state.attendance_mode != "checkin":
                st.session_state.attendance_mode = "checkin"
                if ctx and ctx.video_processor:
                    ctx.video_processor.mode = "checkin"
        with col_i2:
            click_out = st.button(
                "Check-Out",
                key="btn-checkout-inline",
                type=("primary" if st.session_state.attendance_mode == "checkout" else "secondary"),
                use_container_width=True,
            )
            if click_out and st.session_state.attendance_mode != "checkout":
                st.session_state.attendance_mode = "checkout"
                if ctx and ctx.video_processor:
                    ctx.video_processor.mode = "checkout"

    recognized_box = st.empty()
    # Sync current mode to processor instance so overlay updates immediately
    if ctx.video_processor:
        ctx.video_processor.mode = st.session_state.get("attendance_mode", "checkin")
        ids = sorted(list(ctx.video_processor.recognized_student_ids))
        if ids:
            names = [system.students[i]["name"] for i in ids if i in system.students]
            recognized_box.write({"recognized": [{"id": i, "name": n} for i, n in zip(ids, names)]})


def delete_student_page():
    render_header()
    st.header("Delete Faculty")
    back_to_home_button("back-delete")
    ensure_backend()
    system = st.session_state.system

    students = list(system.students.items())
    if not students:
        st.info("No faculty registered.")
        return

    options = [f"{data['name']} (ID: {sid})" for sid, data in students]
    selected = st.selectbox("Select a faculty to delete", options)

    if st.button("Delete", type="primary"):
        idx = options.index(selected)
        sid, sdata = students[idx]
        # Immediate delete on click (Streamlit has no native confirm dialog)
        del system.students[sid]
        if sid in system.embeddings:
            del system.embeddings[sid]
        system.save_data()
        st.success(f"{sdata['name']} deleted.")


def list_students_page():
    render_header()
    st.header("Registered Faculty")
    back_to_home_button("back-list")
    ensure_backend()
    system = st.session_state.system

    if not system.students:
        st.info("No faculty registered.")
        return

    st.write(f"Total faculty: {len(system.students)}")
    rows = []
    def _fmt_dt(val: str) -> str:
        if not val:
            return ""
        try:
            iso = val.replace("Z", "+00:00")
            dt_obj = datetime.fromisoformat(iso)
            return dt_obj.strftime("%d/%m/%Y %I:%M %p")
        except Exception:
            try:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S").strftime("%d/%m/%Y %I:%M %p")
            except Exception:
                return val
    for sid, data in system.students.items():
        rows.append({
            "ID": sid,
            "Name": data.get("name", ""),
            "Registered": _fmt_dt(data.get("registration_date", "")) or "N/A",
        })
    st.dataframe(rows, use_container_width=True)


def threshold_page():
    render_header()
    st.header("Set Threshold")
    back_to_home_button("back-threshold")
    ensure_backend()
    system = st.session_state.system

    current = float(system.recognition_threshold)
    new_val = st.slider("Recognition threshold", min_value=0.1, max_value=0.9, value=current, step=0.01)
    if st.button("Save"):
        system.recognition_threshold = float(new_val)
        st.success(f"Threshold set to {new_val:.2f}")


def admin_login_page():
    ensure_admin_credentials()
    render_header()
    st.header("Admin Login")
    back_to_home_button("back-admin-login")

    with st.form("admin_login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        if verify_admin_credentials(username.strip(), password):
            st.session_state.is_admin_authenticated = True
            st.success("Login successful")
            navigate("admin_dashboard")
            st.rerun()
        else:
            st.error("Invalid username or password")


def admin_dashboard_page():
    render_header()
    if not st.session_state.get("is_admin_authenticated"):
        navigate("admin_login")
        st.warning("Please login to access the admin dashboard.")
        return

    st.header("Admin Dashboard")
    
    # Logout button in top right
    col_header, col_logout = st.columns([4, 1])
    with col_header:
        back_to_home_button("back-admin")
    with col_logout:
        if st.button("Logout", key="admin-logout", type="secondary", use_container_width=True):
            st.session_state.is_admin_authenticated = False
            st.success("Logged out successfully")
            navigate("home")
            st.rerun()

    # Admin actions
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Delete Faculty", key="admin-nav-delete", use_container_width=True):
            navigate("delete")
            st.rerun()
    with col_b:
        if st.button("List Faculty", key="admin-nav-list", use_container_width=True):
            navigate("list")
            st.rerun()

    # Additional actions per request
    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("Add Faculty", key="admin-nav-add", use_container_width=True):
            navigate("add")
            st.rerun()
    with col_d:
        if st.button("Set Threshold", key="admin-nav-threshold", use_container_width=True):
            navigate("threshold")
            st.rerun()

    files = list_attendance_files()
    if not files:
        st.info("No attendance files found.")
        return

    for date_str, fp in files:
        # Display date in dd - mm - yyyy format without changing underlying filenames/keys
        try:
            dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
            display_date = dt_obj.strftime("%d - %m - %Y")
        except Exception:
            display_date = date_str
        st.subheader(display_date)
        records = read_attendance(fp)
        # Normalize for display (support legacy schema)
        display_records = []
        for r in records:
            if "checkin" in r or "checkout" in r:
                display_records.append(r)
            elif "timestamp" in r:
                display_records.append({
                    "student_id": r.get("student_id", ""),
                    "name": r.get("name", ""),
                    "checkin": r.get("timestamp", ""),
                    "checkout": ""
                })
            else:
                display_records.append(r)
        # Format timestamps to dd/mm/yyyy hh:mm AM/PM for admin display
        def _fmt_dt(val: str) -> str:
            if not val:
                return ""
            try:
                iso = val.replace("Z", "+00:00")
                dt_obj = datetime.fromisoformat(iso)
                return dt_obj.strftime("%d/%m/%Y %I:%M %p")
            except Exception:
                return val

        for rec in display_records:
            if isinstance(rec, dict):
                if rec.get("checkin"):
                    rec["checkin"] = _fmt_dt(rec.get("checkin", ""))
                if rec.get("checkout"):
                    rec["checkout"] = _fmt_dt(rec.get("checkout", ""))

        st.write(f"Total records: {len(display_records)}")

        if records:
            try:
                import pandas as pd  # optional
                df = pd.DataFrame(display_records)
                # Order columns nicely if present
                cols = [c for c in ["student_id", "name", "checkin", "checkout"] if c in df.columns]
                if cols:
                    df = df[cols]
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.json(display_records)
        else:
            st.info("No records for this date.")

        col1, col2 = st.columns(2)
        with col1:
            data_bytes, mime, ext = create_tabular_bytes(display_records)
            st.download_button(
                label="Create Excel",
                data=data_bytes,
                file_name=f"attendance_{date_str}{ext}",
                mime=mime,
                key=f"download-{date_str}",
                use_container_width=True,
            )
        with col2:
            confirm_key = f"confirm-clear-{date_str}"
            if not st.session_state.get(confirm_key):
                if st.button("Clear Data", key=f"clear-{date_str}", type="secondary", use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()
            else:
                st.warning("Are you sure you want to clear this date's data?")
                c1, c2 = st.columns(2)
                if c1.button("Confirm", key=f"confirm-{date_str}", type="primary", use_container_width=True):
                    clear_attendance(fp)
                    st.session_state[confirm_key] = False
                    st.success("Data cleared.")
                    st.rerun()
                if c2.button("Cancel", key=f"cancel-{date_str}", use_container_width=True):
                    st.session_state[confirm_key] = False
                    st.rerun()


def router():
    page = st.session_state.get("page", "home")
    if page == "home":
        home_page()
    elif page == "add":
        add_student_page()
    elif page == "attendance":
        attendance_page()
    elif page == "delete":
        delete_student_page()
    elif page == "list":
        list_students_page()
    elif page == "threshold":
        threshold_page()
    elif page == "admin_login":
        admin_login_page()
    elif page == "admin_dashboard":
        admin_dashboard_page()
    else:
        navigate("home")
        home_page()
    # Global footer for all pages
    render_footer()


if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"
    inject_styles()
    ensure_admin_credentials()
    render_sidebar()
    router()