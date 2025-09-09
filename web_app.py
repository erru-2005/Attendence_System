import os
import sys
import time
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


def ensure_backend():
    if "system" not in st.session_state:
        st.session_state.system = DeepFaceAttendance()


def navigate(page: str):
    st.session_state.page = page


def back_to_home_button(key: str):
    st.button("\u2190 Back", key=key, on_click=lambda: navigate("home"))


RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
})


def home_page():
    st.title("Face Recognition Attendance System")
    st.caption("Web interface using Streamlit")

    st.button("Add Student", key="nav-add", use_container_width=True, on_click=lambda: navigate("add"))
    st.button("Take Attendance", key="nav-attendance", use_container_width=True, on_click=lambda: navigate("attendance"))
    st.button("Delete Student", key="nav-delete", use_container_width=True, on_click=lambda: navigate("delete"))
    st.button("List Students", key="nav-list", use_container_width=True, on_click=lambda: navigate("list"))
    st.button("Set Threshold", key="nav-threshold", use_container_width=True, on_click=lambda: navigate("threshold"))


def add_student_page():
    st.header("Add Student")
    back_to_home_button("back-add")
    ensure_backend()
    system = st.session_state.system

    with st.form("add_form", clear_on_submit=False):
        student_id = st.text_input("Student ID")
        name = st.text_input("Student Name")
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
        st.info("Position your face in the camera. It will auto-capture when detected. Click Stop to end.")

        class AddStudentProcessor(VideoProcessorBase):
            def __init__(self):
                self.system = system
                self.embedding = None
                self.captured = False
                self.capture_attempts = 0
                self.max_attempts = 50

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                faces = self.system.detect_faces(img)

                # Overlay guidance
                cv2 = __import__("cv2")
                cv2.putText(
                    img,
                    f"Looking for face... ({self.capture_attempts}/{self.max_attempts})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    "Position your face in the camera",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                if not self.captured and self.capture_attempts < self.max_attempts:
                    for face in faces:
                        bbox = face.get("bbox")
                        if bbox is None:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(
                            img,
                            "Detecting...",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )

                        embedding = face.get("embedding")
                        if embedding is None:
                            face_img = img[y1:y2, x1:x2]
                            if face_img.size > 0:
                                embedding = self.system.extract_embedding(face_img)

                        if embedding is not None:
                            self.embedding = (
                                embedding if isinstance(embedding, np.ndarray) else np.array(embedding)
                            )
                            self.captured = True
                            break

                    self.capture_attempts += 1
                elif not faces:
                    cv2.putText(
                        img,
                        "No face detected",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                if self.captured:
                    cv2.putText(
                        img,
                        "Face captured! Saving...",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(
            key="add-student",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=AddStudentProcessor,
        )

        if ctx.video_processor and ctx.state.playing:
            proc = ctx.video_processor
            if proc.embedding is not None and not st.session_state.get("add_saved", False):
                # Store captured embedding in session and ask user to Submit
                st.session_state.add_captured_embedding = proc.embedding.tolist()
                st.success("Face captured. Click Submit to save.")
                if st.button("Submit", key="add-submit", type="primary"):
                    sid = st.session_state.add_student_id
                    name = st.session_state.add_student_name
                    # Final validation before save
                    if not sid or not name:
                        st.error("Student ID and Name are required.")
                    elif sid in system.students:
                        st.error("Student already exists.")
                    else:
                        system.students[sid] = {
                            "name": name,
                            "registration_date": datetime.now().isoformat(),
                        }
                        embedding_entry = {
                            "vector": st.session_state.add_captured_embedding,
                            "timestamp": datetime.now().isoformat(),
                            "confidence": 1.0,
                            "uniqueness": 1.0,
                        }
                        system.embeddings[sid] = {
                            "name": name,
                            "embeddings": [embedding_entry],
                            "embedding": st.session_state.add_captured_embedding,
                            "registration_date": datetime.now().isoformat(),
                            "diversity_score": 1 / system.max_embeddings,
                        }
                        system.save_data()
                        st.session_state.add_saved = True
                        st.success(f"Student {name} registered successfully!")
                        st.button("\u2190 Back", key="back-add-done", on_click=lambda: navigate("home"))


def attendance_page():
    st.header("Take Attendance")
    back_to_home_button("back-attendance")
    ensure_backend()
    system = st.session_state.system

    st.info("Camera running... Click Stop to end.")

    class AttendanceProcessor(VideoProcessorBase):
        def __init__(self):
            self.system = system
            self.recognized_student_ids = set()
            self.frame_count = 0
            self.start_time = time.time()
            self.camera_start_time = self.start_time
            self.last_recognition_time = 0.0
            self.recognition_interval = 0.3
            self.last_embedding_update = {}
            self.notification_text = ""
            self.notification_color = (255, 255, 255)
            self.notification_end_time = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            cv2 = __import__("cv2")

            ret_img = img
            self.frame_count += 1
            current_time = time.time()
            faces = self.system.detect_faces(img)

            if current_time - self.last_recognition_time >= self.recognition_interval:
                for face in faces:
                    bbox = face.get("bbox")
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]

                    face_embedding = face.get("embedding")
                    if face_embedding is None:
                        face_img = img[y1:y2, x1:x2]
                        if face_img.size > 0:
                            face_embedding = self.system.extract_embedding(face_img)

                    student_id, confidence = self.system.recognize_face(face_embedding)

                    if student_id and confidence >= self.system.recognition_threshold:
                        name = self.system.students[student_id]["name"]
                        if student_id not in self.recognized_student_ids:
                            recognition_time = current_time - self.camera_start_time
                            # Console output mirrors main.py
                            print(f"{name:<40}{confidence*100:.2f}%{recognition_time:.2f} seconds")

                        self.recognized_student_ids.add(student_id)
                        self.system.log_attendance(student_id, name)
                        self.system.draw_face_box_with_name(img, (x1, y1, x2, y2), name, confidence, True)
                    elif student_id and 0.5 <= confidence < 0.7:
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
                            self.system.log_attendance(student_id, name)

                        self.system.draw_face_box_with_name(img, (x1, y1, x2, y2), name, confidence, True)
                    else:
                        self.system.draw_face_box_with_name(img, (x1, y1, x2, y2), "Unknown", confidence, False)

                self.last_recognition_time = current_time
            else:
                for face in faces:
                    bbox = face.get("bbox")
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(img, "Tracking...", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            fps = self.frame_count / max(1e-6, (current_time - self.start_time))
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Faces Detected: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, f"Recognized: {len(self.recognized_student_ids)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, "Auto-Learning Active", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="attendance",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=AttendanceProcessor,
    )

    recognized_box = st.empty()
    if ctx.video_processor:
        ids = sorted(list(ctx.video_processor.recognized_student_ids))
        if ids:
            names = [system.students[i]["name"] for i in ids if i in system.students]
            recognized_box.write({"recognized": [{"id": i, "name": n} for i, n in zip(ids, names)]})


def delete_student_page():
    st.header("Delete Student")
    back_to_home_button("back-delete")
    ensure_backend()
    system = st.session_state.system

    students = list(system.students.items())
    if not students:
        st.info("No students registered.")
        return

    options = [f"{data['name']} (ID: {sid})" for sid, data in students]
    selected = st.selectbox("Select a student to delete", options)

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
    st.header("Registered Students")
    back_to_home_button("back-list")
    ensure_backend()
    system = st.session_state.system

    if not system.students:
        st.info("No students registered.")
        return

    st.write(f"Total students: {len(system.students)}")
    rows = []
    for sid, data in system.students.items():
        rows.append({
            "ID": sid,
            "Name": data.get("name", ""),
            "Registered": data.get("registration_date", "N/A"),
        })
    st.dataframe(rows, use_container_width=True)


def threshold_page():
    st.header("Set Threshold")
    back_to_home_button("back-threshold")
    ensure_backend()
    system = st.session_state.system

    current = float(system.recognition_threshold)
    new_val = st.slider("Recognition threshold", min_value=0.1, max_value=0.9, value=current, step=0.01)
    if st.button("Save"):
        system.recognition_threshold = float(new_val)
        st.success(f"Threshold set to {new_val:.2f}")


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
    else:
        navigate("home")
        home_page()


if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"
    router()


