## Streamlit Web App

Run the web interface that mirrors the Tkinter GUI with five actions: Add, Take Attendance, Delete, List, Set Threshold.

### Setup

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Start the app:

```bash
python -m streamlit run web_app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

Notes:
- The app uses your webcam via WebRTC. Allow camera permission in the browser.
- Data files are the same as the desktop app under `database/`.


