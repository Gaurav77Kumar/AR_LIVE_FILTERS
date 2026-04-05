# AR Live Filters Web

## What this project is about
This project is a Streamlit web app that applies real-time AR-style visual filters to your webcam video feed.

It also supports simple in-air drawing using hand tracking:
- The app detects your index finger (via MediaPipe Hands)
- It draws a line trail on top of the filtered video
- You can clear the drawing canvas with a button

Main file:
- app.py: Streamlit app, filter logic, webcam stream processing, and hand drawing

## Built with
- Python
- Streamlit
- streamlit-webrtc
- OpenCV
- MediaPipe
- NumPy
- av

## Filters included
- CYBERPUNK
- GLOW
- RETROWAVE
- MATRIX
- DREAMY
- VAPORWAVE

## How to run this project (Windows)
1. Open a terminal in the project folder.
2. (Recommended) Create a virtual environment:

   python -m venv .venv

3. Activate the virtual environment:

   .venv\Scripts\activate

4. Install dependencies:

   pip install -r requirements.txt

5. Run the app:

   streamlit run app.py

6. Open the local URL shown in terminal (usually http://localhost:8501).
7. Allow camera permission in your browser.

## Notes
- If MediaPipe is not available on your system, the app still runs with filters, but hand drawing will be disabled.
- packages.txt contains Linux system packages often used for deployment environments.

## Quick troubleshooting
- Camera not opening:
  - Check browser camera permissions.
  - Close other apps that may be using the webcam.
- Import errors:
  - Make sure the virtual environment is activated.
  - Re-run: pip install -r requirements.txt
- Streamlit command not found:
  - Run: python -m streamlit run app.py
