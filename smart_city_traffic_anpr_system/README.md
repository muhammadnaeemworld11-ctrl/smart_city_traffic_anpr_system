# 🚦 Smart Cities: AI-Based Vehicle Tracking & ANPR System

An AI-powered automated traffic monitoring system that detects vehicles, counts them, and recognizes license plates in real-time for better traffic management and law enforcement.

## 🌟 Key Features
- **Vehicle Detection & Classification:** Real-time stream processing using a YOLOv8 Nano model (detects cars, bikes, trucks, buses).
- **Line Crossing Tracker:** Tracks vehicles across a virtual road line using `ByteTrack` logic to increment custom vehicle-class counters.
- **Automatic Number Plate Recognition (ANPR):** Utilizes Haar Cascades and EasyOCR to accurately crop plates and extract license plate alphanumeric codes.
- **Local SQLite Storage:** Logs occurrences inside `database/anpr.db` holding the Vehicle Type, Plate Number, and precise timestamp.
- **Live Analytics Dashboard:** Interactive `Streamlit` App featuring live frame streaming feeds, active session logging dataframes, and analytical final-run profile bar-charts.

## 🛠️ Tech Stack
- **Python** 
- **Computer Vision:** `OpenCV`, `Ultralytics (YOLOv8)`
- **OCR:** `EasyOCR`
- **Database:** `SQLite3`
- **Dashboard UI:** `Streamlit`, `Pandas` 

## 📂 Project Structure
```text
📦 project_folder
 ┣ 📂 core
 ┃ ┣ 📜 anpr.py                # ALPR logic utilizing Haarcascades & EasyOCR
 ┃ ┗ 📜 vehicle_tracker.py     # YOLOv8 inference, Byetrack Tracking, line crossing logic
 ┣ 📂 database
 ┃ ┣ 📜 db_manager.py          # SQLite database connection & CRUD operations
 ┃ ┗ 📜 anpr.db                # Generated local database file
 ┣ 📂 models                   # Directory to store any custom trained `.pt` ALPR models
 ┣ 📜 streamlit_app.py         # Main web dashboard interface
 ┣ 📜 requirements.txt         # Project pip dependencies
 ┗ 📜 README.md
```

## 🚀 How to Run Locally

1. **Install Dependencies**
   Ensure you have Python installed, then install all required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**
   Launch the web app directly from the terminal using Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Verify Functionality**
   Upload any of the provided test MP4 files directly from the dashboard UI and click "🚀 Start Processing".

## ☁️ Deployment Instructions (Hackathon Deliverables)
1. **GitHub Repository:**
   Push this exact folder structure to a public repository on your GitHub account. Ensure you upload the `sample_video.mp4` sequences to a Google Drive link or directly to the repository if size permits.
   
2. **Streamlit App:**
   - Log into [Streamlit Community Cloud](https://share.streamlit.io/).
   - Link it to your GitHub Repository.
   - Point the Main file path to `streamlit_app.py`.
   - Click **Deploy!**

## 💡 Notes on ANPR Detection Quality
Currently, the pipeline attempts to localize plates utilizing the Haar Cascades protocol. For the highly robust "YOLO ANPR" deliverable mentioned in the requirements, the provided Roboflow Universal Dataset (`roboflow-universe-projects/license-plate-recognitionrxg4e`) can be used to locally train a custom Object Detection model via Ultralytics padding. Upon completion, store `best.pt` in the `/models` directory and import it interchangeably in `core/anpr.py`.
