import streamlit as st
import cv2
import pandas as pd
import tempfile
import sqlite3
import os
import easyocr
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# --------------------------
# 1. Database Setup
# --------------------------
DB_NAME = 'traffic_data.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vehicle_logs
                 (vehicle_type TEXT, plate_number TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def log_vehicle(v_type, plate):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO vehicle_logs VALUES (?, ?, ?)", (v_type, plate, now_str))
    conn.commit()
    conn.close()

def get_logs():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM vehicle_logs ORDER BY timestamp DESC LIMIT 10", conn)
    conn.close()
    return df

def get_counts():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT vehicle_type, COUNT(*) as count FROM vehicle_logs GROUP BY vehicle_type")
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def clear_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM vehicle_logs")
    conn.commit()
    conn.close()

# --------------------------
# 2. AI Models Setup
# --------------------------
@st.cache_resource
def load_yolo():
    # Load YOLO model only once
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_ocr():
    # Load EasyOCR only once
    return easyocr.Reader(['en'], gpu=False)

def read_license_plate(image, reader):
    # Use Haar Cascade to find a plate
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = cascade.detectMultiScale(gray, 1.1, 4)
    
    plate_img = None
    if len(plates) > 0:
        x, y, w, h = plates[0]
        plate_img = image[y:y+h, x:x+w]
    else:
        # Fallback to bottom 30% of vehicle
        h, w = image.shape[:2]
        plate_img = image[int(h*0.7):h, :]

    if plate_img is None or plate_img.size == 0:
         return "UNKNOWN"
         
    # OCR reading
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray_plate)
    
    text = "".join([res[1] for res in result])
    cleaned = ''.join(e for e in text if e.isalnum()).upper()
    return cleaned if len(cleaned) >= 4 else "UNKNOWN"

# --------------------------
# 3. Streamlit Dashboard App
# --------------------------
st.set_page_config(layout="wide")
st.title("🚦 Smart City: Live Traffic Tracker")
st.write("Upload a traffic video to detect vehicles, track them across the virtual line, and read their license plates.")

init_db()
yolo_model = load_yolo()
ocr_reader = load_ocr()

video_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi"])
start_btn = st.button("🚀 Start Tracking")

col1, col2 = st.columns([2, 1])
frame_window = col1.empty()
metrics_window = col2.empty()
logs_window = st.empty()
chart_window = st.empty()

if start_btn and video_file:
    clear_db() # reset database for new video
    
    # Save video locally so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    target_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    tracked_vehicles = set()
    previous_y = {}
    
    line_y = None
    frame_idx = 0
    
    st.success("Processing Video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if frame_idx % 2 != 0: # Process every 2nd frame
            continue
            
        if line_y is None:
            line_y = int(frame.shape[0] * 0.25)
            
        # Draw strictly crossing line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)

        # Detect and Track
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id not in target_classes:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                v_type = target_classes[class_id]
                
                # Draw boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # Logic for line crossing
                crossed = False
                if track_id in previous_y:
                    if (previous_y[track_id] < line_y and cy >= line_y) or (previous_y[track_id] > line_y and cy <= line_y):
                        crossed = True
                
                previous_y[track_id] = cy
                
                # If newly crossed
                if crossed and track_id not in tracked_vehicles:
                    tracked_vehicles.add(track_id)
                    
                    # Read License Plate
                    plate = read_license_plate(frame[y1:y2, x1:x2], ocr_reader)
                    
                    # Save to DB
                    log_vehicle(v_type, plate)
                    
                    cv2.putText(frame, f"Logged {v_type}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Update Video preview
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        # Update metrics every 10 processed frames
        if frame_idx % 20 == 0:
            counts = get_counts()
            if counts:
                with metrics_window.container():
                    st.write("### Live Counts")
                    for t, c in counts.items():
                        st.metric(t.capitalize(), c)
            
            df = get_logs()
            if not df.empty:
                logs_window.dataframe(df, use_container_width=True, hide_index=True)

    cap.release()
    try: os.remove(tfile.name)
    except: pass
    
    # Final Chart
    st.write("### Final Vehicle Counts")
    final_counts = get_counts()
    if final_counts:
        df_chart = pd.DataFrame(list(final_counts.items()), columns=["Type", "Count"]).set_index("Type")
        chart_window.bar_chart(df_chart)
    
    st.balloons()