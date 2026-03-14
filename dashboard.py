import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import os
import time
from ultralytics import YOLO
from twilio.rest import Client
import av

# Page Setup
st.set_page_config(page_title="AI Wildlife Monitor", layout="wide")
st.title("🦒 Live Wildlife Detection & Alert System")

# 1. LOAD SECRETS (From Hugging Face Settings)
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_PHONE")
to_number = os.getenv("YOUR_PHONE")

# Initialize Twilio
client = Client(account_sid, auth_token) if account_sid else None

# 2. LOAD MODEL
model = YOLO("yolov8n.pt")
wild_animals = ["elephant", "bear", "zebra", "giraffe"]

# 3. DETECTION LOGIC
class VideoProcessor:
    def __init__(self):
        self.last_alert_time = 0
        self.cooldown = 30  # seconds

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        
        # Analyze detections
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                current_time = time.time()

                if label in wild_animals and conf > 0.60:
                    # Trigger Alert if cooldown passed
                    if current_time - self.last_alert_time > self.cooldown:
                        if client:
                            try:
                                client.messages.create(
                                    from_=from_number,
                                    body=f'🚨 ALERT: {label} detected!',
                                    to=to_number
                                )
                                self.last_alert_time = current_time
                            except Exception as e:
                                print(f"Alert Error: {e}")

        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

# 4. START WEBCAM ON THE WEB
st.write("Click 'Start' below to enable your camera for real-time monitoring.")
webrtc_streamer(key="wildlife", video_frame_callback=VideoProcessor().recv)

st.info("Note: The first time you run this, your browser will ask for camera permission.")
