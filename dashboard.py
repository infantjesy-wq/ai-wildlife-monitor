import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import os
import time
from ultralytics import YOLO
from twilio.rest import Client
import av

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="AI Wildlife Monitor", layout="wide")
st.title("🦒 Live Wildlife Detection & Alert System")

# -------------------------------
# LOAD SECRETS (Hugging Face / .env)
# -------------------------------
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_PHONE")
to_number = os.getenv("YOUR_PHONE")

# Initialize Twilio client
client = None
if account_sid and auth_token:
    client = Client(account_sid, auth_token)

# -------------------------------
# LOAD MODEL (AUTO DOWNLOAD)
# -------------------------------
model = YOLO("yolov8n.pt")

# Wild animals list
wild_animals = ["elephant", "bear", "zebra", "giraffe"]

# -------------------------------
# VIDEO PROCESSOR CLASS
# -------------------------------
class VideoProcessor:
    def __init__(self):
        self.last_alert_time = 0
        self.cooldown = 30  # seconds

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)

        current_time = time.time()

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)

                # Check for wild animal
                if label in wild_animals and conf > 0.60:
                    if current_time - self.last_alert_time > self.cooldown:
                        if client:
                            try:
                                client.messages.create(
                                    from_=from_number,
                                    body=f"🚨 ALERT: {label} detected!",
                                    to=to_number
                                )
                                print(f"Alert sent for {label}")
                                self.last_alert_time = current_time
                            except Exception as e:
                                print("Twilio Error:", e)

        # Return annotated frame
        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")


# -------------------------------
# WEBRTC CONFIG (IMPORTANT FIX)
# -------------------------------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# -------------------------------
# START CAMERA
# -------------------------------
st.write("👉 Click START and allow camera access")

webrtc_streamer(
    key="wildlife",
    video_frame_callback=VideoProcessor().recv,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.info("📌 Allow camera permission when prompted. Alerts will be sent on detecting wild animals.")