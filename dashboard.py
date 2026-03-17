import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2
import os
import time
from ultralytics import YOLO
from twilio.rest import Client
from dotenv import load_dotenv

# -------------------------------
# LOAD ENV VARIABLES
# -------------------------------
load_dotenv()

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="AI Wildlife Monitor", layout="wide")
st.title("🦒 Live Wildlife Detection & Alert System")

# -------------------------------
# TWILIO CONFIG
# -------------------------------
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_PHONE")
to_number = os.getenv("YOUR_PHONE")

client = None
if account_sid and auth_token:
    client = Client(account_sid, auth_token)

st.write("Twilio Loaded:", client is not None)

# -------------------------------
# TEST BUTTON
# -------------------------------
if client:
    if st.button("📩 Test WhatsApp Alert"):
        try:
            client.messages.create(
                from_=from_number,
                body="✅ Test message from AI Wildlife App",
                to=to_number
            )
            st.success("Test alert sent successfully!")
        except Exception as e:
            st.error(f"Twilio Error: {e}")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("yolov8n.pt")

# IMPORTANT: include 'cat' (since tiger detected as cat)
wild_animals = ["elephant", "bear", "zebra", "giraffe", "cat"]

# -------------------------------
# VIDEO PROCESSOR
# -------------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_alert_time = 0
        self.cooldown = 30  # seconds

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img)
        current_time = time.time()

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    label = model.names[int(box.cls)]
                    conf = float(box.conf)

                    print("Detected:", label, conf)

                    if label in wild_animals and conf > 0.30:
                        if current_time - self.last_alert_time > self.cooldown:
                            print("Triggering alert...")

                            if client:
                                try:
                                    client.messages.create(
                                        from_=from_number,
                                        body=f"🚨 ALERT: {label} detected!",
                                        to=to_number
                                    )
                                    self.last_alert_time = current_time
                                    print("Alert sent!")
                                except Exception as e:
                                    print(f"Twilio Error: {e}")

        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# -------------------------------
# WEBRTC CONFIG
# -------------------------------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# -------------------------------
# START CAMERA
# -------------------------------
st.write("Click **Start** to begin live detection")

webrtc_streamer(
    key="wildlife",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.info("Allow camera access when prompted.")