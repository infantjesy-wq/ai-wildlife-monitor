import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
from twilio.rest import Client

# -------------------------------
# PAGE
# -------------------------------
st.set_page_config(page_title="Wildlife Detection", layout="centered")
st.title("🦒 AI Wildlife Detection + WhatsApp Alert")

# -------------------------------
# LOAD MODEL (SAFE)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------------
# TWILIO WHATSAPP CONFIG
# -------------------------------
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_whatsapp = "whatsapp:" + os.getenv("TWILIO_PHONE", "")
to_whatsapp = "whatsapp:" + os.getenv("YOUR_PHONE", "")

def send_whatsapp(msg):
    if account_sid and auth_token:
        client = Client(account_sid, auth_token)
        client.messages.create(
            body=msg,
            from_=from_whatsapp,
            to=to_whatsapp
        )

# -------------------------------
# ALERT LOGIC
# -------------------------------
def check_alert(results):
    alert_animals = ["dog", "cat", "cow", "horse", "sheep", "elephant", "bear"]

    names = results.names
    found = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        if label in alert_animals:
            found.append(label)

    return list(set(found))

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("🔍 Detect"):
        img = np.array(image)
        img = cv2.resize(img, (640, 480))

        with st.spinner("Detecting..."):
            results = model(img)

        result_img = results[0].plot()
        st.image(result_img, caption="Detection Result")

        detected = check_alert(results[0])

        if detected:
            msg = f"🚨 ALERT! Animals detected: {', '.join(detected)}"
            st.error(msg)
            send_whatsapp(msg)
        else:
            st.success("✅ No animals detected")
