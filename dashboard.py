import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import tempfile

# -------------------------------
# Load Model
# -------------------------------
model = YOLO("yolov8n.pt")

st.title("🦒 AI Wildlife Detection & Alert System")

# -------------------------------
# Alert function
# -------------------------------
def check_alert(detections):
    alert_animals = ["dog", "cat", "cow", "horse", "sheep", "elephant", "bear", "zebra", "giraffe"]

    detected_classes = detections.names
    found = []

    for box in detections.boxes:
        cls_id = int(box.cls[0])
        class_name = detected_classes[cls_id]
        if class_name in alert_animals:
            found.append(class_name)

    return list(set(found))


# -------------------------------
# Select Input
# -------------------------------
option = st.radio("Choose Input Type:", ["Image", "Video"])


# -------------------------------
# IMAGE UPLOAD
# -------------------------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        if st.button("🔍 Detect Animals"):
            img_array = np.array(image)

            with st.spinner("Detecting..."):
                results = model(img_array)

            annotated = results[0].plot()
            st.image(annotated, caption="Detection Result")

            # ALERT
            detected = check_alert(results[0])
            if detected:
                st.error(f"🚨 ALERT! Animals detected: {', '.join(detected)}")
            else:
                st.success("✅ No dangerous animals detected")

# -------------------------------
# VIDEO UPLOAD
# -------------------------------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        alert_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR")

            # ALERT
            detected = check_alert(results[0])
            if detected:
                alert_box.error(f"🚨 ALERT! Animals detected: {', '.join(detected)}")
            else:
                alert_box.success("✅ Safe")

        cap.release()
