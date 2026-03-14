from ultralytics import YOLO
import cv2
import datetime
import os
from twilio.rest import Client
import time
import csv
from dotenv import load_dotenv
import os

load_dotenv()
last_alert_time = 0
cooldown = 30   # seconds


# Load YOLO model
model = YOLO("yolov8n.pt")
# Twilio setup
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_PHONE")
to_number = os.getenv("YOUR_PHONE")

client = Client(account_sid, auth_token)

# List of wild animals
wild_animals = ["elephant", "bear", "zebra", "giraffe"]

# Create folder for captured images
if not os.path.exists("detections"):
    os.makedirs("detections")
    # Create log file if it doesn't exist
if not os.path.exists("detections_log.csv"):
    with open("detections_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "animal", "confidence", "image"])

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:

            class_id = int(box.cls[0])
            label = model.names[class_id]

            confidence = float(box.conf[0])

            current_time = time.time()

            if label in wild_animals and confidence > 0.60 and (current_time - last_alert_time > cooldown):

                print("🚨 WILD ANIMAL DETECTED:", label)

                time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"detections/{label}_{time_now}.jpg"

                cv2.imwrite(filename, frame)
                message = client.messages.create(

                from_=from_number,
                body=f'🚨 Wild Animal Detected: {label}',
                to=to_number
                )

                print("WhatsApp alert sent")

                last_alert_time = current_time

                # timestamp for file name
                time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                filename = f"detections/{label}_{time_now}.jpg"

                cv2.imwrite(filename, frame)

                print("Image saved:", filename)
                # Write detection into log file
                with open("detections_log.csv", "a", newline="") as file:
                     writer = csv.writer(file)
                     writer.writerow([time_now, label, confidence, filename])

    annotated = results[0].plot()

    cv2.imshow("Wildlife Detection System", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()