import streamlit as st
import pandas as pd
import os
from PIL import Image

st.title("AI Wildlife Monitoring Dashboard")

if os.path.exists("detections_log.csv"):

    df = pd.read_csv("detections_log.csv")

    st.subheader("Detection Log")
    st.write(df)

    st.subheader("Detection Statistics")
    st.bar_chart(df["animal"].value_counts())

    st.subheader("Captured Images")

    for img in reversed(os.listdir("detections")):
        image = Image.open(os.path.join("detections", img))
        st.image(image, caption=img)

else:
    st.write("No detection data available")