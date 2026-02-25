import streamlit as st
from google import genai
from ultralytics import YOLO
from PIL import Image
import os

# UI Configuration
st.set_page_config(page_title="YOLO + Gemini Vision AI")
st.header("Computer Vision & GenAI Assistant")

# API Setup
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in environment variables.")
    st.stop()

client = genai.Client(api_key=api_key)

# Load YOLO Model
yolo_model = YOLO('yolov8n.pt')

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    if st.button("Analyze with YOLO & Gemini"):
        with st.spinner('Detecting objects and generating summary...'):

            # Step 1: YOLO Detection
            results = yolo_model(img)

            detected_names = []
            for r in results:
                for c in r.boxes.cls:
                    detected_names.append(yolo_model.names[int(c)])

            unique_objects = ", ".join(list(set(detected_names)))

            if not unique_objects:
                unique_objects = "No clearly identifiable objects"

            # Step 2: Gemini Summary
            prompt = f"""
            I have detected the following objects in an image: {unique_objects}.
            Based on these objects, provide a professional summary of what this scene might be and its context.
            """

            try:
                response = client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=[prompt]
                )
                summary_text = response.text

            except Exception as e:
                st.error(f"Gemini Error: {e}")

                st.subheader("Available Models in Your Project:")
                try:
                    models = client.models.list()
                    for m in models:
                        st.write(m.name)
                except Exception as list_error:
                    st.error(f"Model listing also failed: {list_error}")

                summary_text = "Model error occurred. Please check available models above."

            # Display Results
            st.subheader("YOLO Detected Objects:")
            st.success(unique_objects)

            st.subheader("AI Contextual Summary:")
            st.write(summary_text)
