import streamlit as st
from google import genai
from ultralytics import YOLO
from PIL import Image
import os

# UI Configuration
st.set_page_config(page_title="YOLO + Gemini Vision AI")
st.header("Computer Vision & GenAI Assistant")

# API Setup
api_key = os.getenv("GEMINI_API_KEY") # GitHub Secrets se aayega
client = genai.Client(api_key=api_key)
# Model Load (YOLOv8 Nano - Sabse fast aur light)
yolo_model = YOLO('yolov8n.pt') 

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Analyze with YOLO & Gemini"):
        with st.spinner('Detecting objects and generating summary...'):
            # Step 1: YOLO Detection
            results = yolo_model.predict(source=img)
            
            detected_names = []
            for r in results:
                for c in r.boxes.cls:
                    detected_names.append(yolo_model.names[int(c)])
            
            unique_objects = ", ".join(list(set(detected_names)))
            
            # Step 2: Gemini Summary based on YOLO results
            prompt = f"I have detected the following objects in an image: {unique_objects}. Based on these objects, provide a professional summary of what this scene might be and its context."
            
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[prompt, img]
            )
            
            # Display Results
            st.subheader("YOLO Detected Objects:")
            st.success(unique_objects if unique_objects else "No objects detected.")
            
            st.subheader("AI Contextual Summary:")
            st.write(response.text)
