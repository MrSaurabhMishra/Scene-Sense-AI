# 1. Base Python image
FROM python:3.9-slim

# 2. Install system dependencies for OpenCV/YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Work directory
WORKDIR /app

# 4. Copy files
COPY . .

# 5. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Streamlit port
EXPOSE 8501

# 7. Start App
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
