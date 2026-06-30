# Egyptian License Plate Recognition System (كاشف اللوحات المصرية الذكي)

This project is an AI-powered system designed to detect Egyptian vehicle license plates and perform optical character recognition (OCR) to extract Arabic letters and numbers from the plate. It leverages YOLOv11 and YOLOv12 architectures for high-speed, accurate detection and character recognition.

## 🚀 Features
- **License Plate Detection:** Identifies the location of the license plate within a vehicle image using a fine-tuned YOLO model.
- **Arabic Character Recognition (OCR):** Segments and recognizes individual Arabic letters and numbers on the plate.
- **Image Deskewing:** Pre-processes license plates using OpenCV to correct perspective and rotation for better OCR accuracy.
- **Streamlit Web Application:** Interactive dashboard to upload images/videos, adjust confidence thresholds, and view results with bounding boxes.
- **FastAPI API Server:** High-performance API backend designed for easy deployment (compatible with Docker / Hugging Face Spaces).
- **Real-time Camera/Video Processing:** Python scripts to run detection on live webcams or video files.

---

## 📁 Repository Structure
```
├── Characters/                          # YOLO model details for character recognition
│   ├── characters_yolo12n_best.pt       # YOLO model weights for characters
│   └── train_characters_yolo12n/        # Training graphs and metrics
├── fastapi_app/                         # FastAPI Web Service
│   ├── Dockerfile                       # Docker configuration for deployment
│   ├── main.py                          # FastAPI endpoint code
│   ├── requirements.txt                 # Backend dependencies
│   └── README.md                        # FastAPI documentation
├── train3/                              # Training outputs for license plate detector
│   ├── weights/                         # YOLO model weights (best.pt / last.pt)
│   └── confusion_matrix, curves, etc.   # Evaluation metrics
├── streamlit_app.py                     # Streamlit frontend application
├── run_camera.py                        # Script for real-time camera inference
├── run_camera_local.py                  # Local camera inference script
├── run_realtime.py                      # Real-time video/camera license plate detection script
├── best.pt                              # Fine-tuned license plate detection weights (YOLO)
└── yolo11n.pt                           # Pre-trained base YOLO weights
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/graduationprojects73-a11y/Cai_ai.git
cd Cai_ai
```

### 2. Install dependencies
Ensure you have Python 3.9+ installed. You can install the required packages using:
```bash
pip install -r fastapi_app/requirements.txt
# Additional packages for Streamlit frontend
pip install streamlit opencv-python pillow ultralytics
```

---

## 🖥️ Running the Applications

### Run Streamlit Web Application
To start the interactive web interface:
```bash
streamlit run streamlit_app.py
```

### Run FastAPI Server
To launch the API server locally:
```bash
cd fastapi_app
uvicorn main:app --reload --port 7860
```
Then visit `http://localhost:7860` to view the API documentation.

### Run Real-time Detection
To run the detection system using your webcam or a video file:
```bash
python run_realtime.py
```

---

## 🐳 Docker Deployment
You can build and run the FastAPI server inside a Docker container:
```bash
docker build -t egyptian-plate-ocr fastapi_app/
docker run -p 7860:7860 egyptian-plate-ocr
```
