---
title: Violationn
emoji: 🏎️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Egyptian Plate OCR API (Violationn)

This is a **FastAPI** application for detecting Egyptian license plates and recognizing characters using **YOLO**.

## Features
- **Plate Detection:** Locates the license plate in the image.
- **Character Recognition (OCR):** Reads the letters and numbers on the plate.

## Endpoints
- `GET /`: Health check.
- `POST /process`: Upload an image to get detection and OCR results.

## Deployment on Hugging Face
This Space is configured to run as a **Docker** container. It uses the `Dockerfile` in the root directory to set up the environment, install dependencies from `requirements.txt`, and run the FastAPI server on port `7860`.
