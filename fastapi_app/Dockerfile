# Use official Python image
FROM python:3.10-slim

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Add a non-root user (recommended by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory to the user's home
WORKDIR /home/user/app

# Copy requirements and install
COPY --chown=user requirements.txt .
# Install PyTorch CPU directly to save space
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
