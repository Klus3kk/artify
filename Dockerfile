# Use a lightweight Python image with GPU support (if applicable)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set up environment
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && apt-get clean

# Install project dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . /app

# Expose the API port
EXPOSE 8000

# Set the entry point for FastAPI
CMD ["python", "api/FastAPIHandler.py"]
