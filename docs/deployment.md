# Deployment Guide for Artify

## Introduction
This document explains how to deploy Artify locally and on cloud platforms using Docker and Kubernetes. Artify's deployment process is designed to be simple, scalable, and portable.

## Local Deployment (Without Docker)

### Prerequisites
1. **Python**: Ensure Python 3.8 or later is installed.
2. **Virtual Environment**: Use `venv` or Conda.
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running Artify Locally
1. **CLI Interface**:
   ```bash
   python interface/CLIHandler.py --content <path_to_content_image> --style_category <style_category> --output <path_to_output>
   ```

2. **UI Interface**:
   ```bash
   streamlit run interface/UIHandler.py
   ```

## Deployment with Docker

### Prerequisites
1. **Docker Installed**: Ensure Docker is installed on your system.
2. **Dockerfile**: Artify provides a pre-configured Dockerfile.

### Steps to Build and Run
1. **Build the Docker Image**:
   ```bash
   docker build -t artify .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 artify
   ```

3. **Access the UI**:
   Open your browser and go to `http://localhost:8501`.


## Kubernetes Deployment

### Prerequisites
1. **Kubernetes Cluster**: Set up a Kubernetes cluster using tools like `minikube` or `kubectl`.
2. **Manifest Files**: Artify includes Kubernetes deployment and service manifests in the `k8s/` directory.

### Steps to Deploy
1. **Apply the Deployment and Service Files**:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```

2. **Expose the Service**:
   ```bash
   kubectl expose deployment artify --type=LoadBalancer --name=artify-service
   ```

3. **Access the UI**:
   Get the external IP of the service and navigate to it in your browser.

