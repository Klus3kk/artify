# Artify

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20MacOS-green)](https://github.com/Klus3kk/artify)

**Artify** is a user-friendly tool that applies artistic styles to images using AI. It supports both **CLI** and **UI-based interfaces** and integrates with Hugging Face, Docker, and Kubernetes for scalability and ease of deployment.

## Key Features

- **AI-Powered Style Transfer**: Transform your images using pre-trained style models (e.g., Impressionism, Abstract, Surrealism).
- **Command-Line Interface (CLI)**: Automate your workflows with a robust CLI.
- **Streamlit UI**: Upload images, choose styles, and generate styled results interactively.
- **Logging**: Transparent and detailed logging for all operations.
- **Hugging Face Integration**: Automatically download required models for styles.
- **Docker & Kubernetes (in the future)**: Containerized for easy deployment and scalability.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Miniconda or pip (preferred for environment setup)
- NVIDIA GPU with CUDA (optional but recommended)
- Docker and Kubernetes (in progress)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ClueSec/artify.git
   cd artify
   ```

2. Install dependencies:
   ```bash
   conda create -n artify python=3.10 -y
   conda activate artify
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   
   - Add your Hugging Face token to `.env`:
     ```
     HF_TOKEN=your_huggingface_api_token
     ```

## Usage

### CLI Usage

The CLI provides automation for style transfer:

```bash
python interface/CLIHandler.py --content <path_to_content_image> \
    --style_category <style_category> \
    --output <path_to_output_image>
```

#### Example:

```bash
python interface/CLIHandler.py --content images/content/sample_content.jpg \
    --style_category impressionism \
    --output images/output/styled_image.jpg
```

### UI Usage

Start the interactive Streamlit UI:

```bash
streamlit run interface/UIHandler.py
```

1. Upload your content image.
2. Select a style category (e.g., Impressionism).
3. Generate the styled image and download it.


## Architecture

### High-Level Workflow

1. **Input**: User uploads a content image and selects a style.
2. **Preprocessing**:
   - Image resizing and normalization.
   - Pre-trained model is used for feature extraction.
3. **Style Transfer**:
   - Models trained on specific artistic styles.
   - Gram matrices for style features.
4. **Output**: Styled image is generated and saved.

## Deployment

### Docker

Build the Docker image:
```bash
docker build -t artify .
```

Run the container:
```bash
docker run -p 8501:8501 artify
```

