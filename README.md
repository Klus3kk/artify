# Artify

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20MacOS-green)](https://github.com/Klus3kk/artify)

**Artify** is a user-friendly tool that applies artistic styles to images using AI. It supports both **CLI** and **UI-based interfaces**.

## Features

- **AI-Powered Style Transfer**: transform your images using pre-trained style models (e.g., impressionism, abstract, surrealism).
- **Command-Line Interface**: automate your workflows with a robust CLI.
- **Streamlit UI**: upload images, choose styles and generate styled results interactively.
- **Logging**: transparent and detailed logging for all operations.
- **Hugging Face Integration**: automatically download required models for styles.
- **Docker & Kubernetes (in the future...)**

## Setup Instructions

### Prerequisites

- Python 3.8+
- Miniconda or pip (preferred for environment setup)
- NVIDIA GPU with CUDA (optional but recommended)

### Installation

```bash
git clone https://github.com/ClueSec/artify.git
cd artify
```

```bash
conda create -n artify python=3.10 -y
conda activate artify
pip install -r requirements.txt
```

- Add your Hugging Face token to `.env`

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
2. Select a style category.
3. Generate the styled image and download it.


