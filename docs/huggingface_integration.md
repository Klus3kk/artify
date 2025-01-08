# Hugging Face Integration

Artify integrates with Hugging Face Hub for dynamic model management.

## Purpose
Hugging Face integration allows:
1. Dynamic downloading of pre-trained models.
2. Cache management to avoid redundant downloads.
3. Flexibility in adding new styles by updating the Hugging Face repository.

## Workflow
1. **Token Authentication**:
   - Artify uses the Hugging Face token (`HF_TOKEN`) for authenticated API calls.
2. **Model Download**:
   - The `HuggingFaceHandler` module downloads models dynamically using `snapshot_download`.

### Code Example
```python
from huggingface_hub import snapshot_download

def download_model(repo_id, filename, cache_dir):
    model_path = snapshot_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
```

## Repository Structure
- Models are stored in the `artify-models` repository.
- Each style has a corresponding `.pth` file.

## Adding New Styles
1. Add the model file to the Hugging Face repository.
2. Update the `StyleRegistry` with the new style.