# Command-Line Interface (CLI) Usage

Artify's CLI is designed for advanced users who prefer automation or scripting.

## Basic Usage

Run the following command:

```bash
python interface/CLIHandler.py --content <content_image_path> --style_category <style> --output <output_path>
```

### Required Arguments
- `--content`: Path to the content image (e.g., `images/content/sample.jpg`).
- `--style_category`: Artistic style (e.g., `impressionism`, `abstract`).
- `--output`: Path to save the styled image.

### Example
```bash
python interface/CLIHandler.py --content images/content/sample.jpg --style_category impressionism --output images/output/impressionism.jpg
```

## Error Handling
1. **Invalid Paths**:
   - If the content image path is invalid, an error is logged, and the process halts.
2. **Missing Models**:
   - If the model is not found locally, it is downloaded dynamically from Hugging Face.

## Logs
- Logs are stored in `cli.log`.
- Example:
  ```plaintext
  [INFO] [1/5] Validating input arguments...
  [ERROR] Content image 'invalid_path.jpg' does not exist.
  ```
