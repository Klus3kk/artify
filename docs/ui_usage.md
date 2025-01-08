# Streamlit UI Usage

Artify's Streamlit-based UI provides an intuitive way to perform style transfer.

## Starting the UI
Launch the UI by running:

```bash
streamlit run interface/UIHandler.py
```

## Workflow
1. **Upload Content Image**:
   - Click "Upload Content Image" and select a `.jpg` or `.png` file.
2. **Select Style**:
   - Choose a style category from the dropdown (e.g., `Impressionism`).
3. **Generate Styled Image**:
   - Click "Generate Styled Image" to start the process.
4. **Download Image**:
   - Download the styled image once the process is complete.

## Features
- Live previews of content and style images.
- Progress spinner during style transfer.
- Download button for styled images.

## Error Handling
- Invalid input or missing Hugging Face token will show an error message.

