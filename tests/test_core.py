import pytest
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from PIL import Image
from io import BytesIO

def test_style_transfer_model_load():
    model = StyleTransferModel()
    assert model.model is not None, "Model should be loaded successfully."

def test_image_preprocessor_preprocess():
    processor = ImageProcessor()
    image = Image.new("RGB", (800, 600), color="red")
    image_data = BytesIO()
    image.save(image_data, format="JPEG")
    image_data.seek(0)

    processed_image = processor.preprocess_image(image_data)
    assert processed_image.size == (512, 512), "Image should be resized to 512x512."

def test_image_preprocessor_save(tmp_path):
    processor = ImageProcessor()
    image = Image.new("RGB", (512, 512), color="blue")
    save_path = tmp_path / "test_image.jpg"
    processor.save_image(image, save_path)
    assert save_path.exists(), "Image should be saved successfully."
