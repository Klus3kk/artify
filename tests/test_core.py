import pytest
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from PIL import Image
from io import BytesIO

@pytest.fixture
def mock_image():
    image = Image.new("RGB", (800, 600), color="red")
    image_data = BytesIO()
    image.save(image_data, format="JPEG")
    image_data.seek(0)
    return image_data

def test_style_transfer_model_load():
    model = StyleTransferModel()
    assert model.model is not None, "Model should be loaded successfully."

def test_style_transfer_model_apply(mock_image):
    model = StyleTransferModel()
    content_image = Image.open(mock_image)
    styled_image = model.apply_style(content_image, content_image)
    assert styled_image is not None, "Styled image should be generated successfully."

def test_image_preprocessor_preprocess(mock_image):
    processor = ImageProcessor()
    processed_image = processor.preprocess_image(mock_image)
    assert processed_image.size == (512, 512), "Image should be resized to 512x512."

def test_image_preprocessor_save(tmp_path):
    processor = ImageProcessor()
    image = Image.new("RGB", (512, 512), color="blue")
    save_path = tmp_path / "test_image.jpg"
    processor.save_image(image, save_path)
    assert save_path.exists(), "Image should be saved successfully."

def test_image_preprocessor_invalid_input():
    processor = ImageProcessor()
    with pytest.raises(ValueError):
        processor.preprocess_image(None)
