import os
from core.ImageProcessor import ImageProcessor

def test_preprocess_image():
    processor = ImageProcessor()
    processed_image = processor.preprocess_image("")
    assert processed_image.size == (512, 512)

def test_save_image():
    processor = ImageProcessor()
    sample_image = processor.preprocess_image("")
    processor.save_image(sample_image, "")
    assert os.path.exists("")
