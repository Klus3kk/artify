from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor

if __name__ == "__main__":
    content_image_path = "path/to/content.jpg"
    style_image_path = "path/to/style.jpg"
    output_image_path = "path/to/output.jpg"

    processor = ImageProcessor()
    model = StyleTransferModel()

    content_image = processor.preprocess_image(content_image_path)
    style_image = processor.preprocess_image(style_image_path)

    styled_image = model.apply_style(content_image, style_image)
    processor.save_image(styled_image, output_image_path)

    print("Style transfer completed!")
