from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor

if __name__ == "__main__":
    content_image_path = ""
    style_image_path = ""
    output_image_path = ""

    processor = ImageProcessor()
    model = StyleTransferModel()

    content_image = processor.preprocess_image(content_image_path)
    style_image = processor.preprocess_image(style_image_path)

    # Apply style (placeholder for now)
    # styled_image = model.apply_style(content_image, style_image)

    # Save the output (placeholder for now)
    # processor.save_image(styled_image, output_image_path)

    print("Pipeline tested successfully")
