import argparse
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry

def main():
    parser = argparse.ArgumentParser(description="Artify: Apply artistic styles to images.")
    parser.add_argument("--content", required=True, help="Path to content image")
    parser.add_argument("--style_category", required=True, help="Style category (e.g., impressionism, abstract)")
    parser.add_argument("--output", required=True, help="Path to save the styled image")

    args = parser.parse_args()

    processor = ImageProcessor()
    model = StyleTransferModel()
    registry = StyleRegistry()

    # Preprocess the content image
    content_image = processor.preprocess_image(args.content)

    # Get a random style image
    style_image_path = registry.get_random_style_image(args.style_category)
    style_image = processor.preprocess_image(style_image_path)

    # Apply the style
    styled_image = model.apply_style(content_image, style_image)

    # Save the output
    processor.save_image(styled_image, args.output)
    print(f"Styled image saved to: {args.output}")

if __name__ == "__main__":
    main()
