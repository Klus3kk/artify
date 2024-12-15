import argparse
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor

def main():
    parser = argparse.ArgumentParser(description="Artify: Apply artistic styles to images.")
    parser.add_argument("--content", required=True,help="Path to content image")
    parser.add_argument("--style", required=True, help="Path to style image")
    parser.add_argument("--output", required=True, help="Path to save the styled image")

    args = parser.parse_args()

    processor = ImageProcessor()
    model = StyleTransferModel()

    content_image = processor.preprocess_image(args.content)
    style_image = processor.preprocess_image(args.style)

    styled_image = model.apply_style(content_image, style_image)
    processor.save_image(styled_image, args.output)

    print(f"Styled image saved to {args.output}")

if __name__ == "__main__":
    main()