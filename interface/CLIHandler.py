import argparse
import os
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from utilities.Logger import Logger
import logging

# Set up logger
logger = Logger.setup_logger(log_file="cli.log", log_level=logging.INFO)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Artify: Apply artistic styles to images.")
    parser.add_argument("--content", required=True, help="Path to content image")
    parser.add_argument("--style_category", required=True, help="Style category (e.g., impressionism, abstract)")
    parser.add_argument("--output", required=True, help="Path to save the styled image")
    args = parser.parse_args()

    logger.info("[1/5] Validating input arguments...")
    if not os.path.exists(args.content):
        logger.error(f"Content image '{args.content}' does not exist.")
        return
    if not args.content.lower().endswith(('.jpg', '.png')):
        logger.error(f"Content image '{args.content}' must be a .jpg or .png file.")
        return

    # Initialize components
    logger.info("[2/5] Initializing components...")
    processor = ImageProcessor()
    model = StyleTransferModel()
    registry = StyleRegistry()

    if args.style_category not in registry.styles:
        logger.error(f"Style category '{args.style_category}' not found.")
        logger.info(f"Available categories: {', '.join(registry.styles.keys())}")
        return

    # Preprocess the content image
    try:
        logger.info("[3/5] Preprocessing content image...")
        content_image = processor.preprocess_image(args.content)
    except Exception as e:
        logger.error(f"Error preprocessing content image: {e}")
        return

    # Fetch and preprocess the style image
    try:
        logger.info("[4/5] Loading and preprocessing style image...")
        style_image_path = registry.get_random_style_image(args.style_category)
        style_image = processor.preprocess_image(style_image_path)
    except Exception as e:
        logger.error(f"Error loading style image: {e}")
        return

    # Apply the style
    try:
        logger.info("[5/5] Applying style transfer... This may take a few moments.")
        styled_image = model.apply_style(content_image, style_image)
        logger.info("Style transfer complete!")
    except Exception as e:
        logger.error(f"Error applying style: {e}")
        return

    # Save the output image
    try:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        processor.save_image(styled_image, args.output)
        logger.info(f"Styled image saved to: {args.output}")
    except Exception as e:
        logger.error(f"Error saving styled image: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
