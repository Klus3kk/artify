from PIL import Image, ImageOps
from pathlib import Path
import logging

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path_or_data, size=(512, 512), fill=(0, 0, 0)):
        """
        Preprocess an individual image by resizing while maintaining aspect ratio and padding to a square.
        :param image_path_or_data: Path to the image file or in-memory image data.
        :param size: Target size for the square output (default: 512x512).
        :param fill: Padding color as a tuple (default: black).
        :return: Processed PIL Image object.
        """
        if not isinstance(size, (tuple, list)) or len(size) != 2 or not all(isinstance(dim, int) for dim in size):
            raise ValueError("Size must be a tuple of two integers")
        
        try:
            if isinstance(image_path_or_data, (str, bytes)):
                img = Image.open(image_path_or_data).convert("RGB")
            elif isinstance(image_path_or_data, Image.Image):
                img = image_path_or_data
            else:
                raise ValueError("Invalid input type for image")
        except Exception as e:
            logging.error(f"Failed to open image: {e}")
            raise

        img.thumbnail(size, Image.Resampling.LANCZOS)

        delta_width = size[0] - img.size[0]
        delta_height = size[1] - img.size[1]
        padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)
        padded_img = ImageOps.expand(img, padding, fill=fill)

        return padded_img

    @staticmethod
    def preprocess_images(input_dir, output_dir, size=(512, 512), fill=(0, 0, 0)):
        """
        Batch preprocess images in a directory.
        :param input_dir: Directory containing raw images.
        :param output_dir: Directory to save preprocessed images.
        :param size: Target size for the square output (default: 512x512).
        :param fill: Padding color as a tuple (default: black).
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
        if not any(input_dir.iterdir()):
            logging.warning(f"Input directory {input_dir} is empty.")
            return

        for file_name in input_dir.iterdir():
            if file_name.suffix.lower() in (".jpg", ".png"):
                try:
                    img = ImageProcessor.preprocess_image(file_name, size, fill)
                    output_path = output_dir / f"{file_name.stem}.jpg"
                    img.save(output_path, "JPEG")
                    logging.info(f"Processed: {output_path}")
                except Exception as e:
                    logging.error(f"Failed to process {file_name}: {e}")
