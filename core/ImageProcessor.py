from PIL import Image, ImageOps
import os

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path_or_data, size=(512, 512)):
        """
        Preprocess an individual image by resizing while maintaining aspect ratio and padding to a square.
        :param image_path_or_data: Path to the image file or in-memory image data.
        :param size: Target size for the square output (default: 512x512).
        :return: Processed PIL Image object.
        """
        if isinstance(image_path_or_data, (str, bytes)):
            img = Image.open(image_path_or_data).convert("RGB")
        elif isinstance(image_path_or_data, Image.Image):
            img = image_path_or_data
        else:
            raise ValueError("Invalid input type for image")
        
        img.thumbnail(size, Image.ANTIALIAS)

        # Add padding to make the image square
        delta_width = size[0] - img.size[0]
        delta_height = size[1] - img.size[1]
        padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)
        padded_img = ImageOps.expand(img, padding, fill=(0, 0, 0))

        return padded_img

    @staticmethod
    def preprocess_images(input_dir, output_dir, size=(512, 512)):
        """
        Batch preprocess images in a directory.
        :param input_dir: Directory containing raw images.
        :param output_dir: Directory to save preprocessed images.
        :param size: Target size for the square output (default: 512x512).
        """
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(input_dir):
            if file_name.endswith(('.jpg', '.png')):
                img_path = os.path.join(input_dir, file_name)
                img = ImageProcessor.preprocess_image(img_path, size)

                # Save as JPEG
                output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".jpg")
                img.save(output_path, "JPEG")
                print(f"Processed: {output_path}")
