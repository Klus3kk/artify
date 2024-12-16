from PIL import Image 
from io import BytesIO

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path_or_data, size=512):
        if isinstance(image_path_or_data, (str, bytes)):
            image = Image.open(image_path_or_data).convert("RGB")
        elif isinstance(image_path_or_data, BytesIO):
            image = Image.open(image_path_or_data).convert("RGB")
        else:
            raise ValueError("Invalid input type for image")
        image = image.resize((size, size))
        return image

    
    @staticmethod
    def save_image(image, output_path):
        '''Save the processed image.'''
        image.save(output_path)

        