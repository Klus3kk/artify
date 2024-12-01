from PIL import Image 

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path, size=512):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((size, size))
        return image 
    
    @staticmethod
    def save_image(image, output_path):
        image.save(output_path)

        