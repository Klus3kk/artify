from PIL import Image 

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path, size=512):
        '''Resize and normalize image for input.'''
        image = Image.open(image_path).convert("RGB")
        image = image.resize((size, size))
        return image 
    
    @staticmethod
    def save_image(image, output_path):
        '''Save the processed image.'''
        image.save(output_path)

        