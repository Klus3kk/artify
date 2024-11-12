import torch
from torchvision import transforms
from PIL import Image

# Load model (
def load_model(style_path):
    model = torch.load(style_path)  
    model.eval()
    return model

def apply_style(model, image_path):
    # Open image and prepare it for model input
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_image = transform(image).unsqueeze(0)  

    # Apply style transfer
    with torch.no_grad():
        output = model(input_image)

    # Convert output back to PIL image
    output_image = transforms.ToPILImage()(output.squeeze(0))
    return output_image