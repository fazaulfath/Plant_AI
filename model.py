import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

from torchvision.models import ResNet34_Weights

class Plant_Disease_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        out = self.network(xb)
        return out

transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.ToTensor()
])

num_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
               'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
               'Tomato___healthy']

model = Plant_Disease_Model()

# Function to download the model if not already present
def download_model(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")

model_file_path = './Models/plantDisease-resnet34.pth'
model_url = 'https://github.com/fazaulfath/Plant_AI/releases/download/v1.0/plantDisease-resnet34.pth'

# Check if the model file exists; if not, download it
if not os.path.exists(model_file_path):
    download_model(model_url, model_file_path)

try:
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def predict_image(img):
    try:
        img_pil = Image.open(io.BytesIO(img)).convert("RGB")  # Ensure image is RGB
    except Exception as e:
        print(f"Error opening image: {e}")
        return "Error: Unable to process the image."
    
    tensor = transform(img_pil)
    xb = tensor.unsqueeze(0)
    
    with torch.no_grad():  # Disable gradients for inference
        yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return num_classes[preds[0].item()]
