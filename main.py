from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import io
import os
import requests
from pathlib import Path

app = FastAPI()

# Add CORS middleware with more permissive configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model architecture
class TomatoLeafNet(nn.Module):
    def __init__(self, num_classes=4):
        super(TomatoLeafNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Define the class names
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to download model
def download_model(url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading model from {url}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully")

# Initialize and load the model
model = TomatoLeafNet(num_classes=4)
MODEL_URL = os.getenv('MODEL_URL')  # You'll set this in Vercel environment variables
MODEL_PATH = "model/tomato_model_state.pth"

@app.on_event("startup")
async def startup_event():
    global model
    if MODEL_URL:
        download_model(MODEL_URL, MODEL_PATH)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    else:
        raise ValueError("MODEL_URL environment variable is not set")

def read_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)

@app.get("/")
async def root():
    return {"message": "Welcome to Tomato Leaf Disease Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image
    image_data = await file.read()
    img_tensor = read_image(image_data)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = float(probabilities[predicted_class_idx])
    
    return {
        "class": CLASS_NAMES[predicted_class_idx],
        "confidence": confidence,
        "predictions": {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, probabilities.tolist())
        }
    } 