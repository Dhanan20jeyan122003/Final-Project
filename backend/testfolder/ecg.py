import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ECGPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ECGModel().to(self.device)
        
        # Load model weights
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = OrderedDict({k.replace("module.", ""): v for k, v in state_dict.items()})
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128)) / 255.0
        return torch.tensor(image[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    
    def predict(self, image_path):
        ecg_tensor = self.preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            ecg_score = max(0, self.model(ecg_tensor).item())
        return ecg_score