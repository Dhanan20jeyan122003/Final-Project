import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision
from collections import OrderedDict

class R3D_EF_Predictor(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(R3D_EF_Predictor, self).__init__()
        self.backbone = torchvision.models.video.r3d_18(weights='DEFAULT')
        self.backbone.stem[0] = nn.Conv3d(1, 64, kernel_size=(3,7,7),
                                          stride=(1,2,2), padding=(1,3,3), bias=False)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x

class EchoPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = R3D_EF_Predictor().to(self.device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')  # remove 'module.' prefix
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
    
    def preprocess_video(self, video_path, num_frames=64, img_size=112):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        cap.release()
        
        if len(frames) < num_frames:
            padding = [np.zeros((img_size, img_size), dtype=np.uint8)] * (num_frames - len(frames))
            frames += padding
        else:
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        video_tensor = torch.tensor(np.stack(frames), dtype=torch.float32)
        video_tensor = video_tensor.unsqueeze(0)  # [1, T, H, W]
        video_tensor = video_tensor / 255.0
        mean, std = 0.1728, 0.2047
        video_tensor = (video_tensor - mean) / std
        return video_tensor.unsqueeze(0).to(self.device)  # [1, 1, T, H, W]
    
    def predict(self, video_path):
        try:
            # Preprocess input video
            input_tensor = self.preprocess_video(video_path)
            
            with torch.no_grad():
                raw_score = self.model(input_tensor).item()
                # Normalize to [0, 1] range using sigmoid
                normalized_score = 1.0 / (1.0 + np.exp(-raw_score))
                
            return normalized_score
        except Exception as e:
            print(f"Error processing echo video: {e}")
            return 0.5  # Default fallback value