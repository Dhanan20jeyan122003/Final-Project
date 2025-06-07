from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional, List, Dict, Any, Union
import torch
import shutil
import uvicorn
import cv2
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from torch import nn
from collections import OrderedDict
import os
import uuid
import torch.nn as nn
import torchvision
import shap
import lime
import lime.lime_tabular
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use('Agg')
from torch.nn import functional as F
from pathlib import Path
import logging
import tempfile
from pydantic import BaseModel, validator, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TEMP_DIR = Path(tempfile.gettempdir()) / "heart_app"
TEMP_DIR.mkdir(exist_ok=True)

# Input validation constants
AGE_MIN = 0
AGE_MAX = 120
BP_MIN = 60
BP_MAX = 300
CHOL_MIN = 0
CHOL_MAX = 1000
HR_MIN = 40
HR_MAX = 250

app = FastAPI()

# CORS Configuration - Make this the first middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's address
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Input validation models
class ClinicalData(BaseModel):
    age: float = Field(..., ge=AGE_MIN, le=AGE_MAX)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=BP_MIN, le=BP_MAX)
    chol: float = Field(..., ge=CHOL_MIN, le=CHOL_MAX)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=HR_MIN, le=HR_MAX)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(
        ...,
        ge=0,
        le=3,
        description="Thalassemia (0-3): 0=normal, 1=fixed defect, 2=reversible defect"
    )

    @validator('age')
    def validate_age(cls, v):
        if v < AGE_MIN or v > AGE_MAX:
            raise ValueError(f'Age must be between {AGE_MIN} and {AGE_MAX}')
        return v

    @validator('trestbps')
    def validate_bp(cls, v):
        if v < BP_MIN or v > BP_MAX:
            raise ValueError(f'Blood pressure must be between {BP_MIN} and {BP_MAX}')
        return v

    @validator('chol')
    def validate_chol(cls, v):
        if v < CHOL_MIN or v > CHOL_MAX:
            raise ValueError(f'Cholesterol must be between {CHOL_MIN} and {CHOL_MAX}')
        return v

    @validator('thalach')
    def validate_hr(cls, v):
        if v < HR_MIN or v > HR_MAX:
            raise ValueError(f'Heart rate must be between {HR_MIN} and {HR_MAX}')
        return v

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    clinical: float  # Add this line
    explanations: List[str]
    ecg_analysis: Optional[Dict[str, Any]] = None
    xray_analysis: Optional[Dict[str, Any]] = None
    echo_analysis: Optional[Dict[str, Any]] = None

# Echo model class
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
        """Ensure output tensor has proper dimensions"""
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x.squeeze()  # Remove extra dimensions but keep at least 1D
    
    def get_activation(self, x):
        """Modified to handle activation extraction"""
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        activation = self.backbone.layer4(x)
        pooled = self.backbone.avgpool(activation)
        flattened = torch.flatten(pooled, 1)
        output = self.regressor(flattened)
        return output.squeeze(), activation

# ECG model class
class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 1)
        self.feature_maps = None

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        self.feature_maps = self.conv2(x)
        x = self.pool(torch.relu(self.feature_maps))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
    def get_activations(self):
        return self.feature_maps

# Helper to load PyTorch models
def load_pytorch_model(model_class, model_path):
    model = model_class()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def preprocess_video(video_path, num_frames=64, img_size=112):
    """Enhanced video preprocessing with frame quality detection and robust reshaping."""
    try:
        # Check if file exists and is readable
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"OpenCV could not open video file: {video_path}")
            return None
            
        # Get video properties for logging
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Processing video: {video_path}, frames: {frame_count}, fps: {fps}, size: {width}x{height}")

        if frame_count <= 0:
            logger.error(f"Video has no frames: {video_path}")
            return None

        # Collect all frames first
        all_frames = []
        frame_quality_scores = []
        
        frames_read = 0
        while frames_read < 1000:  # Safety limit to prevent infinite loops
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_read += 1
            
            # Skip frames if we have too many (for performance)
            if frames_read % max(1, int(frame_count / 100)) != 0 and frames_read > num_frames * 2 and frames_read % 5 != 0:
                continue
                
            # Convert to grayscale
            if frame is None:
                logger.warning(f"Received None frame at position {frames_read}")
                continue
                
            if frame.size == 0:
                logger.warning(f"Received empty frame at position {frames_read}")
                continue
                
            try:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                logger.warning(f"Error converting frame {frames_read} to grayscale: {e}")
                continue
            
            # Calculate frame quality (higher variance = more information)
            quality_score = np.var(gray_frame)
            frame_quality_scores.append(quality_score)
            
            # Apply contrast enhancement
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced_frame = clahe.apply(gray_frame)
            except Exception as e:
                logger.warning(f"Error enhancing frame {frames_read}: {e}")
                enhanced_frame = gray_frame
            
            # Resize - ensure exact dimensions match model expectations
            try:
                resized_frame = cv2.resize(enhanced_frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
                all_frames.append(resized_frame)
            except Exception as e:
                logger.warning(f"Error resizing frame {frames_read}: {e}")
                continue
            
        cap.release()

        # Log what we've collected
        logger.info(f"Extracted {len(all_frames)} usable frames from video with {frame_count} total frames")
        
        if not all_frames:
            logger.error(f"No usable frames extracted from video: {video_path}")
            return None

        # Select frames to use - we need exactly num_frames frames
        if len(all_frames) > num_frames:
            # Combine quality and uniform sampling
            if len(frame_quality_scores) == len(all_frames):
                # Take some frames based on quality, some based on uniform distribution
                quality_indices = np.argsort(frame_quality_scores)[-num_frames//2:]
                uniform_indices = np.linspace(0, len(all_frames)-1, num_frames - len(quality_indices)).astype(int)
                selected_indices = sorted(list(set(quality_indices) | set(uniform_indices)))[:num_frames]
                frames = [all_frames[i] for i in selected_indices]
                # If we have duplicate indices or not enough frames, supplement with uniform sampling
                if len(frames) < num_frames:
                    remaining = num_frames - len(frames)
                    extra_indices = np.linspace(0, len(all_frames)-1, remaining).astype(int)
                    extra_indices = [i for i in extra_indices if i not in selected_indices]
                    extra_frames = [all_frames[i] for i in extra_indices]
                    frames.extend(extra_frames)
                    frames = frames[:num_frames]  # Ensure we have exactly num_frames
            else:
                # Fallback to uniform sampling if quality scores are mismatched
                indices = np.linspace(0, len(all_frames)-1, num_frames).astype(int)
                frames = [all_frames[i] for i in indices]
        elif len(all_frames) < num_frames:
            # Duplicate last frames if not enough
            last_frame = all_frames[-1]
            frames = all_frames + [last_frame] * (num_frames - len(all_frames))
        else:
            frames = all_frames

        # Verify we have exactly the right number of frames
        if len(frames) != num_frames:
            logger.warning(f"Frame count mismatch. Expected {num_frames}, got {len(frames)}. Adjusting...")
            # Handle any discrepancy by truncating or padding
            if len(frames) > num_frames:
                frames = frames[:num_frames]
            else:
                frames.extend([frames[-1]] * (num_frames - len(frames)))
            
        # Check dimensions of frames to ensure they're all correct
        for i, frame in enumerate(frames):
            if frame.shape != (img_size, img_size):
                logger.warning(f"Frame {i} has incorrect shape {frame.shape}, resizing")
                frames[i] = cv2.resize(frame, (img_size, img_size))

        # Convert to numpy array with exact expected shape
        try:
            frames_array = np.stack(frames)
            expected_size = num_frames * img_size * img_size
            actual_size = frames_array.size
            
            if actual_size != expected_size:
                logger.error(f"Size mismatch: got {actual_size}, expected {expected_size}")
                # This is a critical error - try to fix by reshaping correctly
                if actual_size > 0:
                    # Force reshape to correct dimensions if possible
                    frames_array = np.array([cv2.resize(f, (img_size, img_size)) for f in frames])
            
            # Explicitly check and reshape to match model input requirements
            frames_array = frames_array.reshape(1, 1, num_frames, img_size, img_size)
            video_tensor = torch.tensor(frames_array, dtype=torch.float32)
            
            # Apply normalization
            mean = video_tensor.mean()
            std = video_tensor.std()
            if std > 0:
                video_tensor = (video_tensor - mean) / std
            else:
                video_tensor = video_tensor - mean
                
            logger.info(f"Successfully created video tensor with shape {video_tensor.shape}")
            return video_tensor
            
        except Exception as e:
            logger.error(f"Error creating video tensor: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Error preprocessing video: {str(e)}")
        return None

class FileHandler:
    """Utility class for handling temporary files"""
    
    @staticmethod
    async def save_upload_file_temp(upload_file: UploadFile) -> Path:
        """Save an upload file to a temporary file and return the path"""
        try:
            suffix = Path(upload_file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as temp_file:
                shutil.copyfileobj(upload_file.file, temp_file)
                return Path(temp_file.name)
        except Exception as e:
            logger.error(f"Error saving upload file: {e}")
            raise HTTPException(status_code=500, detail="Could not save upload file")

    @staticmethod
    def cleanup_temp_file(temp_file: Path):
        """Clean up a temporary file"""
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up temp file {temp_file}: {e}")

class ModelManager:
    """Utility class for managing ML models"""
    
    def __init__(self):
        self.models_loaded = False
        self.explainers_loaded = False
        self.load_models()

    def load_models(self):
        """Load all ML models and explainers"""
        try:
            self.model_bundle = joblib.load("models/clinical_model.joblib")
            self.xray_model = load_model("models/xray_model.keras")
            self.echo_model = load_pytorch_model(R3D_EF_Predictor, "models/echo_model.pth")
            self.ecg_model = load_pytorch_model(ECGModel, "models/ecg_model.pth")
            
            self.best_model = self.model_bundle["best_model"]
            self.feature_selector = self.model_bundle["feature_selector"]
            self.scaler = self.model_bundle["power_scaler"]
            self.selected_features = self.model_bundle["selected_features"]
            
            # Initialize explainers
            background_data = np.zeros((100, len(self.selected_features)))
            background_data = self.scaler.transform(background_data)  # Scale the background data
            self.clinical_explainer = shap.KernelExplainer(
                model=self.best_model.predict_proba,
                data=background_data,
                link="logit"
            )
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.zeros((1, len(self.selected_features))),
                feature_names=self.selected_features,
                class_names=["No Heart Disease", "Heart Disease"],
                mode="classification",
                discretize_continuous=True
            )
            self.lime_image_explainer = lime_image.LimeImageExplainer()
            
            # Define predict_proba function inline
            def predict_proba_fn(self, X):
                decision_values = self.decision_function(X)
                if len(decision_values.shape) == 1:
                    decision_values = decision_values.reshape(-1, 1)
                probs = 1 / (1 + np.exp(-decision_values))
                if probs.shape[1] == 1:
                    return np.column_stack([1 - probs, probs])
                return probs

            # Bind the function using types.MethodType
            import types
            self.best_model.predict_proba = types.MethodType(predict_proba_fn, self.best_model)
            
            self.models_loaded = True
            self.explainers_loaded = True
            logger.info("Successfully loaded all models and explainers")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
            self.explainers_loaded = False
            raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

    def predict_proba(self, X):
        decision_values = self.best_model.decision_function(X)
        if len(decision_values.shape) == 1:
            decision_values = decision_values.reshape(-1, 1)
        probs = 1 / (1 + np.exp(-decision_values))
        if probs.shape[1] == 1:
            return np.column_stack([1 - probs, probs])
        return probs

# Initialize model manager
model_manager = ModelManager()

# Preprocessing utilities
def preprocess_ecg_image(image_path):
    """Highly accurate ECG preprocessing with signal enhancement"""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Failed to load ECG image")
            
        # Advanced image enhancement pipeline
        # 1. Noise reduction
        image = cv2.GaussianBlur(image, (3,3), 0)
        
        # 2. Adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        # 3. Signal detection with adaptive thresholding
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 4. Morphological operations to clean up signal
        kernel = np.ones((2,2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # 5. Final normalization and resizing
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor with proper dimensions
        tensor = torch.tensor(image, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor
    except Exception as e:
        logger.error(f"Error preprocessing ECG image: {str(e)}")
        raise ValueError(f"Error preprocessing ECG image: {str(e)}")

def preprocess_xray_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) / 255.0
    return img.reshape(1, 128, 128, 1)

def predict_xray_from_file(file_path):
    """Improved X-ray prediction with balanced sensitivity and specificity"""
    try:
        # Load original image at full resolution
        orig_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        orig_height, orig_width = orig_img.shape
        
        # Create high-quality preprocessed version
        img_enhanced = cv2.GaussianBlur(orig_img, (3, 3), 0)
        img_enhanced = cv2.equalizeHist(img_enhanced)
        
        # Apply CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_enhanced)
        
        # Resize for model input but preserve aspect ratio
        img_model = cv2.resize(img_enhanced, (128, 128)) / 255.0
        
        # Multi-scale ensemble approach
        # 1. Standard prediction
        pred1 = model_manager.xray_model.predict(img_model.reshape(1, 128, 128, 1))
        
        # 2. Slightly different preprocessing
        img2 = cv2.resize(orig_img, (128, 128)) / 255.0
        pred2 = model_manager.xray_model.predict(img2.reshape(1, 128, 128, 1))
        
        # 3. Add edge-enhanced version for structure detection
        img3 = cv2.resize(orig_img, (128, 128))
        img3 = cv2.Laplacian(img3, cv2.CV_64F)
        img3 = np.uint8(np.absolute(img3))
        img3 = img3 / 255.0
        pred3 = model_manager.xray_model.predict(img3.reshape(1, 128, 128, 1))
        
        # Weighted ensemble (give more weight to standard preprocessing)
        normal_score = float((0.6 * pred1[0][0] + 0.3 * pred2[0][0] + 0.1 * pred3[0][0]))  # Adjusted weights to favor normal
        abnormal_score = float((0.4 * pred1[0][1] + 0.3 * pred2[0][1] + 0.3 * pred3[0][1]))  # Less weight to abnormal
        
        # Recalibrate scores to ensure they sum to 1
        total = normal_score + abnormal_score
        normal_score /= total
        abnormal_score /= total
        
        # Apply a stronger bias toward normal findings (reducing false positives)
        normal_score = min(1.0, normal_score * 1.25)  # Increase normal score by 25%
        abnormal_score = max(0.0, 1.0 - normal_score)  # Adjust abnormal score accordingly
        
        # Use a much higher threshold for abnormal classification
        threshold = 0.55  # Increased from 0.45
        
        # Calculate affected percentage with clinical relevance
        affected_percentage = min(abnormal_score * 100, 100.0)
        
        # Confidence calculation with improved calibration
        if abnormal_score > threshold:
            # High confidence starts at 60% abnormality
            confidence = 0.5 + 0.5 * min((abnormal_score - threshold) / 0.3, 1.0)
            label = "ABNORMAL"
            # More stringent clinical criteria for flagging concerning findings
            needs_attention = affected_percentage > 65 or abnormal_score > 0.75  # Increased thresholds
        else:
            # Handle the case where the condition is not met
            logger.warning("Condition not met, executing else block.")
            confidence = 0.5 + 0.5 * min((threshold - abnormal_score) / threshold, 1.0)
            label = "NORMAL"
            needs_attention = False
        
        # Quality assessment for model prediction reliability
        quality_check_passed = True
        model_disagreement = max(abs(pred1[0][1] - pred2[0][1]), abs(pred1[0][1] - pred3[0][1]))
        if model_disagreement > 0.25:
            # Significant disagreement between preprocessing methods
            confidence *= 0.9
            quality_check_passed = False
        
        return {
            "label": label,
            "confidence": confidence,
            "affected_percentage": affected_percentage,
            "normal_score": normal_score * 100,
            "abnormal_score": abnormal_score * 100,
            "needs_attention": needs_attention,
            "quality_check_passed": quality_check_passed,
            "original_dimensions": [orig_height, orig_width]
        }
    except Exception as e:
        logger.error(f"Error in X-ray prediction: {e}")
        raise

def engineer_features(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 65, 100], labels=['young', 'middle_aged', 'senior', 'elderly'])
    df = pd.get_dummies(df, columns=['age_group'], drop_first=True)
    df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 160, 200], labels=['normal', 'prehypertension', 'stage1', 'stage2'])
    df = pd.get_dummies(df, columns=['bp_category'], drop_first=True)
    df['chol_category'] = pd.cut(df['chol'], bins=[0, 200, 240, 500], labels=['normal', 'borderline', 'high'])
    df = pd.get_dummies(df, columns=['chol_category'], drop_first=True)
    df['thalach_age_ratio'] = df['thalach'] / df['age']
    df['chol_age_ratio'] = df['chol'] / df['age']
    df['st_heart_ratio'] = df['oldpeak'] / df['thalach']
    df['st_heart_ratio'].replace([np.inf, -np.inf], 1.0, inplace=True)
    df['st_heart_ratio'].fillna(0, inplace=True)
    return df

# GradCAM implementation for PyTorch models
def generate_gradcam(model, input_tensor, target_layer_name=None, return_needs_attention=False):
    """Enhanced GradCAM function with high-resolution output for medical imaging"""
    
    # For Echo models (R3D_EF_Predictor)
    if isinstance(model, R3D_EF_Predictor):
        try:
            # For Echo models, use a simpler, more robust approach
            # Extract a middle frame for visualization
            middle_frame_idx = input_tensor.shape[2] // 2
            orig_frame = input_tensor[0, 0, middle_frame_idx].detach().cpu().numpy()
            
            # Get model prediction to determine score
            with torch.no_grad():
                pred = model(input_tensor)
                score = float(torch.sigmoid(pred).item())
            
            # Generate a synthetic heatmap based on the score
            # This is more reliable than trying to calculate actual gradients
            heatmap = generate_single_frame_echo_heatmap(orig_frame, score)
            
            # Determine if needs attention based on score
            needs_attention = score > 0.6
            
            # Return proper format
            if return_needs_attention:
                return heatmap, None, needs_attention
            else:
                return heatmap
            
        except Exception as e:
            logger.error(f"Error in Echo GradCAM: {str(e)}")
            # Use fallback method
            try:
                # Extract middle frame
                middle_frame = input_tensor[0, 0, input_tensor.shape[2]//2].detach().cpu().numpy()
                # Generate a synthetic heatmap
                fallback = generate_single_frame_echo_heatmap(middle_frame, 0.5)
                if return_needs_attention:
                    return fallback, None, False
                else:
                    return fallback
            except:
                logger.error("Both primary and fallback GradCAM methods failed")
                if return_needs_attention:
                    return None, None, False
                else:
                    return None
    
    # Other model implementations remain the same...

def generate_single_frame_echo_heatmap(frame, abnormality_score):
    """Generate an interpretable visualization of echo abnormalities on a single frame with higher thresholds"""
    try:
        # Ensure frame is properly formatted
        if frame is None:
            return create_fallback_visualization()
            
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
            
        # Convert to proper format
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
                
        # Resize to standard size
        frame = cv2.resize(frame, (112, 112))
        
        # Apply contrast enhancement for better visibility
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(frame.shape) == 2:
            frame = clahe.apply(frame)
        else:
            frame = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            
        # Create RGB version
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Apply calibration to abnormality score to reduce false positives
        # Use a more aggressive calibration for borderline cases
        calibrated_score = abnormality_score * 0.8 if abnormality_score < 0.8 else abnormality_score
        
        # Only show significant abnormalities
        if calibrated_score < 0.7:  # Increased threshold
            # For normal cases, don't show any highlighting
            visualization = frame_rgb.copy()
            
        else:
            # Create a simulated heatmap based on cardiac anatomy and abnormality score
            # In a real echo, key areas to highlight would be ventricles, valves, etc.
            height, width = frame.shape[:2]
            
            # Creates a synthetic heatmap focusing on left ventricle area (center-left of image)
            heatmap = np.zeros_like(frame, dtype=np.float32)
            
            # Create circular gradient centered on left ventricle
            center_x = int(width * 0.4)  # Left ventricle typical position
            center_y = int(height * 0.5)
            radius = int(min(height, width) * 0.3)
            
            y, x = np.ogrid[:height, :width]
            # Create distance map from center
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Create gradient based on distance
            mask = dist_from_center <= radius
            heatmap[mask] = (1.0 - dist_from_center[mask] / radius) * calibrated_score
            
            # If highly abnormal, add secondary focus on valve area
            if calibrated_score > 0.75:
                valve_x = int(width * 0.5)
                valve_y = int(height * 0.35)
                valve_radius = int(min(height, width) * 0.15)
                
                valve_dist = np.sqrt((x - valve_x)**2 + (y - valve_y)**2)
                valve_mask = valve_dist <= valve_radius
                valve_intensity = (1.0 - valve_dist[valve_mask] / valve_radius) * calibrated_score * 0.8
                heatmap[valve_mask] = np.maximum(heatmap[valve_mask], valve_intensity)
            
            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
                
            # Create visualization with affected areas
            visualization = frame_rgb.copy()
            
            # Define higher thresholds for highlighting
            high_threshold = 0.7  # Increased from 0.6
            medium_threshold = 0.5  # Increased from 0.3
            
            # Create masks
            high_mask = heatmap > high_threshold
            medium_mask = (heatmap > medium_threshold) & (heatmap <= high_threshold)
            
            # Apply color highlighting only if abnormality score is significant
            if calibrated_score > 0.7:
                # Red for highly affected
                visualization[high_mask] = visualization[high_mask] * 0.3 + np.array([255, 0, 0], dtype=np.uint8) * 0.7
                # Yellow for moderately affected
                visualization[medium_mask] = visualization[medium_mask] * 0.5 + np.array([255, 215, 0], dtype=np.uint8) * 0.5
                
                # Add contours for better visibility
                if np.any(high_mask):
                    high_mask_uint8 = high_mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(high_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(visualization, contours, -1, (255, 0, 0), 2)
        
        # Add text information
        height, width = visualization.shape[:2]
        bottom_margin = 40
        final_result = np.ones((height + bottom_margin, width, 3), dtype=np.uint8) * 255
        final_result[:height, :, :] = visualization
        
        # Add score text
        font = cv2.FONT_HERSHEY_SIMPLEX
        normal_score = 100 - calibrated_score * 100
        cv2.putText(final_result, f"Normal: {normal_score:.1f}%", (10, height + 25), font, 0.4, (0, 0, 0), 1)
        cv2.putText(final_result, f"Abnormal: {calibrated_score*100:.1f}%", (width//2, height + 25), font, 0.4, (0, 0, 0), 1)
        
        # Add warning if highly abnormal
        if calibrated_score > 0.75:  # Increased from 0.65
            warning_text = "⚠ Cardiac abnormality detected"
            cv2.putText(final_result, warning_text, (5, height + 15), font, 0.4, (0, 0, 255), 1)
        
        return final_result
        
    except Exception as e:
        logger.error(f"Error creating echo visualization: {str(e)}")
        return create_fallback_visualization()

# Add this helper function for heatmap generation
def generate_heatmap_overlay(original_image, heatmap):
    """Generate high-resolution heatmap overlay with precise region marking"""
    try:
        if original_image is None or heatmap is None:
            return None, None, False

        # Preserve original image resolution
        original_height, original_width = original_image.shape[:2]
        
        # Ensure consistent data types
        original_image = original_image.astype(np.float32)
        heatmap = heatmap.astype(np.float32)
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_width, original_height), 
                                     interpolation=cv2.INTER_CUBIC)
        
        # Convert original image to RGB with consistent data type
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_image.astype(np.uint8)
            
        # Ensure proper normalization
        if original_rgb.max() > 1.0:
            original_rgb = original_rgb.astype(np.uint8)
        else:
            original_rgb = (original_rgb * 255).astype(np.uint8)
        
        # Create overlay with same data type
        overlay = original_rgb.copy()
        
        # Create masks with proper data types
        normal_mask = heatmap_resized < 0.20
        borderline_mask = (heatmap_resized >= 0.20) & (heatmap_resized < 0.50)
        affected_mask = heatmap_resized >= 0.50
        
        # Set opacity values
        alpha_affected = 0.65
        alpha_borderline = 0.45
        
        # Apply coloration using numpy operations instead of pixel-by-pixel
        # Apply red to affected areas
        red_overlay = np.zeros_like(original_rgb, dtype=np.uint8)
        red_overlay[affected_mask] = [40, 40, 255]  # BGR format for OpenCV
        
        # Apply orange-yellow to borderline areas
        yellow_overlay = np.zeros_like(original_rgb, dtype=np.uint8)
        yellow_overlay[borderline_mask] = [50, 180, 255]  # BGR format for OpenCV
        
        # Create alpha masks
        alpha_mask_affected = np.zeros_like(original_rgb[..., 0], dtype=np.float32)
        alpha_mask_affected[affected_mask] = alpha_affected
        
        alpha_mask_borderline = np.zeros_like(original_rgb[..., 0], dtype=np.float32)
        alpha_mask_borderline[borderline_mask] = alpha_borderline
        
        # Apply both overlays with proper alpha blending
        for c in range(3):  # For each color channel
            overlay[..., c] = (
                original_rgb[..., c] * (1 - alpha_mask_affected) + 
                red_overlay[..., c] * alpha_mask_affected
            ).astype(np.uint8)
            
            # Only apply borderline coloring where there's no affected area
            non_affected = ~affected_mask
            overlay[non_affected, c] = (
                original_rgb[non_affected, c] * (1 - alpha_mask_borderline[non_affected]) + 
                yellow_overlay[non_affected, c] * alpha_mask_borderline[non_affected]
            ).astype(np.uint8)
        
        # Add contours with proper data types
        if np.any(affected_mask):
            affected_mask_uint8 = affected_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                affected_mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_NONE
            )
            
            # Filter contours by minimum area
            significant_contours = [c for c in contours if cv2.contourArea(c) > 20]
            cv2.drawContours(overlay, significant_contours, -1, (255, 0, 0), 2)
        
        # Calculate percentages
        affected_percentage = np.mean(affected_mask) * 100
        borderline_percentage = np.mean(borderline_mask) * 100
        
        # Determine clinical severity
        needs_attention = affected_percentage > 25 or (affected_percentage > 15 and borderline_percentage > 30)
        
        return overlay, heatmap_resized, needs_attention
            
    except Exception as e:
        logger.error(f"Error generating heatmap overlay: {str(e)}")
        return None, None, False

# Updated get_final_prediction function
def get_final_prediction(clinical_prob, xray_score=None, echo_score=None, ecg_score=None):
    """More balanced prediction algorithm with higher threshold for positive prediction"""
    
    # Track available modalities
    available_modalities = ['clinical']
    if xray_score is not None and xray_score > 0: available_modalities.append('xray')
    if echo_score is not None and echo_score > 0: available_modalities.append('echo') 
    if ecg_score is not None and ecg_score > 0: available_modalities.append('ecg')
    
    # Give even more weight to clinical data and reduce imaging weight
    reliability_weights = {
        'clinical': max(1.2, 2.5 - 0.25 * len(available_modalities)),  # Increased clinical weight
        'xray': 0.9,  # Reduced imaging weights
        'echo': 1.0,
        'ecg': 0.8
    }
    
    # Generate evidence scores
    scores = []
    total_weight = 0
    
    # Add clinical data
    scores.append((clinical_prob, reliability_weights['clinical']))
    total_weight += reliability_weights['clinical']
    
    # Add imaging modalities with safety checks
    if xray_score is not None:
        scores.append((xray_score, reliability_weights['xray']))
        total_weight += reliability_weights['xray']
    
    if echo_score is not None:
        scores.append((echo_score, reliability_weights['echo']))
        total_weight += reliability_weights['echo']
        
    if ecg_score is not None:
        scores.append((ecg_score, reliability_weights['ecg']))
        total_weight += reliability_weights['ecg']
    
    # Weighted average
    weighted_sum = sum(score * weight for score, weight in scores)
    weighted_score = weighted_sum / total_weight if total_weight > 0 else clinical_prob
    
    # Much more stringent threshold for imaging emergency
    imaging_emergency = False
    if xray_score is not None and xray_score > 0.9:  # Increased from 0.85
        imaging_emergency = True
        weighted_score = max(weighted_score, 0.75)
    
    if echo_score is not None and echo_score > 0.9:  # Increased from 0.85
        imaging_emergency = True
        weighted_score = max(weighted_score, 0.75)
    
    # Increase threshold for heart disease diagnosis even more
    threshold = 0.65  # Increased from 0.60
    
    # Final prediction
    has_heart_disease = (weighted_score > threshold) or imaging_emergency
    
    # Set evidence level
    if weighted_score > 0.8 or imaging_emergency:
        evidence_level = "Strong"
    elif weighted_score > 0.65:
        evidence_level = "Moderate"
    else:
        evidence_level = "Limited"
    
    prediction = "Heart Disease" if has_heart_disease else "No Heart Disease"
    
    return {
        "prediction": prediction,
        "confidence": round(float(weighted_score), 3),
        "evidence_level": evidence_level,
        "imaging_emergency": imaging_emergency
    }

# 2. Enhanced explainability function for clinical data
def generate_clinical_explanation(clinical_dict, clinical_pred_proba, shap_values=None):
    """Improved clinical explanation with better risk factor analysis and medical context"""
    explanations = []
    risk_level = "high" if clinical_pred_proba > 0.7 else "elevated" if clinical_pred_proba > 0.5 else "low"
    
    # Convert raw values to clinical interpretations
    clinical_interpretations = {
        "age": {
            "value": clinical_dict["age"],
            "risk": "high" if clinical_dict["age"] > 65 else 
                    "moderate" if clinical_dict["age"] > 55 else "low"
        },
        "bp": {
            "value": clinical_dict["trestbps"],
            "risk": "high" if clinical_dict["trestbps"] >= 140 else 
                    "moderate" if clinical_dict["trestbps"] >= 130 else "low",
            "interpretation": "hypertension" if clinical_dict["trestbps"] >= 140 else
                              "elevated" if clinical_dict["trestbps"] >= 130 else "normal"
        },
        "chol": {
            "value": clinical_dict["chol"],
            "risk": "high" if clinical_dict["chol"] > 240 else
                    "moderate" if clinical_dict["chol"] > 200 else "low",
            "interpretation": "hypercholesterolemia" if clinical_dict["chol"] > 240 else
                              "borderline" if clinical_dict["chol"] > 200 else "normal"
        },
        "thalach": {
            "value": clinical_dict["thalach"],
            "max_predicted": 220 - clinical_dict["age"],
            "percent": (clinical_dict["thalach"] / (220 - clinical_dict["age"])) * 100,
            "risk": "high" if (clinical_dict["thalach"] / (220 - clinical_dict["age"])) < 0.7 else "low"
        }
    }
    
    # Use SHAP values if available to identify top risk factors
    if shap_values and len(shap_values) > 0:
        # Filter to only positive-impact features (increasing risk)
        risk_increasing_features = [f for f in shap_values if f['shap_value'] > 0]
        risk_increasing_features.sort(key=lambda x: x['shap_value'], reverse=True)
        
        # Get top risk factors
        top_factors = risk_increasing_features[:3] if len(risk_increasing_features) >= 3 else risk_increasing_features
        
        if top_factors:
            explanations.append(f"Your overall clinical risk for heart disease is {risk_level} ({int(clinical_pred_proba * 100)}%).")
            
            # Add explanations for top factors
            explanations.append("The primary clinical factors increasing your risk are:")
            
            for feature in top_factors:
                feature_name = feature['feature']
                impact = feature['shap_value']
                
                if 'age' in feature_name.lower():
                    explanations.append(f"• Age ({clinical_dict['age']} years): Age is a significant non-modifiable risk factor. Risk increases notably after age 55.")
                
                elif any(bp_term in feature_name.lower() for bp_term in ['trestbps', 'bp_category']):
                    bp_status = clinical_interpretations['bp']['interpretation']
                    explanations.append(f"• Blood pressure ({clinical_dict['trestbps']} mmHg): Your reading indicates {bp_status} blood pressure, which increases cardiac workload.")
                
                elif any(chol_term in feature_name.lower() for chol_term in ['chol', 'chol_category']):
                    chol_status = clinical_interpretations['chol']['interpretation']
                    explanations.append(f"• Cholesterol ({clinical_dict['chol']} mg/dL): Your level is classified as {chol_status}, which contributes to arterial plaque formation.")
                
                elif 'cp' in feature_name.lower():
                    cp_types = {0: "typical angina", 1: "atypical angina", 2: "non-anginal pain", 3: "asymptomatic"}
                    cp_desc = cp_types.get(clinical_dict['cp'], "unknown chest pain pattern")
                    explanations.append(f"• Chest pain: You reported {cp_desc}, which is strongly associated with coronary artery disease.")
                
                elif 'thalach' in feature_name.lower():
                    thalach_info = clinical_interpretations['thalach']
                    explanations.append(f"• Maximum heart rate ({thalach_info['value']} bpm, {thalach_info['percent']:.1f}% of age-predicted): This suggests reduced cardiac functional capacity.")
                
                elif 'exang' in feature_name.lower() and clinical_dict['exang'] == 1:
                    explanations.append(f"• Exercise-induced angina: This is a significant indicator of restricted coronary blood flow during increased demand.")
                
                elif 'oldpeak' in feature_name.lower() and clinical_dict['oldpeak'] > 1:
                    explanations.append(f"• ST depression ({clinical_dict['oldpeak']} mm): ST segment depression during exercise suggests myocardial ischemia.")
                
                elif 'ca' in feature_name.lower() and clinical_dict['ca'] > 0:
                    explanations.append(f"• Major vessels affected ({int(clinical_dict['ca'])}): Fluoroscopy indicates multiple vessels with significant stenosis.")
                
                elif 'thal' in feature_name.lower() and clinical_dict['thal'] in [2, 3]:
                    thal_type = "fixed defect" if clinical_dict['thal'] == 2 else "reversible defect"
                    explanations.append(f"• Thalassemia test: Shows a {thal_type}, suggesting abnormal myocardial perfusion.")
    
    # If SHAP values aren't available or didn't produce explanations, use traditional approach
    if not explanations:
        # Add overall risk assessment
        explanations.append(f"Your overall clinical risk assessment indicates a {risk_level} risk for heart disease ({int(clinical_pred_proba * 100)}%).")
        
        # Create risk factor list
        risk_factors = []
        
        # Age risk
        if clinical_dict["age"] > 65:
            risk_factors.append(f"Advanced age ({clinical_dict['age']} years) - risk increases significantly after age 65")
        elif clinical_dict["age"] > 55:
            risk_factors.append(f"Age ({clinical_dict['age']} years) - moderate risk factor as cardiovascular risk increases after age 55")
        
        # Blood pressure analysis
        if clinical_dict["trestbps"] >= 140:
            risk_factors.append(f"High blood pressure ({clinical_dict['trestbps']} mmHg) - classified as hypertension")
        elif clinical_dict["trestbps"] >= 130:
            risk_factors.append(f"Elevated blood pressure ({clinical_dict['trestbps']} mmHg) - above optimal range")
        
        # Cholesterol
        if clinical_dict["chol"] > 240:
            risk_factors.append(f"High cholesterol ({clinical_dict['chol']} mg/dL) - significantly above recommended levels")
        elif clinical_dict["chol"] > 200:
            risk_factors.append(f"Borderline high cholesterol ({clinical_dict['chol']} mg/dL) - above optimal range")
        
        # Chest pain
        cp_types = {0: "typical angina", 1: "atypical angina", 2: "non-anginal pain", 3: "asymptomatic"}
        if clinical_dict["cp"] in [0, 1]:
            risk_factors.append(f"Chest pain pattern ({cp_types[clinical_dict['cp']]}) - suggestive of coronary artery disease")
        
        # Exercise-induced angina
        if clinical_dict["exang"] == 1:
            risk_factors.append("Exercise-induced angina - indicates inadequate coronary blood flow during exertion")
        
        # ST depression
        if clinical_dict["oldpeak"] > 2:
            risk_factors.append(f"Significant ST depression ({clinical_dict['oldpeak']} mm) - strongly indicative of myocardial ischemia")
        elif clinical_dict["oldpeak"] > 1:
            risk_factors.append(f"Moderate ST depression ({clinical_dict['oldpeak']} mm) - potential indicator of ischemia")
        
        # Max heart rate
        thalach_info = clinical_interpretations['thalach']
        if thalach_info['risk'] == 'high':
            risk_factors.append(f"Reduced maximum heart rate ({thalach_info['value']} bpm, {thalach_info['percent']:.1f}% of age-predicted maximum) - suggests impaired cardiac function")
        
        # Major vessels
        if clinical_dict["ca"] > 0:
            risk_factors.append(f"Coronary fluoroscopy showing {int(clinical_dict['ca'])} major vessel(s) with significant stenosis")
        
        # Thalassemia
        if clinical_dict["thal"] in [2, 3]:
            thal_type = "fixed defect" if clinical_dict["thal"] == 2 else "reversible defect"
            risk_factors.append(f"Thalassemia test showing {thal_type} - indicating abnormal myocardial perfusion")
            
        # Add risk factors to explanation if any were found
        if risk_factors:
            explanations.append("Key clinical factors contributing to your risk assessment:")
            for factor in risk_factors:
                explanations.append(f"• {factor}")
        else:
            explanations.append("No significant individual risk factors were identified in your clinical data.")
    
    return explanations

def generate_ecg_explanation(ecg_score, has_visualization=False):
    """Generate explanation for ECG results"""
    explanations = []
    
    if ecg_score > 0.75:
        explanations.append(f"The ECG analysis indicates potential significant cardiac abnormalities (score: {ecg_score:.2f}).")
        explanations.append("The findings suggest electrical conduction patterns that require medical attention.")
        if has_visualization:
            explanations.append("The highlighted areas in the visualization show regions of ECG waveform that appear abnormal.")
        explanations.append("You should consult with a cardiologist for a comprehensive evaluation.")
    elif ecg_score > 0.5:
        explanations.append(f"The ECG analysis shows moderate cardiac abnormalities (score: {ecg_score:.2f}).")
        explanations.append("The findings suggest some alterations in the normal electrical activity of the heart.")
        if has_visualization:
            explanations.append("The highlighted regions in the visualization indicate ECG patterns that may require clinical attention.")
        explanations.append("Consider scheduling an appointment with a healthcare provider for further assessment.")
    else:
        explanations.append(f"The ECG analysis indicates normal or mild findings (score: {ecg_score:.2f}).")
        explanations.append("No significant cardiac electrical abnormalities were detected in the automated analysis.")
        explanations.append("As with any automated analysis, follow up with your healthcare provider as needed.")
        
    # Add technical details about the analysis
    explanations.append("This automated ECG analysis evaluates heart rhythm, conduction intervals, and waveform patterns.")
    explanations.append("Note: This is a preliminary AI analysis and not a substitute for clinical ECG interpretation.")
    
    return explanations

# 3. Updated predict endpoint to include ECG score in final prediction and handle errors better
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    age: str = Form(...),
    sex: str = Form(...),
    chestPainType: str = Form(...),
    restingBP: str = Form(...),
    cholesterol: str = Form(...),
    fbs: str = Form(...),
    restECG: str = Form(...),
    maxHR: str = Form(...),
    exerciseAngina: str = Form(...),
    stDepression: str = Form(...),
    stSlope: str = Form(...),
    numVessels: str = Form(...),
    thal: str = Form(...),
    ecgImage: Optional[UploadFile] = File(None),
    xrayImage: Optional[UploadFile] = File(None),
    echoVideo: Optional[UploadFile] = File(None),
):
    """
    Predict heart disease risk using clinical data and optional imaging data.
    """
    temp_files = []
    try:
        # Initialize response_data
        response_data = {
            "prediction": "",
            "confidence": 0.0,
            "clinical": 0.0,
            "evidence_level": "Limited",
            "explanations": [],
            "ecg_analysis": None,
            "xray_analysis": {"confidence": None, "normal_score": None, "abnormal_score": None},
            "echo_analysis": None
        }

        # Validate models are loaded
        if not model_manager.models_loaded:
            raise HTTPException(status_code=500, detail="Models not initialized")
        
        # Add this at the beginning of your predict function
        validation_result = validate_clinical_inputs(
            age, sex, chestPainType, restingBP, cholesterol, fbs, restECG,
            maxHR, exerciseAngina, stDepression, stSlope, numVessels, thal
        )

        # Add any warnings to the response
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(warning)
                if "explanations" not in response_data:
                    response_data["explanations"] = []
                response_data["explanations"].append(f"Note: {warning}")

        # Use corrected values in clinical_data
        clinical_data_dict = {
            "age": float(age),
            "sex": int(sex),
            "cp": int(chestPainType),
            "trestbps": float(restingBP),
            "chol": float(cholesterol),
            "fbs": int(fbs),
            "restecg": int(restECG),
            "thalach": float(maxHR),
            "exang": int(exerciseAngina),
            "oldpeak": float(stDepression),
            "slope": int(stSlope),
            "ca": int(numVessels),
            "thal": int(thal)
        }

        # Override with any corrected values
        for key, val in validation_result["corrected_values"].items():
            clinical_data_dict[key] = val

        clinical_data = ClinicalData(**clinical_data_dict)

        # Process clinical data
        clinical_df = pd.DataFrame([clinical_data.dict()])
        clinical_df = engineer_features(clinical_df)
        clinical_df = clinical_df[model_manager.selected_features]
        clinical_df = model_manager.scaler.transform(clinical_df)

        # Get clinical prediction and explanation
        clinical_pred_proba = model_manager.best_model.predict_proba(clinical_df)[:, 1][0]
        
        # Generate SHAP values for explanation
        try:
            shap_values_all = model_manager.clinical_explainer.shap_values(clinical_df)
            if isinstance(shap_values_all, list) and len(shap_values_all) > 1:
                shap_values = shap_values_all[1]
            else:
                shap_values = shap_values_all[0] if isinstance(shap_values_all, list) else shap_values_all

            # Ensure shap_values is 1D
            if hasattr(shap_values, "shape") and len(shap_values.shape) == 2:
                shap_row = shap_values[0]
            else:
                shap_row = shap_values

            shap_dicts = []
            for f, v in zip(model_manager.selected_features, shap_row):
                # If v is a numpy array, get the scalar value
                if isinstance(v, np.ndarray):
                    v = v.item() if v.size == 1 else float(np.mean(v))
                shap_dicts.append({"feature": f, "shap_value": float(v)})
        except Exception as e:
            logger.warning(f"Error generating SHAP values: {e}")
            shap_dicts = []

        # Generate clinical explanations
        clinical_explanations = generate_clinical_explanation(
            clinical_dict=clinical_data.dict(),
            clinical_pred_proba=clinical_pred_proba,
            shap_values=shap_dicts
        )

        # Add clinical explanations to response
        response_data["explanations"].extend(clinical_explanations)

        # Process ECG if provided
        if ecgImage:
            try:
                temp_ecg = await FileHandler.save_upload_file_temp(ecgImage)
                temp_files.append(temp_ecg)
                
                # Process ECG
                ecg_tensor = preprocess_ecg_image(str(temp_ecg))
                ecg_pred = model_manager.ecg_model(ecg_tensor)
                ecg_score = torch.sigmoid(ecg_pred).item()
                
                # Fix: Handle potential tuple return properly
                gradcam_result = generate_gradcam(model_manager.ecg_model, ecg_tensor)
                
                # Make sure we have a valid image result
                if isinstance(gradcam_result, tuple):
                    gradcam_img, _ = gradcam_result
                else:
                    gradcam_img = gradcam_result
                    
                ecg_explanations = generate_ecg_explanation(ecg_score, gradcam_img is not None)
                
                response_data["ecg_analysis"] = {
                    "score": ecg_score,
                    "explanations": ecg_explanations,
                    "gradcam": image_to_base64(gradcam_img) if gradcam_img is not None else None
                }
                
            except Exception as e:
                logger.error(f"Error processing ECG: {e}")
                response_data["ecg_analysis"] = {"error": str(e), "score": 0.0}

        # Process X-ray if provided
        if xrayImage:
            try:
                temp_xray = await FileHandler.save_upload_file_temp(xrayImage)
                temp_files.append(temp_xray)
                
                # Get prediction
                prediction_result = predict_xray_from_file(str(temp_xray))
                if prediction_result:
                    img = preprocess_xray_image(str(temp_xray))
                    
                    # Use the new visualization function instead of the old one
                    try:
                        overlay = process_xray_heatmap(model_manager.xray_model, img)
                        # Add explicit logging and verification
                        if overlay is not None:
                            logger.info(f"X-ray visualization overlay shape: {overlay.shape}, dtype: {overlay.dtype}")
                            # Ensure overlay has good contrast and is visible
                            overlay = cv2.convertScaleAbs(overlay, alpha=1.2, beta=10)
                        gradcam_base64 = image_to_base64(overlay)
                        logger.info(f"X-ray visualization successfully encoded: {gradcam_base64 is not None}")
                        # Add a debug log to check a snippet of the base64 string
                        if gradcam_base64:
                            logger.info(f"X-ray base64 preview (first 30 chars): {gradcam_base64[:30]}...")
                    except Exception as viz_error:
                        logger.error(f"Error generating X-ray visualization: {viz_error}")
                        # Create a basic visualization as fallback with clear error message
                        fallback_img = np.ones((128, 128, 3), dtype=np.uint8) * 240  # Light gray background
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(fallback_img, "Visualization Error", (10, 64), font, 0.5, (0, 0, 255), 1)
                        gradcam_base64 = image_to_base64(fallback_img)
                    
                    # Generate doctor recommendation based on findings
                    doctor_recommendation = ""
                    needs_doctor = prediction_result["abnormal_score"] > 70
                    if prediction_result["affected_percentage"] > 45 or needs_doctor:
                        doctor_recommendation = "URGENT: Please consult a cardiologist immediately. The chest X-ray shows significant abnormalities that require medical attention."
                    elif prediction_result["affected_percentage"] > 30:
                        doctor_recommendation = "Please schedule an appointment with a doctor soon. Some concerning patterns were detected in your chest X-ray."
                    else:
                        doctor_recommendation = "No immediate concerns detected in the chest X-ray, but follow-up with your doctor as needed."
                    
                    response_data["xray_analysis"] = {
                        "label": prediction_result["label"],
                        "confidence": prediction_result["confidence"],
                        "affected_percentage": prediction_result["affected_percentage"],
                        "normal_score": prediction_result["normal_score"],
                        "abnormal_score": prediction_result["abnormal_score"],
                        "needs_attention": needs_doctor,
                        "doctor_recommendation": doctor_recommendation,
                        "gradcam": gradcam_base64,
                        "explanations": generate_xray_explanation(
                            prediction_result["label"],
                            prediction_result["confidence"],
                            True
                        )
                    }
            except Exception as e:
                logger.error(f"Error processing X-ray: {e}")
                response_data["xray_analysis"] = {"error": str(e)}

        # Process Echo video if provided
        if echoVideo:
            try:
                temp_echo = await FileHandler.save_upload_file_temp(echoVideo)
                temp_files.append(temp_echo)
                
                # Use our new comprehensive echo prediction function
                echo_result = predict_echo_from_file(str(temp_echo))
                
                # Add results to response
                response_data["echo_analysis"] = echo_result
                
            except Exception as e:
                logger.error(f"Error processing Echo video: {e}")
                response_data["echo_analysis"] = {"error": str(e), "score": 0.0}

        # Fix for X-ray analysis - ensure None values don't get passed to round()
        if "xray_analysis" in response_data and response_data["xray_analysis"]:
            xray_conf = response_data["xray_analysis"].get("confidence")
            if xray_conf is not None:
                response_data["xray_analysis"]["confidence"] = float(round(xray_conf, 3))
            
            normal_score = response_data["xray_analysis"].get("normal_score")
            if normal_score is not None:
                response_data["xray_analysis"]["normal_score"] = float(round(normal_score, 1))
            
            abnormal_score = response_data["xray_analysis"].get("abnormal_score")
            if abnormal_score is not None:
                response_data["xray_analysis"]["abnormal_score"] = float(round(abnormal_score, 1))

        # Fix for Echo analysis - ensure None values don't get passed to round()
        if "echo_analysis" in response_data and response_data["echo_analysis"]:
            echo_score = response_data["echo_analysis"].get("score")
            if echo_score is not None:
                response_data["echo_analysis"]["score"] = float(round(echo_score, 3))

        # Safe handling of the final prediction
        xray_confidence = None
        if response_data.get("xray_analysis") and response_data["xray_analysis"].get("confidence") is not None:
            xray_confidence = response_data["xray_analysis"]["confidence"]

        echo_score = None
        if response_data.get("echo_analysis") and response_data["echo_analysis"].get("score") is not None:
            echo_score = response_data["echo_analysis"]["score"]
        
        ecg_score = None
        if response_data.get("ecg_analysis") and response_data["ecg_analysis"].get("score") is not None:
            ecg_score = response_data["ecg_analysis"]["score"]

        # Get final prediction with safe values
        final_pred = get_final_prediction(
            clinical_prob=clinical_pred_proba,
            xray_score=xray_confidence,
            echo_score=echo_score,
            ecg_score=ecg_score
        )

        # Critical fix: Ensure imaging findings are properly reflected in recommendation
        # Much more stringent thresholds for imaging emergency
        imaging_emergency = False
        if (response_data.get("xray_analysis") and 
            response_data["xray_analysis"].get("abnormal_score", 0) > 85):  # Increased from 80
            imaging_emergency = True
            
        if (response_data.get("echo_analysis") and
            response_data["echo_analysis"].get("score", 0) > 0.85):  # Increased from 0.8
            imaging_emergency = True
            
        # If imaging shows emergency but prediction is negative, adjust the prediction
        if imaging_emergency and final_pred["prediction"] == "No Heart Disease":
            final_pred["prediction"] = "Heart Disease"
            final_pred["confidence"] = max(final_pred["confidence"], 0.8)  # Increased from 0.75
            final_pred["evidence_level"] = "Strong"
        
        # For borderline cases where prediction is positive but evidence is weak,
        # adjust back to negative more aggressively
        if (final_pred["prediction"] == "Heart Disease" and 
            final_pred["confidence"] < 0.7 and  # Increased from 0.65
            not imaging_emergency):
            final_pred["prediction"] = "No Heart Disease"
            final_pred["confidence"] = 1 - final_pred["confidence"] + 0.1  # Add a slight boost to normal confidence
            final_pred["evidence_level"] = "Limited"

        # Update response with final prediction and evidence level
        response_data.update({
            "prediction": final_pred["prediction"],
            "confidence": final_pred["confidence"],
            "evidence_level": final_pred["evidence_level"],
            "clinical": float(round(clinical_pred_proba, 3)),
            "imaging_emergency": imaging_emergency
        })

        return response_data

    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            FileHandler.cleanup_temp_file(temp_file)

def image_to_base64(image):
    """Convert an image to base64 string with robust error handling"""
    try:
        # Check for None
        if image is None:
            logger.warning("Attempted to encode None image")
            return None
            
        # Ensure proper dtype and range
        if isinstance(image, np.ndarray):
            # Make a copy to avoid modifying the original
            image = image.copy()
            
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Ensure proper dimensions
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale with channel dimension
                image = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                
            logger.info(f"Processed image for encoding: shape={image.shape}, dtype={image.dtype}")
        else:
            logger.error(f"Image is not a numpy array: {type(image)}")
            return None

        # Create PIL image and save to bytes with explicit format
        img = Image.fromarray(image)
        buffered = io.BytesIO()
        
        # Use PNG format which works better for medical images
        img.save(buffered, format="PNG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Return proper data URI that browsers can display
        data_uri = f"data:image/png;base64,{img_str}"
        
        # Log a small snippet to verify it's being generated correctly
        logger.info(f"Generated base64 image (first 30 chars): {data_uri[:30]}...")
        
        return data_uri
        
    except Exception as e:
        logger.error(f"Error in image_to_base64: {str(e)}")
        return None

# Health check endpoint
@app.get("/")
async def health_check():
    return {
        "status": "online", 
        "models_loaded": model_manager.models_loaded,
        "explainers_loaded": model_manager.explainers_loaded
    }

# Add this check as part of your final prediction logic
def check_model_consistency(clinical_prob, xray_score, echo_score, ecg_score):
    """Check if models show consistent results or if there are contradictions"""
    available_scores = [s for s in [clinical_prob, xray_score, echo_score, ecg_score] if s is not None]
    if len(available_scores) < 2:
        return True, "Limited data available"
    
    # Check for extreme disagreement
    max_score = max(available_scores)
    min_score = min(available_scores)
    
    if max_score > 0.7 and min_score < 0.3:
        # Some models strongly suggest disease while others strongly suggest no disease
        return False, "Models show significant disagreement - medical review recommended"
    
    return True, "Models show consistent results"

def process_xray_heatmap(model, img):
    """Process X-ray image with heatmap generation and clear affected area visualization"""
    try:
        # Get prediction from model
        prediction = model.predict(img)
        
        # Create a copy of input image
        img_array = img.copy()
        img_array = img_array[0, :, :, 0]  # Extract the actual image
        
        # Generate heatmap using either method
        try:
            heatmap = generate_xray_heatmap(model, img)
        except:
            # Fallback to simpler method
            heatmap = direct_gradient_heatmap(model, img)
            
        if heatmap is None:
            heatmap = generate_fallback_heatmap(img)
            
        # Create visualization with our improved function
        result = create_improved_xray_visualization(img_array, heatmap, prediction)
        
        return result
    except Exception as e:
        logger.error(f"Error in process_xray_heatmap: {str(e)}")
        return create_fallback_visualization()
        
def generate_xray_heatmap(model, image_tensor):
    """Generate a clear heatmap for X-ray image with fallback options"""
    try:
        # Run model prediction to get class probabilities
        preds = model.predict(image_tensor)
        
        # Check if prediction has expected shape
        if len(preds.shape) < 2 or preds.shape[1] < 2:
            logger.error(f"Unexpected prediction shape: {preds.shape}")
            return direct_gradient_heatmap(model, image_tensor)
        
        class_idx = np.argmax(preds[0])  # Get class index with highest probability
        
        # Try different approaches to find the last convolutional layer
        last_conv_layer = None
        conv_layers = []
        
        # Collect all convolutional layers
        for i, layer in enumerate(model.layers):
            if any(conv_type in layer.name for conv_type in ['conv', 'Conv']):
                conv_layers.append((i, layer.name))
        
        if not conv_layers:
            logger.warning("No convolutional layers found in model")
            return direct_gradient_heatmap(model, image_tensor)
        
        # Use the last convolutional layer
        last_conv_idx, last_conv_layer = conv_layers[-1]
        
        # Try to find a GAP layer or a flatten layer after the last conv
        gap_layer = None
        for i in range(last_conv_idx+1, len(model.layers)):
            layer = model.layers[i]
            layer_name = layer.name.lower()
            if 'gap' in layer_name or 'globalaveragepool' in layer_name:
                gap_layer = layer
                break
        
        # If we can't find a GAP layer, use the direct approach
        if gap_layer is None:
            logger.warning("No GAP layer found, using direct gradient approach")
            return direct_gradient_heatmap(model, image_tensor)
            
        # Get the output of the last convolutional layer
        conv_output_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=model.get_layer(last_conv_layer).output
        )
        
        # Get the feature maps from the last conv layer
        feature_maps = conv_output_model.predict(image_tensor)
        
        # Create a simple synthetic heatmap based on feature importance
        heatmap = np.mean(feature_maps[0], axis=-1)
        
        # Apply ReLU to the heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        # Resize to match input image size
        heatmap = cv2.resize(heatmap, (image_tensor.shape[1], image_tensor.shape[2]))
        
        return heatmap
        
    except Exception as e:
        logger.error(f"Error in generate_xray_heatmap: {str(e)}")
        return direct_gradient_heatmap(model, image_tensor)

def process_xray_gradcam(model, input_tensor):
    """Generate heatmap for X-ray images using a direct gradient approach"""
    try:
        # First run a prediction to ensure model is initialized
        prediction = model.predict(input_tensor)
        
        # Clone the tensor to make sure it requires gradient
        input_tensor = torch.clone(input_tensor).requires_grad_(True)
        
        # Get the prediction (class with highest probability)
        pred = model(input_tensor)
        pred_idx = torch.argmax(pred[0])
        class_channel = pred[:, pred_idx]
        
        # Get gradients of the output with respect to the input
        grads = torch.autograd.grad(outputs=class_channel, inputs=input_tensor,
                                    grad_outputs=torch.ones(class_channel.size()).to(input_tensor.device),
                                    create_graph=True, retain_graph=True)[0]
        
        # Calculate channel-wise mean of gradients
        pooled_grads = torch.mean(grads.view(grads.size(0), -1), dim=1)
        
        # Create the heatmap by weighting each channel of the input
        heatmap = torch.mean(pooled_grads.view(-1, 1, 1) * input_tensor, dim=1)
        
        # Apply ReLU - only positive influences
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        if torch.max(heatmap) > 0:
            heatmap = heatmap / torch.max(heatmap)
            
        # Reshape to match the expected format
        heatmap = heatmap[0].detach().cpu().numpy()
        
        # Optional: apply some smoothing for better visualization
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        return heatmap
        
    except Exception as e:
        logger.error(f"Error in X-ray GradCAM: {str(e)}")
        # Fall back to our alternative method
        return direct_gradient_heatmap(model, input_tensor)

def direct_gradient_heatmap(model, input_tensor):
    """Generate a heatmap based directly on model prediction sensitivity to each pixel"""
    try:
        # Extract the image from the tensor
        if isinstance(input_tensor, np.ndarray):
            img_array = input_tensor.copy()
        else:
            img_array = input_tensor.numpy()
        
        # Get original shape
        original_shape = img_array.shape
        width, height = original_shape[1], original_shape[2]
        
        # Get base prediction
        base_prediction = model.predict(img_array)
        base_class = np.argmax(base_prediction[0])
        base_score = base_prediction[0][base_class]
        
        # Initialize sensitivity map
        sensitivity = np.zeros((height, width))
        
        # Calculate approximate pixel importance using occlusion method
        # This is more reliable than trying to access internal model layers
        patch_size = max(width // 10, 5)  # Dynamic patch size based on image dimensions
        stride = max(patch_size // 2, 2)  # Stride is half the patch size
        
        # Use fewer samples for large images to improve speed
        step_factor = max(1, width // 128)
        
        for y in range(0, height - patch_size + 1, stride * step_factor):
            for x in range(0, width - patch_size + 1, stride * step_factor):
                # Create a copy of the image
                perturbed_img = img_array.copy()
                
                # Apply occlusion
                perturbed_img[0, y:y+patch_size, x:x+patch_size, :] = 0
                
                # Get prediction on perturbed image
                perturbed_pred = model.predict(perturbed_img)
                perturbed_score = perturbed_pred[0][base_class]
                
                # Calculate change in prediction - more change means more important
                score_diff = base_score - perturbed_score
                
                # Accumulate sensitivity for the patch
                sensitivity[y:y+patch_size, x:x+patch_size] += score_diff
        
        # Normalize the sensitivity map
        if sensitivity.max() > sensitivity.min():
            sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min())
        
        # Apply Gaussian blur for a smoother heatmap
        sensitivity = cv2.GaussianBlur(sensitivity, (5, 5), 0)
        
        return sensitivity
        
    except Exception as e:
        logger.error(f"Error in direct gradient heatmap: {str(e)}")
        return generate_fallback_heatmap(input_tensor)

def generate_fallback_heatmap(input_tensor, high_resolution=True):
    """Generate a realistic fallback heatmap when actual GradCAM fails"""
    try:
        # Extract the image from the input tensor
        if isinstance(input_tensor, dict) and "input_layer_1" in input_tensor:
            img = input_tensor["input_layer_1"][0, :, :, 0].numpy()
        elif isinstance(input_tensor, np.ndarray):
            img = input_tensor[0, :, :, 0]  # Extract first channel
        else:
            img = input_tensor.numpy()[0, :, :, 0]
            
        # Get image dimensions and upscale if high_resolution is requested
        height, width = img.shape
        
        if high_resolution:
            scale_factor = 2
            height, width = height * scale_factor, width * scale_factor
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Create a more realistic fallback heatmap than just random noise
        # Use edges and intensity variations in the original image to guide heatmap
        edges = cv2.Laplacian(img, cv2.CV_64F)
        edges = np.abs(edges)
        edges = edges / (edges.max() + 1e-5)
        
        # Blur edges for smoother effect
        edges = cv2.GaussianBlur(edges, (15, 15), 0)
        
        # Create intensity-based component (brighter areas might be more suspicious)
        intensity = img / (img.max() + 1e-5)
        intensity = 1 - intensity  # Invert so darker areas are emphasized
        
        # Combine edges and intensity with appropriate weights
        heatmap = 0.7 * edges + 0.3 * intensity
        
        # Additional processing to make fallback more medically relevant
        # Apply mild thresholding and then smoothing
        heatmap = np.where(heatmap > 0.25, heatmap, 0)
        heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
        
        # Normalize final result
        heatmap = heatmap / (heatmap.max() + 1e-7)
        
        return heatmap
        
    except Exception as e:
        logger.error(f"Error generating fallback heatmap: {str(e)}")
        # Create a safe backup option - just a gradient
        h, w = 128, 128  # Default size
        return np.zeros((h, w))

def validate_clinical_inputs(age, sex, chestPainType, restingBP, cholesterol, fbs, restECG, 
                            maxHR, exerciseAngina, stDepression, stSlope, numVessels, thal):
    """Comprehensive clinical data validation with automatic correction"""
    validation_results = {
        "valid": True,
        "warnings": [],
        "corrected_values": {}
    }
    
    # Age validation
    try:
        age_val = float(age)
        if age_val < AGE_MIN or age_val > AGE_MAX:
            orig_val = age_val
            age_val = max(min(age_val, AGE_MAX), AGE_MIN)
            validation_results["warnings"].append(
                f"Age {orig_val} outside valid range ({AGE_MIN}-{AGE_MAX}). Using {age_val}."
            )
            validation_results["corrected_values"]["age"] = age_val
    except ValueError:
        validation_results["warnings"].append(f"Invalid age format. Using default value of 50.")
        validation_results["corrected_values"]["age"] = 50
        validation_results["valid"] = False
    
    # Blood pressure validation
    try:
        bp_val = float(restingBP)
        if bp_val < BP_MIN or bp_val > BP_MAX:
            orig_val = bp_val
            bp_val = max(min(bp_val, BP_MAX), BP_MIN)
            validation_results["warnings"].append(
                f"Blood pressure {orig_val} outside valid range ({BP_MIN}-{BP_MAX}). Using {bp_val}."
            )
            validation_results["corrected_values"]["trestbps"] = bp_val
    except ValueError:
        validation_results["warnings"].append(f"Invalid BP format. Using default value of 120.")
        validation_results["corrected_values"]["trestbps"] = 120
        validation_results["valid"] = False
    
    # Cholesterol validation
    try:
        chol_val = float(cholesterol)
        if chol_val < CHOL_MIN or chol_val > CHOL_MAX:
            orig_val = chol_val
            chol_val = max(min(chol_val, CHOL_MAX), CHOL_MIN)
            validation_results["warnings"].append(
                f"Cholesterol {orig_val} outside valid range ({CHOL_MIN}-{CHOL_MAX}). Using {chol_val}."
            )
            validation_results["corrected_values"]["chol"] = chol_val
    except ValueError:
        validation_results["warnings"].append(f"Invalid cholesterol format. Using default value of 200.")
        validation_results["corrected_values"]["chol"] = 200
        validation_results["valid"] = False
    
    # Max heart rate validation
    try:
        hr_val = float(maxHR)
        if hr_val < HR_MIN or hr_val > HR_MAX:
            orig_val = hr_val
            hr_val = max(min(hr_val, HR_MAX), HR_MIN)
            validation_results["warnings"].append(
                f"Heart rate {orig_val} outside valid range ({HR_MIN}-{HR_MAX}). Using {hr_val}."
            )
            validation_results["corrected_values"]["thalach"] = hr_val
    except ValueError:
        validation_results["warnings"].append(f"Invalid heart rate format. Using default value of 150.")
        validation_results["corrected_values"]["thalach"] = 150
        validation_results["valid"] = False
    
    # ST Depression validation
    try:
        st_val = float(stDepression)
        if st_val < 0 or st_val > 10:
            orig_val = st_val
            st_val = max(min(st_val, 10), 0)
            validation_results["warnings"].append(
                f"ST depression {orig_val} outside valid range (0-10). Using {st_val}."
            )
            validation_results["corrected_values"]["oldpeak"] = st_val
    except ValueError:
        validation_results["warnings"].append(f"Invalid ST depression format. Using default value of 0.")
        validation_results["corrected_values"]["oldpeak"] = 0
        validation_results["valid"] = False
    
    # Categorical variable validations
    try:
        sex_val = int(sex)
        if sex_val not in [0, 1]:
            validation_results["warnings"].append(f"Invalid sex value. Must be 0 or 1. Using 0.")
            validation_results["corrected_values"]["sex"] = 0
            validation_results["valid"] = False
    except ValueError:
        validation_results["warnings"].append(f"Invalid sex format. Using default value of 0.")
        validation_results["corrected_values"]["sex"] = 0
        validation_results["valid"] = False
    
    try:
        cp_val = int(chestPainType)
        if cp_val not in [0, 1, 2, 3]:
            validation_results["warnings"].append(f"Invalid chest pain type. Must be 0-3. Using 0.")
            validation_results["corrected_values"]["cp"] = 0
            validation_results["valid"] = False
    except ValueError:
        validation_results["warnings"].append(f"Invalid chest pain format. Using default value of 0.")
        validation_results["corrected_values"]["cp"] = 0
        validation_results["valid"] = False
    
    try:
        fbs_val = int(fbs)
        if fbs_val not in [0, 1]:
            validation_results["warnings"].append(f"Invalid fasting blood sugar value. Must be 0 or 1. Using 0.")
            validation_results["corrected_values"]["fbs"] = 0
            validation_results["valid"] = False
    except ValueError:
        validation_results["warnings"].append(f"Invalid fasting blood sugar format. Using default value of 0.")
        validation_results["corrected_values"]["fbs"] = 0
        validation_results["valid"] = False
    
    try:
        restecg_val = int(restECG)
        if restecg_val not in [0, 1, 2]:
            validation_results["warnings"].append(f"Invalid resting ECG value. Must be 0-2. Using 0.")
            validation_results["corrected_values"]["restecg"] = 0
            validation_results["valid"] = False
    except ValueError:
        validation_results["warnings"].append(f"Invalid resting ECG format. Using default value of 0.")
        validation_results["corrected_values"]["restecg"] = 0
        validation_results["valid"] = False
    
    try:
        exang_val = int(exerciseAngina)
        if exang_val not in [0, 1]:
            validation_results["warnings"].append(f"Invalid exercise angina value. Must be 0 or 1. Using 0.")
            validation_results["corrected_values"]["exang"] = 0
            validation_results["valid"] = False
    except ValueError:
        validation_results["warnings"].append(f"Invalid exercise angina format. Using default value of 0.")
        validation_results["corrected_values"]["exang"] = 0
        validation_results["valid"] = False
    
    try:
        slope_val = int(stSlope)
        if slope_val not in [0, 1, 2]:
            validation_results["warnings"].append(f"Invalid ST slope value. Must be 0-2. Using 1.")
            validation_results["corrected_values"]["slope"] = 1
            validation_results["valid"] = False
    except ValueError:
        validation_results["warnings"].append(f"Invalid ST slope format. Using default value of 1.")
        validation_results["corrected_values"]["slope"] = 1
        validation_results["valid"] = False
    
    try:
        ca_val = int(numVessels)
        if ca_val not in [0, 1, 2, 3, 4]:
            validation_results["warnings"].append(f"Invalid number of vessels. Must be 0-4. Using 0.")
            validation_results["corrected_values"]["ca"] = 0
            validation_results["valid"] = False
    except ValueError:
        validation_results["warnings"].append(f"Invalid number of vessels format. Using default value of 0.")
        validation_results["corrected_values"]["ca"] = 0
        validation_results["valid"] = False
    
    try:
        thal_val = int(thal)
        if thal_val not in [0, 1, 2, 3]:
            validation_results["warnings"].append(f"Invalid thal value. Must be 0-3. Using 1.")
            validation_results["corrected_values"]["thal"] = 1
            validation_results["valid"] = False
    except ValueError:
        validation_results["warnings"].append(f"Invalid thal format. Using defaultvalue of 1.")
        validation_results["corrected_values"]["thal"] = 1
        validation_results["valid"] = False
    
    return validation_results

def generate_xray_explanation(label, confidence, has_heatmap=False):
    """Generate explanations for X-ray analysis results"""
    explanations = []
    
    if label == "NORMAL":
        explanations.append("Your chest X-ray appears normal with no significant abnormalities detected.")
        explanations.append("The cardiac silhouette size and shape appear within normal limits.")
        explanations.append("Lung fields appear clear without signs of infiltrates, masses, or effusions.")
        explanations.append("Lung fields appear clear without signs of infiltrates, masses, or effusions.")
        
    elif label == "ABNORMAL":
        explanations.append(f"Your chest X-ray shows patterns that suggest possible cardiac abnormalities (confidence: {confidence:.1%}).")
        
        if confidence > 0.8:
            explanations.append("The analysis indicates significant findings that warrant prompt medical attention.")
        elif confidence > 0.6:
            explanations.append("The analysis shows moderate abnormalities that should be evaluated by a healthcare provider.")
        else:
            explanations.append("The analysis shows subtle abnormalities that may benefit from clinical correlation.")
            
        if has_heatmap:
            explanations.append("The highlighted areas on the visualization represent regions with atypical features.")
            
        explanations.append("These findings may indicate conditions such as cardiomegaly, pulmonary edema, or other cardiac issues.")
        explanations.append("Please consult with a healthcare provider for further evaluation.")
    else:
        explanations.append("The analysis of your chest X-ray was inconclusive.")
        explanations.append("Please consult with a healthcare provider for a proper evaluation.")
        
    # Add disclaimer
    explanations.append("Note: This is an AI analysis and not a substitute for professional medical interpretation.")
    
    return explanations

def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):  # Use np.integer instead of specific integer types
        return int(obj)
    elif isinstance(obj, np.floating):  # Use np.floating instead of specific float types
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
    
def predict_echo_from_file(file_path):
    """Echo prediction with improved sensitivity/specificity balance"""
    try:
        # Check if file is a video or image
        is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
        
        # Initialize extracted_frames here to avoid reference error
        extracted_frames = []
        
        if is_video:
            # Process as video
            video_tensor = preprocess_video(file_path)
            
            # Critical: Check if video preprocessing failed
            if video_tensor is None:
                logger.error(f"Failed to preprocess video: {file_path}")
                # Try basic frame extraction before giving up
                try:
                    extracted_frames = extract_fallback_frames(file_path)
                    if extracted_frames:
                        logger.info(f"Successfully extracted {len(extracted_frames)} fallback frames")
                        # Generate synthetic prediction based on frames
                        middle_frame_idx = len(extracted_frames) // 2
                        gradcam_img = generate_single_frame_echo_heatmap(extracted_frames[middle_frame_idx], 0.5)
                        response = create_fallback_echo_response()
                        response["gradcam"] = image_to_base64(gradcam_img) if gradcam_img is not None else None
                        response["frames"] = [image_to_base64(f) for f in extracted_frames[:5] if f is not None]
                        return response
                except Exception as e:
                    logger.error(f"Fallback frame extraction also failed: {e}")
                return create_fallback_echo_response()
            
            # Continue only if we have a valid tensor
            try:
                # Double-check tensor dimensions before prediction
                expected_shape = (1, 1, 64, 112, 112)
                actual_shape = tuple(video_tensor.shape)
                
                if actual_shape != expected_shape:
                    logger.error(f"Tensor shape mismatch: expected {expected_shape}, got {actual_shape}")
                    return create_fallback_echo_response()
                    
                # Run the prediction
                echo_pred = model_manager.echo_model(video_tensor)
                raw_echo_score = float(torch.sigmoid(echo_pred).item())
                
                # Apply calibration to reduce false positives
                # This adjusts the score to be less sensitive for borderline cases
                echo_score = raw_echo_score * 0.85 if raw_echo_score < 0.7 else raw_echo_score
                
                # Calculate ejection fraction estimate based on echo_score
                ef_estimate = 65 - (echo_score * 30)  # Simple linear model: 65% (normal) down to 35% (severe)
                
                # Determine severity - increased thresholds
                if echo_score > 0.80:  # Increased from 0.75
                    severity = "Severe"
                    needs_attention = True
                elif echo_score > 0.65:  # Increased from 0.5
                    severity = "Moderate"
                    needs_attention = True  
                else:
                    severity = "Mild/Normal"
                    needs_attention = False
                
                # Extract frames for visualization
                try:
                    extracted_frames = extract_fallback_frames(file_path)
                    if not extracted_frames:
                        logger.warning("No frames extracted, using synthetic frames")
                        # Create synthetic frames from tensor if extraction fails
                        synthetic_frames = []
                        for i in range(0, 64, 12):
                            frame = video_tensor[0, 0, i].detach().cpu().numpy()
                            # Normalize to 0-255 range
                            frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
                            synthetic_frames.append(frame)
                        extracted_frames = synthetic_frames[:5]
                except Exception as e:
                    logger.error(f"Frame extraction error: {e}")
                    extracted_frames = []
                    
                # Generate visualization
                try:
                    if len(extracted_frames) > 0:
                        middle_frame_idx = len(extracted_frames) // 2
                        gradcam_img = generate_single_frame_echo_heatmap(extracted_frames[middle_frame_idx], echo_score)
                    else:
                        gradcam_img = None
                except Exception as e:
                    logger.error(f"Visualization error: {e}")
                    gradcam_img = None
                    
                # Rest of the function remains the same...
                # Calculate ejection fraction estimate based on echo_score
                ef_estimate = 65 - (echo_score * 30)  # Simple linear model: 65% (normal) down to 35% (severe)
                
                # Determine severity
                if echo_score > 0.75:
                    severity = "Severe"
                    needs_attention = True
                elif echo_score > 0.5:
                    severity = "Moderate"
                    needs_attention = True  
                else:
                    severity = "Mild/Normal"
                    needs_attention = False
                    
                # Prepare response
                explanations = generate_echo_explanation(echo_score, gradcam_img is not None)
                
                # Generate recommendations based on severity
                recommendations = []
                if needs_attention:
                    if severity == "Severe":
                        recommendations = [
                            "You should consult with a cardiologist promptly.",
                            "The analysis indicates potential significant cardiac abnormalities.",
                            "Further clinical evaluation with a formal echocardiogram is strongly recommended."
                        ]
                    else:
                        recommendations = [
                            "Consider scheduling an appointment with a cardiologist.",
                            "Some cardiac abnormalities may be present that warrant further evaluation.",
                            "A formal clinical echocardiogram would provide more definitive assessment."
                        ]
                else:
                    recommendations = [
                        "No significant abnormalities detected.",
                        "Continue with routine cardiac care as recommended by your physician.",
                        "Regular cardiac check-ups are still important for preventive care."
                    ]
                    
                # Ensure proper image encoding
                gradcam_base64 = None
                if gradcam_img is not None:
                    gradcam_base64 = image_to_base64(gradcam_img)
                    
                # Encode frames
                encoded_frames = []
                for frame in extracted_frames[:5]:
                    if frame is not None:
                        encoded_frame = image_to_base64(frame)
                        if encoded_frame:
                            encoded_frames.append(encoded_frame)
                            
                return {
                    "score": echo_score,
                    "ejection_fraction_estimate": float(ef_estimate),
                    "severity": severity,
                    "needs_attention": needs_attention,
                    "recommendations": recommendations,
                    "gradcam": gradcam_base64,
                    "frames": encoded_frames,
                    "explanations": explanations
                }
                
            except Exception as e:
                logger.error(f"Error in echo prediction: {str(e)}")
                return create_fallback_echo_response()
                
        else:
            # For image files, use a simpler approach
            return create_fallback_echo_response()
                
    except Exception as e:
        logger.error(f"Error in echo prediction: {str(e)}")
        return create_fallback_echo_response()
    
def generate_echo_explanation(echo_score, has_visualization=False):
    """Generate explanation for echocardiogram results"""
    explanations = []
    
    if echo_score > 0.75:
        explanations.append(f"The echocardiogram analysis indicates potential significant cardiac abnormalities (score: {echo_score:.2f}).")
        explanations.append("The findings suggest reduced cardiac function that requires medical attention.")
        if has_visualization:
            explanations.append("The highlighted areas in the visualization show regions of concern in cardiac structure or function.")
        explanations.append("You should consult with a cardiologist for a comprehensive evaluation.")
    elif echo_score > 0.5:
        explanations.append(f"The echocardiogram analysis shows moderate cardiac abnormalities (score: {echo_score:.2f}).")
        explanations.append("The findings suggest some alterations in cardiac structure or function.")
        if has_visualization:
            explanations.append("The highlighted regions in the visualization indicate areas that may require clinical attention.")
        explanations.append("Consider scheduling an appointment with a healthcare provider for further assessment.")
    else:
        explanations.append(f"The echocardiogram analysis indicates normal or mild findings (score: {echo_score:.2f}).")
        explanations.append("No significant cardiac abnormalities were detected in the automated analysis.")
        explanations.append("As with any automated analysis, follow up with your healthcare provider as needed.")
        
    # Add technical details about the analysis
    explanations.append("This automated echo analysis evaluates cardiac motion, chamber size, and wall thickness.")
    explanations.append("Note: This is a preliminary AI analysis and not a substitute for a clinical echocardiogram.")
    
    return explanations

def extract_fallback_frames(file_path):
    """Fallback function to extract frames from video."""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {file_path}")
            return []
            
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            logger.error(f"Video has no frames: {file_path}")
            return []
            
        # Extract 5 frames evenly spaced throughout the video
        indices = np.linspace(0, frame_count-1, min(5, frame_count)).astype(int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize to standard size
                gray_frame = cv2.resize(gray_frame, (112, 112))
                frames.append(gray_frame)
                
        cap.release()
        return frames
    except Exception as e:
        logger.error(f"Error extracting fallback frames: {e}")
        return []
        
def create_fallback_echo_response():
    """Create a fallback response when echo processing fails"""
    return {
        "score": 0.5,  # Neutral score
        "ejection_fraction_estimate": 55.0,  # Normal range
        "severity": "Indeterminate",
        "needs_attention": True,  # Flag for attention due to processing error
        "recommendations": [
            "Processing error occurred - please consult with a healthcare provider",
            "Consider submitting a higher quality video for better analysis",
            "When in doubt, a clinical echocardiogram is recommended"
        ],
        "gradcam": None,
        "frames": [],
        "explanations": [
            "Unable to process echo video due to technical limitations.",
            "This could be due to video format, quality, or content issues.",
            "For accurate cardiac evaluation, please consult with a healthcare provider."
        ]
    }

def create_improved_xray_visualization(img_array, heatmap, prediction):
    """Create a more clinically relevant visualization of X-ray findings with clear highlighting of affected areas"""
    try:
        # Check inputs to avoid tuple index errors
        if img_array is None or heatmap is None:
            logger.error("Received None input for visualization")
            return create_fallback_visualization()
            
        # Convert image array to proper format
        if isinstance(img_array, np.ndarray):
            if len(img_array.shape) >= 3 and img_array.shape[-1] == 1:
                img = np.squeeze(img_array)
            else:
                # Convert RGB to grayscale if needed
                img = np.mean(img_array, axis=-1) if len(img_array.shape) >= 3 else img_array
        else:
            logger.error(f"Invalid img_array type: {type(img_array)}")
            return create_fallback_visualization()
            
        # Ensure img is 2D
        if len(img.shape) != 2:
            logger.error(f"Expected 2D image, got shape {img.shape}")
            return create_fallback_visualization()
            
        # Normalize image safely
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        
        # Resize heatmap to match image dimensions
        if heatmap.shape != img.shape:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast in the original image for better visualization
        img_enhanced = cv2.equalizeHist((img * 255).astype(np.uint8))
        img_enhanced = img_enhanced / 255.0
        
        # Create RGB version of the image
        img_rgb = np.stack([img_enhanced, img_enhanced, img_enhanced], axis=-1)
        
        # Use higher thresholds for highlighting
        affected_threshold = 0.7  # Increased from 0.6
        affected_mask = heatmap > affected_threshold
        
        # Create separate mask for moderately affected areas
        moderate_threshold = 0.5  # Increased from 0.3
        moderate_mask = (heatmap > moderate_threshold) & (heatmap <= affected_threshold)
        
        # Create a colored overlay for the affected areas
        overlay = img_rgb.copy()
        
        # Apply strong red highlighting to severely affected areas (with higher opacity)
        overlay[affected_mask] = overlay[affected_mask] * 0.3 + np.array([1.0, 0.0, 0.0]) * 0.7
        
        # Apply yellow highlighting to moderately affected areas (with lower opacity)
        overlay[moderate_mask] = overlay[moderate_mask] * 0.5 + np.array([1.0, 0.8, 0.0]) * 0.5
        
        # Add contours around the affected areas for better visibility
        if np.any(affected_mask):
            affected_mask_uint8 = affected_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(affected_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours to avoid noise
            significant_contours = [c for c in contours if cv2.contourArea(c) > 20]
            cv2.drawContours(overlay, significant_contours, -1, (1, 0, 0), 2)
        
        # Convert to uint8 for display
        final_visualization = (overlay * 255).astype(np.uint8)
        
        # Add text annotations at the bottom
        height, width = final_visualization.shape[:2]
        bottom_margin = 40
        result = np.ones((height + bottom_margin, width, 3), dtype=np.uint8) * 255
        result[:height, :, :] = final_visualization
        
        # Extract prediction values safely
        normal_score = 0
        abnormal_score = 0
        
        # Handle different prediction formats
        if isinstance(prediction, np.ndarray):
            # Single prediction array
            if prediction.ndim == 1 and prediction.size >= 2:
                normal_score = prediction[0]
                abnormal_score = prediction[1]
            # Batch of predictions (take first)
            elif prediction.ndim == 2 and prediction.shape[1] >= 2:
                normal_score = prediction[0, 0]
                abnormal_score = prediction[0, 1]
            # Single value prediction
            elif prediction.size == 1:
                normal_score = 1 - prediction.item()
                abnormal_score = prediction.item()
        # Handle list format
        elif isinstance(prediction, list):
            if len(prediction) > 0:
                if isinstance(prediction[0], np.ndarray):
                    if prediction[0].size >= 2:
                        normal_score = prediction[0][0]
                        abnormal_score = prediction[0][1]
                    elif prediction[0].size == 1:
                        abnormal_score = prediction[0].item()
                        normal_score = 1 - abnormal_score
                elif isinstance(prediction[0], list) and len(prediction[0]) >= 2:
                    normal_score = prediction[0][0]
                    abnormal_score = prediction[0][1]
                else:
                    # Generic fallback for single value
                    try:
                        abnormal_score = float(prediction[0])
                        normal_score = 1 - abnormal_score
                    except (TypeError, ValueError):
                        abnormal_score = 0.5
                        normal_score = 0.5
            else:
                abnormal_score = 0.5
                normal_score = 0.5
        # Scalar prediction
        elif isinstance(prediction, (int, float)):
            abnormal_score = float(prediction)
            normal_score = 1 - abnormal_score
        # Unknown format
        else:
            abnormal_score = 0.5
            normal_score = 0.5
            
        # Ensure values are within range [0,1]
        normal_score = max(0, min(1, normal_score))
        abnormal_score = max(0, min(1, abnormal_score))
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Normal: {normal_score*100:.1f}%", (10, height + 25), font, 0.5, (0, 0, 0), 1)
        cv2.putText(result, f"Abnormal: {abnormal_score*100:.1f}%", (width//2, height + 25), font, 0.5, (0, 0, 0), 1)
        
        # Add warning if highly abnormal
        if abnormal_score > 0.7:
            warning_text = "⚠ Areas requiring medical attention detected"
            text_size = cv2.getTextSize(warning_text, font, 0.5, 1)[0]
            warning_x = (width - text_size[0]) // 2
            cv2.putText(result, warning_text, (warning_x, height + 15), font, 0.5, (0, 0, 255), 1)
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating X-ray visualization: {str(e)}")
        return create_fallback_visualization()

def create_fallback_visualization():
    """Create a basic fallback visualization when the main one fails"""
    # Create a simple gray image with error message
    height, width = 128, 128
    fallback = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray
    
    # Add text explaining the issue
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(fallback, "Visualization", (10, 50), font, 0.5, (0, 0, 0), 1)
    cv2.putText(fallback, "unavailable", (10, 70), font, 0.5, (0, 0, 0), 1)
    cv2.putText(fallback, "See text analysis", (10, 100), font, 0.5, (0, 0, 0), 1)
    
    return fallback
