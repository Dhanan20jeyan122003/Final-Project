from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
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
import asyncio
import gc
import functools
import concurrent.futures
import hashlib

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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()

# CORS Configuration - Make this the first middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's address
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Add compression middleware for better performance
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # Only compress responses larger than 1KB
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
    clinical: float
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
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x.squeeze()
    
    def get_activation(self, x):
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
    """Optimized model manager with lazy loading and caching"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.models_loaded = False
            cls._instance.explainers_loaded = False
            cls._instance._model_cache = {}
        return cls._instance
        
    def load_models(self):
        """Load models only when needed with caching"""
        if self.models_loaded:
            return
            
        try:
            # Load clinical model bundle - most frequently used
            self.model_bundle = joblib.load("models/clinical_model.joblib")
            self.best_model = self.model_bundle["best_model"]
            self.feature_selector = self.model_bundle["feature_selector"]
            self.scaler = self.model_bundle["power_scaler"]
            self.selected_features = self.model_bundle["selected_features"]
            
            # Model caching - load other models on first use
            self._model_cache = {}
            self.models_loaded = True
            logger.info("Clinical model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
            raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
    
    def get_xray_model(self):
        """Lazy-load X-ray model"""
        if "xray_model" not in self._model_cache:
            logger.info("Loading X-ray model")
            self._model_cache["xray_model"] = load_model("models/xray_model.keras")
        return self._model_cache["xray_model"]
    
    def get_echo_model(self):
        """Lazy-load Echo model"""
        if "echo_model" not in self._model_cache:
            logger.info("Loading Echo model")
            self._model_cache["echo_model"] = load_pytorch_model(R3D_EF_Predictor, "models/echo_model.pth")
        return self._model_cache["echo_model"]
    
    def get_ecg_model(self):
        """Lazy-load ECG model"""
        if "ecg_model" not in self._model_cache:
            logger.info("Loading ECG model")
            self._model_cache["ecg_model"] = load_pytorch_model(ECGModel, "models/ecg_model.pth")
        return self._model_cache["ecg_model"]
    
    @property
    def xray_model(self):
        return self.get_xray_model()
    
    @property
    def echo_model(self):
        return self.get_echo_model()
    
    @property
    def ecg_model(self):
        return self.get_ecg_model()
    
    def load_explainers(self):
        """Load explainers only when needed"""
        if self.explainers_loaded:
            return
            
        try:
            # Initialize explainers with minimal background data
            background_data = np.zeros((50, len(self.selected_features)))  # Reduced from 100
            background_data = self.scaler.transform(background_data)
            
            # Simplified explainers
            self.clinical_explainer = shap.KernelExplainer(
                model=self.best_model.predict_proba,
                data=background_data,
                link="logit"
            )
            
            # Simple mapping function instead of binding method
            def predict_proba_wrapper(X):
                decision_values = self.best_model.decision_function(X)
                probs = 1 / (1 + np.exp(-decision_values))
                return np.column_stack([1 - probs, probs]) if probs.ndim == 1 else probs
                
            self.predict_proba = predict_proba_wrapper
            self.explainers_loaded = True
            
        except Exception as e:
            logger.warning(f"Error loading explainers: {e}")
            self.explainers_loaded = False

# Initialize model manager
model_manager = ModelManager()

def validate_clinical_inputs(age, sex, chestPainType, restingBP, cholesterol, fbs, restECG, 
                         maxHR, exerciseAngina, stDepression, stSlope, numVessels, thal):
    """Validate and correct clinical inputs with comprehensive error checking"""
    # Initialize response
    validation_results = {
        "valid": True,
        "warnings": [],
        "corrected_values": {}
    }
    
    # Validate age
    try:
        age_val = float(age)
        if age_val < AGE_MIN:
            validation_results["warnings"].append(f"Age {age_val} is below expected minimum {AGE_MIN}. Using minimum value.")
            validation_results["corrected_values"]["age"] = AGE_MIN
        elif age_val > AGE_MAX:
            validation_results["warnings"].append(f"Age {age_val} is above expected maximum {AGE_MAX}. Using maximum value.")
            validation_results["corrected_values"]["age"] = AGE_MAX
    except ValueError:
        validation_results["warnings"].append(f"Invalid age format: {age}. Using default value.")
        validation_results["corrected_values"]["age"] = 50.0
    
    # Validate sex (0 = female, 1 = male)
    try:
        sex_val = int(float(sex))
        if sex_val not in [0, 1]:
            validation_results["warnings"].append(f"Sex must be 0 (female) or 1 (male). Using 0.")
            validation_results["corrected_values"]["sex"] = 0
    except ValueError:
        validation_results["warnings"].append(f"Invalid sex format: {sex}. Using default value.")
        validation_results["corrected_values"]["sex"] = 0
    
    # Validate chest pain type (0-3)
    try:
        cp_val = int(float(chestPainType))
        if cp_val < 0 or cp_val > 3:
            validation_results["warnings"].append(f"Chest pain type must be between 0-3. Using 0.")
            validation_results["corrected_values"]["cp"] = 0
    except ValueError:
        validation_results["warnings"].append(f"Invalid chest pain type format: {chestPainType}. Using default value.")
        validation_results["corrected_values"]["cp"] = 0
    
    # Validate resting BP
    try:
        bp_val = float(restingBP)
        if bp_val < BP_MIN:
            validation_results["warnings"].append(f"Resting BP {bp_val} is below expected minimum {BP_MIN}. Using minimum value.")
            validation_results["corrected_values"]["trestbps"] = BP_MIN
        elif bp_val > BP_MAX:
            validation_results["warnings"].append(f"Resting BP {bp_val} is above expected maximum {BP_MAX}. Using maximum value.")
            validation_results["corrected_values"]["trestbps"] = BP_MAX
    except ValueError:
        validation_results["warnings"].append(f"Invalid resting BP format: {restingBP}. Using default value.")
        validation_results["corrected_values"]["trestbps"] = 120.0
    
    # Validate cholesterol
    try:
        chol_val = float(cholesterol)
        if chol_val < CHOL_MIN:
            validation_results["warnings"].append(f"Cholesterol {chol_val} is below expected minimum {CHOL_MIN}. Using minimum value.")
            validation_results["corrected_values"]["chol"] = CHOL_MIN
        elif chol_val > CHOL_MAX:
            validation_results["warnings"].append(f"Cholesterol {chol_val} is above expected maximum {CHOL_MAX}. Using maximum value.")
            validation_results["corrected_values"]["chol"] = CHOL_MAX
    except ValueError:
        validation_results["warnings"].append(f"Invalid cholesterol format: {cholesterol}. Using default value.")
        validation_results["corrected_values"]["chol"] = 200.0
    
    # Validate fasting blood sugar (0 = false, 1 = true)
    try:
        fbs_val = int(float(fbs))
        if fbs_val not in [0, 1]:
            validation_results["warnings"].append(f"Fasting blood sugar must be 0 (< 120 mg/dl) or 1 (> 120 mg/dl). Using 0.")
            validation_results["corrected_values"]["fbs"] = 0
    except ValueError:
        validation_results["warnings"].append(f"Invalid fasting blood sugar format: {fbs}. Using default value.")
        validation_results["corrected_values"]["fbs"] = 0
    
    # Validate resting ECG (0-2)
    try:
        restecg_val = int(float(restECG))
        if restecg_val < 0 or restecg_val > 2:
            validation_results["warnings"].append(f"Resting ECG must be between 0-2. Using 0.")
            validation_results["corrected_values"]["restecg"] = 0
    except ValueError:
        validation_results["warnings"].append(f"Invalid resting ECG format: {restECG}. Using default value.")
        validation_results["corrected_values"]["restecg"] = 0
    
    # Validate max heart rate
    try:
        hr_val = float(maxHR)
        if hr_val < HR_MIN:
            validation_results["warnings"].append(f"Max heart rate {hr_val} is below expected minimum {HR_MIN}. Using minimum value.")
            validation_results["corrected_values"]["thalach"] = HR_MIN
        elif hr_val > HR_MAX:
            validation_results["warnings"].append(f"Max heart rate {hr_val} is above expected maximum {HR_MAX}. Using maximum value.")
            validation_results["corrected_values"]["thalach"] = HR_MAX
    except ValueError:
        validation_results["warnings"].append(f"Invalid max heart rate format: {maxHR}. Using default value.")
        validation_results["corrected_values"]["thalach"] = 150.0
    
    # Validate exercise-induced angina (0 = no, 1 = yes)
    try:
        exang_val = int(float(exerciseAngina))
        if exang_val not in [0, 1]:
            validation_results["warnings"].append(f"Exercise-induced angina must be 0 (no) or 1 (yes). Using 0.")
            validation_results["corrected_values"]["exang"] = 0
    except ValueError:
        validation_results["warnings"].append(f"Invalid exercise angina format: {exerciseAngina}. Using default value.")
        validation_results["corrected_values"]["exang"] = 0
    
    # Validate ST depression
    try:
        oldpeak_val = float(stDepression)
        if oldpeak_val < 0:
            validation_results["warnings"].append(f"ST depression {oldpeak_val} is negative. Using 0.")
            validation_results["corrected_values"]["oldpeak"] = 0.0
        elif oldpeak_val > 10:
            validation_results["warnings"].append(f"ST depression {oldpeak_val} is unusually high. Using 5.0.")
            validation_results["corrected_values"]["oldpeak"] = 5.0
    except ValueError:
        validation_results["warnings"].append(f"Invalid ST depression format: {stDepression}. Using default value.")
        validation_results["corrected_values"]["oldpeak"] = 0.0
    
    # Validate ST slope (0-2)
    try:
        slope_val = int(float(stSlope))
        if slope_val < 0 or slope_val > 2:
            validation_results["warnings"].append(f"ST slope must be between 0-2. Using 1.")
            validation_results["corrected_values"]["slope"] = 1
    except ValueError:
        validation_results["warnings"].append(f"Invalid ST slope format: {stSlope}. Using default value.")
        validation_results["corrected_values"]["slope"] = 1
    
    # Validate number of major vessels (0-4)
    try:
        ca_val = int(float(numVessels))
        if ca_val < 0 or ca_val > 4:
            validation_results["warnings"].append(f"Number of major vessels must be between 0-4. Using 0.")
            validation_results["corrected_values"]["ca"] = 0
    except ValueError:
        validation_results["warnings"].append(f"Invalid number of vessels format: {numVessels}. Using default value.")
        validation_results["corrected_values"]["ca"] = 0
    
    # Validate thalassemia (0-3)
    try:
        thal_val = int(float(thal))
        if thal_val < 0 or thal_val > 3:
            validation_results["warnings"].append(f"Thalassemia value must be between 0-3. Using 2.")
            validation_results["corrected_values"]["thal"] = 2
    except ValueError:
        validation_results["warnings"].append(f"Invalid thalassemia format: {thal}. Using default value.")
        validation_results["corrected_values"]["thal"] = 2
    
    # Add serious warning if too many inputs needed correction
    if len(validation_results["corrected_values"]) > 3:
        validation_results["warnings"].insert(0, "Multiple input values required correction. Results may be less reliable.")
    
    # Return validation results
    return validation_results

def engineer_features(df):
    """Engineer clinical features for improved heart disease prediction"""
    # Make a copy of the dataframe to avoid modifying the original
    df_engineered = df.copy()
    
    # Create age groups (10-year bins)
    df_engineered['age_group'] = pd.cut(
        df_engineered['age'], 
        bins=[0, 40, 50, 60, 70, 100], 
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    
    # Add age group dummies for the model
    df_engineered['age_group_young'] = (df_engineered['age'] < 45).astype(int)
    df_engineered['age_group_middle'] = ((df_engineered['age'] >= 45) & (df_engineered['age'] < 60)).astype(int)
    df_engineered['age_group_senior'] = (df_engineered['age'] >= 60).astype(int)
    
    # Create BP categories
    df_engineered['bp_category'] = pd.cut(
        df_engineered['trestbps'], 
        bins=[0, 120, 140, 160, 300], 
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # Create cholesterol categories
    df_engineered['chol_category'] = pd.cut(
        df_engineered['chol'], 
        bins=[0, 200, 240, 300, 1000], 
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # Add cholesterol category dummies for the model
    df_engineered['chol_category_normal'] = (df_engineered['chol'] < 200).astype(int)
    df_engineered['chol_category_borderline'] = ((df_engineered['chol'] >= 200) & (df_engineered['chol'] < 240)).astype(int)
    df_engineered['chol_category_high'] = (df_engineered['chol'] >= 240).astype(int)
    
    # Heart rate as percentage of age-predicted maximum
    df_engineered['heart_rate_pct'] = df_engineered['thalach'] / (220 - df_engineered['age'])
    
    # Ratio features
    df_engineered['thalach_age_ratio'] = df_engineered['thalach'] / df_engineered['age']
    df_engineered['chol_age_ratio'] = df_engineered['chol'] / df_engineered['age']
    df_engineered['st_heart_ratio'] = df_engineered['oldpeak'] / df_engineered['thalach']
    
    # Create pressure-rate product (BP × HR), a measure of cardiac workload
    df_engineered['pressure_rate_product'] = df_engineered['trestbps'] * df_engineered['thalach'] / 10000
    
    # Flag for multiple risk factors
    df_engineered['multiple_risks'] = (
        (df_engineered['chol'] > 240).astype(int) + 
        (df_engineered['trestbps'] > 140).astype(int) + 
        (df_engineered['fbs'] > 0).astype(int) + 
        (df_engineered['age'] > 60).astype(int)
    )
    
    # Return the dataframe with engineered features
    return df_engineered

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
    """Optimized prediction endpoint with parallel processing"""
    temp_files = []
    
    try:
        # Initialize response
        response_data = {
            "prediction": "",
            "confidence": 0.0,
            "clinical": 0.0,
            "explanations": [],
            "ecg_analysis": None,
            "xray_analysis": None,
            "echo_analysis": None
        }
        
        # Process and validate clinical data first - always required
        validation_result = validate_clinical_inputs(
            age, sex, chestPainType, restingBP, cholesterol, fbs, restECG,
            maxHR, exerciseAngina, stDepression, stSlope, numVessels, thal
        )
        
        # Use corrected values from validation
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
        
        # Override with corrected values
        for key, val in validation_result["corrected_values"].items():
            clinical_data_dict[key] = val
            
        # Get clinical prediction
        clinical_df = pd.DataFrame([clinical_data_dict])
        clinical_df = engineer_features(clinical_df)
        clinical_df = clinical_df[model_manager.selected_features]
        clinical_df = model_manager.scaler.transform(clinical_df)
        clinical_pred_proba = model_manager.best_model.predict_proba(clinical_df)[:, 1][0]
        
        # Create tasks for image processing to run in parallel
        tasks = []
        
        # Save uploaded files concurrently
        if ecgImage:
            tasks.append(process_ecg(ecgImage, temp_files))
            
        if xrayImage:
            tasks.append(process_xray(xrayImage, temp_files))
            
        if echoVideo:
            tasks.append(process_echo(echoVideo, temp_files))
        
        # Run all image processing tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and update response data
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task error: {result}")
                    continue
                    
                if result and isinstance(result, dict):
                    if 'type' in result:
                        if result['type'] == 'ecg':
                            response_data['ecg_analysis'] = result['data']
                        elif result['type'] == 'xray':
                            response_data['xray_analysis'] = result['data']
                        elif result['type'] == 'echo':
                            response_data['echo_analysis'] = result['data']
        
        # Generate clinical explanations (always needed)
        response_data["explanations"] = generate_clinical_explanation(
            clinical_dict=clinical_data_dict,
            clinical_pred_proba=clinical_pred_proba
        )
        
        # Extract scores for final prediction
        xray_score = response_data.get("xray_analysis", {}).get("confidence")
        echo_score = response_data.get("echo_analysis", {}).get("score")
        ecg_score = response_data.get("ecg_analysis", {}).get("score")
        
        # Get final prediction
        final_pred = get_final_prediction(
            clinical_prob=clinical_pred_proba,
            xray_score=xray_score,
            echo_score=echo_score,
            ecg_score=ecg_score
        )
        
        # Update response with final prediction
        response_data.update({
            "prediction": final_pred["prediction"],
            "confidence": final_pred["confidence"],
            "evidence_level": final_pred["evidence_level"],
            "clinical": float(round(clinical_pred_proba, 3)),
            "imaging_emergency": final_pred.get("imaging_emergency", False)
        })
        
        # Apply edge case corrections
        response_data = check_and_correct_prediction_edge_cases(
            prediction=response_data,
            clinical_score=clinical_pred_proba,
            xray_score=xray_score,
            echo_score=echo_score,
            ecg_score=ecg_score
        )
        
        return response_data
        
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            FileHandler.cleanup_temp_file(temp_file)

# Helper functions for parallel processing
async def process_ecg(ecgImage, temp_files):
    """Process ECG image asynchronously"""
    try:
        temp_ecg = await FileHandler.save_upload_file_temp(ecgImage)
        temp_files.append(temp_ecg)
        
        # Use ThreadPoolExecutor for CPU-bound processing
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            ecg_tensor = await loop.run_in_executor(
                pool, preprocess_ecg_image, str(temp_ecg))
            
            ecg_pred = await loop.run_in_executor(
                pool, lambda: model_manager.ecg_model(ecg_tensor))
            
            ecg_score = torch.sigmoid(ecg_pred).item()
            
            gradcam_result = await loop.run_in_executor(
                pool, lambda: generate_gradcam(model_manager.ecg_model, ecg_tensor))
        
        # Make sure we have a valid image result
        if isinstance(gradcam_result, tuple):
            gradcam_img, _ = gradcam_result
        else:
            gradcam_img = gradcam_result
            
        ecg_explanations = generate_ecg_explanation(ecg_score, gradcam_img is not None)
        
        return {
            'type': 'ecg',
            'data': {
                "score": ecg_score,
                "explanations": ecg_explanations,
                "gradcam": image_to_base64(gradcam_img) if gradcam_img is not None else None
            }
        }
    except Exception as e:
        logger.error(f"Error processing ECG: {e}")
        return {
            'type': 'ecg',
            'data': {"error": str(e), "score": 0.0}
        }

async def process_xray(xrayImage, temp_files):
    """Process X-ray image asynchronously"""
    try:
        temp_xray = await FileHandler.save_upload_file_temp(xrayImage)
        temp_files.append(temp_xray)
        
        # Use ThreadPoolExecutor for CPU-bound processing
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Preprocess the image
            xray_img = await loop.run_in_executor(
                pool, preprocess_xray_image, str(temp_xray))
            
            # Run model prediction
            xray_pred = await loop.run_in_executor(
                pool, lambda: model_manager.xray_model.predict(xray_img))
            
            # Generate heatmap visualization
            visualization = await loop.run_in_executor(
                pool, lambda: process_xray_heatmap(model_manager.xray_model, xray_img))
                
        # Extract scores
        normal_score = float(xray_pred[0][0])
        abnormal_score = float(xray_pred[0][1])
        
        # Get base64 encoded heatmap
        heatmap_base64 = image_to_base64(visualization) if visualization is not None else None
        
        # Generate explanations
        label = "NORMAL" if normal_score > abnormal_score else "ABNORMAL"
        explanations = generate_xray_explanation(
            label, 
            max(normal_score, abnormal_score), 
            heatmap_base64 is not None
        )
        
        return {
            'type': 'xray',
            'data': {
                "normal_score": normal_score,
                "abnormal_score": abnormal_score,
                "confidence": max(normal_score, abnormal_score),
                "label": label,
                "needs_attention": abnormal_score > 0.65,
                "heatmap": heatmap_base64,
                "explanations": explanations
            }
        }
    except Exception as e:
        logger.error(f"Error processing X-ray: {e}")
        return {
            'type': 'xray',
            'data': {"error": str(e), "abnormal_score": 0.0}
        }

async def process_echo(echoVideo, temp_files):
    """Process echo video asynchronously"""
    try:
        temp_echo = await FileHandler.save_upload_file_temp(echoVideo)
        temp_files.append(temp_echo)
        
        # Use ThreadPoolExecutor for CPU-bound processing
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Preprocess the video
            video_tensor = await loop.run_in_executor(
                pool, preprocess_video, str(temp_echo))
            
            # Check if preprocessing was successful
            if video_tensor is None:
                # Try fallback processing
                video_tensor = await loop.run_in_executor(
                    pool, lambda: preprocess_video(str(temp_echo), num_frames=32, img_size=112))
                
                if video_tensor is None:
                    raise ValueError("Failed to process echo video file")
                
            # Run model prediction
            with torch.no_grad():
                echo_pred = await loop.run_in_executor(
                    pool, lambda: model_manager.echo_model(video_tensor))
                
            echo_score = float(torch.sigmoid(echo_pred).item())
            
            # Extract visualization frames
            middle_frame_idx = video_tensor.shape[2] // 2
            middle_frame = video_tensor[0, 0, middle_frame_idx].detach().cpu().numpy()
            
            # Generate heatmap
            heatmap_img = await loop.run_in_executor(
                pool, lambda: generate_single_frame_echo_heatmap(middle_frame, echo_score))
        
        # Generate explanations
        explanations = generate_echo_explanation(echo_score, heatmap_img is not None)
        
        # Calculate ejection fraction estimate
        ef_estimate = 65 - (echo_score * 25)
        
        # Determine severity
        if echo_score > 0.9:
            severity = "Severe"
            needs_attention = True
        elif echo_score > 0.8:
            severity = "Moderate"
            needs_attention = True
        else:
            severity = "Mild/Normal"
            needs_attention = False
        
        # Extract frames for display
        extracted_frames = []
        try:
            frame_indices = list(range(0, video_tensor.shape[2], video_tensor.shape[2] // 5))[:5]
            for idx in frame_indices:
                frame = video_tensor[0, 0, idx].detach().cpu().numpy()
                frame_normalized = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
                extracted_frames.append(frame_normalized)
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            
        return {
            'type': 'echo',
            'data': {
                "score": echo_score,
                "ejection_fraction_estimate": float(ef_estimate),
                "severity": severity,
                "needs_attention": needs_attention,
                "gradcam": image_to_base64(heatmap_img) if heatmap_img is not None else None,
                "frames": [image_to_base64(f) for f in extracted_frames if f is not None],
                "explanations": explanations
            }
        }
    except Exception as e:
        logger.error(f"Error processing echo: {e}")
        return {
            'type': 'echo',
            'data': {"error": str(e), "score": 0.0}
        }

@app.on_event("startup")
async def startup_event():
    # Configure frameworks on startup
    configure_framework_options()
    
    # Load clinical model only (others will be lazy loaded)
    model_manager.load_models()
    
    # Setup memory cleanup task
    asyncio.create_task(periodic_memory_cleanup())

async def periodic_memory_cleanup():
    """Run memory cleanup periodically"""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        clear_memory_periodic()
        logger.info("Performed memory cleanup")

def clear_memory_periodic():
    """Force garbage collection to clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory cleanup completed")

def configure_framework_options():
    """Configure ML frameworks for better performance"""
    # TensorFlow optimizations
    tf.config.threading.set_inter_op_parallelism_threads(4)  # Between-graph parallelism
    tf.config.threading.set_intra_op_parallelism_threads(2)  # Within-graph parallelism
    
    # PyTorch optimizations
    torch.set_num_threads(4)  # Set number of threads
    
    # Set optimization flags
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['TF_JIT_ENABLE'] = '1'
    
    logger.info("ML framework options configured for optimal performance")
    return True

# Health check endpoint
@app.get("/")
async def health_check():
    return {
        "status": "online", 
        "models_loaded": model_manager.models_loaded,
        "explainers_loaded": model_manager.explainers_loaded
    }

def generate_gradient_heatmap(model, image_tensor, target_layer_name=None):
    """Generate a gradient-based heatmap with safe tensor hashing"""
    try:
        # Handle different tensor types safely
        if isinstance(image_tensor, tf.Tensor):
            tensor_bytes = image_tensor.numpy().tobytes()
        elif isinstance(image_tensor, torch.Tensor):
            tensor_bytes = image_tensor.cpu().numpy().tobytes()
        elif isinstance(image_tensor, np.ndarray):
            tensor_bytes = image_tensor.tobytes()
        else:
            tensor_bytes = str(image_tensor).encode('utf-8')
            
        tensor_hash = hashlib.md5(tensor_bytes).hexdigest()
        
        return _cached_gradient_heatmap(
            model.__class__.__name__, 
            target_layer_name, 
            tensor_hash, 
            model, 
            image_tensor, 
            target_layer_name
        )
    except Exception as e:
        logger.warning(f"Error in gradient heatmap hash generation: {e}")
        # Skip caching if hashing fails
        return _generate_gradient_heatmap_impl(model, image_tensor, target_layer_name)

@functools.lru_cache(maxsize=32)
def _cached_gradient_heatmap(model_name, target_name, tensor_hash, model, image_tensor, target_layer_name=None):
    """Cached implementation of gradient heatmap generation"""
    return _generate_gradient_heatmap_impl(model, image_tensor, target_layer_name)

def _generate_gradient_heatmap_impl(model, image_tensor, target_layer_name=None):
    """Implementation of gradient-based heatmap generation"""
    try:
        # Make sure the model has been called at least once
        # This is crucial - models need to be called before layers have outputs defined
        _ = model(image_tensor)
        
        # Find appropriate layer if not specified
        if target_layer_name is None:
            # Try common layer names for CNN architectures
            potential_layers = ['conv2d_2', 'block5_conv3', 'conv_final', 'features.denseblock4.denselayer24.conv2',
                              'layer4.2.conv3', 'mixed7', 'sequential_1']
            
            # Get actual model layer names
            model_layers = [layer.name for layer in model.layers if hasattr(layer, 'output')]
            logger.info(f"Available model layers: {model_layers}")
            
            # Find the first available target layer from our potential layers list
            target_layer_name = None
            for layer_name in potential_layers:
                if layer_name in model_layers:
                    target_layer_name = layer_name
                    break
            
            # If no suitable layer found, use the last convolutional layer
            if target_layer_name is None:
                # Find the last convolutional layer
                conv_layers = [layer.name for layer in model.layers 
                              if 'conv' in layer.name.lower() and hasattr(layer, 'output')]
                if conv_layers:
                    target_layer_name = conv_layers[-1]
                    logger.info(f"Using last convolutional layer: {target_layer_name}")
                else:
                    # Last resort: use the last layer with an output
                    valid_layers = [layer.name for layer in model.layers if hasattr(layer, 'output')]
                    if valid_layers:
                        target_layer_name = valid_layers[-1]
                        logger.info(f"Using last layer with output: {target_layer_name}")
                    else:
                        raise ValueError("No suitable layer found for gradient visualization")
        
        # Get the target layer
        try:
            target_layer = model.get_layer(target_layer_name)
            logger.info(f"Using layer {target_layer_name} for gradient visualization")
        except Exception as e:
            logger.warning(f"Could not find layer {target_layer_name}: {e}")
            # Try to get the last convolutional layer as a fallback
            conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
            if conv_layers:
                target_layer = conv_layers[-1]
                logger.info(f"Falling back to layer: {target_layer.name}")
            else:
                raise ValueError("No convolutional layer found for visualization")
        
        # Define the gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[target_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            layer_output, predictions = grad_model(image_tensor)
            class_channel = tf.argmax(predictions[0])
            loss = predictions[:, class_channel]
        
        # Extract gradients
        gradients = tape.gradient(loss, layer_output)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        
        # Weight feature maps with gradients
        heatmap = layer_output[0] @ pooled_gradients[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()
        
        return heatmap
        
    except Exception as e:
        logger.warning(f"Error in gradient-based heatmap generation: {str(e)}")
        logger.warning("Falling back to direct gradient method")
        try:
            return generate_direct_gradient_heatmap(model, image_tensor)
        except Exception as e2:
            logger.error(f"Fallback method also failed: {str(e2)}")
            return None

def generate_direct_gradient_heatmap(model, image_tensor):
    """Simpler gradient visualization that doesn't rely on specific layers"""
    with tf.GradientTape() as tape:
        # Watch the input tensor
        tape.watch(image_tensor)
        # Get model prediction
        predictions = model(image_tensor)
        # Use the predicted class or the highest activation
        try:
            class_idx = tf.argmax(predictions[0])
            class_output = predictions[0, class_idx]
        except:
            # Handle cases where predictions have different shape
            if isinstance(predictions, (int, float, np.number)):
                class_output = tf.constant(predictions)
            else:
                class_output = tf.reduce_mean(predictions)
    
    # Get gradients of output with respect to input
    gradients = tape.gradient(class_output, image_tensor)
    
    # Take absolute value and max across color channels
    gradients = tf.abs(gradients)
    if len(gradients.shape) > 3:  # If we have a batch dimension
        gradients = tf.reduce_max(gradients, axis=-1)[0]  # Take first batch item
    else:
        gradients = tf.reduce_max(gradients, axis=-1)
    
    # Normalize gradients
    max_val = tf.math.reduce_max(gradients)
    if max_val > 0:
        gradients = gradients / max_val
    
    return gradients.numpy()

def image_to_base64(image):
    """Convert image to base64 string with robust error handling"""
    if image is None:
        return None
        
    try:
        # Fast path for numpy arrays
        if isinstance(image, np.ndarray):
            # Ensure uint8 type without copying if possible
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Direct conversion without redundant checks
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
            # Use cv2 for encoding - much faster than PIL
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_str}"
        
        # Handle PIL images
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{img_str}"
        
        else:
            logger.warning(f"Unsupported image type: {type(image)}")
            return None
            
    except Exception as e:
        logger.error(f"Error in image_to_base64: {str(e)}")
        return None

# Run server if main module
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")