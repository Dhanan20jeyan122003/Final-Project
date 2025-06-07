# utils/data_utils.py
import json
import os

def load_clinical_data(json_path):
    """
    Loads clinical data from a JSON file, maps string values to required formats.
    
    Expected JSON structure:
    {
        "age": 65,
        "sex": "Male",
        "chestPainType": "Typical Angina",
        "restingBP": 120,
        "cholesterol": 230,
        "fbs": "True",
        "restECG": "Normal",
        "maxHR": 150,
        "exerciseAngina": "Yes",
        "stDepression": 2.3,
        "stSlope": "Flat",
        "majorVessels": 0,
        "thalassemia": "Normal"
    }
    """
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Map string values to required numeric formats
        mapped_data = {
            "age": float(data.get("age", 0)),
            "sex": {"Male": 1, "Female": 0}.get(data.get("sex", ""), 0),
            "cp": {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}.get(data.get("chestPainType", ""), 0),
            "trestbps": float(data.get("restingBP", 0)),
            "chol": float(data.get("cholesterol", 0)),
            "fbs": {"True": 1, "False": 0}.get(data.get("fbs", ""), 0),
            "restecg": {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2}.get(data.get("restECG", ""), 0),
            "thalach": float(data.get("maxHR", 0)),
            "exang": {"Yes": 1, "No": 0}.get(data.get("exerciseAngina", ""), 0),
            "oldpeak": float(data.get("stDepression", 0)),
            "slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2}.get(data.get("stSlope", ""), 0),
            "ca": float(data.get("majorVessels", 0)),
            "thal": {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}.get(data.get("thalassemia", ""), 1)
        }
        
        return mapped_data
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        return None

def ensure_directory_exists(dir_path):
    """Ensures that the specified directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)