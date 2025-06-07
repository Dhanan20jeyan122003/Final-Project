# start.py

# Use absolute imports
from clinical import ClinicalPredictor
from ecg import ECGPredictor
from xray import XRayPredictor
from echo import EchoPredictor
import argparse
import json
import os

class HeartDiseasePredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        
        # Initialize predictors (handle possible errors)
        try:
            self.clinical_predictor = ClinicalPredictor(os.path.join(models_dir, "clinical_model.joblib"))
            print("Clinical model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load clinical model: {e}")
            self.clinical_predictor = None
            
        try:
            self.ecg_predictor = ECGPredictor(os.path.join(models_dir, "ecg_model.pth"))
            print("ECG model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load ECG model: {e}")
            self.ecg_predictor = None
            
        try:
            self.xray_predictor = XRayPredictor(os.path.join(models_dir, "xray_model.keras"))
            print("X-ray model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load X-ray model: {e}")
            self.xray_predictor = None
            
        try:
            self.echo_predictor = EchoPredictor(os.path.join(models_dir, "echo_model.pth"))
            print("Echo model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Echo model: {e}")
            self.echo_predictor = None
        
    def predict(self, clinical_data=None, ecg_path=None, xray_path=None, echo_path=None):
        results = []
        
        # Clinical data prediction
        clinical_pred_proba = None
        if clinical_data and self.clinical_predictor:
            try:
                clinical_pred_proba = self.clinical_predictor.predict(clinical_data)
                results.append(clinical_pred_proba)
                print(f"Clinical prediction: {clinical_pred_proba:.3f}")
            except Exception as e:
                print(f"Error in clinical prediction: {e}")
        
        # ECG prediction
        ecg_score = None
        if ecg_path and os.path.exists(ecg_path) and self.ecg_predictor:
            try:
                ecg_score = self.ecg_predictor.predict(ecg_path)
                results.append(ecg_score)
                print(f"ECG prediction: {ecg_score:.3f}")
            except Exception as e:
                print(f"Error in ECG prediction: {e}")
        
        # X-ray prediction
        xray_result = {"label": None, "confidence": None}
        if xray_path and os.path.exists(xray_path) and self.xray_predictor:
            try:
                xray_label, xray_score = self.xray_predictor.predict(xray_path)
                xray_result = {"label": xray_label, "confidence": round(xray_score, 3)}
                results.append(xray_score)
                print(f"X-ray prediction: {xray_label} ({xray_score:.3f})")
            except Exception as e:
                print(f"Error in X-ray prediction: {e}")
        
        # Echo prediction
        echo_score = None
        if echo_path and os.path.exists(echo_path) and self.echo_predictor:
            try:
                echo_score = self.echo_predictor.predict(echo_path)
                results.append(echo_score)
                print(f"Echo prediction: {echo_score:.3f}")
            except Exception as e:
                print(f"Error in Echo prediction: {e}")
        
        # Calculate final score and risk
        if not results:
            return {"error": "No valid predictions were made"}
        
        final_score = sum(results) / len(results)
        threshold = 0.7  # Adjust as needed
        prediction = "Heart disease risk is high" if final_score > threshold else "Low risk"
        
        return {
            "prediction": prediction,
            "confidence": round(final_score, 3),
            "clinical": round(clinical_pred_proba, 3) if clinical_pred_proba is not None else None,
            "ecg": round(ecg_score, 3) if ecg_score is not None else None,
            "xray": xray_result if xray_result["label"] is not None else None,
            "echo": round(echo_score, 3) if echo_score is not None else None
        }

def load_clinical_data(json_path):
    """Load clinical data from a JSON file"""
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

def main():
    parser = argparse.ArgumentParser(description="Heart Disease Prediction System")
    parser.add_argument("--clinical", type=str, help="Path to JSON file with clinical data")
    parser.add_argument("--ecg", type=str, help="Path to ECG image file")
    parser.add_argument("--xray", type=str, help="Path to X-ray image file")
    parser.add_argument("--echo", type=str, help="Path to Echo video file")
    parser.add_argument("--models_dir", default="models", help="Directory containing model files")
    parser.add_argument("--output", type=str, help="Path to save output JSON")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HeartDiseasePredictor(models_dir=args.models_dir)
    
    # Load clinical data if provided
    clinical_data = None
    if args.clinical:
        clinical_data = load_clinical_data(args.clinical)
    
    # Make prediction
    result = predictor.predict(
        clinical_data=clinical_data,
        ecg_path=args.ecg,
        xray_path=args.xray,
        echo_path=args.echo
    )
    
    # Print result
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2))
    
    # Save to file if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()