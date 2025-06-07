# models/clinical_model.py
import pandas as pd
import numpy as np
import joblib

class ClinicalPredictor:
    def __init__(self, model_path):
        self.model_bundle = joblib.load(model_path)
        self.best_model = self.model_bundle["best_model"]
        self.feature_selector = self.model_bundle["feature_selector"]
        self.scaler = self.model_bundle["power_scaler"]
        self.selected_features = self.model_bundle["selected_features"]
    
    def engineer_features(self, df):
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
    
    def predict(self, clinical_data):
        input_df = pd.DataFrame([clinical_data])
        engineered = self.engineer_features(input_df)
        
        # Make sure we have all required features, filling with zeros if needed
        for feature in self.selected_features:
            if feature not in engineered.columns:
                engineered[feature] = 0
        
        # Select only the features used by the model
        engineered = engineered[self.selected_features]
        
        # Scale features
        scaled = self.scaler.transform(engineered)
        
        # Predict
        clinical_pred_proba = self.best_model.predict_proba(scaled)[0][1]
        return clinical_pred_proba
