import cv2
import numpy as np
from tensorflow.keras.models import load_model

class XRayPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128)) / 255.0
        return img.reshape(1, 128, 128, 1)
    
    def predict(self, image_path):
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img)
        class_idx = np.argmax(prediction)
        label = "PNEUMONIA" if class_idx == 1 else "NORMAL"
        return label, float(prediction[0][class_idx])