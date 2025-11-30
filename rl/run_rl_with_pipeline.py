import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
model_path = PROJECT_ROOT / "file" / "model.pkl"
scaler_path = PROJECT_ROOT / "file" / "scaler.pkl"
y_scaler_path = PROJECT_ROOT / "file" / "y_scaler.pkl"
class Pipeline:
    """
    Builds a prediction pipeline using pre-trained model and scaler.
    """
    def __init__(self):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)
        self.n_features = len(self.scaler.feature_names_in_)
        self.feature_names = self.scaler.feature_names_in_
    def predict(self, observation):
       
        obs_df = pd.DataFrame([observation], columns=self.feature_names)
        obs_scaled = self.scaler.transform(obs_df)
        prediction = self.model.predict(obs_scaled)
        prediction = self.y_scaler.inverse_transform(prediction.reshape(-1, 1))
        return prediction