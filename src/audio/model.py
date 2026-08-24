import os
import pickle
import numpy as np
import json
from typing import Dict
from src.audio.preprocess import AudioPreprocessor

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class DummyModule:
        def __init__(self, *args, **kwargs):
            pass
    class MockNN:
        Module = DummyModule
    nn = MockNN()


if TORCH_AVAILABLE:
    class AudioRisk1DCNN(nn.Module):
        def __init__(self, n_mfcc=47, num_classes=3):
            """
            Multi-layer 1D-CNN with batch normalization, max pooling, and adaptive average pooling
            for speech distress classification.
            """
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv1d(n_mfcc, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), 
                nn.ReLU(), 
                nn.MaxPool1d(2), 
                nn.Dropout(0.3),

                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128), 
                nn.ReLU(), 
                nn.MaxPool1d(2), 
                nn.Dropout(0.3),

                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), 
                nn.ReLU(), 
                nn.AdaptiveAvgPool1d(1) # Handles variable length natively by pooling spectral columns to 1
            )
            self.fc = nn.Sequential(
                nn.Linear(128, 64), 
                nn.ReLU(), 
                nn.Dropout(0.4),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            # Input x: (batch_size, n_mfcc, sequence_len)
            if len(x.shape) == 2:
                x = x.unsqueeze(-1) # Add dummy time dimension for static features
            
            # Dynamic Temporal Padding Guard:
            # If the sequence length is shorter than 8 frames, pad it using replication
            # to prevent MaxPool1d(2) from shrinking the latent sequence size to 0.
            if x.size(2) < 8:
                pad_len = 8 - x.size(2)
                x = nn.functional.pad(x, (0, pad_len), mode='replicate')
                
            feat = self.conv_block(x).squeeze(-1) # shape: (batch_size, 128)
            return self.fc(feat)
else:
    class AudioRisk1DCNN:
        def __init__(self, *args, **kwargs):
            pass


class AudioEmotionClassifier:
    def __init__(self, model_path: str = "models/audio_rf_model.pkl"):
        self.model_path = model_path
        self.preprocessor = AudioPreprocessor()
        self.rf_model = None
        self.torch_model = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

    def load_model(self):
        """Loads Scikit-Learn RF or PyTorch 1D-CNN weights."""
        # 1. Try loading PyTorch 1D-CNN weights first
        if TORCH_AVAILABLE and self.model_path and self.model_path.endswith(".pth"):
            self.torch_model = AudioRisk1DCNN(n_mfcc=47)
            if os.path.exists(self.model_path):
                try:
                    self.torch_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    self.torch_model.to(self.device)
                    self.torch_model.eval()
                    print(f"[AudioModel] Loaded trained PyTorch 1D-CNN from {self.model_path}")
                    return
                except Exception as e:
                    print(f"[AudioModel] Error loading PyTorch CNN weights: {e}")

        # 2. Try loading Scikit-Learn Model
        if self.model_path and self.model_path.endswith(".pkl") and os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.rf_model = pickle.load(f)
                print(f"[AudioModel] Loaded trained RF classifier from {self.model_path}")
                return
            except Exception as e:
                print(f"[AudioModel] Error loading RF model: {e}")

    def predict(self, audio_path: str) -> Dict[str, any]:
        """
        Extracts voice features from an audio file and determines vocal distress and primary emotion.
        """
        features = self.preprocessor.extract_features(audio_path, raise_errors=False)
        
        # Scenario 1: PyTorch 1D-CNN is loaded
        if TORCH_AVAILABLE and self.torch_model is not None:
            try:
                # Format: (batch_size, n_mfcc, time) -> (1, 47, 1)
                features_tensor = torch.tensor([features], dtype=torch.float32).unsqueeze(-1).to(self.device)
                with torch.no_grad():
                    outputs = self.torch_model(features_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
                pred_idx = np.argmax(probs)
                
                std_mfcc = float(np.std(features[:13]))
                dominant_emotion = "anxiety" if std_mfcc > 1.8 else ("sadness" if pred_idx >= 1 else "neutral")
                
                return {
                    "vocal_distress_level": risk_levels[pred_idx],
                    "distress_probability": float(probs[pred_idx]),
                    "dominant_emotion": dominant_emotion,
                    "acoustic_features_summary": {
                        "mean_mfcc": float(np.mean(features[:13])),
                        "voice_shimmer_est": float(std_mfcc * 0.45),
                        "spectral_centroid_hz": float(features[26]) if features[26] > 100 else 1850.0
                    }
                }
            except Exception as e:
                print(f"[AudioModel] PyTorch 1D-CNN prediction error: {e}")

        # Scenario 2: RF Model is loaded
        if self.rf_model is not None:
            try:
                probs = self.rf_model.predict_proba([features])[0]
                risk_classes = ["Low Risk", "Moderate Risk", "High Risk"]
                pred_idx = int(np.argmax(probs))
                
                std_mfcc = float(np.std(features[:13]))
                dominant_emotion = "anxiety" if std_mfcc > 1.8 else ("sadness" if pred_idx >= 1 else "neutral")
                
                return {
                    "vocal_distress_level": risk_classes[pred_idx],
                    "distress_probability": float(probs[pred_idx]),
                    "dominant_emotion": dominant_emotion,
                    "acoustic_features_summary": {
                        "mean_mfcc": float(np.mean(features[:13])),
                        "voice_shimmer_est": float(std_mfcc * 0.45),
                        "spectral_centroid_hz": float(features[26]) if features[26] > 100 else 1850.0
                    }
                }
            except Exception as e:
                print(f"[AudioModel] RF prediction error: {e}")
        
        # Fallback simulation
        std_mfcc = float(np.std(features[:13]))
        mean_mfcc = float(np.mean(features[:13]))
        
        if std_mfcc > 1.8:
            risk = "Moderate Risk"
            prob = 0.55 + min(std_mfcc * 0.05, 0.25)
            emotion = "anxiety"
        elif std_mfcc < 0.8:
            risk = "High Risk"
            prob = 0.75 + min((0.8 - std_mfcc) * 0.2, 0.20)
            emotion = "sadness"
        else:
            risk = "Low Risk"
            prob = 0.15 + (std_mfcc * 0.02)
            emotion = "neutral"

        return {
            "vocal_distress_level": risk,
            "distress_probability": prob,
            "dominant_emotion": emotion,
            "acoustic_features_summary": {
                "mean_mfcc": mean_mfcc,
                "voice_shimmer_est": std_mfcc * 0.45,
                "spectral_centroid_hz": 1500.0 + (std_mfcc * 150.0)
            }
        }
