import os
import pickle
import json
import numpy as np
from typing import Dict, Tuple

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
    class TextRiskBiLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim=300, hidden_dim=128, 
                     num_layers=2, num_classes=3, pretrained_embeddings=None):
            """
            Bi-directional LSTM with Attention Pooling for Suicide Risk Detection.
            Attention pooling captures crisis phrases wherever they appear in the post,
            solving the long-term dependency bottleneck of last-hidden-state pooling.
            """
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(pretrained_embeddings)
                self.embedding.weight.requires_grad = True  # Fine-tune embeddings during training

            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0.0)
            
            # Linear projection layer for attention weights
            self.attention = nn.Linear(hidden_dim * 2, 1)
            self.dropout = nn.Dropout(0.4)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x, lengths=None):
            embedded = self.embedding(x)
            
            if lengths is not None:
                lengths_cpu = torch.as_tensor(lengths, dtype=torch.int64, device='cpu')
                packed = nn.utils.rnn.pack_padded_sequence(
                    embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
                packed_out, _ = self.lstm(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
            else:
                out, _ = self.lstm(embedded)

            attn_scores = self.attention(out).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)

            return self.fc(self.dropout(context))
else:
    class TextRiskBiLSTM:
        def __init__(self, *args, **kwargs):
            pass


class TextRiskClassifier:
    def __init__(self, model_path: str = "models/text_rf_model.pkl", vocab_path: str = "models/vocab.json"):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.torch_model = None
        self.rf_model = None
        
        # Load vocab size dynamically
        self.vocab_size = 5000
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                    self.vocab_size = len(vocab)
            except Exception:
                pass
                
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
        # Keywords baseline
        self.suicidal_keywords = {
            'high': ['suicide', 'kill myself', 'end my life', 'want to die', 'suicidal', 'kill me', 'hanging', 'overdose', 'commit suicide', 'hang myself'],
            'moderate': ['hopeless', 'depressed', 'no way out', 'worthless', 'give up', 'better off dead', 'can\'t go on', 'tired of living', 'self-harm', 'cutting']
        }
        self.emotion_lexicon = {
            'sadness': ['sad', 'cry', 'grief', 'lonely', 'miserable', 'heartbroken', 'unhappy', 'tears', 'hurt'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'pissed', 'annoyed', 'rage', 'frustrated'],
            'anxiety': ['anxious', 'scared', 'afraid', 'panic', 'nervous', 'worried', 'stress', 'fear'],
            'neutral': ['ok', 'fine', 'normal', 'today', 'going', 'yesterday', 'home', 'work', 'school']
        }

    def load_model(self):
        """Loads Scikit-Learn RF or PyTorch Bi-LSTM parameters."""
        # 1. Try loading PyTorch Bi-LSTM parameters first
        if TORCH_AVAILABLE and self.model_path and self.model_path.endswith(".pth"):
            # Initialize with the precise 128/64/1 layout optimized in train.py
            self.torch_model = TextRiskBiLSTM(
                vocab_size=self.vocab_size,
                embed_dim=128,
                hidden_dim=64,
                num_layers=1
            )
            if os.path.exists(self.model_path):
                try:
                    self.torch_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    self.torch_model.to(self.device)
                    self.torch_model.eval()
                    print(f"[TextModel] Loaded trained PyTorch Bi-LSTM from {self.model_path}")
                    return
                except Exception as e:
                    print(f"[TextModel] Error loading PyTorch weights: {e}")

        # 2. Try loading Scikit-Learn Model
        if self.model_path and self.model_path.endswith(".pkl") and os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.rf_model = pickle.load(f)
                print(f"[TextModel] Loaded trained RF classifier from {self.model_path}")
                return
            except Exception as e:
                print(f"[TextModel] Error loading RF model: {e}")

    def text_to_features(self, text: str) -> np.ndarray:
        """Converts raw text to a 50-dimensional feature vector matching RF input."""
        features = np.zeros(50)
        words = text.lower().split()
        for w in words:
            idx = abs(hash(w)) % 50
            features[idx] += 1.0
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features

    def heuristic_predict(self, text: str) -> Dict[str, any]:
        text_lower = text.lower()
        risk_score = 0.0
        risk_level = "Low Risk"
        
        high_matches = sum(1 for kw in self.suicidal_keywords['high'] if kw in text_lower)
        mod_matches = sum(1 for kw in self.suicidal_keywords['moderate'] if kw in text_lower)
        
        if high_matches > 0:
            risk_level = "High Risk"
            risk_score = min(0.85 + (high_matches * 0.05), 1.0)
        elif mod_matches > 0:
            risk_level = "Moderate Risk"
            risk_score = min(0.40 + (mod_matches * 0.08), 0.84)
        else:
            risk_level = "Low Risk"
            risk_score = min(0.05 + (len(text_lower.split()) * 0.005), 0.39)

        emotion_counts = {emotion: 0 for emotion in self.emotion_lexicon}
        for emotion, keywords in self.emotion_lexicon.items():
            for kw in keywords:
                if kw in text_lower:
                    emotion_counts[emotion] += 1
        
        if max(emotion_counts.values()) > 0:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        else:
            dominant_emotion = "sadness" if risk_level in ["High Risk", "Moderate Risk"] else "neutral"
                
        emotions_probs = {"neutral": 0.6, "sadness": 0.2, "anxiety": 0.1, "anger": 0.1}
        if risk_level == "High Risk":
            emotions_probs = {"sadness": 0.7, "anxiety": 0.2, "anger": 0.1, "neutral": 0.0}
        elif risk_level == "Moderate Risk":
            emotions_probs = {"sadness": 0.5, "anxiety": 0.3, "anger": 0.1, "neutral": 0.1}

        return {
            "risk_level": risk_level,
            "risk_probability": risk_score,
            "dominant_emotion": dominant_emotion,
            "emotion_probabilities": emotions_probs
        }

    def predict(self, text: str) -> Dict[str, any]:
        baseline = self.heuristic_predict(text)
        
        # Scenario 1: PyTorch Bi-LSTM Model is loaded
        if TORCH_AVAILABLE and self.torch_model is not None:
            try:
                words = text.lower().split()
                vocab = {"<PAD>": 0, "<UNK>": 1}
                if os.path.exists(self.vocab_path):
                    with open(self.vocab_path, "r", encoding="utf-8") as f:
                        vocab = json.load(f)
                        
                tokens = [vocab.get(w, vocab["<UNK>"]) for w in words]
                if len(tokens) < 50:
                    tokens = tokens + [vocab["<PAD>"]] * (50 - len(tokens))
                else:
                    tokens = tokens[:50]
                
                input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
                lengths = torch.tensor([min(50, max(1, len(words)))], dtype=torch.long)
                
                with torch.no_grad():
                    outputs = self.torch_model(input_tensor, lengths=lengths)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                risk_classes = ["Low Risk", "Moderate Risk", "High Risk"]
                pred_idx = np.argmax(probs)
                
                if baseline["risk_level"] == "High Risk" and risk_classes[pred_idx] != "High Risk":
                    return baseline
                
                return {
                    "risk_level": risk_classes[pred_idx],
                    "risk_probability": float(probs[pred_idx]),
                    "dominant_emotion": baseline["dominant_emotion"],
                    "emotion_probabilities": baseline["emotion_probabilities"]
                }
            except Exception as e:
                print(f"[TextModel] PyTorch prediction error: {e}")

        # Scenario 2: RF Model is loaded
        if self.rf_model is not None:
            try:
                features = self.text_to_features(text)
                probs = self.rf_model.predict_proba([features])[0]
                risk_classes = ["Low Risk", "Moderate Risk", "High Risk"]
                pred_idx = int(np.argmax(probs))
                
                if baseline["risk_level"] == "High Risk" and risk_classes[pred_idx] != "High Risk":
                    return baseline
                
                return {
                    "risk_level": risk_classes[pred_idx],
                    "risk_probability": float(probs[pred_idx]),
                    "dominant_emotion": baseline["dominant_emotion"],
                    "emotion_probabilities": baseline["emotion_probabilities"]
                }
            except Exception as e:
                print(f"[TextModel] RF Prediction failure: {e}")
                
        return baseline
