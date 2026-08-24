import os
import pickle
import numpy as np
from typing import Dict, List, Optional


class NumPyEarlyFusionMLP:
    def __init__(self, text_dim=50, audio_dim=47, vision_dim=128, iot_dim=8, latent_dim=16, hidden_dim=32, num_classes=3):
        """
        A complete, multi-head Early Fusion Deep Neural Network built entirely from scratch in NumPy.
        Projects raw sensory inputs into a joint latent space, concatenates them, and classifies risk.
        """
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim
        self.iot_dim = iot_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Initialize network parameters using Xavier/Glorot Normal initialization
        np.random.seed(42)
        
        # --- Multi-Head Projections (Linear layers + biases) ---
        self.W_text = np.random.randn(text_dim, latent_dim) * np.sqrt(2.0 / (text_dim + latent_dim))
        self.b_text = np.zeros((1, latent_dim))
        
        self.W_audio = np.random.randn(audio_dim, latent_dim) * np.sqrt(2.0 / (audio_dim + latent_dim))
        self.b_audio = np.zeros((1, latent_dim))
        
        self.W_vision = np.random.randn(vision_dim, latent_dim) * np.sqrt(2.0 / (vision_dim + latent_dim))
        self.b_vision = np.zeros((1, latent_dim))
        
        self.W_iot = np.random.randn(iot_dim, latent_dim) * np.sqrt(2.0 / (iot_dim + latent_dim))
        self.b_iot = np.zeros((1, latent_dim))

        # --- Classification Head (64 -> 32 -> 3) ---
        fused_dim = latent_dim * 4 # 16 * 4 = 64
        self.W_h1 = np.random.randn(fused_dim, hidden_dim) * np.sqrt(2.0 / (fused_dim + hidden_dim))
        self.b_h1 = np.zeros((1, hidden_dim))
        
        self.W_out = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / (hidden_dim + num_classes))
        self.b_out = np.zeros((1, num_classes))

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability (prevent overflow)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, text_x: np.ndarray, audio_x: np.ndarray, vision_x: np.ndarray, iot_x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward Pass:
        1. Project each sensor modality to a shared 16-dimensional embedding.
        2. Concatenate embeddings into a 64-dimensional fused joint feature representation (Early Fusion).
        3. Classify through the MLP layers.
        """
        # Multi-Head projections
        z_text = np.dot(text_x, self.W_text) + self.b_text
        z_audio = np.dot(audio_x, self.W_audio) + self.b_audio
        z_vision = np.dot(vision_x, self.W_vision) + self.b_vision
        z_iot = np.dot(iot_x, self.W_iot) + self.b_iot
        
        # Non-linear activations for latent spaces
        a_text = self.relu(z_text)
        a_audio = self.relu(z_audio)
        a_vision = self.relu(z_vision)
        a_iot = self.relu(z_iot)
        
        # EARLY FUSION: Concatenate all latent features along the horizontal axis
        # Shape: (batch_size, 64)
        fused = np.hstack((a_text, a_audio, a_vision, a_iot))
        
        # Classification layer 1
        z_h1 = np.dot(fused, self.W_h1) + self.b_h1
        a_h1 = self.relu(z_h1)
        
        # Output layer
        z_out = np.dot(a_h1, self.W_out) + self.b_out
        probs = self.softmax(z_out)
        
        return {
            "a_text": a_text, "a_audio": a_audio, "a_vision": a_vision, "a_iot": a_iot,
            "fused": fused, "z_h1": z_h1, "a_h1": a_h1, "probs": probs
        }

    def train_step(self, text_x: np.ndarray, audio_x: np.ndarray, vision_x: np.ndarray, iot_x: np.ndarray, y: np.ndarray, lr=0.01) -> float:
        """
        Custom Backpropagation Step:
        Computes gradients of Cross-Entropy Loss with respect to all projection heads 
        and classification weight matrices, and updates them using gradient descent.
        """
        batch_size = text_x.shape[0]
        
        # 1. Forward Pass
        f_cache = self.forward(text_x, audio_x, vision_x, iot_x)
        probs = f_cache["probs"]
        fused = f_cache["fused"]
        a_h1 = f_cache["a_h1"]
        z_h1 = f_cache["z_h1"]
        
        # Compute Categorical Cross Entropy Loss
        loss = -np.mean(np.log(probs[np.arange(batch_size), y] + 1e-15))
        
        # 2. Backpropagation gradients
        # Gradient of loss with respect to output logits (z_out)
        d_z_out = probs.copy()
        d_z_out[np.arange(batch_size), y] -= 1.0
        d_z_out = d_z_out / batch_size # Normalize by batch size
        
        # Gradients for Output layer weights
        dW_out = np.dot(a_h1.T, d_z_out)
        db_out = np.sum(d_z_out, axis=0, keepdims=True)
        
        # Gradient back-propagated to hidden layer 1
        d_a_h1 = np.dot(d_z_out, self.W_out.T)
        d_z_h1 = d_a_h1 * self.relu_derivative(z_h1)
        
        # Gradients for Hidden 1 weights
        dW_h1 = np.dot(fused.T, d_z_h1)
        db_h1 = np.sum(d_z_h1, axis=0, keepdims=True)
        
        # Gradient back-propagated to fused joint feature vector
        d_fused = np.dot(d_z_h1, self.W_h1.T)
        
        # Split gradients back into individual modality projection heads
        d_a_text = d_fused[:, 0:16]
        d_a_audio = d_fused[:, 16:32]
        d_a_vision = d_fused[:, 32:48]
        d_a_iot = d_fused[:, 48:64]
        
        # Modality 1: Text
        d_z_text = d_a_text * self.relu_derivative(f_cache["a_text"])
        dW_text = np.dot(text_x.T, d_z_text)
        db_text = np.sum(d_z_text, axis=0, keepdims=True)
        
        # Modality 2: Audio
        d_z_audio = d_a_audio * self.relu_derivative(f_cache["a_audio"])
        dW_audio = np.dot(audio_x.T, d_z_audio)
        db_audio = np.sum(d_z_audio, axis=0, keepdims=True)
        
        # Modality 3: Vision
        d_z_vision = d_a_vision * self.relu_derivative(f_cache["a_vision"])
        dW_vision = np.dot(vision_x.T, d_z_vision)
        db_vision = np.sum(d_z_vision, axis=0, keepdims=True)
        
        # Modality 4: IoT
        d_z_iot = d_a_iot * self.relu_derivative(f_cache["a_iot"])
        dW_iot = np.dot(iot_x.T, d_z_iot)
        db_iot = np.sum(d_z_iot, axis=0, keepdims=True)
        
        # 3. Update network parameters using standard Stochastic Gradient Descent (SGD)
        self.W_text -= lr * dW_text
        self.b_text -= lr * db_text
        self.W_audio -= lr * dW_audio
        self.b_audio -= lr * db_audio
        self.W_vision -= lr * dW_vision
        self.b_vision -= lr * db_vision
        self.W_iot -= lr * dW_iot
        self.b_iot -= lr * db_iot
        
        self.W_h1 -= lr * dW_h1
        self.b_h1 -= lr * db_h1
        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out
        
        return loss

    def save_weights(self, file_path: str):
        """Serializes weight matrices and biases into a portable pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        state_dict = {
            "W_text": self.W_text, "b_text": self.b_text,
            "W_audio": self.W_audio, "b_audio": self.b_audio,
            "W_vision": self.W_vision, "b_vision": self.b_vision,
            "W_iot": self.W_iot, "b_iot": self.b_iot,
            "W_h1": self.W_h1, "b_h1": self.b_h1,
            "W_out": self.W_out, "b_out": self.b_out
        }
        with open(file_path, "wb") as f:
            pickle.dump(state_dict, f)
        print(f"[EarlyFusionMLP] Serialized trained network to: {file_path}")

    def load_weights(self, file_path: str) -> bool:
        """Loads serialized weight matrices and biases."""
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, "rb") as f:
                state_dict = pickle.load(f)
            self.W_text = state_dict["W_text"]
            self.b_text = state_dict["b_text"]
            self.W_audio = state_dict["W_audio"]
            self.b_audio = state_dict["b_audio"]
            self.W_vision = state_dict["W_vision"]
            self.b_vision = state_dict["b_vision"]
            self.W_iot = state_dict["W_iot"]
            self.b_iot = state_dict["b_iot"]
            
            self.W_h1 = state_dict["W_h1"]
            self.b_h1 = state_dict["b_h1"]
            self.W_out = state_dict["W_out"]
            self.b_out = state_dict["b_out"]
            print(f"[EarlyFusionMLP] Loaded weights from {file_path}")
            return True
        except Exception as e:
            print(f"[EarlyFusionMLP] Failed to load weights: {e}")
            return False


class MultimodalDecisionFusion:
    def __init__(self, early_fusion_path: str = "models/early_fusion_weights.pkl"):
        # Late Fusion weights
        self.default_weights = {
            "text": 0.40,
            "iot": 0.25,
            "audio": 0.20,
            "vision": 0.15
        }
        
        # Instantiate and load the pure NumPy Early Fusion Deep Neural Network
        self.early_fusion_model = NumPyEarlyFusionMLP()
        self.early_fusion_loaded = self.early_fusion_model.load_weights(early_fusion_path)

    def fuse_predictions(self, 
                         text_res: Optional[Dict[str, any]] = None,
                         audio_res: Optional[Dict[str, any]] = None,
                         vision_res: Optional[Dict[str, any]] = None,
                         iot_res: Optional[Dict[str, any]] = None,
                         fusion_method: str = "Late Decision Fusion",
                         raw_text_x: Optional[np.ndarray] = None,
                         raw_audio_x: Optional[np.ndarray] = None,
                         raw_vision_x: Optional[np.ndarray] = None,
                         raw_iot_x: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Cross-modal fusion layer supporting:
        1. 'Late Decision Fusion' (weighted averages of individual model outcomes)
        2. 'Early Feature Fusion' (joint feature projections via NumPy Deep MLP model)
        """
        # SCENARIO A: Early Feature Fusion (requires all four sensory heads to be present)
        if fusion_method == "Early Feature Fusion" and self.early_fusion_loaded:
            # Fallback if any features are missing: generate empty zero features
            t_feat = raw_text_x if raw_text_x is not None else np.zeros((1, 50))
            a_feat = raw_audio_x if raw_audio_x is not None else np.zeros((1, 39))
            v_feat = raw_vision_x if raw_vision_x is not None else np.zeros((1, 128))
            i_feat = raw_iot_x if raw_iot_x is not None else np.zeros((1, 4))
            
            try:
                outputs = self.early_fusion_model.forward(t_feat, a_feat, v_feat, i_feat)
                probs = outputs["probs"][0] # Shape: (3,)
                
                risk_classes = ["Low Risk", "Moderate Risk", "High Risk"]
                pred_idx = np.argmax(probs)
                fused_risk_level = risk_classes[pred_idx]
                fused_risk_score = float(probs[1]*0.4 + probs[2]*0.95) # Scale risk score to [0,1]
                
                # Take dominant emotion from late inputs as linguistic/expressive vote
                dominant_emo = "neutral"
                if text_res:
                    dominant_emo = text_res.get("dominant_emotion", "neutral")
                elif vision_res:
                    dominant_emo = vision_res.get("dominant_emotion", "neutral")
                    
                recommendation = self._generate_recommendation(fused_risk_level)
                
                return {
                    "fused_suicide_risk_level": fused_risk_level,
                    "fused_risk_score": float(np.round(fused_risk_score, 3)),
                    "dominant_state_emotion": dominant_emo,
                    "active_modalities": ["text", "audio", "vision", "iot"],
                    "modality_weights": {"Early Fusion Network": 1.0},
                    "clinical_recommendation": recommendation,
                    "early_fusion_probabilities": {risk_classes[i]: float(probs[i]) for i in range(3)}
                }
            except Exception as e:
                print(f"[DecisionFusion] Early fusion pass failure: {e}. Defaulting to Late Decision Fusion.")

        # SCENARIO B: Late Decision Fusion (Dynamic sensory weight-rebalancing)
        active_inputs = {}
        if text_res is not None:
            active_inputs["text"] = text_res
        if audio_res is not None:
            active_inputs["audio"] = audio_res
        if vision_res is not None:
            active_inputs["vision"] = vision_res
        if iot_res is not None:
            active_inputs["iot"] = iot_res

        if not active_inputs:
            return {
                "fused_suicide_risk_level": "Low Risk",
                "fused_risk_score": 0.0,
                "dominant_state_emotion": "neutral",
                "active_modalities": [],
                "message": "No sensor or linguistic modality data provided."
            }

        total_available_weight = sum(self.default_weights[mod] for mod in active_inputs)
        normalized_weights = {mod: self.default_weights[mod] / total_available_weight for mod in active_inputs}

        fused_risk_score = 0.0
        for mod, res in active_inputs.items():
            mod_weight = normalized_weights[mod]
            
            if mod == "text":
                score = res["risk_probability"]
            elif mod == "audio":
                score = res["distress_probability"]
            elif mod == "vision":
                score = res.get("facial_distress_score", res.get("video_distress_score", 0.0))
            elif mod == "iot":
                score = res["physiological_stress_score"]
            else:
                score = 0.0
                
            fused_risk_score += (score * mod_weight)

        if fused_risk_score >= 0.70:
            fused_risk_level = "High Risk"
        elif fused_risk_score >= 0.35:
            fused_risk_level = "Moderate Risk"
        else:
            fused_risk_level = "Low Risk"

        emotion_votes = {}
        for mod, res in active_inputs.items():
            mod_weight = normalized_weights[mod]
            dom_emo = res.get("dominant_emotion", "neutral")
            emotion_votes[dom_emo] = emotion_votes.get(dom_emo, 0.0) + mod_weight

        fused_dominant_emotion = max(emotion_votes, key=emotion_votes.get) if emotion_votes else "neutral"
        recommendation = self._generate_recommendation(fused_risk_level)

        return {
            "fused_suicide_risk_level": fused_risk_level,
            "fused_risk_score": float(np.round(fused_risk_score, 3)),
            "dominant_state_emotion": fused_dominant_emotion,
            "active_modalities": list(active_inputs.keys()),
            "modality_weights": {k: float(np.round(v, 3)) for k, v in normalized_weights.items()},
            "clinical_recommendation": recommendation
        }

    def _generate_recommendation(self, fused_risk_level: str) -> str:
        if fused_risk_level == "High Risk":
            return (
                "CRITICAL WARNING: The multi-modal analysis indicates an extremely elevated level of psychological and "
                "physiological distress. Immediate outreach, crisis intervention, or contacting emergency professional mental "
                "health services (e.g., local suicide helpline) is strongly advised."
            )
        elif fused_risk_level == "Moderate Risk":
            return (
                "MODERATE CAUTION: Emotional distress and elevated somatic arousal detected. Recommending check-in by "
                "support system, scheduling professional counseling, and close monitoring of behavioral changes."
            )
        else:
            return (
                "NORMAL: Physiological and emotional levels fall within safe baselines. Recommend continuing routine mental wellness "
                "practices and healthy stress management techniques."
            )
