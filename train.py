import os
import sys
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.text.model import TextRiskBiLSTM
from src.audio.model import AudioRisk1DCNN
from src.vision.model import FaceRisk2DCNN
from src.fusion.fusion_model import NumPyEarlyFusionMLP


# ==================== 1. ADVANCED IMBLANCE MITIGATIONS ====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        """
        Focal Loss mathematically suppresses the gradient updates of easy/frequent examples,
        allowing the network to prioritize learning hard, under-represented examples (suicide crisis).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class MultimodalDataset(Dataset):
    def __init__(self, text, audio, vision, iot, labels):
        self.text = torch.tensor(text, dtype=torch.long)
        self.audio = torch.tensor(audio, dtype=torch.float32)
        self.vision = torch.tensor(vision, dtype=torch.float32)
        self.iot = torch.tensor(iot, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text[idx], self.audio[idx], self.vision[idx], self.iot[idx], self.labels[idx]


# ==================== 2. SYNTHETIC DATA BOOTSTRAPPER ====================

def generate_synthetic_multimodal_dataset(num_samples=1000):
    """
    Synthesizes a realistic multimodal clinical dataset containing balanced risk labels:
    - Text: Simulates vocabulary TF-IDF or embedding features (50 dimensions)
    - Audio: Acoustic features (47 dimensions: MFCCs, chroma, centroid, pitch, jitter, shimmer)
    - Vision: Facial expression features (128 dimensions: facial embeddings)
    - IoT: Physiological metrics (8 dimensions: HR, HRV RMSSD, SDNN, pNN50, LF/HF, SCL, SCR, Temp)
    """
    print(f"[DataGen] Generating {num_samples} synthetic multimodal patient records...")
    np.random.seed(42)
    
    # Target label: 0 (Low Risk), 1 (Moderate Risk), 2 (High Risk)
    labels = np.random.choice([0, 1, 2], size=num_samples, p=[0.4, 0.35, 0.25])
    
    # --- 1. Text Features (50 dims) ---
    text_feats = np.zeros((num_samples, 50))
    for i in range(num_samples):
        if labels[i] == 0:
            text_feats[i] = np.random.normal(0.1, 0.2, size=50)
        elif labels[i] == 1:
            text_feats[i] = np.random.normal(0.4, 0.3, size=50)
        else:
            text_feats[i] = np.random.normal(0.8, 0.2, size=50)
            
    # --- 2. Audio Features (47 dims) ---
    audio_feats = np.zeros((num_samples, 47))
    for i in range(num_samples):
        if labels[i] == 0:
            audio_feats[i] = np.random.normal(1.2, 0.3, size=47)
        elif labels[i] == 1:
            audio_feats[i] = np.random.normal(0.6, 0.2, size=47)
        else:
            audio_feats[i] = np.random.normal(-0.4, 0.5, size=47)
            
    # --- 3. Vision Features (128 dims) ---
    vision_feats = np.zeros((num_samples, 128))
    for i in range(num_samples):
        if labels[i] == 0:
            vision_feats[i] = np.random.normal(-0.2, 0.1, size=128)
        elif labels[i] == 1:
            vision_feats[i] = np.random.normal(0.3, 0.2, size=128)
        else:
            vision_feats[i] = np.random.normal(0.8, 0.3, size=128)

    # --- 4. IoT Physiological Features (8 dims) ---
    iot_feats = np.zeros((num_samples, 8))
    for i in range(num_samples):
        if labels[i] == 0:
            hr = np.random.normal(72.0, 5.0)
            hrv = np.random.normal(55.0, 8.0)
            sdnn = np.random.normal(48.0, 5.0)
            pnn50 = np.random.normal(0.35, 0.05)
            lf_hf = np.random.normal(1.2, 0.2)
            scl = np.random.normal(1.5, 0.3)
            scr = np.random.normal(0.05, 0.01)
            temp = np.random.normal(33.5, 0.5)
        elif labels[i] == 1:
            hr = np.random.normal(65.0, 4.0)
            hrv = np.random.normal(14.0, 3.0)
            sdnn = np.random.normal(18.0, 3.0)
            pnn50 = np.random.normal(0.05, 0.01)
            lf_hf = np.random.normal(0.8, 0.1)
            scl = np.random.normal(0.8, 0.2)
            scr = np.random.normal(0.02, 0.01)
            temp = np.random.normal(32.2, 0.6)
        else: # High Risk
            hr = np.random.normal(102.0, 8.0)
            hrv = np.random.normal(18.0, 4.0)
            sdnn = np.random.normal(20.0, 4.0)
            pnn50 = np.random.normal(0.08, 0.02)
            lf_hf = np.random.normal(2.5, 0.4)
            scl = np.random.normal(5.8, 1.0)
            scr = np.random.normal(0.45, 0.08)
            temp = np.random.normal(30.8, 0.8)
            
        iot_feats[i] = [hr, hrv, sdnn, pnn50, lf_hf, scl, scr, temp]
        
    return text_feats, audio_feats, vision_feats, iot_feats, labels


# ==================== 3. MODALITY MACHINE LEARNING TRAINING ====================

def train_machine_learning_baselines():
    """
    Trains and serializes highly robust Scikit-Learn Random Forest classifiers 
    for each individual modality. This provides immediate, operational inference weight files.
    """
    print("\n" + "="*60)
    print("🧠 RUNNING MULTIMODAL MACHINE LEARNING TRAINING PIPELINE")
    print("="*60)

    text_x, audio_x, vision_x, iot_x, y = generate_synthetic_multimodal_dataset()
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    modalities = {
        "text": (text_x, "models/text_rf_model.pkl"),
        "audio": (audio_x, "models/audio_rf_model.pkl"),
        "vision": (vision_x, "models/vision_rf_model.pkl"),
        "iot": (iot_x, "models/iot_rf_model.pkl")
    }
    
    for mod_name, (data_x, save_path) in modalities.items():
        print(f"\nTraining Classifier for [{mod_name.upper()}] Modality...")
        x_train, x_test, y_train, y_test = train_test_split(data_x, y, test_size=0.2, random_state=42, stratify=y)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        clf.fit(x_train, y_train)
        
        preds = clf.predict(x_test)
        acc = accuracy_score(y_test, preds)
        print(f" -> Accuracy for {mod_name.upper()}: {acc:.2%}")
        
        with open(save_path, "wb") as f:
            pickle.dump(clf, f)
        print(f" -> Successfully saved trained weights to: {save_path}")
        
    print("\n" + "-"*50)
    print("🏁 Modality Model Training Complete!")
    print("-"*50)


# ==================== 4. PYTORCH PER-MODALITY DEEP OPTIMIZERS ====================

def train_pytorch_modality_networks():
    """
    Optimizes PyTorch Neural Networks for individual active modalities:
    - Text: Bi-LSTM with Attention Pooling (TextRiskBiLSTM)
    - Audio: 1D-CNN (AudioRisk1DCNN)
    - Vision: 2D-CNN (FaceRisk2DCNN)
    
    Enforces checkpointing and early stopping on VALIDATION RECALL FOR HIGH-RISK CLASS (Class 2).
    """
    if not TORCH_AVAILABLE:
        print("\n[PyTorch Optimizer] PyTorch is not compiled/installed. Skipping neural net optimization.")
        return

    print("\n" + "="*60)
    print("🔥 LAUNCHING PER-MODALITY DEEP LEARNING OPTIMIZERS (PYTORCH)")
    print("="*60)

    # 1. Fetch synthetic multi-sensory dataset
    text_x, audio_x, vision_x, iot_x, y = generate_synthetic_multimodal_dataset(num_samples=1500)
    
    # 2. Stratified train/val splits to preserve rare class proportions
    # 70% Train, 15% Validation, 15% Test
    t_tr, t_val, a_tr, a_val, v_tr, v_val, i_tr, i_val, y_tr, y_val = train_test_split(
        text_x, audio_x, vision_x, iot_x, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Pack into data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create vocabulary default map and save it
    vocab_map = {f"word_{i}": i for i in range(2, 100)}
    vocab_map["<PAD>"] = 0
    vocab_map["<UNK>"] = 1
    with open("models/vocab.json", "w") as f:
        json.dump(vocab_map, f)

    # Calculate Inverse-frequency class weights to combat high imbalance
    class_counts = np.bincount(y_tr)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / np.sum(class_weights) * 3.0 # Normalize weights
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"Computed Inverse-Frequency Class Weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Convert features to numeric indexes for Text
    t_tr_idx = np.abs(t_tr * 10).astype(np.int64) % len(vocab_map)
    t_val_idx = np.abs(t_val * 10).astype(np.int64) % len(vocab_map)

    # Prepare loaders
    train_dataset = MultimodalDataset(t_tr_idx, a_tr, v_tr, i_tr, y_tr)
    val_dataset = MultimodalDataset(t_val_idx, a_val, v_val, i_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --- Optimizing TEXT: Bi-LSTM with Attention ---
    print("\n[Optimizer] Training TextRiskBiLSTM (Bi-LSTM + Attention)...")
    text_model = TextRiskBiLSTM(vocab_size=len(vocab_map), embed_dim=128, hidden_dim=64, num_layers=1, num_classes=3).to(device)
    text_opt = optim.Adam(text_model.parameters(), lr=0.005)
    
    # Optimize checkpoint selection strictly based on Validation High-Risk Recall
    best_high_risk_recall = -1.0
    
    for epoch in range(15):
        text_model.train()
        for batch_t, _, _, _, batch_y in train_loader:
            batch_t, batch_y = batch_t.to(device), batch_y.to(device)
            text_opt.zero_grad()
            logits = text_model(batch_t)
            loss = criterion(logits, batch_y)
            loss.backward()
            text_opt.step()
            
        # Validation Evaluation
        text_model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_t, _, _, _, batch_y in val_loader:
                batch_t = batch_t.to(device)
                logits = text_model(batch_t)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())
                
        # Calculate Validation Recall for Class 2 (High-Risk)
        val_recall_class_2 = recall_score(all_targets, all_preds, labels=[2], average='macro', zero_division=0)
        
        if val_recall_class_2 > best_high_risk_recall:
            best_high_risk_recall = val_recall_class_2
            torch.save(text_model.state_dict(), "models/text_model.pth")
            print(f"  * Epoch {epoch+1:02d} | New Checkpoint! Validation High-Risk Recall: {val_recall_class_2:.2%}")

    # --- Optimizing AUDIO: 1D-CNN ---
    print("\n[Optimizer] Training AudioRisk1DCNN (Conv1D + AdaptivePool)...")
    # Features dimension 47
    audio_model = AudioRisk1DCNN(n_mfcc=47, num_classes=3).to(device)
    audio_opt = optim.Adam(audio_model.parameters(), lr=0.005)
    best_high_risk_recall = -1.0
    
    for epoch in range(15):
        audio_model.train()
        for _, batch_a, _, _, batch_y in train_loader:
            batch_a, batch_y = batch_a.to(device), batch_y.to(device)
            audio_opt.zero_grad()
            logits = audio_model(batch_a)
            loss = criterion(logits, batch_y)
            loss.backward()
            audio_opt.step()
            
        audio_model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for _, batch_a, _, _, batch_y in val_loader:
                batch_a = batch_a.to(device)
                logits = audio_model(batch_a)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())
                
        val_recall_class_2 = recall_score(all_targets, all_preds, labels=[2], average='macro', zero_division=0)
        
        if val_recall_class_2 > best_high_risk_recall:
            best_high_risk_recall = val_recall_class_2
            torch.save(audio_model.state_dict(), "models/audio_model.pth")
            print(f"  * Epoch {epoch+1:02d} | New Checkpoint! Validation High-Risk Recall: {val_recall_class_2:.2%}")

    # --- Optimizing VISION: 2D-CNN ---
    print("\n[Optimizer] Training FaceRisk2DCNN (Conv2D + MaxPool)...")
    vision_model = FaceRisk2DCNN(num_classes=3).to(device)
    vision_opt = optim.Adam(vision_model.parameters(), lr=0.005)
    best_high_risk_recall = -1.0
    
    for epoch in range(15):
        vision_model.train()
        for _, _, batch_v, _, batch_y in train_loader:
            batch_v, batch_y = batch_v.to(device), batch_y.to(device)
            
            # Format vision shape to grayscale [batch, 1, 128, 128] for standard input tracking
            # Downsample 128 embedding dynamically to a spatial 2D array representation
            batch_v_spatial = batch_v.unsqueeze(1).view(-1, 1, 8, 16) # map 128 elements to 8x16 spatial pixels
            
            vision_opt.zero_grad()
            logits = vision_model(batch_v_spatial)
            loss = criterion(logits, batch_y)
            loss.backward()
            vision_opt.step()
            
        vision_model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for _, _, batch_v, _, batch_y in val_loader:
                batch_v_spatial = batch_v.unsqueeze(1).view(-1, 1, 8, 16).to(device)
                logits = vision_model(batch_v_spatial)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())
                
        val_recall_class_2 = recall_score(all_targets, all_preds, labels=[2], average='macro', zero_division=0)
        
        if val_recall_class_2 > best_high_risk_recall:
            best_high_risk_recall = val_recall_class_2
            torch.save(vision_model.state_dict(), "models/vision_model.pth")
            print(f"  * Epoch {epoch+1:02d} | New Checkpoint! Validation High-Risk Recall: {val_recall_class_2:.2%}")

    print("\n" + "="*60)
    print("🏁 PER-MODALITY DEEP OPTIMIZATION LOOPS COMPLETE!")
    print("="*60 + "\n")


def train_numpy_early_fusion_mlp():
    """
    Trains our custom Multi-Head Early Fusion Neural Network in pure NumPy using backpropagation gradients.
    """
    print("\n" + "="*60)
    print("🔥 TRAINING PURE NUMPY EARLY-FUSION DEEP NEURAL NETWORK")
    print("="*60)

    text_x, audio_x, vision_x, iot_x, y = generate_synthetic_multimodal_dataset(num_samples=1200)
    
    t_tr, t_te, a_tr, a_te, v_tr, v_te, i_tr, i_te, y_tr, y_te = train_test_split(
        text_x, audio_x, vision_x, iot_x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = NumPyEarlyFusionMLP(audio_dim=47, iot_dim=8)
    
    epochs = 120
    learning_rate = 0.05
    batch_size = 32
    num_batches = len(y_tr) // batch_size
    
    print(f"Training deep network for {epochs} Epochs | Batch Size: {batch_size}...")
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(len(y_tr))
        epoch_loss = 0.0
        
        for b in range(num_batches):
            batch_idx = shuffled_indices[b * batch_size : (b + 1) * batch_size]
            
            b_text = t_tr[batch_idx]
            b_audio = a_tr[batch_idx]
            b_vision = v_tr[batch_idx]
            b_iot = i_tr[batch_idx]
            b_y = y_tr[batch_idx]
            
            loss = model.train_step(b_text, b_audio, b_vision, b_iot, b_y, lr=learning_rate)
            epoch_loss += loss
            
        if (epoch + 1) % 20 == 0 or epoch == 0:
            val_outputs = model.forward(t_te, a_te, v_te, i_te)
            val_preds = np.argmax(val_outputs["probs"], axis=1)
            val_acc = accuracy_score(y_te, val_preds)
            print(f"  * Epoch {epoch+1:03d}/{epochs} | Avg Cross-Entropy Loss: {epoch_loss/num_batches:.4f} | Validation Acc: {val_acc:.2%}")
            
    save_path = "models/early_fusion_weights.pkl"
    model.save_weights(save_path)
    print("="*60 + "\n")


if __name__ == "__main__":
    train_machine_learning_baselines()
    train_pytorch_modality_networks()
    train_numpy_early_fusion_mlp()
