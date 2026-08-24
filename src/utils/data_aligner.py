import os
import json
import numpy as np
from typing import Dict, List, Any, Optional


class CrossModalRiskTaxonomy:
    """
    Defines the unified 3-tier cross-modal clinical risk taxonomy:
    0: Low Risk (Calm, baseline, healthy positive affect)
    1: Moderate Risk (Distress, sadness, physiological somatic stress)
    2: High Risk (Severe distress, acute crisis, panic, hyperarousal, suicide ideation)
    """
    LOW_RISK = 0
    MODERATE_RISK = 1
    HIGH_RISK = 2

    # --- 1. Text Taxonomy Map (Reddit SuicideWatch / CLPsych) ---
    TEXT_MAP = {
        "non-suicidal": LOW_RISK,
        "neutral": LOW_RISK,
        "safe": LOW_RISK,
        "depressed": MODERATE_RISK,
        "hopeless": MODERATE_RISK,
        "anxious": MODERATE_RISK,
        "suicide": HIGH_RISK,
        "crisis": HIGH_RISK,
        "suicidal": HIGH_RISK
    }

    # --- 2. Audio Taxonomy Map (RAVDESS / SAVEE / IEMOCAP) ---
    AUDIO_MAP = {
        "calm": LOW_RISK,
        "happy": LOW_RISK,
        "neutral": LOW_RISK,
        "surprise": LOW_RISK,
        "sad": MODERATE_RISK,
        "disgust": MODERATE_RISK,
        "fearful": HIGH_RISK,
        "angry": HIGH_RISK
    }

    # --- 3. Vision Taxonomy Map (FER2013 / CK+) ---
    VISION_MAP = {
        "happy": LOW_RISK,
        "neutral": LOW_RISK,
        "surprise": LOW_RISK,
        "sad": MODERATE_RISK,
        "disgust": MODERATE_RISK,
        "fear": HIGH_RISK,
        "angry": HIGH_RISK
    }

    # --- 4. IoT Physiological Map (WESAD wearable sensors) ---
    IoT_MAP = {
        "baseline": LOW_RISK,
        "meditation": LOW_RISK,
        "amusement": LOW_RISK,
        "stress": MODERATE_RISK,
        "anxiety_panic": HIGH_RISK,
        "depressive_flat": HIGH_RISK
    }

    @classmethod
    def align_label(cls, modality: str, raw_label: str) -> int:
        """
        Translates a raw heterogeneous label from a specific modality dataset
        into our unified clinical risk taxonomy score (0, 1, or 2).
        """
        mod = modality.lower().strip()
        raw = str(raw_label).lower().strip()

        if mod == "text":
            return cls.TEXT_MAP.get(raw, cls.LOW_RISK)
        elif mod == "audio":
            return cls.AUDIO_MAP.get(raw, cls.LOW_RISK)
        elif mod == "vision":
            return cls.VISION_MAP.get(raw, cls.LOW_RISK)
        elif mod == "iot":
            return cls.IoT_MAP.get(raw, cls.LOW_RISK)
        else:
            raise ValueError(f"Unknown modality: {modality}. Must be 'text', 'audio', 'vision', or 'iot'.")


class MultimodalDataIndexAligner:
    def __init__(self, data_root="data"):
        self.data_root = data_root
        self.processed_dir = os.path.join(data_root, "processed")
        self.metadata_path = os.path.join(self.processed_dir, "multimodal_aligned_index.json")
        os.makedirs(self.processed_dir, exist_ok=True)

    def generate_aligned_metadata_entry(self, 
                                        patient_id: str, 
                                        text_raw_label: str,
                                        audio_raw_label: str,
                                        vision_raw_label: str,
                                        iot_raw_label: str,
                                        text_file_path: Optional[str] = None,
                                        audio_file_path: Optional[str] = None,
                                        vision_file_path: Optional[str] = None,
                                        iot_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates a structured, aligned data dictionary metadata entry representing
        a single patient/subject session across all four modalities.
        """
        text_risk = CrossModalRiskTaxonomy.align_label("text", text_raw_label)
        audio_risk = CrossModalRiskTaxonomy.align_label("audio", audio_raw_label)
        vision_risk = CrossModalRiskTaxonomy.align_label("vision", vision_raw_label)
        iot_risk = CrossModalRiskTaxonomy.align_label("iot", iot_raw_label)

        # Compute consensus clinical label
        cons_scores = [text_risk, audio_risk, vision_risk, iot_risk]
        fused_consensus_risk = int(np.round(np.mean(cons_scores)))

        return {
            "patient_id": patient_id,
            "fused_consensus_risk": fused_consensus_risk,
            "modalities": {
                "text": {
                    "raw_dataset_label": text_raw_label,
                    "aligned_risk_score": text_risk,
                    "file_path": text_file_path or ""
                },
                "audio": {
                    "raw_dataset_label": audio_raw_label,
                    "aligned_risk_score": audio_risk,
                    "file_path": audio_file_path or ""
                },
                "vision": {
                    "raw_dataset_label": vision_raw_label,
                    "aligned_risk_score": vision_risk,
                    "file_path": vision_file_path or ""
                },
                "iot": {
                    "raw_dataset_label": iot_raw_label,
                    "aligned_risk_score": iot_risk,
                    "file_path": iot_file_path or ""
                }
            }
        }

    # ==================== SHARED FEATURE STORE ====================

    def save_session_features(self, 
                              session_id: str, 
                              text_feat: Optional[np.ndarray], 
                              audio_feat: Optional[np.ndarray], 
                              vision_feat: Optional[np.ndarray], 
                              iot_feat: Optional[np.ndarray], 
                              aligned_label: int):
        """
        Saves individual modality numpy features as a compressed, serialized .npz file 
        under the centralized processed Feature Store.
        """
        save_path = os.path.join(self.processed_dir, f"{session_id}_features.npz")
        
        # Save array mapping dictionaries (pack missing streams as empty zero vectors)
        np.savez_compressed(
            save_path,
            text=text_feat if text_feat is not None else np.zeros((50,)),
            audio=audio_feat if audio_feat is not None else np.zeros((47,)),
            vision=vision_feat if vision_feat is not None else np.zeros((128,)),
            iot=iot_feat if iot_feat is not None else np.zeros((8,)),
            label=np.array([aligned_label], dtype=np.int32)
        )
        print(f"[FeatureStore] Saved aligned features for session {session_id} to: {save_path}")

    def load_session_features(self, session_id: str) -> Dict[str, Any]:
        """Loads serialized feature tensors from the compressed Feature Store."""
        load_path = os.path.join(self.processed_dir, f"{session_id}_features.npz")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Feature file not found in Feature Store: {load_path}")
            
        data = np.load(load_path)
        return {
            "text": data["text"],
            "audio": data["audio"],
            "vision": data["vision"],
            "iot": data["iot"],
            "label": int(data["label"][0])
        }

    def save_aligned_indexing_ledger(self, aligned_records: List[Dict[str, Any]]):
        """Saves a master JSON index ledger aligning all dataset files to the same index taxonomy."""
        with open(self.metadata_path, "w") as f:
            json.dump(aligned_records, f, indent=4)
        print(f"[DataAligner] Saved aligned multi-modal index ledger to: {self.metadata_path}")
