import time
import numpy as np
from typing import Dict, Any, Optional
from src.fusion.fusion_model import MultimodalDecisionFusion


class TelemetryStreamManager:
    def __init__(self, 
                 fusion_engine: Optional[MultimodalDecisionFusion] = None,
                 text_ttl_sec: float = 300.0,    # Text state lasts 5 mins
                 audio_ttl_sec: float = 600.0,   # Voice check-in state lasts 10 mins
                 vision_ttl_sec: float = 120.0,  # Facial crop state lasts 2 mins
                 iot_ttl_sec: float = 30.0):     # Smartwatch metrics expire in 30 seconds
        """
        An active stream manager that integrates asynchronous, multi-sensor telemetry packets.
        Implements circular buffer queues and mathematical exponential time-decay for fusion weights.
        """
        self.fusion_engine = fusion_engine if fusion_engine is not None else MultimodalDecisionFusion()
        
        # Modality Time-To-Live (TTL) parameters
        self.ttls = {
            "text": text_ttl_sec,
            "audio": audio_ttl_sec,
            "vision": vision_ttl_sec,
            "iot": iot_ttl_sec
        }

        # Active telemetry state-buffer stores: (latest_prediction_dict, raw_feature_array, timestamp)
        self.buffer: Dict[str, Dict[str, Any]] = {
            "text": {"res": None, "feat": None, "timestamp": 0.0},
            "audio": {"res": None, "feat": None, "timestamp": 0.0},
            "vision": {"res": None, "feat": None, "timestamp": 0.0},
            "iot": {"res": None, "feat": None, "timestamp": 0.0}
        }

    def ingest_packet(self, modality: str, prediction_res: Dict[str, Any], raw_feature_vector: np.ndarray):
        """
        Ingests a single asynchronous sensor packet on-the-fly and writes it to the active buffer.
        """
        mod = modality.lower().strip()
        if mod not in self.buffer:
            raise ValueError(f"Unknown modality: {modality}. Must be 'text', 'audio', 'vision', or 'iot'.")
            
        self.buffer[mod]["res"] = prediction_res
        self.buffer[mod]["feat"] = raw_feature_vector
        self.buffer[mod]["timestamp"] = time.time()
        print(f"[StreamManager] Ingested fresh packet for [{mod.upper()}] at timestamp {self.buffer[mod]['timestamp']:.2f}")

    def calculate_temporal_decay_weight(self, modality: str, current_time: float) -> float:
        """
        Computes the mathematical exponential temporal weight decay factor:
        w_t = exp(-t_delta / TTL)
        If the sensor goes silent, its fusion significance decays from 1.0 to 0.0.
        """
        mod = modality.lower()
        last_timestamp = self.buffer[mod]["timestamp"]
        
        if last_timestamp == 0.0:
            return 0.0 # No packet has ever been ingested
            
        t_delta = current_time - last_timestamp
        ttl = self.ttls[mod]
        
        if t_delta >= ttl:
            return 0.0 # Packet has completely expired
            
        # Exponential decay calculation
        decay_factor = np.exp(-t_delta / (ttl / 2.0)) # (reaches ~13% at half-life decay)
        return float(decay_factor)

    def get_active_realtime_fusion(self, fusion_method: str = "Late Decision Fusion") -> Dict[str, Any]:
        """
        Gathers buffered data, evaluates sensor temporal decay states, 
        and executes real-time cross-modal fusion.
        
        Dynamically filters out completely expired packets and passes active modalities
        to the fusion engine, automatically re-adjusting fusion weights.
        """
        now = time.time()
        active_text, active_audio, active_vision, active_iot = None, None, None, None
        raw_text_x, raw_audio_x, raw_vision_x, raw_iot_x = None, None, None, None
        
        active_modalities = []
        expired_modalities = []
        
        # 1. Evaluate temporal decay states
        for mod in ["text", "audio", "vision", "iot"]:
            decay_weight = self.calculate_temporal_decay_weight(mod, now)
            
            if decay_weight > 0.0:
                # Modality is active!
                active_modalities.append(f"{mod} (decay: {decay_weight:.1%})")
                
                # Retrieve buffered calculations
                res = self.buffer[mod]["res"]
                feat = self.buffer[mod]["feat"]
                
                if mod == "text":
                    active_text = res
                    raw_text_x = feat
                elif mod == "audio":
                    active_audio = res
                    raw_audio_x = feat
                elif mod == "vision":
                    active_vision = res
                    raw_vision_x = feat
                elif mod == "iot":
                    active_iot = res
                    raw_iot_x = feat
            else:
                if self.buffer[mod]["timestamp"] > 0.0:
                    expired_modalities.append(mod)

        # 2. Run Decision-level or Feature-level fusion on active sensors
        print(f"[StreamManager] Active sensors: {', '.join(active_modalities)}")
        if expired_modalities:
            print(f"[StreamManager] Expired sensors (silent past TTL): {', '.join(expired_modalities)}")
            
        fused_report = self.fusion_engine.fuse_predictions(
            text_res=active_text,
            audio_res=active_audio,
            vision_res=active_vision,
            iot_res=active_iot,
            fusion_method=fusion_method,
            raw_text_x=raw_text_x,
            raw_audio_x=raw_audio_x,
            raw_vision_x=raw_vision_x,
            raw_iot_x=raw_iot_x
        )
        
        # Add metadata telemetry audit trail
        fused_report["stream_metadata"] = {
            "fusion_method": fusion_method,
            "active_modality_decays": {
                mod: self.calculate_temporal_decay_weight(mod, now) for mod in ["text", "audio", "vision", "iot"]
            },
            "timestamp": now
        }
        
        return fused_report
