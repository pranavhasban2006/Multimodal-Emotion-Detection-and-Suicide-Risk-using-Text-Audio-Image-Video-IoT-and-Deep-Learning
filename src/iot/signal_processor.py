import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any


class IoTSignalProcessor:
    def __init__(self, model_path: str = "models/iot_rf_model.pkl"):
        self.model_path = model_path
        self.rf_model = None
        self._load_model()

    def _load_model(self):
        """Loads Scikit-Learn RF parameters if present on disk."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.rf_model = pickle.load(f)
                print(f"[IoTProcessor] Loaded trained RF classifier from {self.model_path}")
            except Exception as e:
                print(f"[IoTProcessor] Error loading RF model: {e}")

    # ==================== ADVANCED DSP FEATURE ENGINEERING ====================

    def calculate_hrv_rmssd(self, rr_intervals_ms: List[float]) -> float:
        """Calculates RMSSD (Root Mean Square of Successive Differences)."""
        if len(rr_intervals_ms) < 2:
            return 50.0
        diffs = np.diff(rr_intervals_ms)
        return float(np.sqrt(np.mean(diffs ** 2)))

    def calculate_hrv_sdnn(self, rr_intervals_ms: List[float]) -> float:
        """Calculates SDNN (Standard Deviation of R-R intervals), proxy for overall autonomic variability."""
        if len(rr_intervals_ms) < 2:
            return 50.0
        return float(np.std(rr_intervals_ms))

    def calculate_hrv_pnn50(self, rr_intervals_ms: List[float]) -> float:
        """Calculates pNN50: Percentage of successive interval differences greater than 50 ms."""
        if len(rr_intervals_ms) < 2:
            return 0.0
        diffs = np.abs(np.diff(rr_intervals_ms))
        nn50 = sum(1 for d in diffs if d > 50.0)
        return float(nn50 / len(diffs))

    def calculate_hrv_lf_hf_ratio(self, rr_intervals_ms: List[float], sampling_hz: float = 4.0) -> float:
        """
        Approximates the LF/HF spectral power ratio using Discrete Fourier Transform (DFT).
        """
        if len(rr_intervals_ms) < 10:
            return 1.5

        try:
            mean_rr = np.mean(rr_intervals_ms)
            time_points = np.cumsum(rr_intervals_ms) / 1000.0
            uniform_time = np.arange(time_points[0], time_points[-1], 1.0 / sampling_hz)
            
            uniform_rr = np.interp(uniform_time, time_points, rr_intervals_ms)
            uniform_rr_detrend = uniform_rr - np.mean(uniform_rr)
            
            fft_vals = np.fft.rfft(uniform_rr_detrend)
            fft_freqs = np.fft.rfftfreq(len(uniform_rr_detrend), d=1.0/sampling_hz)
            psd = np.abs(fft_vals) ** 2
            
            lf_mask = (fft_freqs >= 0.04) & (fft_freqs < 0.15)
            hf_mask = (fft_freqs >= 0.15) & (fft_freqs <= 0.40)
            
            lf_power = np.sum(psd[lf_mask])
            hf_power = np.sum(psd[hf_mask])
            
            if hf_power > 0:
                return float(lf_power / hf_power)
        except Exception:
            pass
        return 1.5

    def decompose_eda(self, eda_signal: np.ndarray, sampling_rate_hz: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decomposes Electrodermal Activity (EDA/GSR) into Tonic and Phasic components in pure NumPy.
        Capping window filter dimension at max length of eda_signal to prevent broadcast shape errors.
        """
        if len(eda_signal) < 1:
            return eda_signal, np.zeros_like(eda_signal)
        
        # Enforce that the convolution filter B is smaller or equal to input signal A
        window_len = min(int(8.0 * sampling_rate_hz), len(eda_signal))
        window_len = max(1, window_len) # Ensure positive dimension

        if window_len % 2 == 0 and window_len > 1:
            window_len += 1
            # Adjust back to cap if increased
            window_len = min(window_len, len(eda_signal))
            
        tonic_scl = np.convolve(eda_signal, np.ones(window_len)/window_len, mode='same')
        phasic_scr = eda_signal - tonic_scl
        phasic_scr = np.maximum(0, phasic_scr)
        
        return tonic_scl, phasic_scr

    # ==================== WESAD SUBJECT PICKLE LOADER ====================

    def parse_wesad_subject(self, pickle_path: str, window_size_sec: float = 60.0, overlap_pct: float = 0.5) -> List[Dict[str, Any]]:
        """
        Loads and parses a raw WESAD subject pickle file (e.g. S2.pkl).
        """
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"WESAD pickle file not found at: {pickle_path}")

        print(f"[WESAD] Parsing WESAD raw file: {pickle_path}...")
        with open(pickle_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        wrist_data = data["signal"]["wrist"]
        fs_eda = 4.0
        fs_bvp = 64.0
        fs_temp = 4.0

        raw_eda = wrist_data["EDA"].flatten()
        raw_bvp = wrist_data["BVP"].flatten()
        raw_temp = wrist_data["TEMP"].flatten()
        
        labels_700hz = data["label"].flatten()
        downsample_factor = int(700 / fs_eda)
        labels_4hz = labels_700hz[::downsample_factor][:len(raw_eda)]

        window_samples = int(window_size_sec * fs_eda)
        step_samples = int(window_samples * (1.0 - overlap_pct))
        
        extracted_windows = []
        num_samples = len(raw_eda)

        for start in range(0, num_samples - window_samples, step_samples):
            end = start + window_samples
            
            win_labels = labels_4hz[start:end]
            valid_mask = (win_labels >= 1) & (win_labels <= 4)
            if len(win_labels[valid_mask]) < (window_samples * 0.5):
                continue
                
            majority_raw_label = int(np.bincount(win_labels[valid_mask]).argmax())
            aligned_label = 1 if majority_raw_label == 2 else 0

            win_eda = raw_eda[start:end]
            scl, scr = self.decompose_eda(win_eda, sampling_rate_hz=fs_eda)
            
            bvp_start = int(start * (fs_bvp / fs_eda))
            bvp_end = int(end * (fs_bvp / fs_eda))
            win_bvp = raw_bvp[bvp_start:bvp_end]
            
            peaks = []
            for i in range(1, len(win_bvp) - 1):
                if win_bvp[i] > win_bvp[i-1] and win_bvp[i] > win_bvp[i+1] and win_bvp[i] > 0.05:
                    peaks.append(i)
            
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) * (1000.0 / fs_bvp)
            else:
                rr_intervals = np.array([800.0, 800.0])

            hrv_rmssd = self.calculate_hrv_rmssd(rr_intervals)
            hrv_sdnn = self.calculate_hrv_sdnn(rr_intervals)
            hrv_pnn50 = self.calculate_hrv_pnn50(rr_intervals)
            hrv_lf_hf = self.calculate_hrv_lf_hf_ratio(rr_intervals, sampling_hz=fs_eda)
            
            mean_rr_sec = np.mean(rr_intervals) / 1000.0
            heart_rate = 60.0 / mean_rr_sec if mean_rr_sec > 0 else 72.0

            win_temp = raw_temp[start:end]
            mean_temp = float(np.mean(win_temp))

            features = np.array([
                heart_rate,
                hrv_rmssd,
                hrv_sdnn,
                hrv_pnn50,
                hrv_lf_hf,
                float(np.mean(scl)),
                float(np.mean(scr)),
                mean_temp
            ])

            extracted_windows.append({
                "aligned_label": aligned_label,
                "features": features
            })

        print(f"[WESAD] Extracted {len(extracted_windows)} aligned windows.")
        return extracted_windows

    @staticmethod
    def train_test_split_by_subject(subjects_data: List[Dict[str, Any]], test_size: float = 0.25) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Subject-Independent Splitting Scheme.
        """
        subjects = list(set(item.get("subject_id") for item in subjects_data if item.get("subject_id") is not None))
        subjects.sort()

        split_idx = int(len(subjects) * (1 - test_size))
        train_subjects = set(subjects[:split_idx])

        train_set = []
        test_set = []

        for item in subjects_data:
            subj = item.get("subject_id")
            if subj in train_subjects:
                train_set.append(item)
            else:
                test_set.append(item)

        print(f"[IoTSplit] Subjects - Train: {len(train_subjects)}, Test: {len(subjects) - len(train_subjects)}")
        print(f"[IoTSplit] Records  - Train: {len(train_set)}, Test: {len(test_set)}")
        return train_set, test_set

    # ==================== RISK COMPOSITING ====================

    def calculate_stress_index(self, hr: float, hrv_rmssd: float, eda_microsiemens: float, skin_temp_c: float) -> Dict[str, any]:
        """
        Inference entrypoint. Synthesizes biometric inputs into a normalized index [0, 1].
        """
        hrv_sdnn = hrv_rmssd * 0.95
        hrv_pnn50 = 0.35 if hrv_rmssd > 35 else 0.05
        hrv_lf_hf = 1.2 if hrv_rmssd > 35 else 2.5
        mean_scl = eda_microsiemens * 0.85
        mean_scr = eda_microsiemens * 0.15
        
        features = [hr, hrv_rmssd, hrv_sdnn, hrv_pnn50, hrv_lf_hf, mean_scl, mean_scr, skin_temp_c]
        
        if self.rf_model is not None:
            try:
                probs = self.rf_model.predict_proba([features])[0]
                risk_classes = ["Low Risk", "Moderate Risk", "High Risk"]
                pred_idx = int(np.argmax(probs))
                
                stress_score = float(probs[1]*0.45 + probs[2]*0.95)
                stress_score = min(max(stress_score, 0.0), 1.0)
                
                states = {
                    "Low Risk": "Calm / Low Stress",
                    "Moderate Risk": "Moderate Autonomic Stress",
                    "High Risk": "High Autonomic Distress"
                }
                
                return {
                    "physiological_stress_score": stress_score,
                    "physiological_state": states[risk_classes[pred_idx]],
                    "physiological_risk_level": risk_classes[pred_idx],
                    "metrics": {
                        "heart_rate_bpm": hr,
                        "hrv_rmssd_ms": hrv_rmssd,
                        "skin_conductance_us": eda_microsiemens,
                        "skin_temperature_c": skin_temp_c
                    }
                }
            except Exception as e:
                print(f"[IoTProcessor] RF prediction failure: {e}")

        # Fallback Heuristics
        hr_stress = max(0.0, min(1.0, (hr - 60.0) / 40.0))
        hrv_stress = max(0.0, min(1.0, (70.0 - hrv_rmssd) / 50.0))
        eda_stress = max(0.0, min(1.0, eda_microsiemens / 8.0))
        temp_stress = max(0.0, min(1.0, (34.0 - skin_temp_c) / 4.0))

        stress_score = (hr_stress * 0.2) + (hrv_stress * 0.35) + (eda_stress * 0.35) + (temp_stress * 0.1)
        stress_score = float(np.clip(stress_score, 0.0, 1.0))
        
        if stress_score > 0.70:
            physiological_state = "High Autonomic Distress"
            risk_level = "High Risk"
        elif stress_score > 0.40:
            physiological_state = "Moderate Autonomic Stress"
            risk_level = "Moderate Risk"
        else:
            physiological_state = "Calm / Low Stress"
            risk_level = "Low Risk"

        return {
            "physiological_stress_score": stress_score,
            "physiological_state": physiological_state,
            "physiological_risk_level": risk_level,
            "metrics": {
                "heart_rate_bpm": hr,
                "hrv_rmssd_ms": hrv_rmssd,
                "skin_conductance_us": eda_microsiemens,
                "skin_temperature_c": skin_temp_c
            }
        }
