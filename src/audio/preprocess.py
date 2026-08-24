import os
import json
import warnings
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Try to import librosa, but allow soft warnings inside preprocessors
LIBROSA_AVAILABLE = True
try:
    import librosa
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioPreprocessor:
    def __init__(self, sr=16000, duration=3.0, n_mfcc=13):
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc

    def extract_features(self, file_path: str, raise_errors: bool = False, return_tuple: bool = False) -> Any:
        """
        Loads an audio file and extracts 47-dimensional clinical acoustic indicators:
        - MFCCs (13 means, 13 stds) -> 26 dims
        - Spectral Centroid -> 1 dim
        - Fundamental Frequency / Pitch (F0) Mean & Std -> 2 dims
        - Jitter & Shimmer (Voice perturbation approximations) -> 2 dims
        - RMS Energy Envelope Mean & Std -> 2 dims
        - Silence Ratio -> 1 dim
        - Chroma short-time Fourier transform (13 means) -> 13 dims
        
        Returns:
            If return_tuple is True: Tuple of (features_vector, is_valid_flag)
            If return_tuple is False: features_vector (with zeros if invalid)
        """
        if not LIBROSA_AVAILABLE:
            if raise_errors:
                raise ImportError("Librosa is not installed. Failed to process vocal acoustics.")
            features = np.zeros(47)
            return (features, False) if return_tuple else features

        if not os.path.exists(file_path):
            if raise_errors:
                raise FileNotFoundError(f"Audio file path not found: {file_path}")
            features = np.zeros(47)
            return (features, False) if return_tuple else features

        try:
            # Force load with fixed sampling rate and duration
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
            
            # Pad audio if too short
            expected_length = int(self.sr * self.duration)
            if len(y) < expected_length:
                y = np.pad(y, (0, expected_length - len(y)), 'constant')
            else:
                y = y[:expected_length]

            # --- 1. Pitch / Fundamental Frequency (F0) & Jitter ---
            # Use YIN algorithm for pitch tracking
            f0 = librosa.yin(y, fmin=75, fmax=400, sr=self.sr)
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                f0_mean = float(np.mean(f0_clean))
                f0_std = float(np.std(f0_clean))
                # Jitter: local pitch period variations
                diff_f0 = np.abs(np.diff(f0_clean))
                jitter = float(np.mean(diff_f0) / f0_mean) if f0_mean > 0 else 0.0
            else:
                f0_mean, f0_std, jitter = 0.0, 0.0, 0.0

            # --- 2. RMS Energy & Shimmer ---
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            
            if rms_mean > 0:
                # Shimmer: local peak-to-peak amplitude variations
                diff_rms = np.abs(np.diff(rms))
                shimmer = float(np.mean(diff_rms) / rms_mean)
            else:
                shimmer = 0.0

            # --- 3. Silence / Pause Ratio ---
            # Ratio of frames below 10% of mean RMS power (capturing flat hesitation)
            silence_threshold = 0.1 * rms_mean if rms_mean > 0 else 0.01
            silent_frames = sum(1 for r in rms if r < silence_threshold)
            silence_ratio = float(silent_frames / len(rms)) if len(rms) > 0 else 0.0

            # --- 4. Spectral Centroid ---
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
            spec_cent_mean = float(np.mean(spec_cent))

            # --- 5. MFCCs (Mel-Scale Spectrums) ---
            mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            # --- 6. Chroma ---
            chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_chroma=13)
            chroma_mean = np.mean(chroma, axis=1)

            # Combine all features sequentially into a 47-dimensional vector
            features = np.concatenate([
                mfcc_mean,          # 13 dims
                mfcc_std,           # 13 dims
                [spec_cent_mean],   # 1 dim
                [f0_mean, f0_std],  # 2 dims
                [jitter, shimmer],  # 2 dims
                [rms_mean, rms_std],# 2 dims
                [silence_ratio],    # 1 dim
                chroma_mean         # 13 dims
            ])

            return (features, True) if return_tuple else features

        except Exception as e:
            if raise_errors:
                raise ValueError(f"Fails to extract features from WAV: {e}")
            features = np.zeros(47)
            return (features, False) if return_tuple else features

    @staticmethod
    def train_test_split_by_actor(audio_files: List[Dict[str, Any]], test_size: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits RAVDESS audio datasets by Actor ID (Actors 01-24) to prevent speaker voice bleeding.
        """
        actors = list(set(item.get("actor_id") for item in audio_files if item.get("actor_id") is not None))
        actors.sort()

        # Deterministic split: use even actors for test partition or simple split
        split_idx = int(len(actors) * (1 - test_size))
        train_actors = set(actors[:split_idx])

        train_set = []
        test_set = []

        for item in audio_files:
            actor = item.get("actor_id")
            if actor in train_actors:
                train_set.append(item)
            else:
                test_set.append(item)

        print(f"[AudioSplit] Actors - Train: {len(train_actors)}, Test: {len(actors) - len(train_actors)}")
        print(f"[AudioSplit] Files - Train: {len(train_set)}, Test: {len(test_set)}")
        return train_set, test_set


class AudioFeatureScaler:
    def __init__(self, config_path: str = "models/audio_scaler.json"):
        self.config_path = config_path
        self.means = None
        self.stds = None

    def fit(self, X: np.ndarray):
        """Calculates means and standard deviations from a training matrix."""
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        # Avoid divide by zero
        self.stds[self.stds == 0.0] = 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies Z-Score standard scaling."""
        if self.means is None or self.stds is None:
            raise ValueError("AudioFeatureScaler must be fitted before running transform.")
        return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self):
        """Serializes scaling configurations to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
        config = {
            "means": self.means.tolist() if self.means is not None else [],
            "stds": self.stds.tolist() if self.stds is not None else []
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"[AudioFeatureScaler] Saved scaling configuration to {self.config_path}")

    def load(self) -> bool:
        """Loads scaling configurations from disk."""
        if not os.path.exists(self.config_path):
            return False
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            self.means = np.array(config["means"])
            self.stds = np.array(config["stds"])
            print(f"[AudioFeatureScaler] Loaded scaling configuration from {self.config_path}")
            return True
        except Exception as e:
            print(f"[AudioFeatureScaler] Failed to load config: {e}")
            return False
