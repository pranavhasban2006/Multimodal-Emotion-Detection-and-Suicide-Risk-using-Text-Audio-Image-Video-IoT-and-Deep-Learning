import os
import sys
import numpy as np
from src.text.model import TextRiskClassifier
from src.audio.model import AudioEmotionClassifier
from src.vision.model import VisionEmotionClassifier
from src.iot.signal_processor import IoTSignalProcessor
from src.iot.mock_sensor import IoTSensorSimulator
from src.fusion.fusion_model import MultimodalDecisionFusion
from src.utils.helpers import setup_all_sample_assets

def print_separator(title=""):
    if title:
        print(f"\n=== {title} ===" + "=" * (60 - len(title)))
    else:
        print("\n" + "=" * 64)

def run_pipeline_demo():
    print_separator("Multimodal Suicide Risk Detection Framework")
    print("Welcome to the CLI demonstration of the Multimodal Distress Analysis pipeline.")
    
    # 1. Setup local raw data files
    print("\n[Step 1/5] Initializing local sample assets...")
    raw_data_dir = "data/raw"
    setup_all_sample_assets(raw_data_dir)
    
    # 2. Instantiate and Load Models (dynamic weight routing)
    print("\n[Step 2/5] Initializing AI models and preprocessing pipelines...")
    text_model_path = "models/text_model.pth" if os.path.exists("models/text_model.pth") else "models/text_rf_model.pkl"
    audio_model_path = "models/audio_model.pth" if os.path.exists("models/audio_model.pth") else "models/audio_rf_model.pkl"
    vision_model_path = "models/vision_model.pth" if os.path.exists("models/vision_model.pth") else "models/vision_rf_model.pkl"
    
    text_model = TextRiskClassifier(model_path=text_model_path)
    audio_model = AudioEmotionClassifier(model_path=audio_model_path)
    vision_model = VisionEmotionClassifier(model_path=vision_model_path)
    
    text_model.load_model()
    audio_model.load_model()
    vision_model.load_model()
    
    # 3. Choose a clinical Scenario for simulation
    print("\n[Step 3/5] Running End-to-End Inference Scenario: 'High Distress Case'")
    
    text_input = "I can't take this anymore. Everything is falling apart and I feel totally hopeless. Goodbye everyone."
    print(f" -> Text Input: \"{text_input}\"")
    
    sample_wav_path = os.path.join(raw_data_dir, "sample_vocal_normal.wav")
    print(f" -> Audio Path: {sample_wav_path}")
    
    sample_img_path = os.path.join(raw_data_dir, "sample_face_sad.png")
    print(f" -> Facial Image Path: {sample_img_path}")
    
    # Generate mock reading for distressed patient
    iot_simulator = IoTSensorSimulator(state="depressed")
    iot_reading = iot_simulator.generate_reading()
    print(" -> IoT Smartwatch Realtime biometrics:")
    print(f"    * Heart Rate: {iot_reading['heart_rate']} BPM")
    print(f"    * GSR/EDA Conductance: {iot_reading['eda_microsiemens']} uS")
    print(f"    * Skin Temp: {iot_reading['skin_temp_c']} °C")

    # --- Feature Extraction to Match 2.5 Structured Schema ---
    raw_text_x = text_model.text_to_features(text_input).reshape(1, -1)
    raw_audio_x = audio_model.preprocessor.extract_features(sample_wav_path, raise_errors=False).reshape(1, -1)
    
    preprocessed_img = vision_model.preprocessor.preprocess_image(sample_img_path)
    raw_vision_x = vision_model.image_to_embedding(preprocessed_img).reshape(1, -1)
    
    # Extract precise 8-dimensional IoT features for the custom Deep Neural Net
    iot_processor = IoTSignalProcessor()
    hrv_rmssd = iot_processor.calculate_hrv_rmssd(iot_reading['rr_intervals_ms'])
    hrv_sdnn = iot_processor.calculate_hrv_sdnn(iot_reading['rr_intervals_ms'])
    hrv_pnn50 = iot_processor.calculate_hrv_pnn50(iot_reading['rr_intervals_ms'])
    hrv_lf_hf = iot_processor.calculate_hrv_lf_hf_ratio(iot_reading['rr_intervals_ms'])
    
    scl, scr = iot_processor.decompose_eda(np.array([iot_reading['eda_microsiemens']]*10))
    mean_scl = float(np.mean(scl))
    mean_scr = float(np.mean(scr))

    raw_iot_x = np.array([[
        iot_reading['heart_rate'], 
        hrv_rmssd, 
        hrv_sdnn, 
        hrv_pnn50, 
        hrv_lf_hf, 
        mean_scl, 
        mean_scr, 
        iot_reading['skin_temp_c']
    ]])

    # 4. Process each Modality individually
    print("\n[Step 4/5] Executing individual modality classifiers...")
    
    # Text
    text_res = text_model.predict(text_input)
    print(f"    [Text] Classified Level: {text_res['risk_level']} (Prob: {text_res['risk_probability']:.2f})")
    
    # Audio
    audio_res = audio_model.predict(sample_wav_path)
    print(f"    [Audio] Vocal Distress: {audio_res['vocal_distress_level']} (Prob: {audio_res['distress_probability']:.2f})")
    
    # Vision
    vision_res = vision_model.predict_image(sample_img_path)
    print(f"    [Vision] Facial Expression: {vision_res['dominant_emotion']} | Risk level: {vision_res['facial_distress_level']} (Score: {vision_res['facial_distress_score']:.2f})")
    
    # IoT Physiological
    iot_res = iot_processor.calculate_stress_index(
        hr=iot_reading['heart_rate'],
        hrv_rmssd=hrv_rmssd,
        eda_microsiemens=iot_reading['eda_microsiemens'],
        skin_temp_c=iot_reading['skin_temp_c']
    )
    print(f"    [IoT] Autonomic Stress: {iot_res['physiological_state']} (Score: {iot_res['physiological_stress_score']:.2f})")

    # 5. Multimodal Fusion (Demonstrating BOTH Fusion Methods!)
    print("\n[Step 5/5] Combining signals using Multimodal Fusion...")
    fusion_engine = MultimodalDecisionFusion()
    
    # Scenario A: Late Decision Fusion
    late_report = fusion_engine.fuse_predictions(
        text_res=text_res,
        audio_res=audio_res,
        vision_res=vision_res,
        iot_res=iot_res,
        fusion_method="Late Decision Fusion"
    )
    
    # Scenario B: Early Feature Fusion via Deep NumPy MLP
    early_report = fusion_engine.fuse_predictions(
        text_res=text_res,
        audio_res=audio_res,
        vision_res=vision_res,
        iot_res=iot_res,
        fusion_method="Early Feature Fusion",
        raw_text_x=raw_text_x,
        raw_audio_x=raw_audio_x,
        raw_vision_x=raw_vision_x,
        raw_iot_x=raw_iot_x
    )
    
    # --- Print Fusion Outputs Side-by-Side ---
    print_separator("MULTIMODAL FUSION REPORT")
    print(f" METHOD A: LATE DECISION FUSION")
    print(f"   * Fused Risk Level  : {late_report['fused_suicide_risk_level'].upper()}")
    print(f"   * Severity Score    : {late_report['fused_risk_score']:.2f} / 1.00")
    print(f"   * Dominant Emotion  : {late_report['dominant_state_emotion'].upper()}")
    
    print(f"\n METHOD B: EARLY FEATURE FUSION (NUMPY DEEP NEURAL NETWORK)")
    if fusion_engine.early_fusion_loaded:
        print(f"   * Fused Risk Level  : {early_report['fused_suicide_risk_level'].upper()}")
        print(f"   * Severity Score    : {early_report['fused_risk_score']:.2f} / 1.00")
        print(f"   * Class Probabilities:")
        for k, v in early_report["early_fusion_probabilities"].items():
            print(f"     - {k}: {v:.1%}")
    else:
        print("   * Early Fusion Model weights not found. Run 'python3 train.py' to generate weights.")
        
    print("\n CLINICAL RECOMMENDATION (LATE FUSION):")
    print(f"   {late_report['clinical_recommendation']}")
    print_separator()

if __name__ == "__main__":
    run_pipeline_demo()
