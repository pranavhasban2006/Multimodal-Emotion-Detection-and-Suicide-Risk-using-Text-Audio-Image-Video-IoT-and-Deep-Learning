import os
import sys
import numpy as np

STREAMLIT_AVAILABLE = True
try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    STREAMLIT_AVAILABLE = False


if not STREAMLIT_AVAILABLE:
    print("\n" + "=" * 80)
    print(" STREAMLIT WEB APP PREPARED!")
    print("=" * 80)
    print("To launch the interactive Multimodal Emotion & Suicide Risk dashboard:")
    print("\n1. Install Streamlit and visualization tools:")
    print("   $ pip install streamlit plotly matplotlib")
    print("\n2. Run the application:")
    print("   $ streamlit run app.py")
    print("=" * 80 + "\n")
    sys.exit(0)


# Initialize our individual modules
from src.text.model import TextRiskClassifier
from src.audio.model import AudioEmotionClassifier
from src.vision.model import VisionEmotionClassifier
from src.iot.signal_processor import IoTSignalProcessor
from src.iot.mock_sensor import IoTSensorSimulator
from src.fusion.fusion_model import MultimodalDecisionFusion
from src.utils.helpers import setup_all_sample_assets

# Setup page config
st.set_page_config(
    page_title="Multimodal Distress & Suicide Risk Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize sample assets
RAW_DATA_DIR = "data/raw"
setup_all_sample_assets(RAW_DATA_DIR)

# Instantiate models
@st.cache_resource
def load_models_cached():
    text_model_path = "models/text_model.pth" if os.path.exists("models/text_model.pth") else "models/text_rf_model.pkl"
    audio_model_path = "models/audio_model.pth" if os.path.exists("models/audio_model.pth") else "models/audio_rf_model.pkl"
    vision_model_path = "models/vision_model.pth" if os.path.exists("models/vision_model.pth") else "models/vision_rf_model.pkl"
    
    text_clf = TextRiskClassifier(model_path=text_model_path)
    audio_clf = AudioEmotionClassifier(model_path=audio_model_path)
    vision_clf = VisionEmotionClassifier(model_path=vision_model_path)
    
    text_clf.load_model()
    audio_clf.load_model()
    vision_clf.load_model()
    
    return text_clf, audio_clf, vision_clf, IoTSignalProcessor(), MultimodalDecisionFusion()

text_clf, audio_clf, vision_clf, iot_processor, fusion_engine = load_models_cached()


# SIDEBAR: Clinical Scenario Controller and Simulation State
st.sidebar.title("🛠️ Clinical Simulation Control")
st.sidebar.markdown("Use this panel to configure the emotional states of the various modalities.")

sim_scenario = st.sidebar.selectbox(
    "Choose Presets Scenario",
    ["Manual Adjustment", "Severe Distress/Crisis Case", "Anxious Panic Attack", "Stable & Calm Control"]
)

# Map preset states
if sim_scenario == "Severe Distress/Crisis Case":
    preset_text = "I feel so lonely and hopeless. I don't want to live anymore, it's better if I just disappear. Goodbye."
    preset_iot_state = "depressed"
    preset_image_path = os.path.join(RAW_DATA_DIR, "sample_face_sad.png")
elif sim_scenario == "Anxious Panic Attack":
    preset_text = "My chest is tight and I am terrified that something awful is happening. I can't breathe or calm down!"
    preset_iot_state = "stressed/anxious"
    preset_image_path = os.path.join(RAW_DATA_DIR, "sample_face_angry.png")
elif sim_scenario == "Stable & Calm Control":
    preset_text = "I had a great day today at the park. Everything is peaceful and I'm feeling quite relaxed."
    preset_iot_state = "normal"
    preset_image_path = os.path.join(RAW_DATA_DIR, "sample_face_happy.png")
else:
    preset_text = ""
    preset_iot_state = "normal"
    preset_image_path = os.path.join(RAW_DATA_DIR, "sample_face_sad.png")


st.sidebar.subheader("🔋 IoT Wearable Live Stream")
sensor_state = st.sidebar.radio(
    "Autonomic State:", 
    ["normal", "stressed/anxious", "depressed"],
    index=0 if preset_iot_state == "normal" else (1 if preset_iot_state == "stressed/anxious" else 2)
)

st.sidebar.subheader("⚡ Fusion Hyperparameters")
fusion_mode = st.sidebar.radio(
    "Select Fusion Algorithm:",
    ["Late Decision Fusion", "Early Feature Fusion"]
)

if fusion_mode == "Early Feature Fusion" and not fusion_engine.early_fusion_loaded:
    st.sidebar.warning("⚠️ Early Fusion Model weights are missing! Run `python3 train.py` to train the deep neural net.")

# MAIN HEADER
st.title("🧠 Multimodal Emotion Detection & Suicide Risk Assessment")
st.markdown(
    """
    This prototype integrates **Text Sentiment (NLP)**, **Vocal Acoustics (Audio)**, 
    **Facial Expression Analysis (Vision)**, and **Physiological Telemetry (IoT)** to assess human 
    emotional distress and suicide risk levels in real-time using Deep Learning.
    """
)

# TABS
tab1, tab2, tab3 = st.tabs(["🖥️ Interactive Inference Dashboard", "📂 Project Architecture & Guide", "📊 Datasets & Training Guide"])

# ==================== TAB 1: INTERACTIVE INFERENCE ====================
with tab1:
    st.header("🔬 Live Multimodal Diagnostics Playground")
    st.info("💡 Tip: Try choosing a preset scenario from the sidebar to auto-populate distress levels across all 4 sensor fields!")

    # Col 1: Text & Audio Input | Col 2: Image & IoT Input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 1. Textual / Linguistic Modality")
        text_input = st.text_area(
            "Enter speech transcript or social post:",
            value=preset_text if sim_scenario != "Manual Adjustment" else "Type something here...",
            height=120
        )
        
        st.subheader("🎙️ 2. Vocal / Acoustic Modality")
        audio_choice = st.selectbox(
            "Select Audio File Source:",
            ["Generated Sample (Tone 440Hz)", "Upload Custom WAV file"]
        )
        audio_file_path = os.path.join(RAW_DATA_DIR, "sample_vocal_normal.wav")
        
        if audio_choice == "Upload Custom WAV file":
            uploaded_audio = st.file_uploader("Upload WAV audio:", type=["wav"])
            if uploaded_audio:
                audio_file_path = os.path.join(RAW_DATA_DIR, "uploaded_temp.wav")
                with open(audio_file_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
        
        st.audio(audio_file_path, format="audio/wav")

    with col2:
        st.subheader("📷 3. Facial Expression (Image/Video) Modality")
        image_choice = st.selectbox(
            "Select Face Image Source:",
            ["Sad Face Expression (Sample)", "Happy Face Expression (Sample)", "Angry Face Expression (Sample)", "Upload Custom Image"]
        )
        
        img_path = preset_image_path
        if image_choice == "Sad Face Expression (Sample)":
            img_path = os.path.join(RAW_DATA_DIR, "sample_face_sad.png")
        elif image_choice == "Happy Face Expression (Sample)":
            img_path = os.path.join(RAW_DATA_DIR, "sample_face_happy.png")
        elif image_choice == "Angry Face Expression (Sample)":
            img_path = os.path.join(RAW_DATA_DIR, "sample_face_angry.png")
        elif image_choice == "Upload Custom Image":
            uploaded_img = st.file_uploader("Upload Image (JPG/PNG):", type=["png", "jpg", "jpeg"])
            if uploaded_img:
                img_path = os.path.join(RAW_DATA_DIR, "uploaded_face_temp.png")
                with open(img_path, "wb") as f:
                    f.write(uploaded_img.getbuffer())
                    
        if os.path.exists(img_path):
            st.image(img_path, caption="Active Facial Analysis Frame", width=160)

        st.subheader("⌚ 4. IoT Physiological Wearable Modality")
        simulator = IoTSensorSimulator(state=sensor_state)
        iot_reading = simulator.generate_reading()
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Heart Rate", f"{iot_reading['heart_rate']} bpm")
        metric_col2.metric("EDA/GSR", f"{iot_reading['eda_microsiemens']} µS")
        metric_col3.metric("Skin Temp", f"{iot_reading['skin_temp_c']} °C")
        
        rmssd_val = iot_processor.calculate_hrv_rmssd(iot_reading['rr_intervals_ms'])
        metric_col4.metric("HRV (RMSSD)", f"{rmssd_val:.1f} ms")

    # RUN FUSION BUTTON
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader(f"⚡ 5. Cross-Modal Fusion ({fusion_mode})")
    
    if st.button("🚀 EXECUTE MULTIMODAL INFERENCE FUSION", use_container_width=True):
        with st.spinner("Processing modalities and applying neural matrices..."):
            
            # 1. Process Text
            text_res = text_clf.predict(text_input)
            
            # 2. Process Audio
            audio_res = audio_clf.predict(audio_file_path)
            
            # 3. Process Vision
            vision_res = vision_clf.predict_image(img_path)
            
            # 4. Process IoT
            iot_res = iot_processor.calculate_stress_index(
                hr=iot_reading['heart_rate'],
                hrv_rmssd=rmssd_val,
                eda_microsiemens=iot_reading['eda_microsiemens'],
                skin_temp_c=iot_reading['skin_temp_c']
            )
            
            # Create feature representations for Early Feature Fusion
            raw_text_x = text_clf.text_to_features(text_input).reshape(1, -1)
            raw_audio_x = audio_clf.preprocessor.extract_features(audio_file_path, raise_errors=False).reshape(1, -1)
            preprocessed_img = vision_clf.preprocessor.preprocess_image(img_path)
            raw_vision_x = vision_clf.image_to_embedding(preprocessed_img).reshape(1, -1)
            raw_iot_x = np.array([[iot_reading['heart_rate'], rmssd_val, iot_reading['eda_microsiemens'], iot_reading['skin_temp_c']]])

            # Expand IoT to 8-dimensional space if needed for early fusion alignment
            if raw_iot_x.shape[1] == 4:
                # Same somatic scaling/projection as signal_processor.py
                hrv_sdnn = rmssd_val * 0.95
                hrv_pnn50 = 0.35 if rmssd_val > 35 else 0.05
                hrv_lf_hf = 1.2 if rmssd_val > 35 else 2.5
                mean_scl = iot_reading['eda_microsiemens'] * 0.85
                mean_scr = iot_reading['eda_microsiemens'] * 0.15
                raw_iot_x = np.array([[
                    iot_reading['heart_rate'], 
                    rmssd_val, 
                    hrv_sdnn, 
                    hrv_pnn50, 
                    hrv_lf_hf, 
                    mean_scl, 
                    mean_scr, 
                    iot_reading['skin_temp_c']
                ]])

            # 5. Fused output
            fused_report = fusion_engine.fuse_predictions(
                text_res=text_res,
                audio_res=audio_res,
                vision_res=vision_res,
                iot_res=iot_res,
                fusion_method=fusion_mode,
                raw_text_x=raw_text_x,
                raw_audio_x=raw_audio_x,
                raw_vision_x=raw_vision_x,
                raw_iot_x=raw_iot_x
            )
            
            # Display metrics
            f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
            
            with f_col1:
                st.metric("Fused Distress Severity Score", f"{fused_report['fused_risk_score']:.2f} / 1.00")
                st.subheader("Risk Rating:")
                if fused_report["fused_suicide_risk_level"] == "High Risk":
                    st.error(f"🚨 {fused_report['fused_suicide_risk_level'].upper()}")
                elif fused_report["fused_suicide_risk_level"] == "Moderate Risk":
                    st.warning(f"⚠️ {fused_report['fused_suicide_risk_level'].upper()}")
                else:
                    st.success(f"✅ {fused_report['fused_suicide_risk_level'].upper()}")
                    
            with f_col2:
                st.metric("Fused Emotional State", fused_report['dominant_state_emotion'].upper())
                
                # Plotly Gauge Chart for Suicide Severity Risk
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = fused_report['fused_risk_score'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Clinical Risk Index %", 'font': {'size': 14}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkred" if fused_report['fused_risk_score'] > 0.6 else "orange"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 35], 'color': 'lightgreen'},
                            {'range': [35, 70], 'color': 'khaki'},
                            {'range': [70, 100], 'color': 'lightcoral'}],
                    }
                ))
                fig.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with f_col3:
                st.markdown("### 📋 Clinician Guidance & Action Steps")
                st.info(fused_report['clinical_recommendation'])
                
            # Breakdown accordions
            with st.expander("🔍 Deep Modality Breakdown & Model Confidence Scores"):
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                
                with b_col1:
                    st.markdown("##### 📝 Text Analysis")
                    st.write(f"**Risk Level:** {text_res['risk_level']}")
                    st.write(f"**Confidence Score:** {text_res['risk_probability']:.2f}")
                    st.write(f"**Emotion:** {text_res['dominant_emotion']}")
                    
                with b_col2:
                    st.markdown("##### 🎙️ Audio Analysis")
                    st.write(f"**Vocal Distress:** {audio_res['vocal_distress_level']}")
                    st.write(f"**Distress Prob:** {audio_res['distress_probability']:.2f}")
                    st.write(f"**Est. Emotion:** {audio_res['dominant_emotion']}")
                    
                with b_col3:
                    st.markdown("##### 📷 Face Analysis")
                    st.write(f"**Facial Distress:** {vision_res['facial_distress_level']}")
                    st.write(f"**Distress Score:** {vision_res['facial_distress_score']:.2f}")
                    st.write(f"**Facial Emotion:** {vision_res['dominant_emotion']}")
                    
                with b_col4:
                    st.markdown("##### ⌚ Physiological IoT")
                    st.write(f"**Autonomic Stress:** {iot_res['physiological_state']}")
                    st.write(f"**Autonomic Score:** {iot_res['physiological_stress_score']:.2f}")
                    st.write(f"**Heart Rate:** {iot_reading['heart_rate']} bpm")

                if fusion_mode == "Early Feature Fusion" and "early_fusion_probabilities" in fused_report:
                    st.markdown("---")
                    st.markdown("##### 🔗 Early Fusion Deep Neural Network Output Distribution")
                    eb1, eb2, eb3 = st.columns(3)
                    eb1.metric("Low Risk Probability", f"{fused_report['early_fusion_probabilities']['Low Risk']:.1%}")
                    eb2.metric("Moderate Risk Probability", f"{fused_report['early_fusion_probabilities']['Moderate Risk']:.1%}")
                    eb3.metric("High Risk Probability", f"{fused_report['early_fusion_probabilities']['High Risk']:.1%}")


# ==================== TAB 2: ARCHITECTURE & GUIDE ====================
with tab2:
    st.header("🏗️ System Architecture & Workflow")
    st.markdown(
        """
        The diagram below explains how streaming telemetry from multiple sources (smartwatches, mic, cameras, and chat loggers)
        is preprocessed, analyzed by deep neural networks, and synthesized in a decision-fusion server.
        """
    )
    
    st.code(
        """
        +------------------+     +------------------------+     +----------------------------+
        |   Input Source   | --> |      Preprocessing     | --> |   Deep Learning Inference  |
        +------------------+     +------------------------+     +----------------------------+
        
        [1. Chat Log Text]  -->  (Tokenization & Cleaning) -->  [Bi-LSTM Ideation Model]  -----\\
                                                                                               \\
        [2. Microphone Wav] -->  (Librosa MFCC Extraction) -->  [Acoustic CNN Distress Model] -->\\ [Late Decision Fusion]
                                                                                               /  (Weighted average of scores)
        [3. Camera Frame]   -->  (Haar Cascade Face Crop)  -->  [FER Facial Expression CNN] ---/               |
                                                                                               /                v
        [4. IoT Wearable]   -->  (RMSSD, EDA Peak Analysis)-->  [Physiological Stress Index]-/         [Risk Level Output]
                                                                                                      (Low, Moderate, High)
        """,
        language="text"
    )
    
    st.subheader("🚀 Step-by-Step Developer Setup Guide")
    st.markdown(
        """
        ### Step 1: Install System & Python Libraries
        Ensure you have your Python package environment activated. Install the required deep learning dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        
        ### Step 2: Run Classifier & Fusion Training
        Generate the synthetic clinical record dataset and optimize the classifiers:
        ```bash
        python3 train.py
        ```
        
        ### Step 3: Running the End-to-End Command Line Demo
        Run the fully automated pipeline CLI command to verify that all directories, asset helper scripts, and decision algorithms work:
        ```bash
        python3 main.py
        ```
        """
    )


# ==================== TAB 3: DATASETS & TRAINING ====================
with tab3:
    st.header("📊 Recommended Research Datasets")
    st.markdown(
        """
        To successfully train high-accuracy models, we recommend training the modules on the following standard open-source datasets:
        
        #### 1. Text Modality (Suicide Ideation Detection)
        *   **Reddit SuicideWatch Dataset:** Labeled posts from support subreddits vs. neutral content.
        *   **5-class Suicide Ideation Dataset:** Available on Kaggle, containing tweets and posts categorized from "safe" to "highly suicidal".
        
        #### 2. Audio Modality (Speech Emotion Recognition)
        *   **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):** Contains emotional expressions (calm, happy, sad, angry, fearful, surprise).
        
        #### 3. Vision Modality (Facial Expression Recognition)
        *   **FER2013:** Standard Kaggle dataset with 35,000+ 48x48 grayscale face images belonging to 7 emotions.
        
        #### 4. IoT Physiological Wearable Modality
        *   **WESAD (Wearable Stress and Affect Detection):** A public dataset of multimodal physiological signals (ECG, EDA, EMG, respiration, skin temp) collected from smartwatches.
        """
    )
    
    st.subheader("📈 Recommended Training & Fusion Loop")
    st.markdown(
        """
        For optimal research results, implement an **Early Fusion Multi-Head Model**:
        1. Initialize an **Embedding encoder** for text, a **1D-CNN encoder** for speech MFCCs, a **ResNet encoder** for face frames, and a **dense MLP encoder** for heart rate / GSR.
        2. Concatenate the output latent embeddings into a single vector of shape `[Batch, Combined_Dimension]`.
        3. Pass the fused representation through a **Cross-Attention layer** or a **Multimodal Transformer block** to capture inter-modal dynamics.
        4. Predict classification probabilities with a final Softmax output.
        """
    )
