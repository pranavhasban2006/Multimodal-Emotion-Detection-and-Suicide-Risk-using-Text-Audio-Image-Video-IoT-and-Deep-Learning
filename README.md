# 🧠 Multimodal Emotion Detection and Suicide Risk Assessment using Text, Audio, Image, Video, IoT, and Deep Learning

This repository contains a comprehensive, production-grade, modular framework for detecting human emotional states and assessing suicide risk levels using a combination of **Natural Language Processing (NLP)**, **Vocal Acoustics**, **Facial Expression Analysis**, and **Wearable Physiological Sensors (IoT)**.

By leveraging multimodal fusion, the system synthesizes linguistic cues, vocal trembling/affect flatness, facial micro-expressions, and biometric stress indicators (GSR, HRV) into a unified, reliable clinical distress index.

---

## ⚖️ Ethical Charter & Scoping (Mandatory Phase 0)

Before reviewing or executing the codebase, please note that this project is governed by a strict ethical charter to ensure clinical safety, patient privacy, and data protection:
* **Triage vs Diagnosis:** The classified risk outputs represent **screening and routing proxies**, *never* final clinical diagnoses.
* **Human-in-the-Loop:** All automated alerts must be reviewed by a licensed clinical professional before triggering outreach or intervention.
* **Data Minimization & De-identification:** Strict zero-persistence data standards are mapped out for sensitive audio, text, and visual modalities.

👉 **Read the full [Phase 0 — Ethical Charter and Clinical Scoping Guide](ETHICS_AND_SCOPING.md) for full compliance details.**

---

## 🌟 Key Features

- **📝 Text (NLP) Sentiment Modality:** Identifies crisis triggers, hopelessness, and suicide ideation using sequence modeling (Bi-LSTM architecture) with a robust rule-based baseline fallback.
- **🎙️ Speech/Audio Modality:** Analyzes vocal tone, pitch variation, and acoustic features (MFCCs, Spectral Centroid, Chroma) to identify vocal trembling, flat affect, or hyperarousal using a 1D-CNN.
- **📷 Vision (Image/Video) Modality:** Tracks facial micro-expressions, distress markers, and emotional states (sadness, anger, fear, neutral) across static images or sequential video frames.
- **⌚ IoT Wearable Modality:** Processes real-time biometric telemetry (Heart Rate, Heart Rate Variability RMSSD, Electrodermal Activity/GSR, Skin Temperature) to gauge sympathetic nervous system arousal.
- **⚡ Multimodal Decision Fusion:** Synthesizes results using a dynamic decision-level (late) fusion engine that reweights weights automatically if certain modalities are missing. Includes an early-fusion deep learning architecture template.
- **🖥️ Streamlit Interactive Web App:** A full-featured diagnostic dashboard for clinical simulations, live IoT sensor streaming, user text inputs, and image/audio analysis.

---

## 🏗️ Repository Directory Structure

```text
Multimodal-Emotion-Detection-and-Suicide-Risk-.../
├── README.md                      # Extensive Master Guide & Architecture
├── requirements.txt               # Required Python packages
├── app.py                         # Streamlit Interactive Dashboard
├── main.py                        # CLI Quickstart & Inference Pipeline Demo
├── data/                          # Folder for raw/preprocessed datasets
│   ├── raw/                       # Auto-generated and custom sample files
│   └── processed/                 # Feature vectors and preprocessed matrices
├── models/                        # Saved PyTorch weights/checkpoints (.pth)
└── src/                           # Main source code
    ├── __init__.py
    ├── text/                      # Text Modality (NLP, Suicide Ideation Detection)
    │   ├── __init__.py
    │   ├── preprocess.py          # Cleaning, tokenizing, and text preprocessing
    │   └── model.py               # Bi-LSTM model and keyword baseline classifier
    ├── audio/                     # Audio Modality (Speech Emotion Recognition)
    │   ├── __init__.py
    │   ├── preprocess.py          # Wave reader & MFCC/Chroma feature extractor
    │   └── model.py               # 1D-CNN vocal distress classifier
    ├── vision/                    # Vision Modality (Face Expression, Micro-expressions)
    │   ├── __init__.py
    │   ├── preprocess.py          # OpenCV face detector & image normalization
    │   └── model.py               # FER 2D-CNN facial emotion classifier
    ├── iot/                       # IoT / Wearable Sensor Processing
    │   ├── __init__.py
    │   ├── signal_processor.py    # Autonomic Stress Index & RMSSD HRV computer
    │   └── mock_sensor.py         # Real-time smartwatch telemetry stream generator
    ├── fusion/                    # Multimodal Fusion (Feature-level & Decision-level)
    │   ├── __init__.py
    │   └── fusion_model.py        # Early Fusion NN & Late Weighted Decision Fusion
    └── utils/                     # Developer utility helpers
        ├── __init__.py
        └── helpers.py             # Mock audio, face images, and asset setup script
```

---

## 🛠️ Step-by-Step Guide to Getting Started

Follow these steps to set up, run, and customize this project from scratch:

### Step 1: Clone and Set Up the Environment
Create a clean virtual environment and install the required packages.

```bash
# Clone the repository
git clone https://github.com/pranavhasban2006/Multimodal-Emotion-Detection-and-Suicide-Risk-using-Text-Audio-Image-Video-IoT-and-Deep-Learning.git
cd Multimodal-Emotion-Detection-and-Suicide-Risk-using-Text-Audio-Image-Video-IoT-and-Deep-Learning

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

---

### Step 2: Run the Command-Line Interface (CLI) Demo
We have provided a fully automated end-to-end simulation script. Running `main.py` will:
1. Automatically generate high-fidelity sample testing assets (mock WAV audio tone and sad face PNG images).
2. Execute individual feature extraction and classification pipelines for all 4 modalities.
3. Compute and output a comprehensive **Multimodal Decision Fusion Report** showing overall risk severity and clinical recommendations.

```bash
python3 main.py
```

---

### Step 3: Launch the Interactive Web Dashboard
Experience the project through a clean, highly visual, interactive **Streamlit dashboard**!
To run the web app:

```bash
pip install streamlit plotly
streamlit run app.py
```

#### What you can do in the Web App:
- **Select Clinical Presets:** Instantly populate the dashboard with severe crisis, high-anxiety panic, or calm control states.
- **Analyze Text Sentiment:** Type or copy clinical transcripts or posts to see immediate risk scores.
- **Upload Image & Audio:** Upload custom voice clips and face photos to see face-crop and acoustic feature results.
- **IoT Streaming Telemetry:** Toggle between normal, depressed, and anxious states to see heart rate, HRV (RMSSD), skin temperature, and skin conductance respond dynamically.
- **Dynamic Late Fusion:** Observe how the decision layer automatically adjusts weighting percentages if you choose to analyze only text and IoT data.

---

### Step 4: Acquiring Research Datasets for Model Training
To train the deep learning architectures defined in the `src/` modules, you should download and prepare these widely recognized research datasets:

| Modality | Recommended Open Dataset | What it Provides |
| :--- | :--- | :--- |
| **Text** | [Reddit SuicideWatch Dataset](https://www.kaggle.com/datasets/nikhileshwaraji/suicidal-tweet-detection-dataset) | Labeled posts from support subreddits vs. neutral content. |
| **Audio** | [RAVDESS Emotion Speech](https://zenodo.org/records/1188976) | Emotional vocal clips (sad, angry, anxious, calm). |
| **Vision** | [FER2013 Facial Expressions](https://www.kaggle.com/datasets/msambare/fer2013) | 35k grayscale face images labeled with emotions. |
| **IoT** | [WESAD Wearable Stress & Affect](https://archive.ics.uci.edu/ml/datasets/WESAD) | Smartwatch PPG, EDA, Temp telemetry during stress & calm. |

---

### Step 5: Training and Loading Deep Learning Weights
The code is built to run fully functional heuristic algorithms when neural network checkpoints are not present, making it highly portable. To replace the simulations with real trained weights:

1. Follow the model architectures defined in `src/text/model.py` (`TextSuicideRiskModel`), `src/audio/model.py` (`AudioDistressCNN`), and `src/vision/model.py` (`FacialEmotionCNN`) to train on the above datasets.
2. Save your trained PyTorch weights (e.g. `text_lstm.pth`, `audio_cnn.pth`, `vision_cnn.pth`) to the `models/` directory.
3. Load the weights using:
   ```python
   from src.text.model import TextRiskClassifier
   
   classifier = TextRiskClassifier(model_path="models/text_lstm.pth")
   classifier.load_model()
   ```

---

## 📐 System Architecture Diagram

```text
               +-------------------------------------------------+
               |             Real-Time Human Subject             |
               +-------------------------------------------------+
                    |             |               |          |
                    v             v               v          v
               [Chat Log]    [Microphone]     [Camera]    [Smartwatch]
                    |             |               |          |
                    v             v               v          v
                Text NLP        Audio          Vision        IoT
               Preprocess     Preprocess     Preprocess   Preprocess
                    |             |               |          |
                    v             v               v          v
                Bi-LSTM        1D-CNN          2D-CNN     Autonomic
                 Model          Model           Model    Stress Index
                    \             \               /          /
                     \             \             /          /
                      v             v           v          v
                  +------------------------------------------+
                  |         Multimodal Late Fusion           |
                  |     (Dynamic Decision Weighting)         |
                  +------------------------------------------+
                                       |
                                       v
                  +------------------------------------------+
                  |       Unified Clinical Severity          |
                  |     (Low, Moderate, or High Risk)        |
                  +------------------------------------------+
```

---

## ⚕️ Disclaimer

**CRITICAL NOTICE:** This software is a prototype designed for research and educational purposes only. It is **not** a diagnostic medical tool, and should never be used as a substitute for professional clinical judgment, psychiatric assessment, or immediate medical intervention. 

If you or someone you know is struggling with mental health, depression, or suicidal thoughts, please reach out to professional emergency services or your local suicide prevention hotline (e.g., dialing 988 in the USA/Canada, or 111 in the UK).
