# 📊 Phase 1: Data Acquisition, Shared Labeling, & Versioning

To train your deep-learning pipelines, this document outlines the exact acquisition steps, access instructions, and the **Cross-Modal Clinical Risk Taxonomy** used to align all datasets under a unified classification schema.

---

## 📂 1. Recommended Dataset Directory Layout

When acquiring the raw files, assemble them in the `data/raw/` directory according to this standard structure:

```text
data/
├── raw/
│   ├── text_reddit/                   # SuicideWatch CSV / TXT files
│   │   └── suicidal_tweet_detection.csv
│   ├── audio_ravdess/                 # RAVDESS speech directories (Actor_01, etc.)
│   │   └── Actor_01/
│   │       └── 03-01-01-01-01-01-01.wav
│   ├── vision_fer2013/                # FER2013 train/test image directories
│   │   ├── train/
│   │   │   ├── sad/
│   │   │   └── happy/
│   │   └── test/
│   └── iot_wesad/                     # WESAD subject subfolders (S2, S3, etc.)
│       └── S2/
│           ├── S2.pkl                 # RespiBan & Empatica E4 physiological signals
│           └── S2_readme.pdf
└── processed/                         # Standardized tensor/numpy matrices post-alignment
    └── multimodal_aligned_index.json  # Master cross-modal index alignment ledger
```

---

## 📥 2. Modality Dataset Acquisition Guides

| Modality | Target Dataset | Access Link / Download CLI Command |
| :--- | :--- | :--- |
| **Text** | **Reddit SuicideWatch** | Download via Kaggle CLI:<br>`kaggle datasets download -d nikhileshwaraji/suicidal-tweet-detection-dataset` |
| **Audio** | **RAVDESS Emotional Speech** | Download via Zenodo link:<br>`wget https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip` |
| **Vision** | **FER2013 Faces** | Download via Kaggle CLI:<br>`kaggle datasets download -d msambare/fer2013` |
| **IoT** | **WESAD Physiological Wearable** | Download from UCI Machine Learning Repository:<br>`wget https://archive.ics.uci.edu/ml/machine-learning-databases/00465/WESAD.zip` |

---

## 🔄 3. Shared Labeling Alignment Scheme

To perform unified early and late fusion, heterogeneous labels from individual studies are dynamically mapped into a standardized 3-tier clinical risk taxonomy score (`0`, `1`, or `2`) using `src/utils/data_aligner.py`:

```text
+------------------------+-----------------------+-----------------------+
|  Modality Raw Label    |   Aligned Risk Class  |    Clinical Meaning   |
+------------------------+-----------------------+-----------------------+
|  Text: "non-suicidal"  |                       |                       |
|  Audio: "calm/happy"   |     0: LOW RISK       |   Healthy Baseline,   |
|  Vision: "happy/neut"  |                       |   Normal Somatic/     |
|  IoT: "meditation"     |                       |   Affective Baseline  |
+------------------------+-----------------------+-----------------------+
|  Text: "depressed"     |                       |                       |
|  Audio: "sad/disgust"  |   1: MODERATE RISK    |   Somatic Stress,     |
|  Vision: "sad/disgust" |                       |   Mild Affect Flatness|
|  IoT: "stress"         |                       |   Emotional Distress  |
+------------------------+-----------------------+-----------------------+
|  Text: "suicidal"      |                       |                       |
|  Audio: "fear/angry"   |     2: HIGH RISK      |   Acute Panic,        |
|  Vision: "fear/angry"  |                       |   Hyperarousal,       |
|  IoT: "panic/flat"     |                       |   Suicide Ideation    |
+------------------------+-----------------------+-----------------------+
```

---

## 🗃️ 4. Shared Data-Versioning System (DVC)

Because raw biometric, audio, and visual data files exceed standard Git sizes, we use **Data Versioning (DVC)**. DVC allows tracking massive datasets within Git by committing small `.dvc` meta-pointer files while storing raw datasets in S3 or local storage.

### **Step 1: Install DVC**
```bash
pip install dvc
```

### **Step 2: Initialize DVC in the Workspace**
```bash
dvc init
```

### **Step 3: Track Raw Data Directories**
Run this command to track your acquired datasets. DVC will create a small `data/.dvc` pointer file:
```bash
dvc add data/raw/
```

### **Step 4: Commit to Git**
Git will ignore the heavy raw files in `data/raw/` and instead track only the lightweight pointer file `data/raw.dvc`:
```bash
git add data/raw.dvc .gitignore
git commit -m "Track acquired raw modality datasets under DVC versioning"
```

### **Step 5: Pull Data on a New Checkout**
If another developer clones this repo, they can instantly retrieve the full raw data folders by running:
```bash
dvc pull
```
---

## 📝 5. Code Example: Dynamic Cross-Modal Label Alignment

You can run your alignment indexer directly inside Python:

```python
from src.utils.data_aligner import MultimodalDataIndexAligner

aligner = MultimodalDataIndexAligner()

# Create a cross-modal record aligning 4 independent dataset streams
aligned_record = aligner.generate_aligned_metadata_entry(
    patient_id="Subject_002",
    text_raw_label="depressed",         # mapped to 1 (Moderate)
    audio_raw_label="sad",              # mapped to 1 (Moderate)
    vision_raw_label="neutral",          # mapped to 0 (Low)
    iot_raw_label="stress",             # mapped to 1 (Moderate)
    text_file_path="data/raw/text_reddit/Subject_002_posts.txt",
    audio_file_path="data/raw/audio_ravdess/Actor_02/03-01-02-01-01-01-02.wav",
    vision_file_path="data/raw/vision_fer2013/train/sad/im234.png",
    iot_file_path="data/raw/iot_wesad/S2/S2.pkl"
)

# Save the indexer configuration
aligner.save_aligned_indexing_ledger([aligned_record])
```
This guarantees that **all 4 modalities are aligned to the same clinical taxonomy**, unlocking seamless multi-sensory deep-learning optimization!
