# 📈 Phase 2: Feature Extraction & Signal Processing (DSP)

This document explains the mathematical foundations, digital signal processing (DSP) equations, and feature engineering transformations used by the framework to translate raw sensor streams into standardized deep learning tensors.

---

## 📝 1. Text NLP: Semantic Embeddings & Sequence Tokenization

Raw character text is unstructured and high-dimensional. Our text pipeline (`src/text/preprocess.py` & `src/text/model.py`) converts characters into numerical dense representations using sequence tokenization and deterministic word hashing:

### Sequence Tokenization & Padding
To feed text into sequence models (like our Bi-LSTM architecture), words are mapped to numerical indices based on a vocabulary lexicon $\mathcal{V}$:
$$x_t = \text{TokenID}(w_t) \quad \text{where} \quad w_t \in \mathcal{V}$$

Since different sentences have varying word lengths $L$, we apply post-padding or truncation to enforce a standard sequence length $N$ (e.g., $N=50$):
$$\mathbf{x}_{\text{padded}} = [x_1, x_2, \dots, x_L, 0, 0, \dots, 0] \in \mathbb{R}^N$$

### Deterministic Hashing Trick
For lightweight machine learning classification, we apply a hash-tokenization trick that maps word tokens $w_t$ into a fixed-size $D$-dimensional feature vector (e.g., $D=50$):
$$f_i = \sum_{w \in \text{text}} \mathbb{I}(\text{hash}(w) \pmod D == i)$$
We then normalize the vector using the $L_2$ norm to prevent text length bias:
$$\mathbf{\hat{f}} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$$

---

## 🎙️ 2. Vocal Acoustics: Digital Signal Processing (DSP)

Speech signals are highly non-stationary. Our audio pipeline (`src/audio/preprocess.py`) extracts spectral vocal features using **Digital Signal Processing (DSP)** via Fourier Transforms:

```text
+------------------+      +-------------------+      +----------------------+      +----------------------+
| Raw Audio Wave   | ---> |  Windowing (STFT) | ---> |  Mel Filterbank Map  | ---> |  DCT Log (MFCCs)     |
| (1D Amplitude)   |      | (Spectral Energy) |      | (Human Auditory scale|      | (Acoustic Coeffs)    |
+------------------+      +-------------------+      +----------------------+      +----------------------+
```

### Mel-Frequency Cepstral Coefficients (MFCCs)
MFCCs represent the short-term power spectrum of audio on a non-linear Mel-scale matching human hearing:
1.  **Short-Time Fourier Transform (STFT):** Raw signal $y[n]$ is windowed (using a Hann window) and converted to the frequency domain:
    $$X(m, \omega) = \sum_{n=-\infty}^{\infty} y[n] w[n-m] e^{-j\omega n}$$
2.  **Mel-Scale Mapping:** Frequencies in Hz are converted to the Mel scale:
    $$M(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
3.  **Discrete Cosine Transform (DCT):** Taking the logarithm of the Mel-filterbank energies and applying DCT extracts the cepstral coefficients, capturing voice trembling and throat constrictions:
    $$\text{MFCC}[k] = \sum_{m=1}^{M} \log(Y_{\text{mel}}[m]) \cos\left( \frac{\pi k}{M} \left(m - \frac{1}{2}\right) \right)$$

### Spectral Centroid
Indicates the "center of gravity" of the spectrum, strongly correlating with vocal "brightness" or "flatness" (clinical depression affect):
$$\mu_{\text{centroid}} = \frac{\sum_{k=0}^{N-1} f[k] |X[k]|^2}{\sum_{k=0}^{N-1} |X[k]|^2}$$

---

## 📷 3. Computer Vision: Spatial Transformations & Face Cropping

Raw frames from video feeds or images are highly noisy, containing background clutter. Our vision pipeline (`src/vision/preprocess.py`) isolates emotional signals via spatial transforms:

### Face Detection via Haar Cascades
We apply a boosted cascade of simple classifiers that evaluate spatial pixel differences (Haar-like features) over an image window:
$$\text{Feature} = \sum \text{Pixels}_{\text{dark}} - \sum \text{Pixels}_{\text{light}}$$
The sliding window evaluates features and isolates face boundary coordinates $[x_0, y_0, w, h]$.

### Spatial Cropping & Standardized Resolution
We crop the facial bounding box to discard background noise and downsample using bilinear interpolation to a standard resolution (e.g., $128 \times 128$):
$$I_{\text{cropped}} = I[y_0 : y_0 + h, \ x_0 : x_0 + w]$$
$$I_{\text{std}} = \text{Resize}(I_{\text{cropped}}, \ (128, 128))$$

### Pixel Normalization
To prevent neural networks from being sensitive to brightness and lighting changes, we normalize pixel intensities to a $[0, 1]$ interval:
$$\hat{I}[i, j, c] = \frac{I_{\text{std}}[i, j, c]}{255.0}$$

---

## ⌚ 4. IoT Wearables: Biometric Feature Engineering

Physiological signals are indicators of autonomic nervous system (ANS) activity. Our IoT pipeline (`src/iot/signal_processor.py`) extracts somatic distress metrics from smartwatch streams:

### Heart Rate Variability (HRV) RMSSD
HRV represents the beat-to-beat fluctuations controlled by parasympathetic activity. We calculate **RMSSD** (Root Mean Square of Successive Differences) from the sequence of R-R intervals (in milliseconds):
$$\text{RMSSD} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N-1} (RR_{i+1} - RR_i)^2}$$
*   **Clinical Significance:** A healthy resting RMSSD ranges between $30\text{--}70 \text{ ms}$. A collapse in RMSSD ($< 25 \text{ ms}$) indicates sympathetic dominance (high autonomic stress, depression, or acute panic).

### Electrodermal Activity (EDA/GSR) Normalization
EDA/GSR measures skin conductance (in microsiemens $\mu S$), driven by sweat gland activity during emotional arousal:
$$\text{EDA}_{\text{scaled}} = \frac{\text{EDA} - \text{EDA}_{\text{baseline}}}{\text{EDA}_{\text{max}} - \text{EDA}_{\text{baseline}}}$$
*   **Clinical Significance:** Elevated, spiking EDA signals immediate emotional distress and panic, while extremely flat, unresponsive EDA signals clinical flat affect.

### Autonomic Stress Index Synthesis
Our signal processor synthesizes all four somatic signals (Heart Rate $HR$, HRV $RMSSD$, Electrodermal $EDA$, and Skin Temp $T$) into a single normalized index $S_{\text{index}} \in [0, 1]$:
$$S_{\text{index}} = 0.20 \cdot \overline{HR} + 0.35 \cdot (1 - \overline{RMSSD}) + 0.35 \cdot \overline{EDA} + 0.10 \cdot (1 - \overline{T})$$
where each variable represents the locally normalized z-score or min-max scaled telemetry value.
