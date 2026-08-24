# ⚖️ Clinical Scoping & Ethical Charter (Phase 0)

This document establishes the ethical, legal, and clinical scoping charter for the **Multimodal Emotion Detection and Suicide Risk Assessment** project. This framework is designed in accordance with standard bioethical principles, clinical decision support (CDS) guidelines, and data protection standards (e.g., HIPAA, GDPR).

---

## 🎯 1. Clinical Scope: Defining "Risk" as an Output

The core objective of this system is to serve as a **screening and triage decision-support tool**, *never* as an autonomous diagnostic instrument.

```text
+-----------------------+      +-----------------------+      +-------------------------+
|  Multimodal Telemetry  | ---> |  AI Screening Layer   | ---> |  Human Clinical Triage  |
| (Text, Voice, Camera) |      | (Low/Mod/High Triage) |      |   (Licensed Review)     |
+-----------------------+      +-----------------------+      +-------------------------+
                                                                          |
                                                                          v
                                                              +-------------------------+
                                                              |  Diagnostic & Treatment |
                                                              +-------------------------+
```

### 🔍 Risk as Triage, Not Diagnosis
*   **The AI Output is a Triage Signal:** The classified outputs (`Low Risk`, `Moderate Risk`, `High Risk`) represent **statistical proxies of autonomic and psychological distress**. These markers indicate physiological and linguistic arousal correlating with depressive or anxious states.
*   **Non-Diagnostic Limitation:** This system **cannot** render diagnoses of clinical depression, generalized anxiety disorder (GAD), post-traumatic stress disorder (PTSD), or suicidal ideation. Diagnostic assessments must remain the exclusive purview of licensed human psychiatrists, clinical psychologists, and medical doctors.
*   **False-Positive/False-Negative Management:**
    *   **False Positives (High triage when calm):** Managed via gentle clinical routing to avoid distress.
    *   **False Negatives (Calm triage when in distress):** Mitigation requires regular, structured self-reporting and secondary safety nets (e.g., direct panic buttons).

---

## 🧑‍⚕️ 2. Human-in-the-Loop (HITL) Requirements

This framework mandates a strict **human-in-the-loop (HITL)** operational standard. The AI acts as an assistant, with absolute prohibition on autonomous administrative or clinical action.

### 🚫 No Autonomous Clinical Intervention
*   **No Automated Dispensing or Lockdown:** Under no circumstances will this system trigger automated medical intervention, locked facilities, physical constraints, or medical record flagging without prior human authentication.
*   **Clinician-in-the-Loop Triage:** When a `High Risk` or `Moderate Risk` score is generated:
    1.  The telemetry and feature-breakdowns are securely routed to a **licensed mental health professional** or crisis hotline operator.
    2.  The clinician reviews the multimodal raw signals (facial crops, speech logs, physiological curves) alongside the AI confidence score.
    3.  The human clinician conducts clinical evaluation and initiates outreach or emergency intervention if warranted.
*   **Explainable AI (XAI):** The system must present *why* a triage level was predicted (e.g., showing low HRV alongside specific crisis words) so that the human reviewer can interpret the assessment within seconds.

---

## 🔬 3. IRB, Consent, and Data Retention Protocols

If this system is transition from synthetic simulations to active development using **real human subjects**, the following clinical research protocols must be strictly enforced:

### 🏛️ Institutional Review Board (IRB) Oversight
*   **Ethical Approval:** Prior to the collection of any biological, vocal, or physical data from human subjects, full protocol approval must be secured from a registered **Institutional Review Board (IRB)** or Independent Ethics Committee (IEC).
*   **Risk-Benefit Analysis:** The clinical trial protocol must prove that the potential benefit of early suicidal ideation detection significantly outweighs the psychological stress of monitoring.

### ✍️ Multi-Layered Informed Consent
Consent must be obtained via active, multi-layered, and revocable processes:
*   **Modality-Specific Consent:** Subjects must be allowed to opt-in or opt-out of specific sensor streams (e.g., consenting to text and IoT telemetry but denying video facial frame capture) without losing baseline screening access.
*   **Revocability:** Subjects retain the absolute right to withdraw consent and halt data collection at any point during monitoring, triggering an immediate and complete wipe of their historical records.
*   **Age-Specific Protocols:** For minors (under 18), parental/guardian consent alongside minor assent is legally mandatory under pediatric research standards.

### 🔒 Data Minimization and Retention Policies
To protect highly sensitive biometric and psychiatric indicators, the system must enforce strict **zero-persistence** and **local-edge processing** protocols:

| Modality | Collection Practice | Retention Policy | Security Standard |
| :--- | :--- | :--- | :--- |
| **Text (Chat Logs)** | Local transcription only. | Deleted immediately post-inference unless flagged for clinician review (max 24 hours). | AES-256 local database encryption. |
| **Audio (Speech)** | Raw WAV streaming processed in volatile memory. | **Never saved on disk.** Only aggregated 39-dimensional acoustic metrics (MFCCs) are cached. | Memory-buffer scrubbing post-inference. |
| **Vision (Face)** | Live webcam stream processed locally on the client. | **No image files saved.** Frames are cropped, processed, and destroyed within volatile memory. | Local GPU memory scrubbing. |
| **IoT (Physiological)** | Smartwatch Bluetooth/local packet streaming. | Retained up to 7 days for historical trend graphing. | End-to-end encrypted telemetry packets. |

*   **De-identification:** All saved clinician-review files must have personal identifying information (PII) removed or hashed to ensure patient anonymity.
