import string
import numpy as np
from typing import Dict, Any, List, Optional


class ClinicalExplainabilityEngine:
    def __init__(self):
        pass

    @staticmethod
    def explain_text_predictions(text_input: str, prediction_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts local explainability for text predictions.
        Identifies active linguistic triggers and self-focus metrics from the text.
        """
        # Strip punctuation from each word for exact matches
        words = [w.strip(string.punctuation) for w in text_input.lower().split() if w.strip(string.punctuation)]
        
        # Identify matched keywords and self-focus pronouns
        crisis_words = ["kill", "suicide", "die", "hopeless", "end", "death", "pain", "goodbye", "depressed", "worthless"]
        pronouns = ["i", "me", "my", "myself", "mine"]
        
        matched_triggers = [w for w in words if w in crisis_words]
        matched_pronouns = [w for w in words if w in pronouns]

        # Calculate word contributions
        explanation = {
            "linguistic_impact_analysis": "Primary risk is driven by linguistic despair indicators." if matched_triggers else "No acute crisis vocabulary detected.",
            "identified_crisis_triggers": list(set(matched_triggers)),
            "self_focus_index": f"{len(matched_pronouns) / max(1, len(words)):.1%} of vocabulary" if matched_pronouns else "Low self-focus markers.",
            "risk_level_rationale": f"Classifier predicted {prediction_res['risk_level']} with {prediction_res['risk_probability']:.1%} confidence."
        }
        return explanation

    @staticmethod
    def explain_audio_predictions(audio_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates raw vocal acoustic features (pitch, jitter, shimmer, silence)
        into a descriptive, clinician-readable acoustic profile.
        """
        summary = audio_res.get("acoustic_features_summary", {})
        shimmer = summary.get("voice_shimmer_est", 0.0)
        silence = summary.get("silence_ratio", 0.0)
        
        # Clinical interpretations of acoustic anomalies
        vocal_profile = []
        if shimmer > 0.8:
            vocal_profile.append("High Jitter/Shimmer detected, indicating severe vocal instability (voice trembling under high anxiety).")
        elif shimmer < 0.3:
            vocal_profile.append("Extremely flat vocal intensity, matching clinical flat affect (severe depression).")
        else:
            vocal_profile.append("Vocal modulation remains within safe baseline physiological bounds.")

        if silence > 0.35:
            vocal_profile.append("Elevated Silence Ratio detected, indicating long pauses and speech hesitation (clinical depressive motor slowing).")

        return {
            "vocal_acoustic_rationales": vocal_profile,
            "estimated_shimmer_db": f"{shimmer:.2f} dB",
            "vocal_flatness_index": "Severe flat affect" if shimmer < 0.3 else "Normal vocal range",
            "hesitation_index": f"{silence:.1%} speech pauses" if silence > 0 else "Normal speech cadence"
        }

    @staticmethod
    def explain_vision_predictions(vision_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interprets facial expression probabilities to identify active facial muscle micro-expressions.
        """
        probs = vision_res.get("emotion_probabilities", {})
        dominant = vision_res.get("dominant_emotion", "neutral")
        
        visual_profile = []
        # Classify high-probability micro-expressions
        for emo, prob in probs.items():
            if prob > 0.40 and emo in ["sadness", "anxiety/fear", "anger"]:
                visual_profile.append(f"Elevated {emo.upper()} facial micro-expression detected (Confidence: {prob:.1%}).")

        if not visual_profile:
            visual_profile.append("Facial musculature displays relaxed, neutral, or happy baselines.")

        return {
            "facial_expression_rationales": visual_profile,
            "dominant_emotional_expressiveness": dominant.upper(),
            "distress_expression_load": f"{vision_res.get('facial_distress_score', 0.0):.1%} of visual grid"
        }

    @staticmethod
    def explain_iot_predictions(iot_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs autonomic biometric attribution, calculating exactly which somatic
        signals (HR, HRV, GSR) contributed most heavily to the Autonomic Stress Index.
        """
        metrics = iot_res.get("metrics", {})
        hr = metrics.get("heart_rate_bpm", 72.0)
        hrv = metrics.get("hrv_rmssd_ms", 50.0)
        eda = metrics.get("skin_conductance_us", 2.0)
        temp = metrics.get("skin_temperature_c", 33.5)

        # 1. Calculate relative weight contribution to Autonomic Stress Index
        # Somatic formula weights: HR (20%), HRV (35%), EDA (35%), Temp (10%)
        hr_contrib = max(0.0, min(1.0, (hr - 60.0) / 40.0)) * 0.20
        hrv_contrib = max(0.0, min(1.0, (70.0 - hrv) / 50.0)) * 0.35
        eda_contrib = max(0.0, min(1.0, eda / 8.0)) * 0.35
        temp_contrib = max(0.0, min(1.0, (34.0 - temp) / 4.0)) * 0.10

        total = hr_contrib + hrv_contrib + eda_contrib + temp_contrib
        if total == 0:
            total = 1e-8

        # Convert to percentage contributions
        attributions = {
            "Heart Rate (BPM) Contribution": f"{hr_contrib / total:.1%}",
            "HRV (RMSSD ms) Contribution": f"{hrv_contrib / total:.1%}",
            "GSR / Skin Conductance Contribution": f"{eda_contrib / total:.1%}",
            "Skin Temperature Contribution": f"{temp_contrib / total:.1%}"
        }

        # 2. Formulate clinical physiological interpretation
        somatic_profile = []
        if hrv < 25.0:
            somatic_profile.append("Severe reduction in Heart Rate Variability (RMSSD), indicating chronic autonomic distress or hyperarousal.")
        if eda > 5.0:
            somatic_profile.append("Spiking Galvanic Skin Conductance, indicating active sympathetic nervous system fight-or-flight panic.")
        if temp < 31.0:
            somatic_profile.append("Peripheral skin temperature dropping, indicating severe stress-induced vasoconstriction.")

        if not somatic_profile:
            somatic_profile.append("Somatic biometrics (HR, HRV, EDA, Temp) sit safely within calm, healthy baselines.")

        return {
            "autonomic_physiological_rationales": somatic_profile,
            "biometric_feature_attributions": attributions,
            "physiological_arousal_state": iot_res.get("physiological_state", "Calm / Low Stress")
        }

    def generate_clinical_xai_report(self, 
                                     text_res: Optional[Dict[str, Any]] = None,
                                     audio_res: Optional[Dict[str, Any]] = None,
                                     vision_res: Optional[Dict[str, Any]] = None,
                                     iot_res: Optional[Dict[str, Any]] = None,
                                     text_input: str = "") -> Dict[str, Any]:
        """
        Synthesizes individual modal explainability structures into a single, cohesive,
        clinician-readable Cross-Modal Explainable AI (XAI) Report.
        """
        report = {
            "clinician_alert_summary": "Unified Clinical Explainability Audit Trail",
            "modality_explanations": {}
        }

        if text_res and text_input:
            report["modality_explanations"]["textual_linguistics_nlp"] = self.explain_text_predictions(text_input, text_res)
        if audio_res:
            report["modality_explanations"]["vocal_speech_acoustics"] = self.explain_audio_predictions(audio_res)
        if vision_res:
            report["modality_explanations"]["facial_expression_cv"] = self.explain_vision_predictions(vision_res)
        if iot_res:
            report["modality_explanations"]["physiological_wearable_iot"] = self.explain_iot_predictions(iot_res)

        return report
