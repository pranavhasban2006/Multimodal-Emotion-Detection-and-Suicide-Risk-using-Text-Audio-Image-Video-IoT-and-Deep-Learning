import unittest
import numpy as np

from src.utils.explainability import ClinicalExplainabilityEngine


class TestClinicalExplainabilityEngine(unittest.TestCase):
    
    def setUp(self):
        self.engine = ClinicalExplainabilityEngine()

    def test_text_explanation(self):
        """Verify linguistic triggers and self-focus indexing compute correctly."""
        text_input = "I feel hopeless and I want to die."
        pred_res = {"risk_level": "High Risk", "risk_probability": 0.95}
        
        explanation = self.engine.explain_text_predictions(text_input, pred_res)
        
        # Verify crisis words are parsed
        self.assertIn("hopeless", explanation["identified_crisis_triggers"])
        self.assertIn("die", explanation["identified_crisis_triggers"])
        
        # Verify self-focus rate counts the pronoun "I"
        self.assertIn("% of vocabulary", explanation["self_focus_index"])

    def test_audio_explanation(self):
        """Verify vocal trembling (shimmer) and speech pauses (silence) trigger correct rationales."""
        # 1. Test high-trembling speech
        audio_res_tremble = {
            "vocal_distress_level": "High Risk",
            "distress_probability": 0.85,
            "acoustic_features_summary": {
                "voice_shimmer_est": 1.2, # high trembling (>0.8)
                "silence_ratio": 0.1
            }
        }
        exp_tremble = self.engine.explain_audio_predictions(audio_res_tremble)
        self.assertTrue(any("trembling" in r for r in exp_tremble["vocal_acoustic_rationales"]))
        
        # 2. Test flat, hesitant speech
        audio_res_flat = {
            "vocal_distress_level": "High Risk",
            "distress_probability": 0.90,
            "acoustic_features_summary": {
                "voice_shimmer_est": 0.1, # extremely flat (<0.3)
                "silence_ratio": 0.45     # high silence (>0.35)
            }
        }
        exp_flat = self.engine.explain_audio_predictions(audio_res_flat)
        self.assertTrue(any("flat affect" in r for r in exp_flat["vocal_acoustic_rationales"]))
        self.assertTrue(any("pauses" in r for r in exp_flat["vocal_acoustic_rationales"]))

    def test_vision_explanation(self):
        """Verify facial expression probabilities output the correct micro-expression rationales."""
        vision_res = {
            "facial_distress_level": "High Risk",
            "facial_distress_score": 0.82,
            "dominant_emotion": "sadness",
            "emotion_probabilities": {
                "sadness": 0.65,
                "neutral": 0.10,
                "anxiety/fear": 0.20,
                "anger": 0.05,
                "happiness": 0.00
            }
        }
        explanation = self.engine.explain_vision_predictions(vision_res)
        self.assertEqual(explanation["dominant_emotional_expressiveness"], "SADNESS")
        self.assertTrue(any("SADNESS" in r for r in explanation["facial_expression_rationales"]))

    def test_iot_explanation_and_attributions(self):
        """Verify biometric attributions calculate correctly and sum up to 100%."""
        iot_res = {
            "physiological_stress_score": 0.85,
            "physiological_state": "High Autonomic Distress",
            "physiological_risk_level": "High Risk",
            "metrics": {
                "heart_rate_bpm": 105.0,        # high
                "hrv_rmssd_ms": 12.0,           # low RMSSD (severe distress)
                "skin_conductance_us": 7.5,     # high GSR (active fight-or-flight)
                "skin_temperature_c": 30.5      # dropping temp
            }
        }
        explanation = self.engine.explain_iot_predictions(iot_res)
        
        # Verify clinical autonomic rationales are populated
        self.assertTrue(any("variability" in r.lower() for r in explanation["autonomic_physiological_rationales"]))
        self.assertTrue(any("sweat" in r.lower() or "sympathetic" in r.lower() for r in explanation["autonomic_physiological_rationales"]))
        self.assertTrue(any("temperature" in r.lower() or "vasoconstriction" in r.lower() for r in explanation["autonomic_physiological_rationales"]))
        
        # Verify feature contribution string formats
        contribs = explanation["biometric_feature_attributions"]
        for key, val in contribs.items():
            self.assertTrue(val.endswith("%"))


if __name__ == "__main__":
    unittest.main()
