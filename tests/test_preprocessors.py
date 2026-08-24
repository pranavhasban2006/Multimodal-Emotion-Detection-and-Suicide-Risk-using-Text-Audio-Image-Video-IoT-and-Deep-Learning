import os
import unittest
import numpy as np
from PIL import Image

from src.text.preprocess import TextPreprocessor
from src.audio.preprocess import AudioPreprocessor
from src.vision.preprocess import VisionPreprocessor
from src.iot.signal_processor import IoTSignalProcessor


class TestClinicalPreprocessors(unittest.TestCase):
    
    def setUp(self):
        self.text_prep = TextPreprocessor(vocab_path="models/test_vocab.json", max_seq_len=50)
        self.audio_prep = AudioPreprocessor()
        self.vision_prep = VisionPreprocessor(target_size=(128, 128))
        self.iot_proc = IoTSignalProcessor()

    def tearDown(self):
        # Clean up test vocab file if generated
        if os.path.exists("models/test_vocab.json"):
            try:
                os.remove("models/test_vocab.json")
            except Exception:
                pass

    # ==================== 1. TEXT PREPROCESSOR TESTS ====================

    def test_text_empty_input(self):
        """Verify text preprocessor handles empty/corrupt strings gracefully."""
        res_clean = self.text_prep.anonymize_and_clean("")
        self.assertEqual(res_clean, "")
        
        features = self.text_prep.extract_linguistic_features("")
        self.assertEqual(features["first_person_rate"], 0.0)
        self.assertEqual(features["crisis_lexicon_rate"], 0.0)

        seq = self.text_prep.text_to_sequence("")
        self.assertEqual(seq.shape, (50,))
        self.assertEqual(seq[0], 0) # PAD token ID

    def test_text_pii_anonymization(self):
        """Verify usernames, social handles, subreddits and URLs are stripped."""
        raw_post = "My post on r/SuicideWatch by /u/john_doe. Reach me at https://help.org or @twitter."
        anonymized = self.text_prep.anonymize_and_clean(raw_post)
        
        self.assertNotIn("john_doe", anonymized)
        self.assertNotIn("SuicideWatch", anonymized)
        self.assertNotIn("https://help.org", anonymized)
        self.assertNotIn("@twitter", anonymized)
        
        self.assertIn("[USER]", anonymized)
        self.assertIn("[SUBREDDIT]", anonymized)
        self.assertIn("[URL]", anonymized)
        self.assertIn("[HANDLE]", anonymized)

    def test_text_linguistic_markers(self):
        """Verify punctuation intensity, pronoun rate and crisis lexicons count correctly."""
        raw_text = "I feel hopeless... I want to end it now!!"
        features = self.text_prep.extract_linguistic_features(raw_text)
        
        # Verify first person pronoun "I" and crisis words "hopeless" are matched
        self.assertGreater(features["first_person_rate"], 0.0)
        self.assertGreater(features["crisis_lexicon_rate"], 0.0)
        self.assertGreater(features["exclamation_rate"], 0.0)
        self.assertGreater(features["ellipsis_rate"], 0.0)

    # ==================== 2. AUDIO PREPROCESSOR TESTS ====================

    def test_audio_corrupt_or_missing_path(self):
        """Verify audio preprocessor raises errors when configured or fails gracefully."""
        # 1. Test failure with raise_errors=True
        with self.assertRaises((FileNotFoundError, ImportError)):
            self.audio_prep.extract_features("data/raw/completely_missing_file_xyz.wav", raise_errors=True)

        # 2. Test fallback handling with raise_errors=False
        feats, is_valid = self.audio_prep.extract_features("data/raw/completely_missing_file_xyz.wav", raise_errors=False, return_tuple=True)
        self.assertFalse(is_valid)
        self.assertEqual(feats.shape, (47,))
        self.assertTrue(np.all(feats == 0.0))

    # ==================== 3. VISION PREPROCESSOR TESTS ====================

    def test_vision_no_face_detection_fallback(self):
        """Verify image preprocessor handles no-face frames safely by returning resized output."""
        # Generate an empty solid canvas containing absolutely no facial structures
        blank_canvas = Image.new('RGB', (300, 300), color=(10, 20, 30))
        processed = self.vision_prep.preprocess_image(blank_canvas)
        
        # Output shape must still match standard target resolution
        self.assertEqual(processed.shape, (128, 128, 3))
        self.assertGreaterEqual(np.min(processed), 0.0)
        self.assertLessEqual(np.max(processed), 1.0)

    # ==================== 4. IoT PHYSIOLOGICAL TESTS ====================

    def test_iot_empty_rr_intervals(self):
        """Verify HRV calculator does not break when interval history is empty or short."""
        rmssd = self.iot_proc.calculate_hrv_rmssd([])
        sdnn = self.iot_proc.calculate_hrv_sdnn([])
        pnn50 = self.iot_proc.calculate_hrv_pnn50([])
        
        # Output must degrade to safe baseline defaults instead of crashing on ZeroDivision
        self.assertEqual(rmssd, 50.0)
        self.assertEqual(sdnn, 50.0)
        self.assertEqual(pnn50, 0.0)

        # Single R-R interval check
        self.assertEqual(self.iot_proc.calculate_hrv_rmssd([820.0]), 50.0)


if __name__ == "__main__":
    unittest.main()
