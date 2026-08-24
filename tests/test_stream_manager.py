import unittest
import time
import numpy as np

from src.fusion.stream_manager import TelemetryStreamManager


class TestTelemetryStreamManager(unittest.TestCase):
    
    def setUp(self):
        # Initialize stream manager with tight TTL bounds for fast test validation
        self.manager = TelemetryStreamManager(
            text_ttl_sec=1.0,     # expires in 1 second
            audio_ttl_sec=10.0,   # expires in 10 seconds
            vision_ttl_sec=5.0,   # expires in 5 seconds
            iot_ttl_sec=0.1       # expires in 0.1 seconds
        )

    def test_packet_ingestion_and_decay(self):
        """Verify that newly ingested packets are active, and decay mathematically over time."""
        now = time.time()
        
        # Fake predictions & feature vectors
        dummy_pred = {"risk_probability": 0.25, "risk_level": "Low Risk", "dominant_emotion": "neutral"}
        dummy_feat = np.zeros((50,))
        
        # Ingest text packet
        self.manager.ingest_packet("text", dummy_pred, dummy_feat)
        
        # 1. Decay immediately after ingestion should be very close to 1.0 (no decay)
        decay_init = self.manager.calculate_temporal_decay_weight("text", current_time=time.time())
        self.assertGreater(decay_init, 0.95)
        self.assertLessEqual(decay_init, 1.0)
        
        # 2. Simulate 0.5 seconds passing: decay factor should decrease
        time.sleep(0.5)
        decay_mid = self.manager.calculate_temporal_decay_weight("text", current_time=time.time())
        self.assertLess(decay_mid, decay_init)
        self.assertGreater(decay_mid, 0.0)

        # 3. Simulate complete expiration (wait 1.1 seconds, exceeding 1.0s TTL)
        time.sleep(0.6)
        decay_expired = self.manager.calculate_temporal_decay_weight("text", current_time=time.time())
        self.assertEqual(decay_expired, 0.0)

    def test_dynamic_reweighted_fusion(self):
        """Verify that expired channels are automatically dropped from the active fusion pool."""
        dummy_pred_text = {"risk_probability": 0.85, "risk_level": "High Risk", "dominant_emotion": "sadness"}
        dummy_pred_audio = {"distress_probability": 0.10, "vocal_distress_level": "Low Risk", "dominant_emotion": "neutral"}
        
        # Ingest both text and audio packets
        self.manager.ingest_packet("text", dummy_pred_text, np.zeros((50,)))
        self.manager.ingest_packet("audio", dummy_pred_audio, np.zeros((47,)))
        
        # Both channels are currently active
        report_init = self.manager.get_active_realtime_fusion()
        self.assertIn("text", report_init["active_modalities"])
        self.assertIn("audio", report_init["active_modalities"])
        
        # Text has a high risk score, audio has a low risk score.
        # Fused score should be a weighted combination
        self.assertGreater(report_init["fused_risk_score"], 0.20)
        
        # Sleep 1.1 seconds (exceeding text's 1.0s TTL, but within audio's 10.0s TTL)
        time.sleep(1.1)
        
        # Retrieve fusion again: text should have expired and dropped out!
        report_post_expire = self.manager.get_active_realtime_fusion()
        
        self.assertNotIn("text", report_post_expire["active_modalities"])
        self.assertIn("audio", report_post_expire["active_modalities"])
        
        # Fused score should now represent ONLY the active audio channel (which is 0.10, Low Risk)
        self.assertEqual(report_post_expire["fused_suicide_risk_level"], "Low Risk")
        self.assertAlmostEqual(report_post_expire["fused_risk_score"], 0.10, places=2)


if __name__ == "__main__":
    unittest.main()
