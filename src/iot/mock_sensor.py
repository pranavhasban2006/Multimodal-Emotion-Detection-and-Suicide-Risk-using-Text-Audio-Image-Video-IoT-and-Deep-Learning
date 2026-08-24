import random
import time
from typing import Dict

class IoTSensorSimulator:
    """
    ⚠️ WARNING: DEMO-ONLY & SIMULATION-ONLY CLASS.
    ---------------------------------------------
    This class generates synthetic hardcoded biometric readings for interactive user interface
    simulations (such as the Streamlit app).
    
    This class MUST NEVER be used to generate training, validation, or testing datasets.
    Real model evaluations must exclusively load and parse the official WESAD clinical dataset.
    """
    def __init__(self, state="normal"):
        """
        state: 'normal', 'depressed', 'stressed/anxious'
        """
        self.state = state
        self.hr_history = []
        self.eda_history = []
        
    def set_state(self, state: str):
        self.state = state

    def generate_reading(self) -> Dict[str, float]:
        """
        Generates a synthetic smartwatch reading for real-time frontend dashboard rendering.
        """
        if self.state == "stressed/anxious":
            hr = random.uniform(92.0, 115.0)
            rmssd = random.uniform(12.0, 24.0)
            eda = random.uniform(5.5, 9.8)
            temp = random.uniform(30.2, 31.8)
        elif self.state == "depressed":
            hr = random.uniform(62.0, 72.0)
            rmssd = random.uniform(10.0, 18.0)
            eda = random.uniform(0.8, 1.8)
            temp = random.uniform(31.5, 32.8)
        else: # normal / calm
            hr = random.uniform(68.0, 78.0)
            rmssd = random.uniform(42.0, 65.0)
            eda = random.uniform(1.2, 3.2)
            temp = random.uniform(33.2, 34.5)

        # Generate mock R-R intervals (in ms) matching target RMSSD
        rr_intervals = []
        base_interval = 60000.0 / hr
        for _ in range(10):
            offset = random.choice([-1, 1]) * rmssd * random.uniform(0.7, 1.3)
            rr_intervals.append(base_interval + offset)

        return {
            "timestamp": time.time(),
            "heart_rate": round(hr, 2),
            "rr_intervals_ms": [round(x, 1) for x in rr_intervals],
            "eda_microsiemens": round(eda, 3),
            "skin_temp_c": round(temp, 2),
            "is_demo_reading": True  # Explicit flag marking this as synthetic UI telemetry
        }
