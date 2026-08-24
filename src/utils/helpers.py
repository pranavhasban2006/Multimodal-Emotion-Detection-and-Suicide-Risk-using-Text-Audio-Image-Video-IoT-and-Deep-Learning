import os
import wave
import struct
import math
from PIL import Image, ImageDraw

def generate_sample_audio(file_path: str, duration_sec: float = 3.0, sample_rate: int = 16000):
    """
    Generates a mock wav audio file containing a pure sine wave (440Hz, A4 note).
    This allows testing the audio pipeline without having to download or record real files first.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # 16-bit audio params
    num_samples = int(duration_sec * sample_rate)
    frequency = 440.0 # Standard A4 tone
    amplitude = 16000 # Max value for 16-bit is 32767
    
    with wave.open(file_path, 'wb') as wav_file:
        # Channels: Mono, Sample Width: 2 bytes (16-bit), Frame Rate: sample_rate
        wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
        
        # Write sine wave values
        for i in range(num_samples):
            t = float(i) / sample_rate
            value = int(amplitude * math.sin(2.0 * math.pi * frequency * t))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)
            
    print(f"[Utils] Created mock audio asset: {file_path}")


def generate_sample_image(file_path: str, emotion_type: str = "sad"):
    """
    Generates a mock PNG image showing a simple expressive face.
    This allows testing the vision pipeline out-of-the-box.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Create 256x256 image with light gray background
    img = Image.new('RGB', (256, 256), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw face circle (yellow-ish/pale)
    draw.ellipse([40, 40, 216, 216], fill=(255, 230, 180), outline=(50, 50, 50), width=3)
    
    # Draw eyes (left and right)
    draw.ellipse([80, 90, 100, 110], fill=(50, 50, 50))
    draw.ellipse([156, 90, 176, 110], fill=(50, 50, 50))
    
    # Draw eyebrows depending on emotion
    if emotion_type == "sad":
        # Sad eyebrows (slanted upwards in the middle)
        draw.line([75, 85, 105, 75], fill=(50, 50, 50), width=3)
        draw.line([151, 75, 181, 85], fill=(50, 50, 50), width=3)
        # Sad frown mouth (arc curving down)
        draw.arc([90, 140, 166, 190], start=180, end=360, fill=(50, 50, 50), width=3)
    elif emotion_type == "angry":
        # Angry eyebrows (slanted downwards in the middle)
        draw.line([75, 75, 105, 85], fill=(50, 50, 50), width=3)
        draw.line([151, 85, 181, 75], fill=(50, 50, 50), width=3)
        # Flat angry mouth
        draw.line([100, 160, 156, 160], fill=(50, 50, 50), width=3)
    else: # happy / calm
        # Curved relaxed eyebrows
        draw.arc([75, 75, 105, 95], start=180, end=360, fill=(50, 50, 50), width=3)
        draw.arc([151, 75, 181, 95], start=180, end=360, fill=(50, 50, 50), width=3)
        # Smiley mouth (arc curving up)
        draw.arc([90, 120, 166, 170], start=0, end=180, fill=(50, 50, 50), width=3)
        
    img.save(file_path)
    print(f"[Utils] Created mock image asset: {file_path}")


def setup_all_sample_assets(data_dir: str = "data/raw"):
    """
    Creates multiple sample files to play with immediately.
    """
    # 1. Text samples
    # Written directly as a list for documentation and easy copy-pasting
    
    # 2. Audio samples
    generate_sample_audio(os.path.join(data_dir, "sample_vocal_normal.wav"), duration_sec=3.0)
    
    # 3. Vision samples
    generate_sample_image(os.path.join(data_dir, "sample_face_sad.png"), emotion_type="sad")
    generate_sample_image(os.path.join(data_dir, "sample_face_happy.png"), emotion_type="happy")
    generate_sample_image(os.path.join(data_dir, "sample_face_angry.png"), emotion_type="angry")
