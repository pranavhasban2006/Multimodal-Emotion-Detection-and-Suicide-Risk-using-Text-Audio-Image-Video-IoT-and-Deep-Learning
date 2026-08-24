import numpy as np
import warnings
import random
from PIL import Image

# Import opencv-python but support fallback if not installed
OPENCV_AVAILABLE = True
try:
    import cv2
except ImportError:
    OPENCV_AVAILABLE = False


class VisionPreprocessor:
    def __init__(self, target_size=(128, 128), grayscale=False, dataset_mode=None):
        """
        target_size: resolution to resize images (width, height)
        grayscale: whether to convert inputs to single-channel
        dataset_mode: 'fer2013' or None (skips face-cropping for already-cropped datasets)
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.dataset_mode = dataset_mode
        
        # Load OpenCV Haar Cascades for face and eye tracking
        self.face_cascade = None
        self.eye_cascade = None
        if OPENCV_AVAILABLE:
            try:
                face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                self.face_cascade = cv2.CascadeClassifier(face_path)
                self.eye_cascade = cv2.CascadeClassifier(eye_path)
            except Exception:
                pass

    def align_face(self, img_np: np.ndarray) -> np.ndarray:
        """
        Detects eyes using Haar Cascades, computes the horizontal tilt angle,
        and rotates the frame to align the face before cropping.
        """
        if not OPENCV_AVAILABLE or self.eye_cascade is None:
            return img_np

        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
            
            # We need exactly 2 detected eyes to compute rotation
            if len(eyes) == 2:
                # Sort eyes by X coordinate: left eye is eye with smaller X
                eyes = sorted(eyes, key=lambda e: e[0])
                left_eye, right_eye = eyes[0], eyes[1]
                
                # Calculate centers
                l_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
                r_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)
                
                # Compute angle
                dy = r_center[1] - l_center[1]
                dx = r_center[0] - l_center[0]
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)
                
                # Center of rotation (between eyes)
                eye_center = ((l_center[0] + r_center[0]) // 2, (l_center[1] + r_center[1]) // 2)
                
                # Get rotation matrix and warp frame
                h, w = img_np.shape[:2]
                rot_matrix = cv2.getRotationMatrix2D(eye_center, angle_deg, scale=1.0)
                rotated = cv2.warpAffine(img_np, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)
                return rotated
        except Exception:
            # Fall back to un-aligned image on any math/tracking errors
            pass
        return img_np

    def preprocess_image(self, img_input, augment: bool = False) -> np.ndarray:
        """
        Preprocesses images:
        1. Anonymizes background by isolating & cropping face (unless dataset_mode == 'fer2013').
        2. Applies eye-alignment rotation.
        3. Normalizes dimensions & pixel weights [0, 1].
        4. Implements dynamic training data augmentations (flip, brightness, rotation).
        """
        img_np = None
        if isinstance(img_input, str):
            if OPENCV_AVAILABLE:
                img_np = cv2.imread(img_input)
                if img_np is not None:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            else:
                with Image.open(img_input) as img:
                    img_np = np.array(img.convert('RGB'))
        elif isinstance(img_input, Image.Image):
            img_np = np.array(img_input.convert('RGB'))
        elif isinstance(img_input, np.ndarray):
            img_np = img_input.copy()
            
        if img_np is None:
            warnings.warn("Invalid image input. Generating blank frame.")
            return np.zeros((self.target_size[0], self.target_size[1], 1 if self.grayscale else 3))

        # Check dataset mode bypass (skip crop step for already cropped sets)
        cropped_face = img_np
        if self.dataset_mode != 'fer2013' and OPENCV_AVAILABLE and self.face_cascade is not None:
            try:
                # 1. Align face
                aligned_frame = self.align_face(img_np)
                
                # 2. Crop face
                gray_img = cv2.cvtColor(aligned_frame, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    cropped_face = aligned_frame[y:y+h, x:x+w]
            except Exception:
                pass

        # Resize and Grayscale conversion
        if OPENCV_AVAILABLE:
            if self.grayscale:
                if len(cropped_face.shape) == 3:
                    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(cropped_face, self.target_size)
                resized = np.expand_dims(resized, axis=-1)
            else:
                if len(cropped_face.shape) == 2:
                    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_GRAY2RGB)
                resized = cv2.resize(cropped_face, self.target_size)
        else:
            pil_img = Image.fromarray(cropped_face)
            if self.grayscale:
                pil_img = pil_img.convert('L')
            pil_img = pil_img.resize(self.target_size)
            resized = np.array(pil_img)
            if self.grayscale:
                resized = np.expand_dims(resized, axis=-1)

        # Normalize pixels to float
        normalized = resized.astype(np.float32) / 255.0

        # Apply augmentation for training partitions
        if augment:
            normalized = self._apply_augmentation(normalized)

        return normalized

    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """
        Applies basic training augmentations in pure numpy/opencv:
        - Random Horizontal Flip (50% prob)
        - Random Brightness adjustment (±15%)
        - Small random rotations (±10 degrees)
        """
        # 1. Horizontal Flip
        if random.random() > 0.5:
            img = np.fliplr(img)

        # 2. Brightness adjustment
        brightness_factor = random.uniform(0.85, 1.15)
        img = np.clip(img * brightness_factor, 0.0, 1.0)

        # 3. Small rotation
        if OPENCV_AVAILABLE:
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            # Re-ensure channel dimension is kept for grayscale
            if self.grayscale and len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)

        return img

    def extract_video_frames(self, video_path: str, fps_sample=1) -> list:
        """
        Reads a video file, samples frames, and crops faces.
        **CRITICAL FIX:** Frames where no face is detected are dropped completely 
        rather than zero-padded, preventing training models on blank black screens.
        """
        frames = []
        if not OPENCV_AVAILABLE:
            warnings.warn("OpenCV is missing. Skipping video processing.")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
        sample_interval = max(1, int(fps / fps_sample))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check face existence inside frame before saving
                if self.face_cascade is not None:
                    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    
                    if len(faces) > 0:
                        # Process and append valid facial frame
                        preprocessed = self.preprocess_image(rgb_frame)
                        frames.append(preprocessed)
                        
            frame_idx += 1
            
        cap.release()
        return frames
