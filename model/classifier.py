"""
model/classifier.py
====================
Image classification pipeline using MobileNetV2 + ImageNet weights.

Pipeline:
  raw image file  →  OpenCV read  →  resize + normalize  →  MobileNetV2  →  top-K labels
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)

# Target input size required by MobileNetV2
INPUT_SIZE = (224, 224)


class ImageClassifier:
    """
    Wraps MobileNetV2 for single-image inference.

    Usage:
        clf = ImageClassifier()
        results = clf.predict("path/to/image.jpg", top_k=3)
        # [{"label": "tabby", "confidence": 94.2}, ...]
    """

    def __init__(self):
        """Load pre-trained MobileNetV2 with ImageNet weights (downloads once)."""
        print("[Classifier] Loading MobileNetV2 with ImageNet weights…")
        self.model = MobileNetV2(weights="imagenet", include_top=True)
        self.model.trainable = False  # Freeze weights — inference only

        # Warm-up pass so the first real prediction isn't slow
        dummy = np.zeros((1, *INPUT_SIZE, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        print("[Classifier] Model loaded and warmed up.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, image_path: str, top_k: int = 3) -> list[dict]:
        """
        Run inference on a single image file.

        Parameters
        ----------
        image_path : str
            Absolute path to the image on disk.
        top_k : int
            Number of top predictions to return.

        Returns
        -------
        list[dict]
            [{"label": str, "confidence": float (0-100)}, …]
        """
        image_array = self._load_and_preprocess(image_path)
        preds = self.model.predict(image_array, verbose=0)
        return self._decode(preds, top_k)

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Use OpenCV to read and prepare the image for MobileNetV2.

        Steps:
          1. Read from disk with OpenCV (returns BGR)
          2. Convert BGR → RGB (Keras expects RGB)
          3. Resize to 224×224
          4. Expand dims → (1, 224, 224, 3)
          5. Apply MobileNetV2-specific normalisation (scale to [-1, 1])
        """
        # 1. Read image (handles EXIF rotation on most platforms)
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"OpenCV could not read image at: {image_path}")

        # 2. BGR → RGB
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # 3. Resize to model input dimensions
        resized = cv2.resize(rgb_img, INPUT_SIZE, interpolation=cv2.INTER_AREA)

        # 4. Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
        batch = np.expand_dims(resized, axis=0)

        # 5. MobileNetV2 preprocessing: uint8 → float32, scale to [-1, 1]
        processed = preprocess_input(batch.astype(np.float32))

        return processed

    @staticmethod
    def _decode(preds: np.ndarray, top_k: int) -> list[dict]:
        """
        Decode raw softmax probabilities into human-readable labels.

        Returns a list of dicts sorted by confidence descending.
        """
        # decode_predictions returns: [[(class_id, label, prob), ...]]
        decoded = decode_predictions(preds, top=top_k)[0]

        results = []
        for _class_id, label, probability in decoded:
            # Clean up underscored ImageNet labels ("great_white_shark" → "Great White Shark")
            clean_label = label.replace("_", " ").title()
            confidence = round(float(probability) * 100, 2)
            results.append({"label": clean_label, "confidence": confidence})

        return results