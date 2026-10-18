"""Frozen MobileNetV3-Small appearance vector for the eight probe frames.

The network is not trained. It only answers "what do these pictures look like?"
Cuts, camera pans, and speech stay in the hand-built clues.
"""

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

_MODEL = None
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def embed_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Mean embedding of the frames. Shape is (576,)."""
    if not frames:
        return np.zeros(576, dtype=np.float64)
    model = _network()
    batch = _batch(frames)
    with torch.inference_mode():
        vectors = model(batch)
    return vectors.mean(dim=0).cpu().numpy().astype(np.float64)


def _network():
    global _MODEL
    if _MODEL is None:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        network = mobilenet_v3_small(weights=weights)
        network.classifier = torch.nn.Identity()
        network.eval()
        _MODEL = network
    return _MODEL


def _batch(frames: list[np.ndarray]) -> torch.Tensor:
    images = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        image = rgb.astype(np.float32) / 255.0
        image = (image - _MEAN) / _STD
        images.append(torch.from_numpy(np.transpose(image, (2, 0, 1))))
    return torch.stack(images)
