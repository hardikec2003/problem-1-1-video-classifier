"""Frozen YOLO11n object counts on the eight spread frames.

The network is not trained. It answers which objects are in the pictures.
Cuts, pans, and speech stay in the hand-built clues, and MobileNet still
answers what the frames look like.
"""

from pathlib import Path

import numpy as np

YOLO_NAMES = (
    "person_mean",
    "person_max",
    "object_mean",
    "screen_frac",
    "animal_frac",
    "sports_frac",
)

_SCREEN = {"tv", "laptop", "cell phone"}
_ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
_SPORTS = {
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
}

_MODEL = None
_WEIGHTS = Path.home() / ".cache" / "video_classification" / "yolo11n.pt"


def detect_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Six counts averaged over the frames. Shape is (6,)."""
    if not frames:
        return np.zeros(len(YOLO_NAMES), dtype=np.float64)
    model = _network()
    results = model.predict(list(frames), imgsz=320, conf=0.25, verbose=False, device="cpu")
    person_counts = []
    object_counts = []
    screen_hits = []
    animal_hits = []
    sports_hits = []
    for result in results:
        names = _class_names(model, result)
        person_counts.append(sum(name == "person" for name in names))
        object_counts.append(len(names))
        screen_hits.append(float(any(name in _SCREEN for name in names)))
        animal_hits.append(float(any(name in _ANIMALS for name in names)))
        sports_hits.append(float(any(name in _SPORTS for name in names)))
    return np.array(
        [
            float(np.mean(person_counts)),
            float(np.max(person_counts)),
            float(np.mean(object_counts)),
            float(np.mean(screen_hits)),
            float(np.mean(animal_hits)),
            float(np.mean(sports_hits)),
        ],
        dtype=np.float64,
    )


def _class_names(model, result) -> list[str]:
    boxes = result.boxes
    if boxes is None or boxes.cls is None or len(boxes.cls) == 0:
        return []
    return [model.names[int(class_id)] for class_id in boxes.cls.tolist()]


def _network():
    global _MODEL
    if _MODEL is None:
        import cv2

        # opencv-python-headless has no windowing. Ultralytics copies imshow at import.
        if not hasattr(cv2, "imshow"):
            cv2.imshow = lambda *_args, **_kwargs: None
        from ultralytics import YOLO

        if not _WEIGHTS.exists():
            _download_weights()
        _MODEL = YOLO(str(_WEIGHTS))
    return _MODEL


def _download_weights() -> None:
    import urllib.request

    _WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    urllib.request.urlretrieve(url, _WEIGHTS)
