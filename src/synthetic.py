"""Tiny videos with an obvious chunking class, so tests run without Video-MME."""

from pathlib import Path

import cv2
import numpy as np

from config import (
    CLASSES,
    HOLDOUT_FROM,
    SYNTHETIC_FPS,
    SYNTHETIC_FRAMES,
    SYNTHETIC_SIZE,
    VIDEOS_PER_CLASS,
)


def build_synthetic_dataset(root: Path) -> list[dict]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in CLASSES:
        for index in range(VIDEOS_PER_CLASS):
            path = root / f"{label}_{index}.avi"
            _write_video(path, _frames(label, index))
            rows.append(
                {
                    "path": path,
                    "video_id": path.stem,
                    "label": label,
                    "split": "holdout" if index >= HOLDOUT_FROM else "train",
                }
            )
    return rows


def _frames(label: str, index: int) -> list[np.ndarray]:
    size = SYNTHETIC_SIZE
    rng = np.random.default_rng(1000 + index)
    frames = []
    for t in range(SYNTHETIC_FRAMES):
        if label == "gameplay":
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            frame[16:-16, 16:-16] = rng.integers(0, 70, (size - 32, size - 32, 3), dtype=np.uint8)
            frame[:12, :] = (0, 255, 0)
            frame[-12:, :] = (0, 80, 255)
            frame[:, :12] = (0, 220, 220)
            frame[:, -12:] = (255, 180, 0)
        elif label == "egocentric":
            picture = np.full((size, size, 3), (30, 40, 50), np.uint8)
            cv2.circle(picture, (28, 36), 18, (40, 180, 220), -1)
            cv2.circle(picture, (68, 58), 12, (20, 40, 180), -1)
            cv2.rectangle(picture, (10, 70), (40, 88), (80, 160, 40), -1)
            frame = np.roll(picture, (t * 4 + index) % size, axis=1)
        elif label == "dialogue":
            frame = np.full((size, size, 3), 160, np.uint8)
            center = size // 2 + (index % 3) - 1
            cv2.circle(frame, (center, center), 18, (200, 210, 220), -1)
        else:
            color = (180, 40, 40) if t < SYNTHETIC_FRAMES // 2 else (40, 140, 40)
            frame = np.full((size, size, 3), color, np.uint8)
        frames.append(frame)
    return frames


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), SYNTHETIC_FPS, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not write video: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
