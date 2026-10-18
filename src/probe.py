"""Read a fixed number of frames from a video of any length."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    BURST_START_FRAC,
    CUT_SCAN_SECONDS,
    DETECT_SHORT_SIDE,
    N_BURST_FRAMES,
    N_SPREAD_FRAMES,
    SHORT_SIDE,
)


class ProbeError(RuntimeError):
    """The file could not be decoded."""


@dataclass
class Probe:
    spread: list[np.ndarray]
    burst: list[np.ndarray]
    cuts: list[np.ndarray]
    n_frames: int
    fps: float
    spread_detect: list[np.ndarray]


def probe_video(path: Path) -> Probe:
    """Seek to 8 frames across the file, 8 frames in one early burst, and one
    frame per second for the first 30 seconds.

    Spread frames show whether the scene changes over the whole video.
    The burst shows what kind of motion happens inside one moment.
    The one-per-second scan counts hard cuts. Eight spread frames often land
    inside a single shot, so they miss the edit rate of a film or a match.
    """
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProbeError(f"Could not open video: {path}")

    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if n_frames <= 0:
        capture.release()
        raise ProbeError(f"Video has no readable frames: {path}")

    spread_idx = _even_indices(n_frames, N_SPREAD_FRAMES)
    burst_idx = _burst_indices(n_frames, N_BURST_FRAMES, BURST_START_FRAC)
    cut_idx = _cut_scan_indices(n_frames, fps)
    spread_set = set(spread_idx)
    frames = {}
    detect_frames = {}
    for index in sorted(set(spread_idx + burst_idx + cut_idx)):
        frame = _read_at(capture, index)
        if frame is None:
            continue
        frames[index] = _resize(frame, SHORT_SIDE)
        if index in spread_set:
            detect_frames[index] = _resize(frame, DETECT_SHORT_SIDE)
    capture.release()

    spread = [frames[i] for i in spread_idx if i in frames]
    spread_detect = [detect_frames[i] for i in spread_idx if i in detect_frames]
    burst = [frames[i] for i in burst_idx if i in frames]
    cuts = [frames[i] for i in cut_idx if i in frames]
    if not spread:
        raise ProbeError(f"Could not decode any frames: {path}")
    return Probe(
        spread=spread,
        burst=burst or spread,
        cuts=cuts or spread,
        n_frames=n_frames,
        fps=fps,
        spread_detect=spread_detect or spread,
    )


def _even_indices(n_frames: int, count: int) -> list[int]:
    if n_frames <= count:
        return list(range(n_frames))
    return [int(round(i * (n_frames - 1) / (count - 1))) for i in range(count)]


def _burst_indices(n_frames: int, count: int, start_frac: float) -> list[int]:
    start = min(int(n_frames * start_frac), max(0, n_frames - 1))
    return [min(start + offset, n_frames - 1) for offset in range(count)]


def _cut_scan_indices(n_frames: int, fps: float) -> list[int]:
    step = max(1, int(round(fps if fps > 1 else 8)))
    last = min(n_frames, step * CUT_SCAN_SECONDS)
    return list(range(0, last, step))[:CUT_SCAN_SECONDS]


def _read_at(capture: cv2.VideoCapture, index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if ok and frame is not None:
        return frame
    return None


def _resize(frame: np.ndarray, short_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = short_side / min(height, width)
    if scale >= 1:
        return frame
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
