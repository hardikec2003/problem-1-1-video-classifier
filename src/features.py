"""Nine numbers that describe how a video should be chunked.

The questions are the ones a person asks while skimming:
did the scene change, did the camera pan, how often does a cut happen,
is the remaining motion local, is there a large face, is the rim brighter
than the center, and is someone speaking?
"""

from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from config import CUT_JUMP

FEATURE_NAMES = (
    "scene_change_mean",
    "scene_change_max",
    "motion_mean",
    "global_flow",
    "residual_flow",
    "face_frac",
    "border_brightness",
    "speech_frac",
    "cut_rate",
)

STRUCTURAL_NAMES = (
    "global_flow",
    "residual_flow",
    "speech_frac",
    "cut_rate",
)


_FACE = None


def structural_vector(
    spread: list[np.ndarray],
    burst: list[np.ndarray],
    speech_frac: float,
    cuts: list[np.ndarray] | None = None,
) -> np.ndarray:
    """The four clues the appearance embedding cannot see: pan, leftover motion, speech, cuts."""
    full = feature_vector(spread, burst, speech_frac, cuts)
    chosen = [FEATURE_NAMES.index(name) for name in STRUCTURAL_NAMES]
    return full[chosen]


def feature_vector(
    spread: list[np.ndarray],
    burst: list[np.ndarray],
    speech_frac: float,
    cuts: list[np.ndarray] | None = None,
) -> np.ndarray:
    scene = _scene_changes(spread)
    motion, global_flow, residual = _motion(burst)
    values = np.array(
        [
            _mean(scene),
            max(scene) if scene else 0.0,
            motion,
            global_flow,
            residual,
            _face_fraction(spread),
            _border_brightness(spread),
            float(speech_frac),
            _cut_rate(cuts if cuts is not None else spread),
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def speech_fraction(video_path: Path) -> float:
    """Share of the first 15 seconds that is louder than room tone.

    A file with no audio returns 0. The other seven features still classify
    the video. The ffmpeg binary is the system one when it exists, otherwise
    the binary shipped with imageio-ffmpeg.
    """
    ffmpeg = _ffmpeg_binary()
    if ffmpeg is None:
        return 0.0
    command = [
        ffmpeg, "-v", "error", "-t", "15", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return 0.0
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    hop = 320
    if samples.size < hop * 5:
        return 0.0
    energies = [
        float(np.sqrt(np.mean(samples[i : i + hop] ** 2)))
        for i in range(0, samples.size - hop, hop)
    ]
    threshold = max(0.02, 0.35 * max(energies))
    return float(np.mean([energy > threshold for energy in energies]))


def _cut_rate(frames: list[np.ndarray]) -> float:
    """Share of one-per-second pairs that jump. Edited video scores high."""
    changes = _scene_changes(frames)
    if not changes:
        return 0.0
    return float(np.mean([change >= CUT_JUMP for change in changes]))


def _scene_changes(frames: list[np.ndarray]) -> list[float]:
    changes = []
    for left, right in zip(frames, frames[1:]):
        similarity = cv2.compareHist(_hue_hist(left), _hue_hist(right), cv2.HISTCMP_CORREL)
        changes.append(float(1.0 - similarity))
    return changes


def _hue_hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    cv2.normalize(hist, hist)
    return hist


def _motion(frames: list[np.ndarray]) -> tuple[float, float, float]:
    """Split motion into a camera pan and everything left over.

    global_flow is how many pixels the whole frame slid (phase correlation).
    residual_flow is the pixel change that remains after that slide is undone.
    If the frame did not slide as a whole, global_flow is 0 and residual_flow
    equals the raw pixel change.
    """
    motions, globals_, residuals = [], [], []
    for left, right in zip(frames, frames[1:]):
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        raw = float(np.mean(np.abs(gray_left.astype(np.float32) - gray_right.astype(np.float32))))
        (dx, dy), response = cv2.phaseCorrelate(np.float32(gray_left), np.float32(gray_right))
        shift = float(np.hypot(dx, dy))
        if response >= 0.15 and shift >= 0.5:
            aligned = _shift(gray_right, -dx, -dy)
            leftover = float(np.mean(np.abs(gray_left.astype(np.float32) - aligned)))
            global_motion = shift
        else:
            leftover = raw
            global_motion = 0.0
        motions.append(raw)
        globals_.append(global_motion)
        residuals.append(leftover)
    return _mean(motions), _mean(globals_), _mean(residuals)


def _shift(gray: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = gray.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        gray, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _face_fraction(frames: list[np.ndarray]) -> float:
    detector = _face_detector()
    if detector is None or not frames:
        return 0.0
    hits = 0
    area_of = lambda frame: frame.shape[0] * frame.shape[1]
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        if any(w * h >= 0.04 * area_of(frame) for (_, _, w, h) in boxes):
            hits += 1
    return hits / len(frames)


def _face_detector():
    global _FACE
    if _FACE is None:
        if not hasattr(cv2, "CascadeClassifier"):
            _FACE = False
            return None
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(path)
        _FACE = detector if not detector.empty() else False
    return _FACE or None


def _border_brightness(frames: list[np.ndarray]) -> float:
    """Rim brightness minus center brightness. HUDs and scorebugs sit on the rim."""
    if not frames:
        return 0.0
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        margin = max(2, int(0.15 * min(gray.shape)))
        border = np.concatenate(
            [
                gray[:margin, :].ravel(),
                gray[-margin:, :].ravel(),
                gray[:, :margin].ravel(),
                gray[:, -margin:].ravel(),
            ]
        )
        center = gray[margin:-margin, margin:-margin]
        center_value = float(center.mean()) if center.size else float(gray.mean())
        scores.append(float(border.mean()) - center_value)
    return _mean(scores)


def _ffmpeg_binary() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
