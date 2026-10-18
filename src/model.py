"""Fit and apply the logistic-regression router.

The input is four structural clues, plus optional frozen MobileNet numbers
and optional frozen YOLO11n object counts.
"""

from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from chunkers import chunker_for
from config import APPEARANCE_DIMS, DEFER_THRESHOLD
from embed import embed_frames
from features import STRUCTURAL_NAMES, speech_fraction, structural_vector
from probe import probe_video

# After scaling, each extra number is quieter than a structural clue,
# so appearance and object counts cannot outvote cut rate, pan, and speech.
APPEARANCE_SCALE = 0.25
FEATURE_SETS = ("mobilenet", "yolo", "both")


class _ScaledClues(BaseEstimator, TransformerMixin):
    def __init__(self, n_structural: int = 4, appearance_scale: float = APPEARANCE_SCALE):
        self.n_structural = n_structural
        self.appearance_scale = appearance_scale

    def fit(self, features, labels=None):
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self

    def transform(self, features):
        scaled = (features - self.mean_) / self.scale_
        scaled = np.array(scaled, copy=True)
        scaled[:, self.n_structural :] *= self.appearance_scale
        return scaled


def train_router(rows: list[dict], features: str = "mobilenet") -> dict:
    _check_features(features)
    cached = [extract_clues(row["path"], use_audio=True, features=features) | {"label": row["label"]} for row in rows]
    return fit_router(cached, features)


def fit_router(cached: list[dict], features: str) -> dict:
    """Fit from clues that were already computed, so a comparison probes once."""
    _check_features(features)
    structural = np.vstack([row["structural"] for row in cached])
    labels = np.array([row["label"] for row in cached])
    pca = None
    appearance = None
    if features in ("mobilenet", "both"):
        appearance = np.vstack([row["appearance"] for row in cached])
        components = min(APPEARANCE_DIMS, len(cached) - 1, appearance.shape[1])
        pca = PCA(n_components=components, random_state=0)
        pca.fit(appearance)
    objects = np.vstack([row["objects"] for row in cached]) if features in ("yolo", "both") else None
    matrix, names = _matrix(structural, appearance, objects, pca, features)
    pipeline = Pipeline(
        [
            ("scale", _ScaledClues(n_structural=structural.shape[1])),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=0)),
        ]
    )
    pipeline.fit(matrix, labels)
    return {
        "pipeline": pipeline,
        "pca": pca,
        "feature_names": names,
        "threshold": DEFER_THRESHOLD,
        "features": features,
    }


def predict_video(bundle: dict, video_path: Path, use_audio: bool = True) -> dict:
    clues = extract_clues(video_path, use_audio=use_audio, features=bundle.get("features", "mobilenet"))
    return predict_clues(bundle, clues)


def predict_clues(bundle: dict, clues: dict) -> dict:
    features = bundle.get("features", "mobilenet")
    vector = _vector(clues, bundle.get("pca"), features)
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba(vector.reshape(1, -1))[0]
    classes = list(pipeline.named_steps["clf"].classes_)
    best = int(np.argmax(probabilities))
    label = classes[best]
    confidence = float(probabilities[best])
    uncertain = confidence < bundle.get("threshold", DEFER_THRESHOLD)
    chosen = "uncertain" if uncertain else label
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "uncertain": uncertain,
        "chunker": chunker_for(chosen),
        "probabilities": {name: round(float(p), 4) for name, p in zip(classes, probabilities)},
        "n_frames": clues["n_frames"],
    }


def extract_clues(video_path: Path, use_audio: bool = True, features: str = "both") -> dict:
    """Probe once and run only the frozen networks this feature set needs."""
    _check_features(features)
    started = time.perf_counter()
    sample = probe_video(video_path)
    probe_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    speech = speech_fraction(video_path) if use_audio else 0.0
    speech_ms = (time.perf_counter() - started) * 1000
    structural = structural_vector(sample.spread, sample.burst, speech, sample.cuts)
    appearance = np.zeros(576, dtype=np.float64)
    objects = np.zeros(6, dtype=np.float64)
    embed_ms = 0.0
    yolo_ms = 0.0
    if features in ("mobilenet", "both"):
        started = time.perf_counter()
        appearance = embed_frames(sample.spread)
        embed_ms = (time.perf_counter() - started) * 1000
    if features in ("yolo", "both"):
        from detect import detect_frames

        started = time.perf_counter()
        objects = detect_frames(sample.spread_detect)
        yolo_ms = (time.perf_counter() - started) * 1000
    return {
        "structural": structural,
        "appearance": appearance,
        "objects": objects,
        "n_frames": sample.n_frames,
        "latency_ms": {
            "probe": probe_ms,
            "speech": speech_ms,
            "embed": embed_ms,
            "yolo": yolo_ms,
        },
    }


def coefficient_table(bundle: dict) -> list[tuple[str, str, float]]:
    clf = bundle["pipeline"].named_steps["clf"]
    names = bundle["feature_names"]
    rows = []
    for class_index, label in enumerate(clf.classes_):
        for name, weight in zip(names, clf.coef_[class_index]):
            rows.append((str(label), name, float(weight)))
    return rows


def save_bundle(path: Path, bundle: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_bundle(path: Path) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def latency_ms(clues: dict, features: str) -> float:
    parts = clues["latency_ms"]
    total = parts["probe"] + parts["speech"]
    if features in ("mobilenet", "both"):
        total += parts["embed"]
    if features in ("yolo", "both"):
        total += parts["yolo"]
    return total


def _matrix(structural, appearance, objects, pca, features: str):
    parts = [structural]
    names = list(STRUCTURAL_NAMES)
    if features in ("mobilenet", "both"):
        reduced = pca.transform(appearance)
        parts.append(reduced)
        names.extend(f"appearance_{index:02d}" for index in range(reduced.shape[1]))
    if features in ("yolo", "both"):
        parts.append(objects)
        from detect import YOLO_NAMES

        names.extend(YOLO_NAMES)
    return np.hstack(parts), names


def _vector(clues: dict, pca, features: str) -> np.ndarray:
    structural = clues["structural"].reshape(1, -1)
    appearance = clues["appearance"].reshape(1, -1) if features in ("mobilenet", "both") else None
    objects = clues["objects"].reshape(1, -1) if features in ("yolo", "both") else None
    matrix, _names = _matrix(structural, appearance, objects, pca, features)
    return matrix[0]


def _check_features(features: str) -> None:
    if features not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {features!r}. Expected one of {FEATURE_SETS}.")
