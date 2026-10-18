"""Probe features and the router, without the Video-MME download."""

import json
from pathlib import Path

import numpy as np

from config import CLASSES
from features import feature_vector
from labels import class_for_subcategory, write_labels_from_videomme
from metrics import score_predictions
from model import predict_video, train_router
from synthetic import build_synthetic_dataset


def _still(color, n=4):
    frame = np.full((96, 96, 3), color, np.uint8)
    return [frame.copy() for _ in range(n)]


def _pan(n=8):
    picture = np.full((96, 96, 3), 30, np.uint8)
    picture[20:50, 10:40] = (40, 180, 220)
    picture[55:80, 50:85] = (20, 40, 180)
    return [np.roll(picture, t * 4, axis=1) for t in range(n)]


def test_a_hard_cut_changes_the_scene_more_than_a_still_clip():
    still = feature_vector(_still((20, 20, 180)), _still((20, 20, 180)), 0.0)
    cut_frames = _still((20, 20, 180), n=4) + _still((20, 160, 20), n=4)
    cut = feature_vector(cut_frames, _still((20, 20, 180)), 0.0)
    assert cut[1] > still[1]
    assert cut[-1] > still[-1]


def test_deferral_costs_less_than_calling_an_edited_film_egocentric():
    raw = score_predictions(["shot_scene"], ["egocentric"])
    deferred = score_predictions(["shot_scene"], ["egocentric"], routed=["uncertain"])
    assert deferred["mean_routing_cost_deferred"] < raw["mean_routing_cost"]


def test_a_pan_has_global_motion_and_a_smaller_residual():
    still = feature_vector(_still(40), _still(40, n=6), 0.0)
    pan = feature_vector(_pan(), _pan(), 0.0)
    assert pan[3] > 2.0
    assert pan[3] > still[3]
    assert pan[4] < pan[2]


def test_subcategory_map_picks_the_chunker():
    assert class_for_subcategory("Esports") == "gameplay"
    assert class_for_subcategory("Daily Life") == "egocentric"
    assert class_for_subcategory("Multilingual") == "dialogue"
    assert class_for_subcategory("Movie & TV Show") == "shot_scene"
    assert class_for_subcategory("Athletics") == "shot_scene"
    assert class_for_subcategory("Exercise") == "shot_scene"


def test_router_learns_the_four_chunkers(tmp_path: Path):
    rows = build_synthetic_dataset(tmp_path)
    bundle = train_router([row for row in rows if row["split"] == "train"])
    truths, guesses = [], []
    for row in rows:
        if row["split"] != "holdout":
            continue
        result = predict_video(bundle, row["path"], use_audio=False)
        truths.append(row["label"])
        guesses.append(result["label"])
        assert "Kuration" in result["chunker"] or "Speech" in result["chunker"] or "cuts" in result["chunker"]
    report = score_predictions(truths, guesses)
    assert report["n"] == 2 * len(CLASSES)
    assert report["macro_f1"] >= 0.75
    assert report["macro_f1_ci95"][0] <= report["macro_f1"] <= report["macro_f1_ci95"][1]
    assert report["mean_routing_cost"] < 0.25


def test_a_wrong_chunker_costs_more_than_the_right_one():
    perfect = score_predictions(["gameplay"], ["gameplay"])
    missed = score_predictions(["gameplay"], ["dialogue"])
    assert missed["mean_routing_cost"] > perfect["mean_routing_cost"]


def test_labels_keep_one_row_per_video_and_a_holdout(tmp_path: Path):
    jsonl = tmp_path / "test.jsonl"
    rows = [
        {"video_id": "aaa", "sub_category": "Esports", "duration": "short", "domain": "Sports Competition"},
        {"video_id": "aaa", "sub_category": "Esports", "duration": "short", "domain": "Sports Competition"},
        {"video_id": "bbb", "sub_category": "Documentary", "duration": "long", "domain": "Film & Television"},
        {"video_id": "ccc", "sub_category": "Documentary", "duration": "medium", "domain": "Film & Television"},
    ]
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "labels.csv"
    assert write_labels_from_videomme(jsonl, out) == 3
    text = out.read_text()
    assert text.count("aaa") == 1
    assert "gameplay" in text
    assert "holdout" in text
