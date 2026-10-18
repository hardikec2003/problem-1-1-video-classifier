"""Starter labels for the Video-MME 10% sample.

A subcategory is a topic. The label is the chunker that topic usually needs.
Geography can be a lecture or a landscape, so the CSV is meant to be skimmed
and corrected before the score is quoted.
"""

import csv
import json
from pathlib import Path

SUBCATEGORY_TO_CLASS = {
    "Esports": "gameplay",
    "Daily Life": "egocentric",
    "Travel": "egocentric",
    "Handicraft": "egocentric",
    "Food": "egocentric",
    "Life Tip": "dialogue",
    "Finance & Commerce": "dialogue",
    "Multilingual": "dialogue",
    "News Report": "dialogue",
    "Literature & Art": "dialogue",
    "Biology & Medicine": "dialogue",
    "Astronomy": "dialogue",
    "Law": "dialogue",
    "Technology": "dialogue",
    "Humanity & History": "dialogue",
    "Geography": "shot_scene",
    "Documentary": "shot_scene",
    "Animation": "shot_scene",
    "Movie & TV Show": "shot_scene",
    "Fashion": "shot_scene",
    "Stage Play": "shot_scene",
    "Variety Show": "shot_scene",
    "Acrobatics": "shot_scene",
    "Magic Show": "shot_scene",
    "Athletics": "shot_scene",
    "Exercise": "shot_scene",
    "Basketball": "shot_scene",
    "Football": "shot_scene",
    "Other Sports": "shot_scene",
}


def class_for_subcategory(name: str) -> str:
    return SUBCATEGORY_TO_CLASS.get(name, "shot_scene")


def write_labels_from_videomme(jsonl_path: Path, out_path: Path) -> int:
    """One row per video. Question text is ignored on purpose."""
    seen = {}
    with Path(jsonl_path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = row["video_id"]
            if video_id in seen:
                continue
            subcategory = row.get("sub_category", "")
            seen[video_id] = {
                "video_id": video_id,
                "label": class_for_subcategory(subcategory),
                "sub_category": subcategory,
                "duration": row.get("duration", ""),
                "domain": row.get("domain", ""),
            }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_id", "label", "sub_category", "duration", "domain", "split"]
    rows = list(seen.values())
    _assign_splits(rows)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_label_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _assign_splits(rows: list[dict]) -> None:
    """Hold out about a quarter of each class. A class with one video stays in train."""
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    for group in by_label.values():
        group.sort(key=lambda item: item["video_id"])
        holdout_n = 0 if len(group) < 2 else max(1, round(0.25 * len(group)))
        for index, row in enumerate(group):
            row["split"] = "holdout" if index < holdout_n else "train"
