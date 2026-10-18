"""Route a video to the chunker that should index it.

The evaluator commands are in run.yaml. A short local check, with no download:

    python src/main.py train --synthetic data/synthetic --model artifacts/router.pkl
    python src/main.py evaluate --synthetic data/synthetic --model artifacts/router.pkl

The Video-MME 10% sample:

    python src/main.py download --out data/videomme
    python src/main.py labels --jsonl data/videomme/qa/test.jsonl --out data/labels.csv
    python src/main.py train --videos data/videomme/videos --labels data/labels.csv --model artifacts/router.pkl
    python src/main.py evaluate --videos data/videomme/videos --labels data/labels.csv --model artifacts/router.pkl --report artifacts/metrics.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLASSES, DATASET_ID
from labels import read_label_csv, write_labels_from_videomme
from metrics import format_report, score_predictions
from model import (
    FEATURE_SETS,
    coefficient_table,
    extract_clues,
    fit_router,
    latency_ms,
    load_bundle,
    predict_clues,
    predict_video,
    save_bundle,
    train_router,
)
from synthetic import build_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a video by the chunker it needs.")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download the Video-MME 10% sample.")
    download.add_argument("--out", default="data/videomme")

    labels = commands.add_parser("labels", help="Write starter chunking labels from Video-MME.")
    labels.add_argument("--jsonl", required=True)
    labels.add_argument("--out", required=True)

    train = commands.add_parser("train", help="Fit the logistic-regression router.")
    _add_data_args(train)
    train.add_argument("--model", required=True)
    train.add_argument("--features", default="mobilenet", choices=FEATURE_SETS)

    compare = commands.add_parser("compare", help="Score MobileNet, YOLO, and both on one probe.")
    _add_data_args(compare)
    compare.add_argument("--report", default="artifacts/comparison.json")

    evaluate = commands.add_parser("evaluate", help="Score the router on held-out videos.")
    _add_data_args(evaluate)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--report", help="Optional JSON path for the metrics.")

    predict = commands.add_parser("predict", help="Print the class and chunker for one video.")
    predict.add_argument("--video")
    predict.add_argument("--videos", help="Used with --labels when --video is omitted.")
    predict.add_argument("--labels")
    predict.add_argument("--model", required=True)

    args = parser.parse_args()
    if args.command == "download":
        _download(Path(args.out))
        return
    if args.command == "labels":
        count = write_labels_from_videomme(Path(args.jsonl), Path(args.out))
        print(f"Wrote {count} video labels to {args.out}")
        print("These labels follow the subcategory-to-chunker map. Skim and edit before quoting the score.")
        return
    if args.command == "predict":
        video = _predict_path(args)
        result = predict_video(load_bundle(Path(args.model)), video)
        result["video"] = str(video)
        print(json.dumps(result, indent=2))
        return

    rows = _load_rows(args)
    if args.command == "compare":
        source = "synthetic" if args.synthetic else "videomme-subcategory-map"
        _compare(rows, Path(args.report), source)
        return
    if args.command == "train":
        fitting = [row for row in rows if row["split"] == "train"]
        _require_two_classes(fitting)
        started = time.perf_counter()
        bundle = train_router(fitting, features=args.features)
        save_bundle(Path(args.model), bundle)
        elapsed = time.perf_counter() - started
        print(f"Trained on {len(fitting)} videos in {elapsed:.1f}s. Saved {args.model}")
        print(f"model_bytes: {Path(args.model).stat().st_size}")
        _print_split(rows)
        print("Largest weights (positive means the feature supports that class):")
        ranked = sorted(coefficient_table(bundle), key=lambda item: abs(item[2]), reverse=True)[:12]
        for label, name, weight in ranked:
            print(f"  {label:12} {name:22} {weight:+.3f}")
        return

    bundle = load_bundle(Path(args.model))
    heldout = [row for row in rows if row["split"] == "holdout"]
    if not heldout:
        raise SystemExit("No holdout videos. Check the split column in the labels file.")
    truths, guesses, routed, latencies = [], [], [], []
    for row in heldout:
        started = time.perf_counter()
        result = predict_video(bundle, row["path"])
        latencies.append((time.perf_counter() - started) * 1000)
        truths.append(row["label"])
        guesses.append(result["label"])
        routed.append("uncertain" if result["uncertain"] else result["label"])
    report = score_predictions(truths, guesses, latencies, routed)
    report["model_bytes"] = Path(args.model).stat().st_size
    report["train_videos"] = sum(row["split"] == "train" for row in rows)
    report["label_source"] = "synthetic" if args.synthetic else "videomme-subcategory-map"
    print(format_report(report))
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {destination}")


def _download(destination: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {DATASET_ID} to {destination}")
    snapshot_download(repo_id=DATASET_ID, repo_type="dataset", local_dir=str(destination))
    print(f"Dataset ready at {destination}")


def _compare(rows: list[dict], report_path: Path, label_source: str) -> None:
    import pickle

    fitting = [row for row in rows if row["split"] == "train"]
    heldout = [row for row in rows if row["split"] == "holdout"]
    _require_two_classes(fitting)
    if not heldout:
        raise SystemExit("No holdout videos. Check the split column in the labels file.")
    print(f"Probing {len(rows)} videos once, then fitting MobileNet, YOLO, and both.")
    cached = []
    for index, row in enumerate(rows, start=1):
        cached.append(extract_clues(row["path"], features="both") | row)
        if index % 10 == 0 or index == len(rows):
            print(f"  probed {index}/{len(rows)}", flush=True)

    reports = {}
    for features in FEATURE_SETS:
        train_rows = [row for row in cached if row["split"] == "train"]
        bundle = fit_router(train_rows, features)
        truths, guesses, routed, latencies = [], [], [], []
        for row in cached:
            if row["split"] != "holdout":
                continue
            result = predict_clues(bundle, row)
            truths.append(row["label"])
            guesses.append(result["label"])
            routed.append("uncertain" if result["uncertain"] else result["label"])
            latencies.append(latency_ms(row, features))
        report = score_predictions(truths, guesses, latencies, routed)
        report["features"] = features
        report["model_bytes"] = len(pickle.dumps(bundle))
        report["train_videos"] = len(train_rows)
        report["label_source"] = label_source
        reports[features] = report
        print(f"\n{features}")
        print(format_report(report))
        ranked = sorted(coefficient_table(bundle), key=lambda item: abs(item[2]), reverse=True)[:8]
        for label, name, weight in ranked:
            print(f"  {label:12} {name:22} {weight:+.3f}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Wrote {report_path}")


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--synthetic", help="Directory for the generated four-class videos.")
    parser.add_argument("--videos", help="Directory of videos named <video_id>.mp4 or .avi.")
    parser.add_argument("--labels", help="CSV with video_id, label, and split.")


def _load_rows(args) -> list[dict]:
    if args.synthetic:
        return build_synthetic_dataset(Path(args.synthetic))
    if not args.videos or not args.labels:
        raise SystemExit("Pass --synthetic DIR, or both --videos DIR and --labels CSV.")
    video_dir = Path(args.videos)
    rows = []
    for record in read_label_csv(Path(args.labels)):
        label = record["label"].strip()
        if label not in CLASSES:
            raise SystemExit(f"Unknown label {label!r}. Expected one of {CLASSES}.")
        split = record.get("split", "").strip() or "train"
        rows.append(
            {
                "path": _find_video(video_dir, record["video_id"]),
                "video_id": record["video_id"],
                "label": label,
                "split": split,
            }
        )
    return rows


def _find_video(video_dir: Path, video_id: str) -> Path:
    for suffix in (".mp4", ".avi", ".mkv", ".webm"):
        candidate = video_dir / f"{video_id}{suffix}"
        if candidate.exists():
            return candidate
    raise SystemExit(f"No video file for {video_id} under {video_dir}")


def _predict_path(args) -> Path:
    if args.video:
        return Path(args.video)
    if not args.videos or not args.labels:
        raise SystemExit("Pass --video PATH, or both --videos DIR and --labels CSV.")
    first = read_label_csv(Path(args.labels))[0]["video_id"]
    return _find_video(Path(args.videos), first)


def _require_two_classes(rows: list[dict]) -> None:
    if len({row["label"] for row in rows}) < 2:
        raise SystemExit("Need at least two classes in the training split.")


def _print_split(rows: list[dict]) -> None:
    for split in ("train", "holdout"):
        counts: dict[str, int] = {}
        for row in rows:
            if row["split"] == split:
                counts[row["label"]] = counts.get(row["label"], 0) + 1
        summary = ", ".join(f"{name}={counts.get(name, 0)}" for name in CLASSES)
        print(f"{split}: {summary}")


if __name__ == "__main__":
    main()
