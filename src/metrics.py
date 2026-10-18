"""Macro-F1, routing cost, and a bootstrap interval around both."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from chunkers import HYBRID_COST, ROUTING_COST
from config import BOOTSTRAP_ROUNDS, CLASSES


def score_predictions(
    true_labels: list[str],
    predicted: list[str],
    latencies_ms: list[float] | None = None,
    routed: list[str] | None = None,
) -> dict:
    labels = list(CLASSES)
    matrix = confusion_matrix(true_labels, predicted, labels=labels)
    per_class = {}
    precision = precision_score(true_labels, predicted, labels=labels, average=None, zero_division=0)
    recall = recall_score(true_labels, predicted, labels=labels, average=None, zero_division=0)
    for index, name in enumerate(labels):
        support = int(matrix[index].sum())
        per_class[name] = {
            "support": support,
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
        }
    costs = [ROUTING_COST[truth][guess] for truth, guess in zip(true_labels, predicted)]
    f1_lo, f1_hi = _bootstrap_ci(true_labels, predicted, _macro_f1)
    cost_lo, cost_hi = _bootstrap_ci(true_labels, predicted, _mean_cost)
    report = {
        "n": len(true_labels),
        "accuracy": round(float(accuracy_score(true_labels, predicted)), 4),
        "macro_f1": round(_macro_f1(true_labels, predicted), 4),
        "macro_f1_ci95": [round(f1_lo, 4), round(f1_hi, 4)],
        "mean_routing_cost": round(float(np.mean(costs)) if costs else 0.0, 4),
        "mean_routing_cost_ci95": [round(cost_lo, 4), round(cost_hi, 4)],
        "per_class": per_class,
        "confusion": matrix.tolist(),
        "class_order": labels,
    }
    if routed is not None:
        deferred = [_route_cost(truth, route) for truth, route in zip(true_labels, routed)]
        report["deferred"] = sum(route == "uncertain" for route in routed)
        report["mean_routing_cost_deferred"] = round(float(np.mean(deferred)) if deferred else 0.0, 4)
    if latencies_ms:
        ordered = np.sort(np.asarray(latencies_ms, dtype=np.float64))
        report["latency_ms"] = {
            "p50": round(float(np.percentile(ordered, 50)), 1),
            "p95": round(float(np.percentile(ordered, 95)), 1),
            "mean": round(float(ordered.mean()), 1),
        }
    return report


def format_report(report: dict) -> str:
    f1_lo, f1_hi = report["macro_f1_ci95"]
    cost_lo, cost_hi = report["mean_routing_cost_ci95"]
    lines = [
        f"videos: {report['n']}",
        f"accuracy: {report['accuracy']:.3f}",
        f"macro_f1: {report['macro_f1']:.3f}  (95% bootstrap CI {f1_lo:.3f} to {f1_hi:.3f})",
        f"mean_routing_cost: {report['mean_routing_cost']:.3f}  (95% bootstrap CI {cost_lo:.3f} to {cost_hi:.3f})",
    ]
    if "mean_routing_cost_deferred" in report:
        lines.append(
            f"mean_routing_cost_deferred: {report['mean_routing_cost_deferred']:.3f}  "
            f"(uncertain videos: {report['deferred']})"
        )
    if "latency_ms" in report:
        latency = report["latency_ms"]
        lines.append(
            f"probe_plus_predict_ms: p50={latency['p50']:.0f}  p95={latency['p95']:.0f}  mean={latency['mean']:.0f}"
        )
    if "model_bytes" in report:
        lines.append(f"model_bytes: {report['model_bytes']}")
    lines.extend(["", "per class:"])
    for name, stats in report["per_class"].items():
        lines.append(
            f"  {name:12} precision={stats['precision']:.3f}  recall={stats['recall']:.3f}  n={stats['support']}"
        )
    lines.extend(
        [
            "",
            "confusion rows are the true class, columns are the prediction, order: "
            + ", ".join(report["class_order"]),
        ]
    )
    for name, row in zip(report["class_order"], report["confusion"]):
        lines.append(f"  {name:12} {row}")
    return "\n".join(lines)


def _macro_f1(true_labels, predicted) -> float:
    return float(
        f1_score(true_labels, predicted, labels=list(CLASSES), average="macro", zero_division=0)
    )


def _mean_cost(true_labels, predicted) -> float:
    costs = [ROUTING_COST[truth][guess] for truth, guess in zip(true_labels, predicted)]
    return float(np.mean(costs)) if costs else 0.0


def _route_cost(truth: str, route: str) -> float:
    if route == "uncertain":
        return HYBRID_COST[truth]
    return ROUTING_COST[truth][route]


def _bootstrap_ci(true_labels, predicted, statistic) -> tuple[float, float]:
    truth = np.asarray(true_labels)
    guess = np.asarray(predicted)
    if len(truth) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    scores = np.empty(BOOTSTRAP_ROUNDS, dtype=np.float64)
    for draw in range(BOOTSTRAP_ROUNDS):
        index = rng.integers(0, len(truth), len(truth))
        scores[draw] = statistic(truth[index], guess[index])
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)
