"""
evaluation/evaluate.py

Evaluation workflow: compares output.csv against ground_truth.csv and
computes Precision, Recall, F1, and Decision Accuracy per class and overall.

Usage:
    python evaluation/evaluate.py \
        --predictions output.csv \
        --ground-truth evaluation/ground_truth.csv

Ground truth CSV expected columns:
    claim_id, decision, damage_type, severity

Design rationale:
- Macro and per-class metrics are both reported. Macro is useful for class-imbalanced
  test sets (e.g. most claims are SUPPORTED) where micro would be misleading.
- We report per-class precision/recall/F1 for the three decision values separately.
- A confusion matrix is printed for quick visual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(predictions_path: str, ground_truth_path: str) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        predictions_path:  Path to output.csv (pipeline predictions).
        ground_truth_path: Path to ground_truth.csv (human-labelled).

    Returns:
        Dictionary of metrics (also printed to stdout).
    """
    pred_df = pd.read_csv(predictions_path)
    gt_df = pd.read_csv(ground_truth_path)

    # Align on claim_id
    merged = pd.merge(gt_df, pred_df, on="claim_id", suffixes=("_true", "_pred"))

    if merged.empty:
        raise ValueError("No matching claim_ids between predictions and ground truth.")

    y_true = merged["decision_true"].str.upper()
    y_pred = merged["decision_pred"].str.upper()
    labels = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]

    # --- Decision-level metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    per_class = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- Damage type accuracy (subset where decision is SUPPORTED) ---
    supported = merged[merged["decision_true"].str.upper() == "SUPPORTED"].copy()
    damage_acc = None
    if not supported.empty and "damage_type_true" in supported.columns:
        damage_acc = accuracy_score(
            supported["damage_type_true"].str.lower(),
            supported["damage_type_pred"].str.lower(),
        )

    # --- Severity accuracy ---
    severity_acc = None
    if "severity_true" in merged.columns and "severity_pred" in merged.columns:
        severity_acc = accuracy_score(
            merged["severity_true"].str.upper(),
            merged["severity_pred"].str.upper(),
        )

    results = {
        "n_claims": len(merged),
        "decision": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": {
                k: {m: round(v, 4) for m, v in v2.items() if isinstance(v2, dict)}
                for k, v2 in per_class.items() if isinstance(v2, dict)
            },
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
            },
        },
        "damage_type_accuracy": round(damage_acc, 4) if damage_acc is not None else None,
        "severity_accuracy": round(severity_acc, 4) if severity_acc is not None else None,
    }

    _print_report(results)
    return results


def _print_report(r: Dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Claims evaluated : {r['n_claims']}")
    d = r["decision"]
    print(f"\nDecision accuracy  : {d['accuracy']:.4f}")
    print(f"Macro Precision    : {d['macro_precision']:.4f}")
    print(f"Macro Recall       : {d['macro_recall']:.4f}")
    print(f"Macro F1           : {d['macro_f1']:.4f}")

    print("\nPer-class breakdown:")
    for label, metrics in d["per_class"].items():
        p = metrics.get("precision", 0)
        rec = metrics.get("recall", 0)
        f = metrics.get("f1-score", 0)
        sup = metrics.get("support", 0)
        print(f"  {label:<30} P={p:.2f}  R={rec:.2f}  F1={f:.2f}  n={sup}")

    print("\nConfusion matrix:")
    labels = d["confusion_matrix"]["labels"]
    mat = d["confusion_matrix"]["matrix"]
    header = " " * 30 + "  ".join(f"{l[:5]:>5}" for l in labels)
    print(header)
    for i, row in enumerate(mat):
        print(f"  {labels[i]:<28}" + "  ".join(f"{v:>5}" for v in row))

    if r["damage_type_accuracy"] is not None:
        print(f"\nDamage type accuracy (supported claims): {r['damage_type_accuracy']:.4f}")
    if r["severity_accuracy"] is not None:
        print(f"Severity accuracy                       : {r['severity_accuracy']:.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate evidence review pipeline predictions.")
    parser.add_argument("--predictions", required=True, help="Path to output.csv")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth.csv")
    parser.add_argument("--json-out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results = evaluate(args.predictions, args.ground_truth)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {args.json_out}")


if __name__ == "__main__":
    main()
