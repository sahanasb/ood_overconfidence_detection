"""Evaluate Layer 2 Decision Tree detectors on held-out and blind-test data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

RANDOM_STATE = 42
FINGERPRINT_ROOT = Path("model_outputs")
DETECTOR_OUTPUT_DIR = Path("detector_model_outputs")
EVAL_OUTPUT_DIR = Path("detector_model_outputs/evaluation")
FEATURE_COLUMNS = ["confidence", "entropy", "margin"]

MODEL_CONFIGS = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "folder": "logistic_regression_fingerprints",
    },
    "svm": {
        "display_name": "SVM",
        "folder": "svm_fingerprints",
    },
    "mlp": {
        "display_name": "MLP",
        "folder": "mlp_fingerprints",
    },
}


def load_blind_test_dataset(model_name: str) -> pd.DataFrame:
    """Load blind-test fingerprints: NSL (trustworthy) + UNSW portion 2 (suspicious)."""
    config = MODEL_CONFIGS[model_name]
    fingerprint_dir = FINGERPRINT_ROOT / config["folder"]

    nsl_path = fingerprint_dir / "nsl_blind_test_fingerprints.csv"
    unsw_path = fingerprint_dir / "unsw_portion2_fingerprints.csv"

    nsl_df = pd.read_csv(nsl_path)
    unsw_df = pd.read_csv(unsw_path)

    nsl_df = nsl_df.copy()
    unsw_df = unsw_df.copy()
    nsl_df["detector_label"] = 0
    unsw_df["detector_label"] = 1

    return pd.concat([nsl_df, unsw_df], ignore_index=True)


def evaluate_detector(
    model_name: str, split_name: str, X: pd.DataFrame, y: np.ndarray
) -> dict:
    """Compute classification metrics and ROC data for one detector."""
    model_dir = DETECTOR_OUTPUT_DIR / f"{model_name}_detector"
    model = joblib.load(model_dir / f"best_{model_name}_detector.joblib")

    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_score)

    metrics = {
        "model": MODEL_CONFIGS[model_name]["display_name"],
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label=1),
        "recall": recall_score(y, y_pred, pos_label=1),
        "f1": f1_score(y, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y, y_score),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "classification_report": classification_report(
            y,
            y_pred,
            target_names=["Trustworthy (NSL)", "Suspicious (UNSW/OOD)"],
        ),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"{metrics['model']} — {metrics['split']}")
    print(f"{'=' * 72}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}  (TP / total positive predictions)")
    print(f"Recall    : {metrics['recall']:.4f}  (TP / actual positives)")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("\nConfusion matrix [rows=actual, cols=predicted]:")
    print("                 Pred Trustworthy  Pred Suspicious")
    cm = metrics["confusion_matrix"]
    print(f"Actual Trustworthy   {cm[0, 0]:>10}  {cm[0, 1]:>10}")
    print(f"Actual Suspicious    {cm[1, 0]:>10}  {cm[1, 1]:>10}")
    print("\nClassification report:")
    print(metrics["classification_report"])


def plot_roc_curves(all_metrics: list[dict]) -> None:
    """Plot ROC curves grouped by evaluation split."""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["Internal Holdout", "Blind Test"]:
        split_metrics = [m for m in all_metrics if m["split"] == split_name]
        if not split_metrics:
            continue

        plt.figure(figsize=(8, 6))
        for metrics in split_metrics:
            plt.plot(
                metrics["fpr"],
                metrics["tpr"],
                label=f"{metrics['model']} (AUC = {metrics['roc_auc']:.4f})",
            )

        plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR / Recall)")
        plt.title(f"Layer 2 ROC Curve — {split_name}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filename = split_name.lower().replace(" ", "_")
        output_path = EVAL_OUTPUT_DIR / f"layer2_roc_{filename}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved ROC plot: {output_path}")


def main() -> None:
    all_metrics: list[dict] = []
    summary_rows: list[dict] = []

    for model_name in MODEL_CONFIGS:
        model_dir = DETECTOR_OUTPUT_DIR / f"{model_name}_detector"

        # The detectors were fitted on DataFrames, so keep the column names and
        # their training order when re-loading the saved arrays.
        X_holdout = pd.DataFrame(
            np.load(model_dir / f"{model_name}_X_test.npy"),
            columns=FEATURE_COLUMNS,
        )
        y_holdout = np.load(model_dir / f"{model_name}_y_test.npy")

        holdout_metrics = evaluate_detector(
            model_name, "Internal Holdout", X_holdout, y_holdout
        )
        print_metrics(holdout_metrics)
        all_metrics.append(holdout_metrics)

        blind_df = load_blind_test_dataset(model_name)
        X_blind = blind_df[FEATURE_COLUMNS]
        y_blind = blind_df["detector_label"].to_numpy()

        blind_metrics = evaluate_detector(model_name, "Blind Test", X_blind, y_blind)
        print_metrics(blind_metrics)
        all_metrics.append(blind_metrics)

        for metrics in (holdout_metrics, blind_metrics):
            summary_rows.append(
                {
                    "model": metrics["model"],
                    "split": metrics["split"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EVAL_OUTPUT_DIR / "layer2_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved metrics summary: {summary_path}")
    print("\nSummary table:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_roc_curves(all_metrics)


if __name__ == "__main__":
    main()
