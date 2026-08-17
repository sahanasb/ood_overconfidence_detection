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
CLASS_LABELS = ["Trustworthy\n(NSL)", "Suspicious\n(UNSW/OOD)"]
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

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


def build_report_table(all_metrics: list[dict]) -> pd.DataFrame:
    """Assemble the report table: per-model metrics plus confusion-matrix counts."""
    rows = []

    for metrics in all_metrics:
        true_neg, false_pos, false_neg, true_pos = metrics["confusion_matrix"].ravel()

        rows.append(
            {
                "Model": metrics["model"],
                "Split": metrics["split"],
                "Samples": int(true_neg + false_pos + false_neg + true_pos),
                "TP": int(true_pos),
                "FP": int(false_pos),
                "TN": int(true_neg),
                "FN": int(false_neg),
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
            }
        )

    return pd.DataFrame(rows)


def format_report_table(report_df: pd.DataFrame) -> pd.DataFrame:
    """Render metric columns with a fixed 4-decimal width for display and export."""
    display_df = report_df.copy()

    for column in METRIC_COLUMNS:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    return display_df


def write_markdown_table(display_df: pd.DataFrame, output_path: Path) -> None:
    """Write the report table as a Markdown table for pasting into a report."""
    columns = list(display_df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in display_df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved report table (Markdown): {output_path}")


def plot_report_table(display_df: pd.DataFrame, output_path: Path) -> None:
    """Render the report table as an image for slides and write-ups."""
    fig, ax = plt.subplots(
        figsize=(1.05 * len(display_df.columns) + 2, 0.4 * len(display_df) + 1.2)
    )
    ax.axis("off")

    table = ax.table(
        cellText=display_df.to_numpy(),
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(display_df.columns))))
    table.scale(1, 1.6)

    for column_index in range(len(display_df.columns)):
        header_cell = table[0, column_index]
        header_cell.set_facecolor("#40466e")
        header_cell.set_text_props(color="white", weight="bold")

    ax.set_title(
        "Layer 2 Detector — Evaluation Report\n"
        "(positive class = Suspicious / UNSW-OOD)",
        fontweight="bold",
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved report table (PNG): {output_path}")


def draw_confusion_matrix(ax, confusion: np.ndarray, title: str) -> None:
    """Draw one annotated confusion matrix with counts and per-row percentages."""
    row_percentages = confusion / confusion.sum(axis=1, keepdims=True) * 100

    ax.imshow(row_percentages, cmap="Blues", vmin=0, vmax=100)

    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            ax.text(
                column,
                row,
                f"{confusion[row, column]:,}\n({row_percentages[row, column]:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if row_percentages[row, column] > 50 else "black",
            )

    ax.set_xticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    ax.set_yticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)


def plot_confusion_matrices(all_metrics: list[dict]) -> None:
    """Save one confusion-matrix figure per model, with a panel for each split."""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, config in MODEL_CONFIGS.items():
        display_name = config["display_name"]
        model_metrics = [m for m in all_metrics if m["model"] == display_name]

        if not model_metrics:
            continue

        fig, axes = plt.subplots(
            1, len(model_metrics), figsize=(5.5 * len(model_metrics), 5)
        )

        for ax, metrics in zip(np.atleast_1d(axes), model_metrics):
            draw_confusion_matrix(
                ax,
                metrics["confusion_matrix"],
                f"{metrics['split']}\n"
                f"Accuracy = {metrics['accuracy']:.4f} | F1 = {metrics['f1']:.4f}",
            )

        fig.suptitle(
            f"Layer 2 Confusion Matrix — {display_name} Detector",
            fontweight="bold",
        )
        fig.tight_layout()

        output_path = EVAL_OUTPUT_DIR / f"layer2_confusion_matrix_{model_name}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix plot: {output_path}")


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

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_df = build_report_table(all_metrics)
    display_df = format_report_table(report_df)

    print(f"\n{'=' * 72}")
    print("LAYER 2 EVALUATION REPORT (positive class = Suspicious / UNSW-OOD)")
    print(f"{'=' * 72}")
    print(display_df.to_string(index=False))
    print()

    summary_path = EVAL_OUTPUT_DIR / "layer2_metrics_summary.csv"
    report_df.to_csv(summary_path, index=False)
    print(f"Saved report table (CSV): {summary_path}")

    write_markdown_table(display_df, EVAL_OUTPUT_DIR / "layer2_report_table.md")
    plot_report_table(display_df, EVAL_OUTPUT_DIR / "layer2_report_table.png")
    plot_confusion_matrices(all_metrics)
    plot_roc_curves(all_metrics)


if __name__ == "__main__":
    main()
