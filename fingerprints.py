from pathlib import Path

import numpy as np
import pandas as pd

# true_label, predicted_label, confidence, entropy, margin in .csv file


def get_fingerprints(probabilities):
    """Calculate confidence, entropy, and prediction margin."""

    confidence = np.max(probabilities, axis=1)

    safe_probs = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(safe_probs * np.log(safe_probs), axis=1)

    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    return confidence, entropy, margin


def calculate_ece(y_true, probabilities, n_bins=10):
    """Calculate Expected Calibration Error."""

    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == y_true

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)

        if np.any(mask):
            bin_accuracy = np.mean(correct[mask])
            bin_confidence = np.mean(confidence[mask])
            bin_weight = np.mean(mask)

            ece += bin_weight * abs(
                bin_accuracy - bin_confidence
            )

    return ece


def save_fingerprints(
    model,
    X,
    y_true,
    model_name,
    dataset_name,
    output_dir="model_outputs",
):
    """
    Generate and save fingerprints from a trained model.

    The model must support predict_proba().
    """

    model_output_dir = (
        Path(output_dir)
        / f"{model_name}_fingerprints"
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)

    confidence, entropy, margin = get_fingerprints(
        probabilities
    )

    fingerprint_df = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": predictions,
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
    })

    filename = f"{dataset_name}_fingerprints"

    csv_path = model_output_dir / f"{filename}.csv"
    numpy_path = model_output_dir / f"{filename}.npy"

    fingerprint_df.to_csv(
        csv_path,
        index=False
    )

    np.save(
        numpy_path,
        fingerprint_df[
            [
                "predicted_label",
                "confidence",
                "entropy",
                "margin",
            ]
        ].to_numpy(),
    )

    ece = calculate_ece(
        y_true,
        probabilities
    )

    print(f"\n{model_name} — {dataset_name}")
    print("Average confidence:", confidence.mean())
    print("Average entropy:", entropy.mean())
    print("Average margin:", margin.mean())
    print("ECE:", ece)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved NumPy: {numpy_path}")

    return fingerprint_df