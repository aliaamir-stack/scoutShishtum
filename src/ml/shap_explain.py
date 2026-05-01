"""Generate SHAP explanations for the Random Forest talent model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.ml.train_models import DEFAULT_FIGURES, DEFAULT_INPUT, build_models, load_dataset


DEFAULT_OUTPUT = DEFAULT_FIGURES / "shap_summary_random_forest.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()

    X, y = load_dataset(args.input)
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = build_models()["random_forest"]
    pipeline.fit(X_train, y_train)

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    sample = X_test.sample(min(args.sample_size, len(X_test)), random_state=42)
    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    positive_class_values = shap_values[1] if isinstance(shap_values, list) else shap_values

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        positive_class_values,
        transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary to {args.output}")


if __name__ == "__main__":
    main()
