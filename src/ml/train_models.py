"""Train and evaluate talent classification models for CruyffAI."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.features import add_custom_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "players_22.csv"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "metrics.json"
DEFAULT_FIGURES = PROJECT_ROOT / "outputs" / "figures"


NUMERIC_FEATURES = [
    "age",
    "height_cm",
    "weight_kg",
    "value_eur",
    "wage_eur",
    "league_level",
    "weak_foot",
    "skill_moves",
    "international_reputation",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "movement_reactions",
    "mentality_composure",
    "performance_index",
    "age_adjusted_potential",
    "market_value_efficiency",
    "wage_value_ratio",
]

CATEGORICAL_FEATURES = [
    "position_group",
    "preferred_foot",
    "work_rate",
    "league_name",
]


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    df = add_custom_features(df)
    features = [column for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES if column in df.columns]
    return df[features], df["top_talent"]


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps)
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, ImbPipeline]:
    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        boosting_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
        boosting_name = "xgboost"
    else:
        boosting_model = GradientBoostingClassifier(random_state=42)
        boosting_name = "gradient_boosting"

    return {
        "logistic_regression": ImbPipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=True)),
                ("smote", SMOTE(random_state=42)),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                ),
            ]
        ),
        "random_forest": ImbPipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=False)),
                ("smote", SMOTE(random_state=42)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        boosting_name: ImbPipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=False)),
                ("smote", SMOTE(random_state=42)),
                ("model", boosting_model),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, ImbPipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    figure_dir: Path,
) -> dict[str, dict[str, object]]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, object]] = {}

    roc_fig, roc_ax = plt.subplots(figsize=(8, 6))

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics[name] = {
            "classification_report": report,
            "roc_auc": roc_auc_score(y_test, y_score),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=None, cmap="Blues")
        plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(figure_dir / f"{name}_confusion_matrix.png", dpi=180)
        plt.close()

        RocCurveDisplay.from_predictions(y_test, y_score, name=name.replace("_", " ").title(), ax=roc_ax)

    roc_ax.set_title("ROC Curves: Talent Classification Models")
    roc_ax.grid(alpha=0.3)
    roc_fig.tight_layout()
    roc_fig.savefig(figure_dir / "roc_curves.png", dpi=180)
    plt.close(roc_fig)

    random_forest = models["random_forest"]
    model = random_forest.named_steps["model"]
    feature_names = random_forest.named_steps["preprocess"].get_feature_names_out()
    importances = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    plt.figure(figsize=(10, 7))
    sns.barplot(data=importances, x="importance", y="feature", color="#2563eb")
    plt.title("Random Forest Top Feature Importances")
    plt.tight_layout()
    plt.savefig(figure_dir / "random_forest_feature_importance.png", dpi=180)
    plt.close()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()

    X, y = load_dataset(args.input)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    metrics = evaluate_models(build_models(), X_train, X_test, y_train, y_test, args.figure_dir)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved metrics to {args.metrics_output}")
    print(f"Saved figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
