# CruyffAI

AI-powered football scouting and transfer path optimizer for the Artificial Intelligence project.

## Project Scope

The project has two required parts:

1. **Informed Search:** A* search on a football transfer graph, compared against Greedy Best-First Search.
2. **Machine Learning:** Binary top-talent classification using the FIFA 22 player dataset.

## Current Dataset

The selected file is:

```text
data/raw/players_22.csv
```

Target label:

```text
top_talent = 1 if overall >= 80 or potential >= 85 else 0
```

The dataset has 19,239 players and 110 columns. The positive class is about 3.36%, so the ML pipeline uses SMOTE.

## Setup

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `python` is not available on PATH, install Python 3.10+ first.

## Run ML Training

```powershell
python -m src.ml.train_models
```

Outputs:

```text
outputs/metrics.json
outputs/figures/roc_curves.png
outputs/figures/*_confusion_matrix.png
outputs/figures/random_forest_feature_importance.png
```

Models trained:

- Logistic Regression
- Random Forest
- XGBoost if installed, otherwise scikit-learn Gradient Boosting

XGBoost is included in `requirements.txt`. If it is unavailable in the active environment, the code automatically falls back to scikit-learn Gradient Boosting so the baseline still runs.

## Run SHAP Explainability

```powershell
python -m src.ml.shap_explain
```

Output:

```text
outputs/figures/shap_summary_random_forest.png
```

## Run Search Demo

```powershell
python -m src.search.run_demo
```

Outputs:

```text
outputs/search_results.json
outputs/figures/transfer_graph_paths.png
```

The demo compares:

- A* Search
- Greedy Best-First Search

Metrics reported:

- final path
- total transfer cost
- nodes expanded
- expansion order

## Folder Structure

```text
data/raw/              Original dataset files
data/processed/        Cleaned or engineered datasets
notebooks/             Submission-friendly notebooks
src/ml/                ML feature engineering and training code
src/search/            A* and Greedy search code
outputs/figures/       Generated plots
reports/               Final report material
```

## Next Implementation Steps

1. Install dependencies.
2. Run `python -m src.ml.train_models`.
3. Review model metrics and generated plots.
4. Run SHAP analysis for the best tree-based model.
5. Run `python -m src.search.run_demo`.
6. Improve the transfer graph using club-level statistics derived from `players_22.csv`.
7. Move analysis screenshots and results into the final report.
