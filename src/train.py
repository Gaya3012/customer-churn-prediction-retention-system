"""Train, compare, report, and persist churn classification pipelines."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import evaluate_model, select_best_model
from src.preprocess import DEFAULT_DATA_PATH, build_preprocessor, load_dataset, split_features_target

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_eda_plots(frame: pd.DataFrame, reports_dir: Path) -> None:
    """Save a churn distribution and numeric-feature EDA plot."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=frame, x="Churn", hue="Churn", order=["No", "Yes"], legend=False, ax=axis)
    axis.set(title="Customer churn distribution", xlabel="Churn", ylabel="Customers")
    fig.tight_layout()
    fig.savefig(reports_dir / "churn_distribution.png", dpi=150)
    plt.close(fig)

    numeric = frame.select_dtypes(include=["number"]).copy()
    numeric["Churn"] = frame["Churn"].map({"No": "No", "Yes": "Yes"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=numeric, x="tenure", hue="Churn", bins=24, element="step", stat="density", common_norm=False, ax=axes[0])
    axes[0].set_title("Tenure by churn")
    sns.boxplot(data=numeric, x="Churn", y="MonthlyCharges", hue="Churn", legend=False, ax=axes[1])
    axes[1].set_title("Monthly charges by churn")
    fig.tight_layout()
    fig.savefig(reports_dir / "customer_feature_eda.png", dpi=150)
    plt.close(fig)


def create_comparison_plot(results: dict[str, dict[str, Any]], reports_dir: Path) -> None:
    metrics = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    axis = metrics.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0)
    axis.set(title="Model comparison", ylabel="Score", xlabel="Model")
    axis.legend(loc="lower right")
    axis.figure.tight_layout()
    axis.figure.savefig(reports_dir / "model_comparison.png", dpi=150)
    plt.close(axis.figure)


def train(data_path: str | Path = DEFAULT_DATA_PATH, models_dir: str | Path | None = None,
          reports_dir: str | Path | None = None) -> dict[str, Any]:
    """Train candidate models on a stratified split and persist the best pipeline."""
    models_path = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
    reports_path = Path(reports_dir) if reports_dir else PROJECT_ROOT / "reports"
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(data_path)
    create_eda_plots(frame, reports_path)
    features, target = split_features_target(frame)
    if target.nunique() != 2 or target.value_counts().min() < 2:
        raise ValueError("Training requires at least two examples of each churn class.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }

    fitted: dict[str, Pipeline] = {}
    results: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        results[name] = evaluate_model(pipeline, x_test, y_test)
        print(f"{name}: ROC-AUC={results[name]['roc_auc']:.4f}, F1={results[name]['f1_score']:.4f}")

    best_name = select_best_model(results)
    best_pipeline = fitted[best_name]
    model_path = models_path / "churn_pipeline.joblib"
    dump(best_pipeline, model_path)

    report = {
        "selection_metric": "ROC-AUC (tie-breakers: F1-score, accuracy)",
        "best_model": best_name,
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "rows": int(len(frame)),
        "features": list(features.columns),
        "models": results,
    }
    (reports_path / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(results).T.to_csv(reports_path / "model_metrics.csv", index_label="model")
    create_comparison_plot(results, reports_path)

    fig, axes = plt.subplots(1, len(fitted), figsize=(15, 4))
    for axis, (name, pipeline) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_estimator(
            pipeline, x_test, y_test, display_labels=["No churn", "Churn"],
            cmap="Blues", colorbar=False, ax=axis,
        )
        axis.set_title(name)
    fig.tight_layout()
    fig.savefig(reports_path / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for name, pipeline in fitted.items():
        RocCurveDisplay.from_estimator(pipeline, x_test, y_test, name=name, ax=axis)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_title("ROC curves (test set)")
    fig.tight_layout()
    fig.savefig(reports_path / "roc_curves.png", dpi=150)
    plt.close(fig)

    print(f"Selected: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Evaluation outputs: {reports_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the IBM Telco CSV.")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.data, arguments.models_dir, arguments.reports_dir)
