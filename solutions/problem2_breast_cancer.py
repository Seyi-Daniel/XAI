"""Problem 2: SHAP for Classification on the Breast Cancer dataset."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from solutions.utils.reporting import dump_json, ensure_dir, write_markdown, write_table_md


plt.switch_backend("Agg")


@dataclass
class ModelResult:
    name: str
    model: Pipeline
    accuracy: float
    precision: float
    recall: float
    f1: float


RANDOM_STATE = 42
REPORT_DIR = Path("reports/problem2")
N_SAMPLES = 5


def _load_dataset():
    dataset = load_breast_cancer()
    feature_names = dataset.feature_names
    X = pd.DataFrame(dataset.data, columns=feature_names)
    y = dataset.target
    return X, y, dataset.target_names


def _build_models() -> Dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=RANDOM_STATE)),
        ]),
    }


def _evaluate(models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, ModelResult]:
    summary: Dict[str, ModelResult] = {}
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        summary[name] = ModelResult(
            name=name,
            model=pipeline,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
        )
    return summary


def _run_shap(best_result: ModelResult, X_train: pd.DataFrame, X_test: pd.DataFrame, class_names: List[str], output_dir: Path):
    background = shap.sample(X_train, 100, random_state=RANDOM_STATE)
    samples = X_test.sample(N_SAMPLES, random_state=RANDOM_STATE)

    def _pipeline_proba(data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=X_train.columns)
        else:
            # Ensure the columns are in the same order as training data
            data = data[X_train.columns]
        return best_result.model.predict_proba(data)

    explainer = shap.KernelExplainer(_pipeline_proba, background)
    shap_values = explainer.shap_values(samples)

    mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
    shap_importances = pd.Series(mean_abs_shap, index=X_train.columns).sort_values(ascending=False)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        samples,
        feature_names=X_train.columns,
        class_names=class_names,
        show=False,
    )
    summary_plot_path = output_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    plt.close()

    # Dependence plot for top three features
    dependence_paths = []
    for feature in shap_importances.index[:3]:
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(
            feature,
            shap_values[1],
            samples,
            feature_names=X_train.columns,
            interaction_index=None,
            show=False,
        )
        path = output_dir / f"dependence_{feature}.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        dependence_paths.append(path)

    dump_json(
        output_dir / "shap_summary.json",
        {
            "samples": samples.index.tolist(),
            "shap_top_features": shap_importances.head(10).round(4).to_dict(),
            "summary_plot": str(summary_plot_path),
            "dependence_plots": [str(path) for path in dependence_paths],
        },
    )

    sample_details = []
    for idx, (sample_index, row) in enumerate(samples.iterrows()):
        contributions = sorted(
            zip(X_train.columns, shap_values[1][idx]),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        contribution_text = ", ".join(f"{feature} ({value:+.3f})" for feature, value in contributions)
        sample_details.append(f"- Sample {sample_index}: {contribution_text}")

    return sample_details


def run(output_dir: Path | str = REPORT_DIR) -> Dict[str, object]:
    output_dir = ensure_dir(output_dir)

    X, y, class_names = _load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = _build_models()
    for model in models.values():
        model.fit(X_train, y_train)

    results = _evaluate(models, X_test, y_test)
    metrics_table = write_table_md(
        headers=["Model", "Accuracy", "Precision", "Recall", "F1"],
        rows=[
            (
                res.name,
                f"{res.accuracy:.3f}",
                f"{res.precision:.3f}",
                f"{res.recall:.3f}",
                f"{res.f1:.3f}",
            )
            for res in results.values()
        ],
    )

    best_model = max(results.values(), key=lambda res: res.f1)
    shap_sample_details = _run_shap(best_model, X_train, X_test, list(class_names), output_dir)

    write_markdown(
        output_dir / "summary.md",
        lines=[
            "# Problem 2 – SHAP for the Breast Cancer Dataset",
            "",
            "## Model Evaluation",
            "",
            metrics_table,
            "",
            f"Best model based on F1-score: **{best_model.name}**.",
            "",
            "## SHAP Explanations for Selected Samples",
            "",
            *shap_sample_details,
            "",
            "Summary and dependence plots saved under `reports/problem2`.",
        ],
    )

    dump_json(
        output_dir / "metrics.json",
        {name: asdict(result) for name, result in results.items()},
    )

    return {
        "metrics": {name: asdict(result) for name, result in results.items()},
        "best_model": best_model.name,
    }


if __name__ == "__main__":
    run()
