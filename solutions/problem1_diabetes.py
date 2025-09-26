"""Problem 1: LIME for Classification on the Diabetes dataset."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lime.lime_tabular import LimeTabularExplainer

from solutions.utils.reporting import dump_json, ensure_dir, write_markdown, write_table_md


@dataclass
class ModelResult:
    name: str
    model: object
    accuracy: float
    precision: float
    recall: float
    f1: float


RANDOM_STATE = 42
N_LIME_SAMPLES = 5
REPORT_DIR = Path("reports/problem1")


def _prepare_dataset(random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dataset = load_diabetes()
    feature_names = list(dataset.feature_names)
    X = dataset.data
    # Binarise the regression target around the median to obtain a balanced classification task.
    y = (dataset.target >= np.median(dataset.target)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def _train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=RANDOM_STATE),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def _evaluate_models(models: Dict[str, object], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, ModelResult]:
    results: Dict[str, ModelResult] = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = ModelResult(
            name=name,
            model=model,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
        )
    return results


def _select_samples(X_test: np.ndarray, y_test: np.ndarray, n_samples: int = N_LIME_SAMPLES) -> List[int]:
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(len(X_test))
    rng.shuffle(indices)
    return indices[:n_samples].tolist()


def _summarise_lime_features(exp, class_idx: int) -> List[Tuple[str, float]]:
    # exp.local_exp maps class index to (feature_id, weight) tuples sorted by absolute contribution.
    return [
        (exp.domain_mapper.feature_names[feature_id], round(weight, 4))
        for feature_id, weight in exp.local_exp[class_idx]
    ]


def _collect_kernel_width_stats(explainer: LimeTabularExplainer, instance: np.ndarray, predict_fn, class_idx: int) -> Dict[str, float]:
    explanation = explainer.explain_instance(instance, predict_fn, num_features=10, num_samples=2000)
    weights = explanation.weights
    return {
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "top_features": _summarise_lime_features(explanation, class_idx)[:5],
    }


def run(output_dir: Path | str = REPORT_DIR) -> Dict[str, object]:
    output_dir = ensure_dir(output_dir)

    X_train, X_test, y_train, y_test, feature_names = _prepare_dataset()
    models = _train_models(X_train, y_train)
    results = _evaluate_models(models, X_test, y_test)

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
    predict_fn = best_model.model.predict_proba

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Low progression", "High progression"],
        discretize_continuous=False,
        random_state=RANDOM_STATE,
    )

    selected_indices = _select_samples(X_test, y_test)
    sample_summaries = []
    lime_json_payload = {}
    for idx in selected_indices:
        instance = X_test[idx]
        explanation = explainer.explain_instance(instance, predict_fn, num_features=10)
        class_idx = int(np.argmax(predict_fn([instance])[0]))
        feature_weights = _summarise_lime_features(explanation, class_idx)
        sample_summary = {
            "sample_index": int(idx),
            "predicted_class": int(class_idx),
            "lime_weights": feature_weights,
        }
        lime_json_payload[f"sample_{idx}"] = sample_summary
        sample_summaries.append(
            f"- Sample {idx}: predicted class {class_idx}; top features: "
            + ", ".join(f"{name} ({weight:+.3f})" for name, weight in feature_weights[:5])
        )

    # Global behaviour: compare feature frequencies with logistic regression coefficients
    logistic_coefficients = np.abs(models["Logistic Regression"].coef_[0])
    ranked_features = [
        feature_names[i]
        for i in np.argsort(logistic_coefficients)[::-1]
    ]

    top_global_features = ranked_features[:10]

    frequency_counter: Dict[str, int] = {name: 0 for name in feature_names}
    for summary in lime_json_payload.values():
        for name, _ in summary["lime_weights"][:5]:
            frequency_counter[name] += 1
    lime_top_features = sorted(frequency_counter.items(), key=lambda item: (-item[1], item[0]))

    # Kernel width sensitivity
    kernel_results = {}
    n_features = X_train.shape[1]
    for idx in selected_indices[:2]:
        instance = X_test[idx]
        class_idx = int(np.argmax(predict_fn([instance])[0]))
        sample_key = f"sample_{idx}"
        kernel_results[sample_key] = {}
        for factor in (0.25, 0.75, 1.0):
            width = factor * np.sqrt(n_features)
            tuned_explainer = LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                class_names=["Low progression", "High progression"],
                discretize_continuous=False,
                kernel_width=width,
                random_state=RANDOM_STATE,
            )
            kernel_results[sample_key][f"kernel_{factor}"] = _collect_kernel_width_stats(
                tuned_explainer, instance, predict_fn, class_idx
            )

    dump_json(output_dir / "lime_explanations.json", lime_json_payload)
    dump_json(output_dir / "kernel_width_analysis.json", kernel_results)

    write_markdown(
        output_dir / "summary.md",
        lines=[
            "# Problem 1 – LIME for the Diabetes Dataset",
            "",
            "## Model Evaluation",
            "",
            metrics_table,
            "",
            f"Best model based on F1-score: **{best_model.name}**.",
            "",
            "## LIME Explanations for Selected Samples",
            "",
            *sample_summaries,
            "",
            "## Global vs. Local Feature Importance",
            "",
            "Top logistic regression features (absolute coefficients):",
            ", ".join(top_global_features),
            "",
            "Most frequent LIME features across samples:",
            ", ".join(f"{name} (count={count})" for name, count in lime_top_features[:10]),
            "",
            "## Kernel Width Sensitivity",
            "",
            "See `kernel_width_analysis.json` for summary statistics of the proximity weights under different kernel widths.",
        ],
    )

    return {
        "metrics": {name: asdict(res) for name, res in results.items()},
        "selected_samples": selected_indices,
        "kernel_analysis": kernel_results,
    }


if __name__ == "__main__":
    run()
