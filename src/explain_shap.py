import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from .utils import FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, ensure_directories

RANDOM_STATE = 42


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(PROCESSED_DIR / "ufc_model_table.csv")
    X = df.drop(columns=["y_red_win"])
    y = df["y_red_win"].astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)


def pick_best_tree_model(metrics_payload: dict) -> str:
    candidates = ["random_forest", "xgboost"]
    rows = []
    for name in candidates:
        m = metrics_payload["metrics"][name]
        rows.append((name, m["f1"], m["roc_auc"]))
    rows = sorted(rows, key=lambda t: (t[1], t[2]), reverse=True)
    return rows[0][0]


def to_explanation_array(shap_values, pos_class_index: int = 1):
    if isinstance(shap_values, list):
        return shap_values[pos_class_index]
    if hasattr(shap_values, "values"):
        values = shap_values.values
        if values.ndim == 3:
            return values[:, :, pos_class_index]
        return values
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, pos_class_index]
    return shap_values


def main() -> None:
    ensure_directories()

    with open(OUTPUTS_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics_payload = json.load(f)

    best_tree = pick_best_tree_model(metrics_payload)
    model_path = MODELS_DIR / ("xgboost.joblib" if best_tree == "xgboost" else "random_forest.joblib")
    model_pipe = joblib.load(model_path)

    X_train, X_test, _, _ = load_split()

    preprocessor = model_pipe.named_steps["preprocessor"]
    estimator = model_pipe.named_steps["model"]

    X_test_sample = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
    X_trans = preprocessor.transform(X_test_sample)
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_model_used = best_tree
    except Exception:
        # XGBoost 3.x can fail in some SHAP versions; fall back to RF tree model.
        fallback_pipe = joblib.load(MODELS_DIR / "random_forest.joblib")
        preprocessor = fallback_pipe.named_steps["preprocessor"]
        estimator = fallback_pipe.named_steps["model"]
        X_trans = preprocessor.transform(X_test_sample)
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(estimator)
        shap_model_used = "random_forest"
    shap_raw = explainer.shap_values(X_trans)
    shap_vals = to_explanation_array(shap_raw)

    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = float(np.array(explainer.expected_value).reshape(-1)[-1])
    else:
        base_value = float(explainer.expected_value)

    plt.figure()
    shap.summary_plot(
        shap_vals,
        X_trans,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_vals,
        X_trans,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    idx = 0
    explanation = shap.Explanation(
        values=shap_vals[idx],
        base_values=base_value,
        data=np.array(X_trans[idx]).ravel(),
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SHAP artifacts saved using tree model: {shap_model_used}")


if __name__ == "__main__":
    main()
