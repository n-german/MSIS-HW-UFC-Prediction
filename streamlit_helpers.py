import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

ROOT = Path(__file__).resolve().parent


def _pick_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _artifact(name: str, kind: str) -> Path:
    if kind == "model":
        return _pick_path([ROOT / "models" / name, ROOT / name])
    if kind == "processed":
        return _pick_path([ROOT / "data" / "processed" / name, ROOT / name])
    if kind == "output":
        return _pick_path([ROOT / "outputs" / name, ROOT / name])
    if kind == "figure":
        return _pick_path([ROOT / "outputs" / "figures" / name, ROOT / name])
    raise ValueError(f"Unknown artifact kind: {kind}")


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(_artifact("ufc_model_table.csv", "processed"))


@st.cache_data
def load_metrics() -> dict:
    with open(_artifact("metrics.json", "output"), "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_best_params() -> dict:
    with open(_artifact("best_params.json", "output"), "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_feature_ranges() -> dict:
    with open(_artifact("feature_ranges.json", "processed"), "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(_artifact("model_comparison.csv", "output"))


@st.cache_resource
def load_models() -> dict:
    models = {
        "logistic_regression": joblib.load(_artifact("logistic_regression.joblib", "model")),
        "decision_tree": joblib.load(_artifact("decision_tree.joblib", "model")),
        "random_forest": joblib.load(_artifact("random_forest.joblib", "model")),
        "xgboost": joblib.load(_artifact("xgboost.joblib", "model")),
        "mlp": None,
        "preprocessor": joblib.load(_artifact("preprocessor.joblib", "model")),
    }
    mlp_path = _artifact("mlp.keras", "model")
    if tf is not None and mlp_path.exists():
        models["mlp"] = tf.keras.models.load_model(mlp_path)
    return models


def available_prediction_models(models_dict: dict) -> list[str]:
    models = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
    if models_dict.get("mlp") is not None:
        models.append("mlp")
    return models


@st.cache_data
def load_feature_columns() -> list[str]:
    df = load_processed_data()
    return [c for c in df.columns if c != "y_red_win"]


def get_best_tree_model_name(metrics: dict) -> str:
    rf = metrics["metrics"]["random_forest"]
    xgb = metrics["metrics"]["xgboost"]
    return "random_forest" if (rf["f1"], rf["roc_auc"]) >= (xgb["f1"], xgb["roc_auc"]) else "xgboost"


def default_input_row(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    defaults = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median()) if df[col].notna().any() else 0.0
        else:
            mode = df[col].mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"
    return defaults


def predict_probability(model_name: str, models_dict: dict, input_df: pd.DataFrame) -> float:
    if model_name == "mlp":
        if models_dict.get("mlp") is None:
            raise ValueError("MLP model unavailable in this runtime.")
        X = models_dict["preprocessor"].transform(input_df)
        return float(models_dict["mlp"].predict(np.asarray(X, dtype=np.float32), verbose=0).ravel()[0])
    return float(models_dict[model_name].predict_proba(input_df)[:, 1][0])


def make_user_shap_waterfall(best_tree_pipeline, input_df: pd.DataFrame):
    preprocessor = best_tree_pipeline.named_steps["preprocessor"]
    estimator = best_tree_pipeline.named_steps["model"]
    transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.TreeExplainer(estimator)
    except Exception:
        fallback_pipe = joblib.load(_artifact("random_forest.joblib", "model"))
        preprocessor = fallback_pipe.named_steps["preprocessor"]
        estimator = fallback_pipe.named_steps["model"]
        transformed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(estimator)

    shap_raw = explainer.shap_values(transformed)
    if isinstance(shap_raw, list):
        values = shap_raw[1][0]
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        values = shap_raw[0, :, 1]
    elif hasattr(shap_raw, "values") and np.asarray(shap_raw.values).ndim == 3:
        values = np.asarray(shap_raw.values)[0, :, 1]
    elif hasattr(shap_raw, "values"):
        values = np.asarray(shap_raw.values)[0]
    else:
        values = np.asarray(shap_raw)[0]

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.array(expected).reshape(-1)[-1])
    else:
        expected = float(expected)

    explanation = shap.Explanation(
        values=values,
        base_values=expected,
        data=np.array(transformed[0]).ravel(),
        feature_names=feature_names,
    )
    return shap.plots.waterfall(explanation, max_display=15, show=False)


def figure_path(name: str) -> Path:
    return _artifact(name, "figure")


PLOT_CAPTIONS = {
    "target_distribution.png": "The target is moderately imbalanced, with RED wins occurring more often than BLUE wins. We address this with class balancing and F1-first model selection.",
    "eda_1.png": "RED reach advantage is associated with higher red-corner win concentration, though overlap remains significant.",
    "eda_2.png": "Higher prior win-rate differences are directionally linked with RED wins, but not deterministically.",
    "eda_3.png": "Red-corner win rate varies by weight class, supporting inclusion of class context in prediction.",
    "eda_4.png": "Reach and prior-win-rate differences together show better separation than either feature alone.",
    "correlation_heatmap.png": "Engineered difference and base features are correlated by construction; models account for this during fitting.",
}