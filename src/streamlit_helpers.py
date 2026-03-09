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

from .utils import FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "ufc_model_table.csv")


@st.cache_data
def load_metrics() -> dict:
    with open(OUTPUTS_DIR / "metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_best_params() -> dict:
    with open(OUTPUTS_DIR / "best_params.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_feature_ranges() -> dict:
    with open(PROCESSED_DIR / "feature_ranges.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")


@st.cache_resource
def load_models() -> dict:
    models = {
        "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
        "decision_tree": joblib.load(MODELS_DIR / "decision_tree.joblib"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest.joblib"),
        "xgboost": joblib.load(MODELS_DIR / "xgboost.joblib"),
        "mlp": None,
        "preprocessor": joblib.load(MODELS_DIR / "preprocessor.joblib"),
    }
    if tf is not None and (MODELS_DIR / "mlp.keras").exists():
        models["mlp"] = tf.keras.models.load_model(MODELS_DIR / "mlp.keras")
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
    rf_pair = (rf["f1"], rf["roc_auc"])
    xgb_pair = (xgb["f1"], xgb["roc_auc"])
    return "random_forest" if rf_pair >= xgb_pair else "xgboost"


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
            raise ValueError("MLP model is unavailable in this deployment environment.")
        preprocessor = models_dict["preprocessor"]
        X = preprocessor.transform(input_df)
        X = np.asarray(X, dtype=np.float32)
        return float(models_dict["mlp"].predict(X, verbose=0).ravel()[0])

    model = models_dict[model_name]
    return float(model.predict_proba(input_df)[:, 1][0])


def make_user_shap_waterfall(best_tree_pipeline, input_df: pd.DataFrame):
    preprocessor = best_tree_pipeline.named_steps["preprocessor"]
    estimator = best_tree_pipeline.named_steps["model"]
    transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.TreeExplainer(estimator)
    except Exception:
        fallback_pipe = joblib.load(MODELS_DIR / "random_forest.joblib")
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
    return FIGURES_DIR / name


PLOT_CAPTIONS = {
    "target_distribution.png": "The target is moderately imbalanced, with RED wins occurring more often than BLUE wins. We address this with class balancing (class_weight and scale_pos_weight) and F1-based model selection.",
    "eda_1.png": "Fights where RED has a reach advantage show a higher concentration of RED wins, but the distributions overlap heavily. Reach contributes signal, yet it is not decisive alone.",
    "eda_2.png": "Prior win-rate advantage is directionally associated with RED winning. The overlap indicates matchup context still matters, motivating multivariate models.",
    "eda_3.png": "RED win rates vary by weight class, reflecting stylistic and competitive differences across divisions. This supports retaining weight_class as a categorical predictor.",
    "eda_4.png": "Reach and prior win-rate differences jointly separate outcomes better than either feature in isolation. The scatter suggests nonlinear interactions that tree models can exploit.",
    "correlation_heatmap.png": "Several engineered difference features are correlated with their base components, which is expected from construction. Tree models are robust to this, while regularized linear models provide a baseline.",
}
