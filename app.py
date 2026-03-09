import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from src.streamlit_helpers import (
        PLOT_CAPTIONS,
        available_prediction_models,
        figure_path,
        get_best_tree_model_name,
        load_best_params,
        load_feature_columns,
        load_feature_ranges,
        load_metrics,
        load_model_comparison,
        load_models,
        load_processed_data,
        make_user_shap_waterfall,
        predict_probability,
        default_input_row,
    )
except Exception:
    from streamlit_helpers import (
        PLOT_CAPTIONS,
        available_prediction_models,
        figure_path,
        get_best_tree_model_name,
        load_best_params,
        load_feature_columns,
        load_feature_ranges,
        load_metrics,
        load_model_comparison,
        load_models,
        load_processed_data,
        make_user_shap_waterfall,
        predict_probability,
        default_input_row,
    )

st.set_page_config(page_title="UFC Pre-Fight Win Predictor", layout="wide")

st.title("UFC Pre-Fight Win Probability Predictor")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

metrics_payload = load_metrics()
comparison_df = load_model_comparison()
best_params = load_best_params()
feature_ranges = load_feature_ranges()
models_dict = load_models()
model_df = load_processed_data()
feature_cols = load_feature_columns()

best_model_name = metrics_payload["best_model"]
best_tree_name = get_best_tree_model_name(metrics_payload)
prediction_models = available_prediction_models(models_dict)

with tab1:
    st.subheader("Prediction Task")
    st.markdown(
        "This project uses UFC event-level and fighter-level tabular data (1993-2025) to build a **binary "
        "classification** model that predicts whether the RED-corner fighter wins **before** the bout starts. "
        "The dependent variable is `y_red_win`, where `1` means RED wins and `0` means BLUE wins. Input features "
        "include pre-fight context (weight class, card details, venue), stable fighter traits (age, height, reach, "
        "stance), and prior-performance summaries computed strictly from each fighter's fight history before the "
        "current event."
    )
    st.markdown(
        "Why this matters: scouting teams, analysts, and coaching staff need structured matchup intelligence before "
        "fight night. A calibrated pre-fight probability can support opponent preparation, roster decisions, and "
        "scenario planning. This app is an analytical estimator for decision support, not a betting recommendation "
        "engine."
    )
    st.markdown(
        "Approach summary: the workflow includes EDA, leakage-controlled feature engineering, and comparison of "
        "five predictive models (Logistic Regression, Decision Tree, Random Forest, XGBoost, and MLP). Model "
        "selection is based on held-out test performance (F1 and ROC-AUC), and explainability is handled with SHAP "
        "on the best tree model. Leakage policy is strict: post-fight/in-fight outcome fields are removed, and "
        "`fighter_stats.csv` is excluded from predictive features."
    )

    best_f1 = metrics_payload["best_model_test_f1"]
    best_auc = metrics_payload["best_model_test_roc_auc"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", best_model_name)
    c2.metric("Best Test F1", f"{best_f1:.3f}")
    c3.metric("Best Test ROC-AUC", f"{best_auc:.3f}")

with tab2:
    st.subheader("Descriptive Analytics")
    for fname in [
        "target_distribution.png",
        "eda_1.png",
        "eda_2.png",
        "eda_3.png",
        "eda_4.png",
        "correlation_heatmap.png",
    ]:
        path = figure_path(fname)
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.caption(PLOT_CAPTIONS.get(fname, ""))
        else:
            st.warning(f"Missing figure: {fname}. Run `make all`.")

with tab3:
    st.subheader("Model Comparison")
    st.dataframe(comparison_df, use_container_width=True)

    f1_fig = figure_path("f1_comparison.png")
    if f1_fig.exists():
        st.image(str(f1_fig), use_container_width=True)

    st.subheader("ROC Curves")
    roc_files = ["roc_logreg.png", "roc_tree.png", "roc_rf.png", "roc_xgb.png", "roc_mlp.png"]
    cols = st.columns(2)
    for i, fname in enumerate(roc_files):
        path = figure_path(fname)
        with cols[i % 2]:
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning(f"Missing {fname}")

    st.subheader("Best Hyperparameters")
    st.json(best_params)

    st.subheader("Bonus: Neural Network Tuning Grid")
    nn_heat = figure_path("nn_tuning_heatmap.png")
    if nn_heat.exists():
        st.image(str(nn_heat), use_container_width=True)
        st.caption("This grid summarizes how hidden units, dropout, and learning rate affect test F1.")
    if "mlp" not in prediction_models:
        st.info("MLP artifact is shown in results, but TensorFlow is unavailable in this deployment runtime.")

with tab4:
    st.subheader("Explainability")
    shap1 = figure_path("shap_summary_beeswarm.png")
    shap2 = figure_path("shap_bar_importance.png")
    s1, s2 = st.columns(2)
    with s1:
        if shap1.exists():
            st.image(str(shap1), use_container_width=True)
    with s2:
        if shap2.exists():
            st.image(str(shap2), use_container_width=True)

    st.markdown(
        "SHAP values show how each feature pushes probability up or down. "
        "Positive SHAP contributions push toward RED winning; negative values push toward BLUE."
    )

    st.subheader("Interactive Pre-Fight Prediction")
    selected_model = st.selectbox(
        "Select model for prediction",
        prediction_models,
        index=min(3, len(prediction_models) - 1),
    )

    defaults = default_input_row(model_df, feature_cols)

    c1, c2, c3 = st.columns(3)

    with c1:
        weight_options = feature_ranges.get("weight_class", [])
        weight_class = st.selectbox(
            "weight_class",
            options=weight_options,
            index=weight_options.index(defaults["weight_class"]) if defaults["weight_class"] in weight_options else 0,
        )
        title_fight = st.selectbox("title_fight", options=[0, 1], index=int(defaults.get("title_fight", 0) >= 0.5))
        card_options = feature_ranges.get("card_section", [])
        card_section = st.selectbox(
            "card_section",
            options=card_options,
            index=card_options.index(defaults["card_section"]) if defaults["card_section"] in card_options else 0,
        )
        cp = feature_ranges.get("card_position", {})
        card_position = st.slider(
            "card_position",
            min_value=int(cp.get("min", 1) or 1),
            max_value=int(cp.get("max", 15) or 15),
            value=int(cp.get("median", defaults.get("card_position", 1)) or 1),
        )

    with c2:
        reach = feature_ranges.get("reach_diff", {})
        reach_diff = st.slider(
            "reach_diff",
            min_value=float(reach.get("p01", -20.0) or -20.0),
            max_value=float(reach.get("p99", 20.0) or 20.0),
            value=float(reach.get("median", 0.0) or 0.0),
        )
        height = feature_ranges.get("height_diff", {})
        height_diff = st.slider(
            "height_diff",
            min_value=float(height.get("p01", -15.0) or -15.0),
            max_value=float(height.get("p99", 15.0) or 15.0),
            value=float(height.get("median", 0.0) or 0.0),
        )
        age = feature_ranges.get("age_diff", {})
        age_diff = st.slider(
            "age_diff",
            min_value=float(age.get("p01", -15.0) or -15.0),
            max_value=float(age.get("p99", 15.0) or 15.0),
            value=float(age.get("median", 0.0) or 0.0),
        )
        stance_options = feature_ranges.get("red_stance", ["Orthodox", "Southpaw", "Switch"])
        red_stance = st.selectbox("red_stance", options=stance_options, index=0)
        blue_options = feature_ranges.get("blue_stance", ["Orthodox", "Southpaw", "Switch"])
        blue_stance = st.selectbox("blue_stance", options=blue_options, index=0)

    with c3:
        pwr = feature_ranges.get("prior_win_rate_diff", {})
        prior_win_rate_diff = st.slider(
            "prior_win_rate_diff",
            min_value=float(pwr.get("p01", -1.0) or -1.0),
            max_value=float(pwr.get("p99", 1.0) or 1.0),
            value=float(pwr.get("median", 0.0) or 0.0),
        )
        pfd = feature_ranges.get("prior_fights_diff", {})
        prior_fights_diff = st.slider(
            "prior_fights_diff",
            min_value=float(pfd.get("p01", -30.0) or -30.0),
            max_value=float(pfd.get("p99", 30.0) or 30.0),
            value=float(pfd.get("median", 0.0) or 0.0),
        )
        pfr = feature_ranges.get("prior_finish_rate_diff", {})
        prior_finish_rate_diff = st.slider(
            "prior_finish_rate_diff",
            min_value=float(pfr.get("p01", -1.0) or -1.0),
            max_value=float(pfr.get("p99", 1.0) or 1.0),
            value=float(pfr.get("median", 0.0) or 0.0),
        )
        dlf = feature_ranges.get("days_since_last_fight_diff", {})
        days_since_last_fight_diff = st.slider(
            "days_since_last_fight_diff",
            min_value=float(dlf.get("p01", -600.0) or -600.0),
            max_value=float(dlf.get("p99", 600.0) or 600.0),
            value=float(dlf.get("median", 0.0) or 0.0),
        )

    venue_options = feature_ranges.get("event_venue_country", [])
    event_venue_country = st.selectbox(
        "event_venue_country",
        options=venue_options,
        index=venue_options.index(defaults["event_venue_country"]) if defaults["event_venue_country"] in venue_options else 0,
    )

    input_payload = defaults.copy()
    input_payload.update(
        {
            "weight_class": weight_class,
            "title_fight": float(title_fight),
            "card_section": card_section,
            "card_position": float(card_position),
            "event_venue_country": event_venue_country,
            "reach_diff": reach_diff,
            "height_diff": height_diff,
            "age_diff": age_diff,
            "prior_win_rate_diff": prior_win_rate_diff,
            "prior_fights_diff": prior_fights_diff,
            "prior_finish_rate_diff": prior_finish_rate_diff,
            "days_since_last_fight_diff": days_since_last_fight_diff,
            "red_stance": red_stance,
            "blue_stance": blue_stance,
            "stance_matchup": f"{red_stance}_vs_{blue_stance}",
        }
    )

    input_df = pd.DataFrame([input_payload])[feature_cols]

    if st.button("Predict RED Win Probability"):
        prob = predict_probability(selected_model, models_dict, input_df)
        label = "RED wins" if prob >= 0.5 else "BLUE wins"
        st.markdown(f"## Prediction: **{label}**")
        st.markdown(f"## p(RED wins): **{prob:.3f}**")

        st.markdown(f"Waterfall explanation below uses best tree model: **{best_tree_name}**.")
        plt.figure(figsize=(9, 5))
        make_user_shap_waterfall(models_dict[best_tree_name], input_df)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption(
            "Read left to right: the base value is the average prediction, and each feature shifts that value "
            "up (toward RED) or down (toward BLUE) until the final probability."
        )
