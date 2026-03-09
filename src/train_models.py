import json
import warnings
from copy import deepcopy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import callbacks, layers, models, optimizers
from xgboost import XGBClassifier

from .utils import FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, ensure_directories

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DIR / "ufc_model_table.csv")
    X = df.drop(columns=["y_red_win"])
    y = df["y_red_win"].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def evaluate_model(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def save_roc_curve(y_true: pd.Series, y_prob: np.ndarray, out_path: str, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / out_path, dpi=150)
    plt.close()


def create_mlp(input_dim: int, hidden_units: int = 128, dropout: float = 0.2, learning_rate: float = 1e-3) -> models.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def save_history_plot(history: tf.keras.callbacks.History) -> None:
    hist_df = pd.DataFrame(history.history)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_df["loss"], label="train_loss")
    if "val_loss" in hist_df:
        plt.plot(hist_df["val_loss"], label="val_loss")
    plt.title("MLP Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_df["auc"], label="train_auc")
    if "val_auc" in hist_df:
        plt.plot(hist_df["val_auc"], label="val_auc")
    plt.title("MLP AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mlp_training_history.png", dpi=150)
    plt.close()


def run_mlp_tuning(
    X_train_proc: np.ndarray,
    y_train: pd.Series,
    X_test_proc: np.ndarray,
    y_test: pd.Series,
) -> pd.DataFrame:
    grid = {
        "hidden_units": [64, 128],
        "dropout": [0.0, 0.2],
        "learning_rate": [1e-3, 5e-4],
    }

    rows = []
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    for hidden_units in grid["hidden_units"]:
        for dropout in grid["dropout"]:
            for learning_rate in grid["learning_rate"]:
                tf.keras.utils.set_random_seed(RANDOM_STATE)
                model = create_mlp(
                    input_dim=X_train_proc.shape[1],
                    hidden_units=hidden_units,
                    dropout=dropout,
                    learning_rate=learning_rate,
                )
                hist = model.fit(
                    X_train_proc,
                    y_train.values,
                    validation_split=0.2,
                    epochs=30,
                    batch_size=64,
                    callbacks=[early_stop],
                    verbose=0,
                )
                test_prob = model.predict(X_test_proc, verbose=0).ravel()
                test_f1 = f1_score(y_test, (test_prob >= 0.5).astype(int), zero_division=0)
                rows.append(
                    {
                        "hidden_units": hidden_units,
                        "dropout": dropout,
                        "learning_rate": learning_rate,
                        "best_val_auc": float(np.max(hist.history.get("val_auc", [np.nan]))),
                        "test_f1": float(test_f1),
                    }
                )

    tune_df = pd.DataFrame(rows).sort_values(["test_f1", "best_val_auc"], ascending=False)
    tune_df.to_csv(OUTPUTS_DIR / "nn_tuning_results.csv", index=False)

    heat_df = tune_df.copy()
    heat_df["config"] = (
        "dropout=" + heat_df["dropout"].astype(str) + ", lr=" + heat_df["learning_rate"].astype(str)
    )
    pivot = heat_df.pivot(index="hidden_units", columns="config", values="test_f1")

    plt.figure(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("NN Tuning Grid: Test F1")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nn_tuning_heatmap.png", dpi=150)
    plt.close()

    return tune_df


def main() -> None:
    ensure_directories()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor, _, _ = build_preprocessor(X)
    preprocessor_for_save = clone(preprocessor)
    preprocessor_for_save.fit(X_train)
    joblib.dump(preprocessor_for_save, MODELS_DIR / "preprocessor.joblib")

    feature_names = preprocessor_for_save.get_feature_names_out().tolist()
    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics = {}
    best_params = {}

    # Model A: Logistic Regression
    logreg_pipe = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    logreg_pipe.fit(X_train, y_train)
    logreg_prob = logreg_pipe.predict_proba(X_test)[:, 1]
    metrics["logistic_regression"] = evaluate_model(y_test, logreg_prob)
    joblib.dump(logreg_pipe, MODELS_DIR / "logistic_regression.joblib")
    save_roc_curve(y_test, logreg_prob, "roc_logreg.png", "ROC - Logistic Regression")

    # Model B: Decision Tree
    dt_pipe = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
        ]
    )
    dt_grid = {
        "model__max_depth": [3, 5, 7, 10],
        "model__min_samples_leaf": [5, 10, 20, 50],
    }
    dt_search = GridSearchCV(dt_pipe, dt_grid, cv=cv, scoring="f1", n_jobs=-1)
    dt_search.fit(X_train, y_train)
    dt_best = dt_search.best_estimator_
    dt_prob = dt_best.predict_proba(X_test)[:, 1]
    metrics["decision_tree"] = evaluate_model(y_test, dt_prob)
    best_params["decision_tree"] = dt_search.best_params_
    joblib.dump(dt_best, MODELS_DIR / "decision_tree.joblib")
    save_roc_curve(y_test, dt_prob, "roc_tree.png", "ROC - Decision Tree")

    # Model C: Random Forest
    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [4, 6, 8],
        "model__min_samples_leaf": [5, 10, 20],
    }
    rf_search = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="f1", n_jobs=-1)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_prob = rf_best.predict_proba(X_test)[:, 1]
    metrics["random_forest"] = evaluate_model(y_test, rf_prob)
    best_params["random_forest"] = rf_search.best_params_
    joblib.dump(rf_best, MODELS_DIR / "random_forest.joblib")
    save_roc_curve(y_test, rf_prob, "roc_rf.png", "ROC - Random Forest")

    # Model D: XGBoost
    pos = int(y_train.sum())
    neg = int((1 - y_train).sum())
    scale_pos_weight = neg / max(pos, 1)

    xgb_pipe = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )
    xgb_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1],
    }
    xgb_search = GridSearchCV(xgb_pipe, xgb_grid, cv=cv, scoring="f1", n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_
    xgb_prob = xgb_best.predict_proba(X_test)[:, 1]
    metrics["xgboost"] = evaluate_model(y_test, xgb_prob)
    best_params["xgboost"] = xgb_search.best_params_
    joblib.dump(xgb_best, MODELS_DIR / "xgboost.joblib")
    save_roc_curve(y_test, xgb_prob, "roc_xgb.png", "ROC - XGBoost")

    # Model E: Neural Network MLP
    mlp_preprocessor = clone(preprocessor)
    X_train_proc = mlp_preprocessor.fit_transform(X_train)
    X_test_proc = mlp_preprocessor.transform(X_test)

    X_train_proc = np.asarray(X_train_proc, dtype=np.float32)
    X_test_proc = np.asarray(X_test_proc, dtype=np.float32)

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    mlp_model = create_mlp(input_dim=X_train_proc.shape[1], hidden_units=128, dropout=0.2, learning_rate=1e-3)
    history = mlp_model.fit(
        X_train_proc,
        y_train.values,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0,
    )

    save_history_plot(history)
    mlp_prob = mlp_model.predict(X_test_proc, verbose=0).ravel()
    metrics["mlp"] = evaluate_model(y_test, mlp_prob)
    mlp_model.save(MODELS_DIR / "mlp.keras")
    save_roc_curve(y_test, mlp_prob, "roc_mlp.png", "ROC - Neural Net MLP")

    # Bonus: NN tuning grid
    tune_df = run_mlp_tuning(X_train_proc, y_train, X_test_proc, y_test)

    # Comparison artifacts
    comparison_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    comparison_df = comparison_df.sort_values(["f1", "roc_auc"], ascending=False)
    comparison_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison_df, x="model", y="f1", palette="crest")
    plt.title("Model Comparison by Test F1")
    plt.xlabel("Model")
    plt.ylabel("F1 score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "f1_comparison.png", dpi=150)
    plt.close()

    best_row = comparison_df.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]
    best_model = best_row["model"]

    metrics_payload = {
        "metrics": metrics,
        "best_model": best_model,
        "best_model_test_f1": float(best_row["f1"]),
        "best_model_test_roc_auc": float(best_row["roc_auc"]),
    }

    with open(OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with open(OUTPUTS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("Training complete. Saved models, metrics, and figures.")
    print("Top models by F1:")
    print(comparison_df[["model", "f1", "roc_auc"]].head().to_string(index=False))
    print(f"NN tuning runs: {len(tune_df)}")


if __name__ == "__main__":
    main()