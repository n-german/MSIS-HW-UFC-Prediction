import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import FIGURES_DIR, PROCESSED_DIR, ensure_directories

sns.set_theme(style="whitegrid")


def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "ufc_model_table.csv")


def print_basic_stats(df: pd.DataFrame) -> None:
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    missing = df.isna().mean().sort_values(ascending=False)

    print(f"n_rows: {n_rows}")
    print(f"n_features (excluding target): {n_cols - 1}")
    print(f"numeric features: {len([c for c in numeric_cols if c != 'y_red_win'])}")
    print(f"categorical features: {len(categorical_cols)}")
    print("Top missingness (%):")
    print((missing.head(20) * 100).round(2).to_string())


def save_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="y_red_win", data=df, palette="Set2", hue="y_red_win", legend=False)
    ax.set_xticklabels(["BLUE wins (0)", "RED wins (1)"])
    ax.set_title("Target Distribution: y_red_win")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()


def save_eda_plots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="y_red_win", y="reach_diff", data=df, inner="quartile", palette="coolwarm", hue="y_red_win", legend=False)
    plt.title("Reach Difference by Fight Outcome")
    plt.xlabel("y_red_win")
    plt.ylabel("reach_diff (RED - BLUE)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.violinplot(
        x="y_red_win",
        y="prior_win_rate_diff",
        data=df,
        inner="quartile",
        palette="viridis",
        hue="y_red_win",
        legend=False,
    )
    plt.title("Prior Win Rate Difference by Fight Outcome")
    plt.xlabel("y_red_win")
    plt.ylabel("prior_win_rate_diff (RED - BLUE)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_2.png", dpi=150)
    plt.close()

    plot_df = df.copy()
    top_classes = plot_df["weight_class"].value_counts().head(11).index.tolist()
    plot_df["weight_class_plot"] = np.where(plot_df["weight_class"].isin(top_classes), plot_df["weight_class"], "Other")

    red_win_rate = (
        plot_df.groupby("weight_class_plot", dropna=False)["y_red_win"].mean().sort_values(ascending=False)
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(x=red_win_rate.index, y=red_win_rate.values, palette="mako")
    plt.title("RED Win Rate by Weight Class")
    plt.xlabel("weight_class")
    plt.ylabel("RED win rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_3.png", dpi=150)
    plt.close()

    scatter_base = df[["reach_diff", "prior_win_rate_diff", "y_red_win"]].dropna()
    if scatter_base.empty:
        sample_df = df[["reach_diff", "prior_win_rate_diff", "y_red_win"]].fillna(0)
    else:
        sample_df = scatter_base.sample(n=min(3000, len(scatter_base)), random_state=42)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=sample_df,
        x="reach_diff",
        y="prior_win_rate_diff",
        hue="y_red_win",
        alpha=0.7,
        palette="Set1",
    )
    plt.title("Reach Diff vs Prior Win Rate Diff")
    plt.xlabel("reach_diff")
    plt.ylabel("prior_win_rate_diff")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_4.png", dpi=150)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["y_red_win"], errors="ignore")
    corr = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_directories()
    df = load_data()
    print_basic_stats(df)
    save_target_distribution(df)
    save_eda_plots(df)
    save_correlation_heatmap(df)
    print("EDA figures saved to outputs/figures")


if __name__ == "__main__":
    main()
