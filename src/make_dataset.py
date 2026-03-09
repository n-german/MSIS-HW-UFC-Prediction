import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import PROCESSED_DIR, RAW_DIR, ensure_directories

RAW_FILES = [
    "ufc-events.csv",
    "fighter_attributes.csv",
    "fighter_history.csv",
    "fighter_stats.csv",
]


def find_archive_path() -> Path:
    candidates = [
        Path.cwd() / "archive.zip",
        Path(__file__).resolve().parent.parent / "archive.zip",
        Path.home() / "Downloads" / "archive.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("archive.zip not found in project root or ~/Downloads")


def extract_raw_files_if_needed() -> None:
    missing = [name for name in RAW_FILES if not (RAW_DIR / name).exists()]
    if not missing:
        return

    archive_path = find_archive_path()
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = {Path(name).name: name for name in zf.namelist()}
        for filename in missing:
            if filename not in members:
                raise FileNotFoundError(f"{filename} not found inside {archive_path}")
            source = members[filename]
            with zf.open(source) as src, open(RAW_DIR / filename, "wb") as dst:
                dst.write(src.read())


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_history_priors(history_df: pd.DataFrame) -> pd.DataFrame:
    history = history_df.copy()
    history["fight_id_prefix"] = history["fight_id"].astype(str).str.split("-").str[0]
    history["fight_id_prefix"] = pd.to_numeric(history["fight_id_prefix"], errors="coerce").astype("Int64")
    history["event_id"] = pd.to_numeric(history["event_id"], errors="coerce").astype("Int64")
    history["fighter_id"] = pd.to_numeric(history["fighter_id"], errors="coerce").astype("Int64")
    history["opponent_id"] = pd.to_numeric(history["opponent_id"], errors="coerce").astype("Int64")
    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
    history["fight_duration"] = _coerce_numeric(history["fight_duration"])

    history = history.dropna(
        subset=["fight_id_prefix", "event_id", "fighter_id", "opponent_id", "event_date"]
    ).copy()
    history = history.sort_values(["fighter_id", "event_date", "fight_id_prefix"]).reset_index(drop=True)

    history["is_win"] = (history["fight_result"].astype(str).str.upper() == "W").astype(int)
    finish_pattern = r"KO|TKO|SUB|SUBMISSION|DQ"
    history["is_finish_win"] = (
        (history["is_win"] == 1)
        & history["fight_result_type"].astype(str).str.contains(finish_pattern, case=False, na=False)
    ).astype(int)

    grouped = history.groupby("fighter_id")
    history["prior_fights"] = grouped.cumcount().astype(float)

    cumulative_wins = grouped["is_win"].cumsum()
    cumulative_finish_wins = grouped["is_finish_win"].cumsum()
    cumulative_duration = grouped["fight_duration"].cumsum()

    history["prior_wins"] = (cumulative_wins - history["is_win"]).astype(float)
    history["prior_finish_wins"] = (cumulative_finish_wins - history["is_finish_win"]).astype(float)

    duration_before = cumulative_duration - history["fight_duration"].fillna(0.0)
    history["prior_avg_duration"] = duration_before / history["prior_fights"].replace(0, np.nan)

    history["prior_win_rate"] = history["prior_wins"] / history["prior_fights"].replace(0, np.nan)
    history["prior_finish_rate"] = history["prior_finish_wins"] / history["prior_wins"].replace(0, np.nan)

    history["days_since_last_fight"] = grouped["event_date"].diff().dt.days.astype(float)

    history["prior_win_rate"] = history["prior_win_rate"].fillna(0.0)
    history["prior_finish_rate"] = history["prior_finish_rate"].fillna(0.0)

    keep_cols = [
        "event_id",
        "fight_id_prefix",
        "fighter_id",
        "opponent_id",
        "prior_fights",
        "prior_wins",
        "prior_win_rate",
        "prior_finish_wins",
        "prior_finish_rate",
        "prior_avg_duration",
        "days_since_last_fight",
    ]
    return history[keep_cols].copy()


def validate_no_leakage(feature_cols: list[str], fighter_stats_columns: set[str]) -> None:
    forbidden = [
        "winner",
        "outcome",
        "end_round",
        "end_clock",
        "fight_result",
        "over_2_5",
        "judge",
        "score",
        "finish",
        "completed",
        "stats",
    ]
    allow_finish_columns = {
        "red_prior_finish_wins",
        "blue_prior_finish_wins",
        "red_prior_finish_rate",
        "blue_prior_finish_rate",
        "prior_finish_rate_diff",
    }

    violations = []
    for col in feature_cols:
        lc = col.lower()
        if col in fighter_stats_columns:
            violations.append(col)
            continue
        if lc in allow_finish_columns:
            continue
        if any(token in lc for token in forbidden):
            violations.append(col)

    if violations:
        raise ValueError(
            "Leakage guard failed. Forbidden feature columns detected: " + ", ".join(sorted(set(violations)))
        )


def build_model_table() -> pd.DataFrame:
    events = pd.read_csv(RAW_DIR / "ufc-events.csv")
    attrs = pd.read_csv(RAW_DIR / "fighter_attributes.csv")
    history = pd.read_csv(RAW_DIR / "fighter_history.csv")
    fighter_stats_cols = set(pd.read_csv(RAW_DIR / "fighter_stats.csv", nrows=0).columns)

    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events["red_corner_id"] = pd.to_numeric(events["red_corner_id"], errors="coerce").astype("Int64")
    events["blue_corner_id"] = pd.to_numeric(events["blue_corner_id"], errors="coerce").astype("Int64")
    events["fight_id"] = pd.to_numeric(events["fight_id"], errors="coerce").astype("Int64")

    base = events.loc[
        (events["fight_completed"] == 1)
        & (events["fight_winner"].isin(["RED", "BLUE"]))
        & events["red_corner_id"].notna()
        & events["blue_corner_id"].notna()
    ].copy()

    base["y_red_win"] = (base["fight_winner"] == "RED").astype(int)

    attrs2 = attrs.copy()
    attrs2["fighter_id"] = pd.to_numeric(attrs2["fighter_id"], errors="coerce").astype("Int64")
    attrs2["dob"] = pd.to_datetime(attrs2["dob"], errors="coerce")
    attrs2["height"] = _coerce_numeric(attrs2["height"])
    attrs2["reach"] = _coerce_numeric(attrs2["reach"])

    attrs_keep = attrs2[["fighter_id", "dob", "height", "reach", "stance"]].copy()

    red_attrs = attrs_keep.rename(
        columns={
            "fighter_id": "red_corner_id",
            "dob": "red_dob",
            "height": "red_height",
            "reach": "red_reach",
            "stance": "red_stance",
        }
    )
    blue_attrs = attrs_keep.rename(
        columns={
            "fighter_id": "blue_corner_id",
            "dob": "blue_dob",
            "height": "blue_height",
            "reach": "blue_reach",
            "stance": "blue_stance",
        }
    )

    model_df = base.merge(red_attrs, on="red_corner_id", how="left").merge(blue_attrs, on="blue_corner_id", how="left")

    model_df["red_age"] = (model_df["event_date"] - model_df["red_dob"]).dt.days / 365.25
    model_df["blue_age"] = (model_df["event_date"] - model_df["blue_dob"]).dt.days / 365.25

    prior_stats = prepare_history_priors(history)

    red_prior = prior_stats.rename(
        columns={
            "event_id": "event_id",
            "fighter_id": "red_corner_id",
            "opponent_id": "blue_corner_id",
            "prior_fights": "red_prior_fights",
            "prior_wins": "red_prior_wins",
            "prior_win_rate": "red_prior_win_rate",
            "prior_finish_wins": "red_prior_finish_wins",
            "prior_finish_rate": "red_prior_finish_rate",
            "prior_avg_duration": "red_prior_avg_duration",
            "days_since_last_fight": "red_days_since_last_fight",
        }
    )

    blue_prior = prior_stats.rename(
        columns={
            "event_id": "event_id",
            "fighter_id": "blue_corner_id",
            "opponent_id": "red_corner_id",
            "prior_fights": "blue_prior_fights",
            "prior_wins": "blue_prior_wins",
            "prior_win_rate": "blue_prior_win_rate",
            "prior_finish_wins": "blue_prior_finish_wins",
            "prior_finish_rate": "blue_prior_finish_rate",
            "prior_avg_duration": "blue_prior_avg_duration",
            "days_since_last_fight": "blue_days_since_last_fight",
        }
    )

    model_df["event_id"] = pd.to_numeric(model_df["event_id"], errors="coerce").astype("Int64")
    model_df = model_df.merge(red_prior, on=["event_id", "red_corner_id", "blue_corner_id"], how="left")
    model_df = model_df.merge(blue_prior, on=["event_id", "blue_corner_id", "red_corner_id"], how="left")

    model_df["age_diff"] = model_df["red_age"] - model_df["blue_age"]
    model_df["height_diff"] = model_df["red_height"] - model_df["blue_height"]
    model_df["reach_diff"] = model_df["red_reach"] - model_df["blue_reach"]
    model_df["prior_fights_diff"] = model_df["red_prior_fights"] - model_df["blue_prior_fights"]
    model_df["prior_win_rate_diff"] = model_df["red_prior_win_rate"] - model_df["blue_prior_win_rate"]
    model_df["prior_finish_rate_diff"] = model_df["red_prior_finish_rate"] - model_df["blue_prior_finish_rate"]
    model_df["prior_avg_duration_diff"] = model_df["red_prior_avg_duration"] - model_df["blue_prior_avg_duration"]
    model_df["days_since_last_fight_diff"] = (
        model_df["red_days_since_last_fight"] - model_df["blue_days_since_last_fight"]
    )

    model_df["title_fight"] = pd.to_numeric(model_df["title_fight"], errors="coerce")
    model_df["card_position"] = pd.to_numeric(model_df["card_position"], errors="coerce")

    for cat_col in ["weight_class", "event_venue_country", "card_section"]:
        counts = model_df[cat_col].value_counts(dropna=False)
        rare_values = counts[counts < 50].index
        model_df.loc[model_df[cat_col].isin(rare_values), cat_col] = "Other"

    model_df["stance_matchup"] = (
        model_df["red_stance"].fillna("Unknown").astype(str)
        + "_vs_"
        + model_df["blue_stance"].fillna("Unknown").astype(str)
    )

    feature_cols = [
        "weight_class",
        "title_fight",
        "card_section",
        "card_position",
        "event_venue_country",
        "red_age",
        "blue_age",
        "red_height",
        "blue_height",
        "red_reach",
        "blue_reach",
        "red_stance",
        "blue_stance",
        "red_prior_fights",
        "blue_prior_fights",
        "red_prior_wins",
        "blue_prior_wins",
        "red_prior_win_rate",
        "blue_prior_win_rate",
        "red_prior_finish_wins",
        "blue_prior_finish_wins",
        "red_prior_finish_rate",
        "blue_prior_finish_rate",
        "red_prior_avg_duration",
        "blue_prior_avg_duration",
        "red_days_since_last_fight",
        "blue_days_since_last_fight",
        "age_diff",
        "height_diff",
        "reach_diff",
        "prior_fights_diff",
        "prior_win_rate_diff",
        "prior_finish_rate_diff",
        "prior_avg_duration_diff",
        "days_since_last_fight_diff",
        "stance_matchup",
    ]

    validate_no_leakage(feature_cols, fighter_stats_cols)

    dataset = model_df[feature_cols + ["y_red_win"]].copy()

    return dataset


def save_feature_ranges(df: pd.DataFrame) -> None:
    feature_ranges: dict[str, dict | list] = {}
    feature_df = df.drop(columns=["y_red_win"])

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    for col in numeric_cols:
        series = pd.to_numeric(feature_df[col], errors="coerce")
        feature_ranges[col] = {
            "min": float(np.nanmin(series)) if not series.isna().all() else None,
            "p01": float(np.nanpercentile(series, 1)) if not series.isna().all() else None,
            "median": float(np.nanmedian(series)) if not series.isna().all() else None,
            "p99": float(np.nanpercentile(series, 99)) if not series.isna().all() else None,
            "max": float(np.nanmax(series)) if not series.isna().all() else None,
        }

    for col in categorical_cols:
        values = sorted(feature_df[col].dropna().astype(str).unique().tolist())
        feature_ranges[col] = values

    with open(PROCESSED_DIR / "feature_ranges.json", "w", encoding="utf-8") as f:
        json.dump(feature_ranges, f, indent=2)


def main() -> None:
    ensure_directories()
    extract_raw_files_if_needed()
    model_df = build_model_table()
    model_df.to_csv(PROCESSED_DIR / "ufc_model_table.csv", index=False)
    save_feature_ranges(model_df)
    print(f"Saved modeling table: {PROCESSED_DIR / 'ufc_model_table.csv'}")
    print(f"Rows: {len(model_df):,}, Features: {model_df.shape[1] - 1}")


if __name__ == "__main__":
    main()
