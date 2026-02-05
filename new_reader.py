import pandas as pd
import os
import json

# ==============================
# PATHS
# ==============================
OURA_PATH = "C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Oura\\"
STRAVA_PATH = "C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Strava\\"
OUTPUT_PATH = "data/processed/athlete_day_features.csv"

# ==============================
# LOAD OURA DATA
# ==============================
df_activity = pd.read_csv(f"{OURA_PATH}oura_activity_rows.csv")
df_hr = pd.read_csv(f"{OURA_PATH}oura_heartrate_rows.csv")
df_readiness = pd.read_csv(f"{OURA_PATH}oura_readiness_rows.csv")
df_sleep = pd.read_csv(f"{OURA_PATH}oura_sleep_rows.csv")

# ------------------------------
# Datetime parsing
# ------------------------------
for df, cols in [
    (df_activity, ["day", "created_at", "updated_at"]),
    (df_hr, ["day", "created_at", "updated_at"]),
    (df_readiness, ["day", "created_at", "updated_at"]),
]:
    for c in cols:
        df[c] = pd.to_datetime(df[c], dayfirst=True)

df_sleep["start_datetime"] = pd.to_datetime(df_sleep["start_datetime"])
df_sleep["end_datetime"] = pd.to_datetime(df_sleep["end_datetime"])
df_sleep["created_at"] = pd.to_datetime(df_sleep["created_at"])
df_sleep["updated_at"] = pd.to_datetime(df_sleep["updated_at"])

# ------------------------------
# Readiness contributors parsing
# ------------------------------
df_readiness["contributors_parsed"] = df_readiness["contributors"].apply(json.loads)
contributors_df = pd.json_normalize(df_readiness["contributors_parsed"])
df_readiness = pd.concat([df_readiness, contributors_df], axis=1)

# ------------------------------
# Sleep aggregation (wake-up day)
# ------------------------------
df_sleep["day"] = df_sleep["end_datetime"].dt.normalize()

df_sleep_daily = (
    df_sleep
    .groupby("day")
    .agg(
        total_sleep_duration=("duration", "sum"),
        avg_sleep_score=("score", "mean"),
        sleep_start=("start_datetime", "min"),
        sleep_end=("end_datetime", "max"),
    )
    .reset_index()
)

df_sleep_daily["total_sleep_hours"] = df_sleep_daily["total_sleep_duration"] / 3600

# Remove timezone everywhere
for df in [df_activity, df_hr, df_readiness, df_sleep_daily]:
    df["day"] = df["day"].dt.tz_localize(None)

# ==============================
# MERGE OURA → DAY TABLE
# ==============================
df_merged = (
    df_activity
    .merge(df_hr, on=["user_id", "day"], how="left")
    .merge(df_readiness, on=["user_id", "day"], how="left")
    .merge(df_sleep_daily, on="day", how="left")
)

# ==============================
# LOAD STRAVA DATA
# ==============================
df_strava = pd.read_csv(f"{STRAVA_PATH}strava_activities_rows.csv")

df_strava["start_date"] = pd.to_datetime(df_strava["start_date"], utc=True)
df_strava["start_date_local"] = df_strava["start_date"].dt.tz_localize(None)

df_strava["day"] = df_strava["start_date_local"].dt.normalize()
df_strava["hour"] = df_strava["start_date_local"].dt.hour

# ------------------------------
# Session classification
# ------------------------------
df_strava["duration_min"] = df_strava["moving_time"] / 60
df_strava["name_clean"] = df_strava["name"].str.lower().fillna("")

df_strava["morning_sessions"] = df_strava["hour"].between(5, 11)
df_strava["afternoon_sessions"] = df_strava["hour"].between(12, 16)
df_strava["evening_sessions"] = df_strava["hour"].between(17, 23)

df_strava["hard_sessions"] = (
    (df_strava["suffer_score"] >= 150) |
    (df_strava["average_heartrate"] >= 0.85 * df_strava["max_heartrate"])
)

df_strava["easy_sessions"] = (
    (df_strava["suffer_score"] < 80) &
    (df_strava["duration_min"] < 60)
)

df_strava["long_sessions"] = (
    (df_strava["duration_min"] >= 90) |
    (df_strava["distance"] >= 20000)
)

df_strava["strength_sessions"] = df_strava["type"].str.contains(
    "Weight|Strength|Gym|Cross", case=False, na=False
)

# ------------------------------
# Aggregate Strava per day
# ------------------------------
df_strava_daily = (
    df_strava
    .groupby(["user_id", "day"])
    .agg(
        session_count=("id", "count"),
        total_moving_time=("moving_time", "sum"),
        total_distance=("distance", "sum"),
        total_elevation_gain=("total_elevation_gain", "sum"),
        avg_session_hr=("average_heartrate", "mean"),
        max_session_hr=("max_heartrate", "max"),
        avg_power=("average_watts", "mean"),
        total_kilojoules=("kilojoules", "sum"),
        total_suffer_score=("suffer_score", "sum"),
        morning_sessions=("morning_sessions", "sum"),
        afternoon_sessions=("afternoon_sessions", "sum"),
        evening_sessions=("evening_sessions", "sum"),
        easy_sessions=("easy_sessions", "sum"),
        hard_sessions=("hard_sessions", "sum"),
        long_sessions=("long_sessions", "sum"),
        strength_sessions=("strength_sessions", "sum"),
    )
    .reset_index()
)

df_strava_daily["training_hours"] = df_strava_daily["total_moving_time"] / 3600

# ==============================
# MERGE STRAVA → DAY TABLE
# ==============================
df_merged = df_merged.merge(
    df_strava_daily,
    on=["user_id", "day"],
    how="left"
)

# Fill session counts
session_cols = [
    "morning_sessions", "afternoon_sessions", "evening_sessions",
    "easy_sessions", "hard_sessions", "long_sessions", "strength_sessions",
]
df_merged[session_cols] = df_merged[session_cols].fillna(0)

# ------------------------------
# Behavioral flags
# ------------------------------
df_merged["hard_day"] = (df_merged["hard_sessions"] > 0).astype(int)
df_merged["multi_session_day"] = (df_merged["session_count"] > 1).astype(int)
df_merged["late_training_day"] = (df_merged["evening_sessions"] > 0).astype(int)

# ==============================
# ROLLING FEATURES (PER USER)
# ==============================
df_merged = df_merged.sort_values(["user_id", "day"])

df_merged["acute_training_hours_7d"] = (
    df_merged.groupby("user_id")["training_hours"]
    .rolling(7, min_periods=2)
    .sum()
    .reset_index(level=0, drop=True)
)

df_merged["chronic_training_hours_28d"] = (
    df_merged.groupby("user_id")["training_hours"]
    .rolling(28, min_periods=5)
    .mean()
    .reset_index(level=0, drop=True)
)

df_merged["acr_training_load"] = (
    df_merged["acute_training_hours_7d"] /
    df_merged["chronic_training_hours_28d"]
)

df_merged["hrv_baseline_14d"] = (
    df_merged.groupby("user_id")["hrv_balance"]
    .rolling(14, min_periods=5)
    .mean()
    .reset_index(level=0, drop=True)
)

df_merged["hrv_deviation"] = df_merged["hrv_balance"] - df_merged["hrv_baseline_14d"]

df_merged["rhr_baseline_14d"] = (
    df_merged.groupby("user_id")["resting_heart_rate"]
    .rolling(14, min_periods=5)
    .mean()
    .reset_index(level=0, drop=True)
)

df_merged["rhr_deviation"] = (
    df_merged["resting_heart_rate"] - df_merged["rhr_baseline_14d"]
)

df_merged["sleep_baseline_7d"] = (
    df_merged.groupby("user_id")["total_sleep_hours"]
    .rolling(7, min_periods=2)
    .mean()
    .reset_index(level=0, drop=True)
)

df_merged["sleep_debt"] = (
    df_merged["sleep_baseline_7d"] - df_merged["total_sleep_hours"]
)

df_merged["training_hours"] = df_merged["training_hours"].fillna(0)
df_merged["session_count"] = df_merged["session_count"].fillna(0)
df_merged = df_merged.sort_values(["user_id", "day"])

df_merged["acute_training_hours_7d"] = (
    df_merged.groupby("user_id")["training_hours"]
    .rolling(7, min_periods=2)
    .sum()
    .reset_index(level=0, drop=True)
)

df_merged["chronic_training_hours_28d"] = (
    df_merged.groupby("user_id")["training_hours"]
    .rolling(28, min_periods=5)
    .mean()
    .reset_index(level=0, drop=True)
)

df_merged["acr_training_load"] = (
    df_merged["acute_training_hours_7d"] /
    df_merged["chronic_training_hours_28d"]
)

# ==============================
# SAVE FINAL DATASET
# ==============================
os.makedirs("data/processed", exist_ok=True)
df_merged.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print(df_merged[[
    "day", "training_hours", "acr_training_load",
    "hard_day", "multi_session_day"
]].tail(10))


