import pandas as pd
import os

folder_path = "C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Oura\\"   # folder containing CSVs

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        print(f"\n--- {file} ---")
        df = pd.read_csv(os.path.join(folder_path, file))
        print(df.columns.tolist())

for col in df.columns:
    print(col, "->", df[col].dtype)

# activity data datetime parsing
df_activity = pd.read_csv("C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Oura\\oura_activity_rows.csv")
df_activity["day"] = pd.to_datetime(df_activity["day"], dayfirst=True)
df_activity["created_at"] = pd.to_datetime(df_activity["created_at"], dayfirst=True)
df_activity["updated_at"] = pd.to_datetime(df_activity["updated_at"], dayfirst=True)
numeric_cols = ["steps", "calories", "active_calories", "active_time"]
df_activity[numeric_cols] = df_activity[numeric_cols].apply(
    pd.to_numeric, errors="coerce"
)


# heart rate data datetime parsing
df_hr = pd.read_csv("C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Oura\\oura_heartrate_rows.csv")
df_hr["day"] = pd.to_datetime(df_hr["day"], dayfirst=True)
df_hr["created_at"] = pd.to_datetime(df_hr["created_at"], dayfirst=True)
df_hr["updated_at"] = pd.to_datetime(df_hr["updated_at"], dayfirst=True)
print(df_hr.dtypes)
print(df_hr.head())


# readiness data datetime parsing
df_readiness = pd.read_csv("C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Oura\\oura_readiness_rows.csv")
df_readiness["day"] = pd.to_datetime(df_readiness["day"], dayfirst=True)
df_readiness["created_at"] = pd.to_datetime(df_readiness["created_at"], dayfirst=True)
df_readiness["updated_at"] = pd.to_datetime(df_readiness["updated_at"], dayfirst=True)
print(df_readiness.dtypes)
print(df_readiness.head())

# sleep data datetime parsing
df_sleep = pd.read_csv("C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Oura\\oura_sleep_rows.csv")
df_sleep["start_datetime"] = pd.to_datetime(df_sleep["start_datetime"], dayfirst=True)
df_sleep["end_datetime"] = pd.to_datetime(df_sleep["end_datetime"], dayfirst=True)
df_sleep["created_at"] = pd.to_datetime(df_sleep["created_at"], dayfirst=True)
df_sleep["updated_at"] = pd.to_datetime(df_sleep["updated_at"], dayfirst=True)
print(df_sleep.dtypes)
print(df_sleep.head())


print(df_readiness["contributors"].head())# contributors column parsing


import json
df_readiness["contributors_parsed"] = df_readiness["contributors"].apply(json.loads)
print(type(df_readiness.loc[0, "contributors_parsed"]))
print(df_readiness.loc[0, "contributors_parsed"])

contributors_df = pd.json_normalize(df_readiness["contributors_parsed"])
df_readiness = pd.concat([df_readiness, contributors_df], axis=1)


print(contributors_df.dtypes)

# Assign sleep to wake-up date
df_sleep["day"] = df_sleep["end_datetime"].dt.normalize()
print(df_sleep[["start_datetime", "end_datetime", "duration", "score", "day"]].head())


df_sleep_daily = (
    df_sleep
    .groupby("day")
    .agg(
        total_sleep_duration=("duration", "sum"),
        avg_sleep_score=("score", "mean"),
        sleep_start=("start_datetime", "min"),
        sleep_end=("end_datetime", "max")
    )
    .reset_index()
)
df_sleep_daily["total_sleep_hours"] = df_sleep_daily["total_sleep_duration"] / 3600

print(df_sleep_daily.head())
print(df_sleep_daily.dtypes)

# Remove timezone info
df_sleep_daily["day"] = df_sleep_daily["day"].dt.tz_convert(None)
df_activity["day"] = df_activity["day"].dt.tz_localize(None)
df_hr["day"] = df_hr["day"].dt.tz_localize(None)
df_readiness["day"] = df_readiness["day"].dt.tz_localize(None)
print(df_sleep_daily.head())
print(df_sleep_daily.dtypes)

df_merged = pd.merge(
    df_activity,
    df_hr,
    on=["day", "user_id"],
    how="left"
)
df_merged = pd.merge(
    df_merged,
    df_readiness,
    on=["day", "user_id"],
    how="left",
    suffixes=("", "_readiness")
)
df_merged = pd.merge(
    df_merged,
    df_sleep_daily,
    on="day",
    how="left"
)
print(df_merged.shape)
print(df_merged.columns)
print(df_merged.head())

print(df_merged.isna().sum().sort_values(ascending=False))
df_merged = df_merged.sort_values("day").reset_index(drop=True)
df_merged.to_csv("athlete_day_table.csv", index=False)
# The final merged DataFrame is saved as 'athlete_day_table.csv'

# Load Strava data
df_strava = pd.read_csv("C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Strava\\strava_activities_rows.csv")  # change filename if needed

# Parse timestamps
df_strava["start_date"] = pd.to_datetime(df_strava["start_date"], utc=True)
df_strava["created_at"] = pd.to_datetime(df_strava["created_at"], utc=True)
df_strava["updated_at"] = pd.to_datetime(df_strava["updated_at"], utc=True)

print(df_strava.dtypes)

# Assign calendar date (normalize to midnight)
df_strava["day"] = df_strava["start_date"].dt.normalize()
df_strava_daily = (
    df_strava
    .groupby(["day", "user_id"])
    .agg(
        session_count=("id", "count"),
        total_moving_time=("moving_time", "sum"),
        total_distance=("distance", "sum"),
        total_elevation_gain=("total_elevation_gain", "sum"),
        avg_session_hr=("average_heartrate", "mean"),
        max_session_hr=("max_heartrate", "max"),
        avg_power=("average_watts", "mean"),
        total_kilojoules=("kilojoules", "sum"),
        total_suffer_score=("suffer_score", "sum")
    )
    .reset_index()
)

# Convert seconds → hours
df_strava_daily["training_hours"] = df_strava_daily["total_moving_time"] / 3600
print(df_strava_daily.head())
print(df_strava_daily.dtypes)
df_strava_daily["day"] = df_strava_daily["day"].dt.tz_convert(None)
df_merged["day"] = df_merged["day"].dt.tz_localize(None)
df_merged = pd.merge(
    df_merged,
    df_strava_daily,
    on=["day", "user_id"],
    how="left"
)
print(df_merged.shape)
print(df_merged[[
    "day",
    "session_count",
    "training_hours",
    "total_distance",
    "avg_session_hr",
    "total_suffer_score"
]].head())

print(df_merged.isna().sum().sort_values(ascending=False))
df_merged.to_csv("athlete_day_table_v2.csv", index=False)
# The final merged DataFrame with Strava data is saved as 'athlete_day_table_v2.csv'


df_merged["acute_training_hours_7d"] = (
    df_merged["training_hours"]
    .rolling(window=7, min_periods=3)
    .sum()
)
df_merged["chronic_training_hours_28d"] = (
    df_merged["training_hours"]
    .rolling(window=28, min_periods=10)
    .mean()
)
df_merged["acr_training_load"] = (
    df_merged["acute_training_hours_7d"] /
    df_merged["chronic_training_hours_28d"]
)
# < 0.8 → detraining
# 0.8–1.3 → optimal
# > 1.5 → injury / overreach risk


df_merged["sleep_baseline_7d"] = (
    df_merged["total_sleep_hours"]
    .rolling(window=7, min_periods=3)
    .mean()
)
df_merged["sleep_debt"] = (
    df_merged["total_sleep_hours"] -
    df_merged["sleep_baseline_7d"]
)
# sleep_debt < -1.0 hours → significant sleep debt

df_merged["hrv_baseline_14d"] = (
    df_merged["hrv_balance"]
    .rolling(window=14, min_periods=5)
    .mean()
)
df_merged["hrv_deviation"] = (
    df_merged["hrv_balance"] -
    df_merged["hrv_baseline_14d"]
)
# negative hrv_deviation may indicate fatigue or illness


df_merged["rhr_baseline_14d"] = (
    df_merged["resting_heart_rate"]
    .rolling(window=14, min_periods=5)
    .mean()
)

df_merged["rhr_deviation"] = (
    df_merged["resting_heart_rate"] -
    df_merged["rhr_baseline_14d"]
)
# elevated rhr_deviation (> +5 bpm) may indicate fatigue or illness

df_merged["load_recovery_stress"] = (
    df_merged["acute_training_hours_7d"] *
    (-df_merged["hrv_deviation"])
)
# higher load_recovery_stress may indicate increased fatigue risk

df_merged["readiness_prev_day"] = df_merged["score"].shift(1)
df_merged["training_hours_prev_day"] = df_merged["training_hours"].shift(1)
df_merged["sleep_hours_prev_day"] = df_merged["total_sleep_hours"].shift(1)
# These lagged features can help model today's readiness based on yesterday's metrics

feature_cols = [
    "acute_training_hours_7d",
    "chronic_training_hours_28d",
    "acr_training_load",
    "sleep_debt",
    "hrv_deviation",
    "rhr_deviation",
    "load_recovery_stress"
]

print(df_merged[feature_cols].describe())
print(df_merged[feature_cols].isna().sum())
# The engineered features are now part of df_merged and can be used for analysis or modeling




# Defensive lowercase for text processing
df_strava["name_clean"] = df_strava["name"].str.lower().fillna("")

# Time-of-day intent
df_strava["is_morning"] = df_strava["name_clean"].str.contains("morning")
df_strava["is_afternoon"] = df_strava["name_clean"].str.contains("afternoon")
df_strava["is_evening"] = df_strava["name_clean"].str.contains("evening|night")

# Session intensity intent
df_strava["is_easy"] = df_strava["name_clean"].str.contains("easy|recovery")
df_strava["is_hard"] = df_strava["name_clean"].str.contains(
    "interval|tempo|threshold|hard|race|vo2"
)

# Session length / purpose
df_strava["is_long"] = df_strava["name_clean"].str.contains("long")
df_strava["is_strength"] = df_strava["name_clean"].str.contains(
    "gym|strength|weights|weight"
)
# These boolean flags can help categorize workouts for further analysis

df_strava["day"] = pd.to_datetime(df_strava["day"]).dt.normalize()

df_strava_name_daily = (
    df_strava
    .assign(
        morning_sessions=df_strava["hour"].between(5, 11),
        afternoon_sessions=df_strava["hour"].between(12, 16),
        evening_sessions=df_strava["hour"].between(17, 23),
        hard_sessions=df_strava["is_hard"],
        easy_sessions=df_strava["is_easy"],
        long_sessions=df_strava["is_long"],
        strength_sessions=df_strava["is_strength"]
    )
    .groupby(["user_id", "day"])
    .sum()
    .reset_index()
)
# The categorized session counts can be merged back into df_merged if needed

print(df_strava.columns)

df_strava_name_daily["day"] = df_strava["day"].dt.tz_localize(None)
df_merged = pd.merge(
    df_merged,
    df_strava_name_daily,
    on=["day", "user_id"],
    how="left"
)


session_count_cols = [
    "morning_sessions",
    "afternoon_sessions",
    "evening_sessions",
    "easy_sessions",
    "hard_sessions",
    "long_sessions",
    "strength_sessions",
]

df_merged[session_count_cols] = df_merged[session_count_cols].fillna(0)

# Example usage of the categorized session counts


df_merged["hard_day"] = (df_merged["hard_sessions"] > 0).astype(int)
# A "hard day" is defined as having at least one hard session
df_merged["multi_session_day"] = (df_merged["session_count"] > 1).astype(int)
# A "multi-session day" has more than one workout session
df_merged["late_training_day"] = (df_merged["evening_sessions"] > 0).astype(int)
# A "late training day" includes evening or night workouts

df_merged["hard_day"] = (df_merged["hard_sessions"] > 0).astype(int)
df_merged["multi_session_day"] = (df_merged["session_count"] > 1).astype(int)
df_merged["late_training_day"] = (df_merged["evening_sessions"] > 0).astype(int)
df_merged["hard_day"] = df_merged["hard_sessions"] > 0
df_merged = df_merged.sort_values(["user_id", "day"])

# Training load
df_merged["acute_training_hours_7d"] = (
    df_merged.groupby("user_id")["training_hours"]
      .rolling(7, min_periods=3)
      .sum()
      .reset_index(level=0, drop=True)
)

df_merged["chronic_training_hours_28d"] = (
    df_merged.groupby("user_id")["training_hours"]
      .rolling(28, min_periods=10)
      .mean()
      .reset_index(level=0, drop=True)
)

df_merged["acr_training_load"] = (
    df_merged["acute_training_hours_7d"] /
    df_merged["chronic_training_hours_28d"]
)
df_merged["hrv_baseline_14d"] = (
    df_merged.groupby("user_id")["hrv_balance"]
      .rolling(14, min_periods=7)
      .mean()
      .reset_index(level=0, drop=True)
)

df_merged["hrv_deviation"] = df_merged["hrv_balance"] - df_merged["hrv_baseline_14d"]

df_merged = df_merged.sort_values(["user_id", "day"])

# ---------------------------
# Training load trends
# ---------------------------
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

# ---------------------------
# HRV & RHR baselines
# ---------------------------
df_merged["hrv_baseline_14d"] = (
    df_merged.groupby("user_id")["hrv_balance"]
      .rolling(14, min_periods=5)
      .mean()
      .reset_index(level=0, drop=True)
)

df_merged["hrv_deviation"] = df_merged["hrv_balance"] - df_merged["hrv_baseline_14d"]

df_merged["rhr_baseline_14d"] = (
    df_merged.groupby("user_id")["resting_heart_rate"]
      .rolling(14, min_periods=7)
      .mean()
      .reset_index(level=0, drop=True)
)

df_merged["rhr_deviation"] = df_merged["resting_heart_rate"] - df_merged["rhr_baseline_14d"]

# ---------------------------
# Sleep baselines
# ---------------------------
df_merged["sleep_baseline_7d"] = (
    df_merged.groupby("user_id")["total_sleep_hours"]
      .rolling(7, min_periods=2)
      .mean()
      .reset_index(level=0, drop=True)
)

df_merged["sleep_debt"] = df_merged["sleep_baseline_7d"] - df_merged["total_sleep_hours"]



name_feature_cols = [
    "morning_sessions",
    "afternoon_sessions",
    "evening_sessions",
    "easy_sessions",
    "hard_sessions",
    "long_sessions",
    "strength_sessions",
    "hard_day",
    "multi_session_day",
    "late_training_day",
]

print(df_merged[name_feature_cols].describe())
print(df_merged[name_feature_cols].isna().sum())
# The new name-based features are now part of df_merged for further analysis

df_merged.to_csv("data/processed/athlete_day_features.csv", index=False)
print(df_merged[[
    "day",
    "training_hours",
    "acute_training_hours_7d",
    "chronic_training_hours_28d",
    "acr_training_load",
    "hard_day",
    "multi_session_day"
]].tail(10))
