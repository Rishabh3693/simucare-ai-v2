import pandas as pd

DATA_PATH = "data/processed/athlete_day_features.csv"

def load_athlete_day(user_id: str, day: str) -> dict:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["day"],
        dayfirst=True
    )

    # Normalize both sides to date-only
    df["day"] = df["day"].dt.normalize()
    day = pd.to_datetime(day, dayfirst=True).normalize()

    row = df[
        (df["user_id"] == user_id) &
        (df["day"] == day)
    ]

    if row.empty:
        raise ValueError(
            f"No data found for user_id={user_id}, day={day.date()}"
        )
    
    return row.iloc[0].to_dict()


def load_athlete_window(user_id: str, start_day: str, days: int):
    df = pd.read_csv(DATA_PATH, parse_dates=["day"], dayfirst=True)
    df["day"] = df["day"].dt.normalize()

    start = pd.to_datetime(start_day, dayfirst=True).normalize()
    end = start + pd.Timedelta(days=days)

    user_df = df[
        (df["user_id"] == user_id) &
        (df["day"] >= start) &
        (df["day"] < end)
    ].sort_values("day")

    return user_df.reset_index(drop=True)

#uvicorn api.main:app --reload
