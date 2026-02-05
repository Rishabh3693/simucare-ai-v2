import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000"
USER_ID = "779b2e98-d061-4748-bcef-78b1c43570ba"
DATA_PATH = "data/processed/athlete_day_features.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["day"], dayfirst=True)

for day in df["day"].dt.strftime("%Y-%m-%d"):
    print(f"Running inference for {day}")

    response = requests.get(
        f"{API_URL}/user/{USER_ID}/daily-insight",
        params={"day": day}
    )

    if response.status_code != 200:
        print(f"❌ Failed for {day}: {response.text}")
    else:
        print(f"✅ Done for {day}")
