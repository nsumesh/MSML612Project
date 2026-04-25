import fastf1
import pandas as pd
from pathlib import Path
import datetime 
Path("data/raw").mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache("data/raw")

FEATURE_COLS = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "SpeedST", "TyreLife", "Compound", "Driver", "LapNumber", "PitInTime", "PitOutTime", "TrackStatus", "Time"]

RACES = [
    # 2024 Season
    ("Bahrain", 2024),
    ("Saudi Arabia", 2024),
    ("Australia", 2024),
    ("Japan", 2024),
    ("China", 2024),
    ("Miami", 2024),
    ("Emilia Romagna", 2024),
    ("Monaco", 2024),
    ("Canada", 2024),
    ("Spain", 2024),
    ("Austria", 2024),
    ("Great Britain", 2024),
    ("Hungary", 2024),
    ("Belgium", 2024),
    ("Netherlands", 2024),
    ("Italy", 2024),
    ("Azerbaijan", 2024),
    ("Singapore", 2024),
    ("United States", 2024),
    ("Mexico City", 2024),
    ("São Paulo", 2024),
    ("Las Vegas", 2024),
    ("Qatar", 2024),
    ("Abu Dhabi", 2024),
    # 2025 Season
    ("Australia", 2025),
    ("China", 2025),
    ("Japan", 2025),
    ("Bahrain", 2025),
    ("Saudi Arabia", 2025),
    ("Miami", 2025),
    ("Emilia Romagna", 2025),
    ("Monaco", 2025),
    ("Spain", 2025),
    ("Canada", 2025),
    ("Austria", 2025),
    ("Great Britain", 2025),
    ("Belgium", 2025),
    ("Hungary", 2025),
    ("Netherlands", 2025),
    ("Italy", 2025),
    ("Azerbaijan", 2025),
    ("Singapore", 2025),
    ("United States", 2025),
    ("Mexico City", 2025),
    ("São Paulo", 2025),
    ("Las Vegas", 2025),
    ("Qatar", 2025),
    ("Abu Dhabi", 2025),
]


def fetch_race(year: int, race_name: str):
    session = fastf1.get_session(year, race_name, "R")
    session.load(telemetry=False, weather=True, messages=False)
    laps = session.laps[FEATURE_COLS].copy()
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[col] = laps[col].dt.total_seconds()

    weather = session.weather_data[["Time", "AirTemp", "TrackTemp", "Rainfall"]].copy()
    laps = pd.merge_asof(
        laps.sort_values("Time"),
        weather.sort_values("Time"),
        on="Time",
    )
    laps = laps.drop(columns=["Time"])

    laps = laps[laps["TrackStatus"] == "1"]
    laps["Year"] = year
    laps["Race"] = race_name
    return laps

def fetch_all_races(race_list : list, save_path: str = "data/raw/all_laps.csv"):
    all_data = []
    for race_name, year in race_list:
        try:
            race = fetch_race(year, race_name)
            all_data.append(race)
        except Exception as e:
            print(f"Error while parsing {e}")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    combined_data.to_csv(save_path, index=False)
    return combined_data
