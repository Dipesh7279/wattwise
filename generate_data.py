"""Generates simulated hourly electricity data for 3 campus buildings.
Run:  python generate_data.py   ->  creates data/campus_energy.csv
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

BUILDINGS = {
    # name: (students, waste_level)  waste_level = chance/day of night-time AC & light waste
    "Hostel A": (120, 0.25),
    "Hostel B": (150, 0.55),
    "Library Block": (60, 0.10),
}
RATING_KW = {"ac": 1.5, "fan": 0.07, "lights": 0.04, "geyser": 2.0, "fridge": 0.15}


def simulate(name, students, waste_level):
    idx = pd.date_range("2026-01-01", "2026-06-30 23:00", freq="h")
    df = pd.DataFrame({"timestamp": idx})
    h, m = df.timestamp.dt.hour, df.timestamp.dt.month
    dow = df.timestamp.dt.dayofweek

    # Outdoor temperature (Nagpur-like: hot in Apr-Jun)
    base = 24 + (m - 1) * 3.0
    df["temp"] = base + 6 * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, 1, len(df))

    occ = np.where((h >= 18) | (h <= 8), 0.85, 0.35)       # hostels fill at night
    if name == "Library Block":
        occ = np.where((h >= 9) & (h <= 20), 0.8, 0.05)
    occ = occ * (1 - 0.15 * (dow >= 5))
    n = students * occ

    hot = np.clip((df.temp - 26) / 12, 0, 1)
    df["ac_kwh"] = RATING_KW["ac"] * (n / 4) * hot * rng.uniform(0.7, 1.0, len(df))
    df["fan_kwh"] = RATING_KW["fan"] * n * rng.uniform(0.8, 1.0, len(df))
    dark = ((h >= 18) | (h <= 6)).astype(float)
    df["lights_kwh"] = RATING_KW["lights"] * n * dark
    geyser_on = ((h >= 5) & (h <= 8)).astype(float) * (df.temp < 30)
    df["geyser_kwh"] = RATING_KW["geyser"] * (students / 15) * geyser_on * rng.uniform(0.6, 1.0, len(df))
    df["fridge_kwh"] = RATING_KW["fridge"] * (students / 20) * rng.uniform(0.9, 1.0, len(df))

    # Inject waste: AC + lights left on in empty rooms at night
    days = df.timestamp.dt.normalize().unique()
    for d in days:
        if rng.random() < waste_level:
            mask = (df.timestamp >= d + pd.Timedelta(hours=1)) & (df.timestamp <= d + pd.Timedelta(hours=5))
            df.loc[mask, "ac_kwh"] += rng.uniform(8, 14) * (students / 120)
            df.loc[mask, "lights_kwh"] += rng.uniform(2, 4)

    cols = ["ac_kwh", "fan_kwh", "lights_kwh", "geyser_kwh", "fridge_kwh"]
    df["total_kwh"] = df[cols].sum(axis=1)
    df[cols + ["total_kwh", "temp"]] = df[cols + ["total_kwh", "temp"]].round(3)
    df.insert(1, "building", name)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    out = pd.concat([simulate(n, *v) for n, v in BUILDINGS.items()], ignore_index=True)
    out.to_csv("data/campus_energy.csv", index=False)
    print("Saved data/campus_energy.csv", out.shape)
