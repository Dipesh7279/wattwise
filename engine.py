"""Core logic: forecasting, anomaly detection, what-if simulation, scoring."""
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest

TARIFF = 8.0          # Rs per kWh (institutional average, change if needed)
CO2_FACTOR = 0.82     # kg CO2 per kWh (India grid average)
APPLIANCES = {"ac": "ac_kwh", "fan": "fan_kwh", "light": "lights_kwh",
              "lights": "lights_kwh", "geyser": "geyser_kwh", "fridge": "fridge_kwh"}
APP_COLS = ["ac_kwh", "fan_kwh", "lights_kwh", "geyser_kwh", "fridge_kwh"]
FEATURES = ["hour", "dow", "month", "is_weekend", "temp"]


def add_time_features(df):
    df = df.copy()
    df["hour"] = df.timestamp.dt.hour
    df["dow"] = df.timestamp.dt.dayofweek
    df["month"] = df.timestamp.dt.month
    df["is_weekend"] = (df.dow >= 5).astype(int)
    return df


def train_model(df):
    """Train on first 80%, report MAPE on last 20%, then refit on all data."""
    df = add_time_features(df)
    split = int(len(df) * 0.8)
    params = dict(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.9)
    m = xgb.XGBRegressor(**params).fit(df[FEATURES][:split], df.total_kwh[:split])
    pred = m.predict(df[FEATURES][split:])
    actual = df.total_kwh[split:].values
    mape = float(np.mean(np.abs(pred - actual) / np.maximum(actual, 1e-6)) * 100)
    model = xgb.XGBRegressor(**params).fit(df[FEATURES], df.total_kwh)
    df["pred"] = model.predict(df[FEATURES])
    return model, df, mape


def forecast_future(model, df, hours=168):
    """Forecast next `hours` using typical temperature for that (month, hour)."""
    typical = df.groupby(["month", "hour"]).temp.mean()
    future = pd.DataFrame({"timestamp": pd.date_range(
        df.timestamp.max() + pd.Timedelta(hours=1), periods=hours, freq="h")})
    future = add_time_features(future)
    last_m = df.month.iloc[-1]   # fall back to the latest month seen if the month is new
    future["temp"] = [typical.get((m, h), typical.get((last_m, h), df.temp.mean()))
                      for m, h in zip(future.month, future.hour)]
    future["pred"] = model.predict(future[FEATURES])
    return future


def detect_anomalies(df):
    """Isolation Forest on forecast residual + hour; keep only over-consumption."""
    d = df.copy()
    d["resid"] = d.total_kwh - d.pred
    iso = IsolationForest(contamination=0.02, random_state=1)
    d["flag"] = iso.fit_predict(d[["resid", "hour"]]) == -1
    d["anomaly"] = d.flag & (d.resid > d.resid.std() * 2)
    return d


def describe_anomalies(d):
    """Group consecutive anomalous hours into readable alerts."""
    a = d[d.anomaly].copy()
    if a.empty:
        return []
    a["day"] = a.timestamp.dt.date
    out = []
    for day, g in a.groupby("day"):
        worst = g[APP_COLS[:3]].sum().idxmax().replace("_kwh", "").upper()
        out.append(f"{day}: {worst} over-use from {g.timestamp.min():%I %p} to "
                   f"{g.timestamp.max():%I %p}, about {g.resid.sum():.0f} kWh above normal")
    return out[-8:][::-1]


def window_mask(hours, start, end):
    """True for hours inside [start, end) and handles overnight windows."""
    return (hours >= start) | (hours < end) if start > end else (hours >= start) & (hours < end)


def simulate_change(df, appliance_col, start, end, cut_pct=100):
    """Monthly savings if `appliance` is cut by cut_pct% between start and end hour."""
    mask = window_mask(df.timestamp.dt.hour, start, end)
    n_days = max(df.timestamp.dt.normalize().nunique(), 1)
    kwh_month = df.loc[mask, appliance_col].sum() / n_days * 30 * cut_pct / 100
    return kwh_month, kwh_month * TARIFF, kwh_month * CO2_FACTOR


def parse_question(text):
    """Very small rule-based parser for questions like:
    'turn off AC after 10 PM', 'reduce lights from 11 pm to 5 am by 50%'."""
    t = text.lower()
    app = next((APPLIANCES[k] for k in APPLIANCES if re.search(rf"\b{k}\b", t)), None)
    if app is None:
        return None
    def to24(h, ap):
        h = int(h) % 12
        return h + 12 if ap == "pm" else h
    start, end, cut = 22, 6, 100
    m = re.search(r"after (\d{1,2})\s*(am|pm)", t)
    if m:
        start = to24(m.group(1), m.group(2))
    m = re.search(r"from (\d{1,2})\s*(am|pm)\s*(?:to|till|until)\s*(\d{1,2})\s*(am|pm)", t)
    if m:
        start, end = to24(m.group(1), m.group(2)), to24(m.group(3), m.group(4))
    m = re.search(r"(\d{1,3})\s*%", t)
    if m:
        cut = min(int(m.group(1)), 100)
    return app, start, end, cut


def action_plan(df):
    """Rank standard energy-saving actions by monthly Rs saved."""
    rules = [
        ("Switch off AC 11 PM to 6 AM in unoccupied areas", "ac_kwh", 23, 6, 60),
        ("Switch off AC 1 AM to 5 AM (late-night waste)", "ac_kwh", 1, 5, 100),
        ("Switch off corridor and room lights 12 AM to 5 AM", "lights_kwh", 0, 5, 70),
        ("Cut fan usage 10 AM to 4 PM when rooms are empty", "fan_kwh", 10, 16, 40),
        ("Use geyser timers, limit to 5 AM to 7 AM", "geyser_kwh", 7, 9, 100),
    ]
    rows = []
    for text, col, s, e, pct in rules:
        kwh, rs, co2 = simulate_change(df, col, s, e, pct)
        rows.append({"Action": text, "kWh saved/month": round(kwh),
                     "Rs saved/month": round(rs), "CO2 avoided (kg)": round(co2)})
    return pd.DataFrame(rows).sort_values("Rs saved/month", ascending=False).reset_index(drop=True)


def green_score(df):
    """0-100. Penalises night-time waste share and over-consumption vs forecast."""
    d = detect_anomalies(add_time_features(df).assign(pred=df["pred"])) if "pred" in df else df
    waste_kwh = d.loc[d.anomaly, "resid"].sum()
    night = d[d.hour.between(1, 5)][["ac_kwh", "lights_kwh"]].sum().sum()
    ratio = (waste_kwh + 0.3 * night) / d.total_kwh.sum()
    return int(np.clip(100 - ratio * 700, 0, 100))
