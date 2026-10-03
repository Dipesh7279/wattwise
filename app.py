import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engine as E

st.set_page_config(page_title="WattWise", page_icon="⚡", layout="wide")
st.title("⚡ WattWise: Energy Twin for Campuses")
st.caption("Find the energy leaks. Fix them before the bill arrives.")


@st.cache_data
def load_default():
    if not os.path.exists("data/campus_energy.csv"):
        import generate_data
        os.makedirs("data", exist_ok=True)
        out = pd.concat([generate_data.simulate(n, *v) for n, v in generate_data.BUILDINGS.items()])
        out.to_csv("data/campus_energy.csv", index=False)
    return pd.read_csv("data/campus_energy.csv", parse_dates=["timestamp"])


@st.cache_data
def prepare(df_building):
    model, d, mape = E.train_model(df_building)
    d = E.detect_anomalies(d)
    fut = E.forecast_future(model, d, 168)
    return d, fut, mape


# ---------- Sidebar: data ----------
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload your CSV (optional)", type="csv")
st.sidebar.caption("Minimum: **timestamp** + **total_kwh** (hourly, 14+ days). "
                   "Optional: temp, ac_kwh, fan_kwh, lights_kwh, geyser_kwh, fridge_kwh, building.")
st.sidebar.download_button("Download sample template", "timestamp,total_kwh\n2026-01-01 00:00:00,12.5\n"
                           "2026-01-01 01:00:00,11.8\n", "template.csv", "text/csv")
try:
    source = pd.read_csv(up) if up else load_default()
    raw, info = E.standardize(source)
except Exception as ex:
    st.error(f"Could not read the file: {ex}")
    st.stop()
if up:
    if not info["has_appliances"]:
        st.warning("Appliance-wise data not found. The breakdown below is an **ESTIMATE** based on typical "
                   "campus usage patterns (simple NILM stand-in). Totals, forecast and waste alerts use your real data.")
    if not info["has_temp"]:
        st.info("No temperature column found. The forecast uses time patterns only.")
    for n in info["notes"]:
        st.info(n)
est = bool(up) and not info["has_appliances"]
building = st.sidebar.selectbox("Building", sorted(raw.building.unique()))
df_b = raw[raw.building == building].reset_index(drop=True)
d, fut, mape = prepare(df_b)

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
monthly_kwh = d.total_kwh.sum() / d.timestamp.dt.normalize().nunique() * 30
c1.metric("Avg monthly usage", f"{monthly_kwh:,.0f} kWh")
c2.metric("Avg monthly bill", f"Rs {monthly_kwh * E.TARIFF:,.0f}")
c3.metric("Forecast accuracy (MAPE)", f"{mape:.1f}%")
c4.metric("Green Score", f"{E.green_score(d)}/100")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Forecast", "Waste alerts", "Appliance breakdown", "What-if simulator", "Action plan & leaderboard"])

# ---------- Tab 1: Forecast ----------
with tab1:
    st.subheader("Actual vs predicted (last 7 days) and next 7 days")
    hist = d.tail(168)
    fig = go.Figure()
    fig.add_scatter(x=hist.timestamp, y=hist.total_kwh, name="Actual", line=dict(color="#1f77b4"))
    fig.add_scatter(x=hist.timestamp, y=hist.pred, name="Model fit", line=dict(color="#ff7f0e", dash="dot"))
    fig.add_scatter(x=fut.timestamp, y=fut.pred, name="Forecast", line=dict(color="#2ca02c"))
    fig.update_layout(yaxis_title="kWh per hour", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig)
    nxt = fut.head(24).pred.sum()
    st.info(f"Expected usage in the next 24 hours: **{nxt:,.0f} kWh** (about Rs {nxt * E.TARIFF:,.0f})")

# ---------- Tab 2: Waste alerts ----------
with tab2:
    st.subheader("Automatic waste detection (Isolation Forest on forecast residuals)")
    alerts = E.describe_anomalies(d, est)
    if alerts:
        for a in alerts:
            st.error("🚨 " + a)
    else:
        st.success("No abnormal usage detected.")
    recent = d.tail(24 * 30)
    fig = px.line(recent, x="timestamp", y="total_kwh", title="Last 30 days, anomalies in red")
    pts = recent[recent.anomaly]
    fig.add_scatter(x=pts.timestamp, y=pts.total_kwh, mode="markers",
                    marker=dict(color="red", size=8), name="Anomaly")
    st.plotly_chart(fig)

# ---------- Tab 3: Breakdown ----------
with tab3:
    st.subheader("Where does the energy go?" + (" (estimated split)" if est else ""))
    share = d[E.APP_COLS].sum().rename(lambda x: x.replace("_kwh", "").title())
    left, right = st.columns(2)
    left.plotly_chart(px.pie(values=share.values, names=share.index, hole=0.45))
    hourly = d.groupby("hour")[E.APP_COLS].mean().reset_index()
    right.plotly_chart(px.area(hourly, x="hour", y=E.APP_COLS, title="Average hourly profile"))

# ---------- Tab 4: What-if ----------
with tab4:
    st.subheader("What-if simulator" + (" (based on estimated appliance split)" if est else ""))
    q = st.text_input("Ask a question",
                      placeholder="e.g. turn off AC after 10 PM   |   reduce lights from 12 am to 5 am by 50%")
    parsed = E.parse_question(q) if q else None
    if q and not parsed:
        st.warning("Couldn't understand. Mention an appliance (AC, fan, lights, geyser, fridge) or use sliders below.")

    st.markdown("**Or use the controls**")
    a, b, c, e = st.columns(4)
    app_name = a.selectbox("Appliance", ["AC", "Fan", "Lights", "Geyser", "Fridge"])
    start = b.slider("From hour", 0, 23, 22)
    end = c.slider("To hour", 0, 23, 6)
    pct = e.slider("Reduce by %", 10, 100, 100, step=10)
    col, s, en, cut = (parsed if parsed else (E.APPLIANCES[app_name.lower()], start, end, pct))

    kwh, rs, co2 = E.simulate_change(d, col, s, en, cut)
    m1, m2, m3 = st.columns(3)
    m1.metric("Energy saved / month", f"{kwh:,.0f} kWh")
    m2.metric("Money saved / month", f"Rs {rs:,.0f}")
    m3.metric("CO2 avoided / month", f"{co2:,.0f} kg", help="Roughly the CO2 absorbed by "
              f"{co2 / 21:,.0f} trees in a month")
    st.caption(f"Scenario: {col.replace('_kwh', '').upper()} cut by {cut}% between {s}:00 and {en}:00")

    # Before/after hourly chart
    base = d.groupby("hour")[col].mean()
    mask = E.window_mask(base.index.to_series(), s, en)
    after = base.where(~mask, base * (1 - cut / 100))
    fig = go.Figure()
    fig.add_bar(x=base.index, y=base.values, name="Current")
    fig.add_bar(x=after.index, y=after.values, name="After change")
    fig.update_layout(barmode="group", xaxis_title="Hour of day", yaxis_title="Avg kWh")
    st.plotly_chart(fig)

# ---------- Tab 5: Action plan & leaderboard ----------
with tab5:
    st.subheader("Smart action plan (ranked by savings)")
    plan = E.action_plan(d)
    st.dataframe(plan, hide_index=True)
    st.success(f"Total potential savings: **Rs {plan['Rs saved/month'].sum():,.0f}/month**, "
               f"**{plan['CO2 avoided (kg)'].sum():,.0f} kg CO2**")

    st.subheader("Green Score leaderboard")
    rows = []
    for name in sorted(raw.building.unique()):
        dd, _, _ = prepare(raw[raw.building == name].reset_index(drop=True))
        rows.append({"Building": name, "Green Score": E.green_score(dd)})
    board = pd.DataFrame(rows).sort_values("Green Score", ascending=False).reset_index(drop=True)
    board.index += 1
    st.dataframe(board)
