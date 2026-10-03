# WattWise: Energy Twin for Campuses

## Run locally
    pip install -r requirements.txt
    python generate_data.py        # optional, app auto-creates data if missing
    streamlit run app.py

## Deploy (free) on Streamlit Community Cloud
1. Push this folder to a public GitHub repo (include data/campus_energy.csv).
2. Go to share.streamlit.io, sign in with GitHub, click "Create app".
3. Pick the repo, branch main, main file app.py, then Deploy.

## Files
- generate_data.py : simulated hourly data for 3 buildings (with injected night-time waste)
- engine.py        : XGBoost forecast, Isolation Forest anomalies, what-if maths, action plan, Green Score
- app.py           : Streamlit dashboard (5 tabs)
