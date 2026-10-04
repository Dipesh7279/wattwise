# WattWise

WattWise is a Python-based energy analytics dashboard for campuses and buildings. It helps monitor electricity consumption, forecast demand, detect abnormal usage, estimate appliance-level energy splits, and suggest practical savings actions to reduce bills and carbon emissions.

The app is built with Streamlit and uses forecasting and anomaly detection models to turn raw hourly energy data into actionable insights.

## Key Features

- Hourly energy forecasting for each building
- Waste/anomaly detection using forecast residuals
- Appliance-wise estimation from aggregate energy data
- What-if simulator for scenario planning
- Action plan ranked by savings in rupees and CO2 avoided
- Green score leaderboard across buildings
- CSV upload support for real meter data

## Why WattWise?

Energy waste in campus or facility operations often goes unnoticed until bills arrive. WattWise makes it easier to:

- understand consumption trends,
- identify abnormal spikes,
- estimate where load is coming from,
- model reductions such as switching off ACs or lights during idle hours,
- plan cost-saving and sustainability actions.

## Demo / Screenshot

The dashboard includes multiple tabs such as:

- Forecast
- Waste alerts
- Appliance breakdown
- What-if simulator
- Action plan and leaderboard

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- XGBoost
- scikit-learn

## Project Structure

```text
wattwise/
├── app.py                 # Streamlit dashboard
├── engine.py              # Core analytics logic
├── generate_data.py       # Simulated campus energy data generator
├── requirements.txt       # Python dependencies
├── data/
│   └── campus_energy.csv  # Generated sample dataset (created automatically if missing)
├── README.md              # Project documentation
└── .gitignore             # Git ignore rules (if present)
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Dipesh7279/wattwise.git
cd wattwise
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

If the dataset does not exist yet, the app will generate sample campus energy data automatically.

## Data Format

The app accepts CSV files with hourly energy data. Minimum required columns:

- `timestamp`
- `total_kwh`

Optional columns include:

- `temp`
- `building`
- `ac_kwh`
- `fan_kwh`
- `lights_kwh`
- `geyser_kwh`
- `fridge_kwh`

Example:

```csv
timestamp,total_kwh
2026-01-01 00:00:00,12.5
2026-01-01 01:00:00,11.8
2026-01-01 02:00:00,10.9
```

If appliance-wise data is missing, WattWise estimates the split using a simple rule-based model and continues with forecasting and anomaly detection.

## How It Works

1. Upload real data or load demo data
2. Standardize timestamp and energy columns
3. Train the forecast model on historical hourly consumption
4. Detect anomalies from prediction residuals
5. Estimate energy split across common campus loads
6. Simulate savings scenarios and rank actions by impact
7. Present the results in an interactive dashboard

## Example Use Cases

- Campus energy monitoring
- Building performance benchmarking
- Detecting night-time HVAC or lighting waste
- Estimating cost reduction from policy changes
- Sustainability tracking and reporting

## Deployment

### Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Create a new app and select the repository.
4. Set the main file to `app.py`.
5. Deploy.

## License

This project is currently distributed without a formal license declaration. If you plan to share or reuse it publicly, consider adding an appropriate open-source license.

## Contributing

Contributions are welcome. If you want to improve the app:

- open an issue for a feature request or bug,
- fork the project,
- create a feature branch,
- submit a pull request with clear notes.

## Contact

For questions or collaboration opportunities, reach out via the GitHub repository or project maintainer.

## Acknowledgements

This app is designed for educational and practical energy-efficiency use cases in campus and building operations.
