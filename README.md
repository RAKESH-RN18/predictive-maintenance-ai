# AI-Powered Predictive Maintenance System

A portfolio-grade machine-health platform combining **machine learning, anomaly detection, a rule-based fault diagnosis layer, FastAPI, Streamlit, SQLite, testing, Docker and CI**.

## What it does

Sensor readings are converted into a machine-health decision:

**NORMAL / WARNING / CRITICAL → anomaly detection → likely fault → maintenance priority → recommended actions**

### Inputs
- Temperature
- Vibration
- Pressure
- RPM
- Voltage
- Current
- Operating hours

### ML
- Logistic Regression baseline
- Random Forest baseline
- XGBoost multiclass classifier
- Isolation Forest anomaly detector
- Macro-F1 used to select the classification model

The included dataset is synthetic and designed for development/portfolio demonstration. It is **not validated for real industrial safety decisions**.

## Architecture

```text
Sensor Input
    ↓
FastAPI REST API
    ↓
XGBoost 3-class classifier ──→ NORMAL/WARNING/CRITICAL
    ↓
Isolation Forest ──→ anomaly + severity
    ↓
Rule-based fault diagnosis
    ↓
Maintenance priority + actions
    ↓
SQLite history
    ↓
Streamlit dashboard
```

## Windows setup

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python scripts\generate_data.py
python scripts\train_model.py
pytest -q
```

If PowerShell blocks activation, use Command Prompt or:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

## Run API

```bat
set PYTHONPATH=.
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Run dashboard

Keep the API running, open a second terminal:

```bat
.venv\Scripts\activate.bat
streamlit run dashboard\streamlit_app.py
```

Dashboard: `http://localhost:8501`

## Example critical request

```json
{
  "temperature": 100,
  "vibration": 0.80,
  "pressure": 4.0,
  "rpm": 1000,
  "voltage": 210,
  "current": 12,
  "operating_hours": 12000
}
```

## API response design

The API intentionally uses `status_probability` rather than `failure_probability` because this is a **three-class status model**. It also returns all three class probabilities.

## Docker

```bash
docker compose up --build
```

API: `http://localhost:8000/docs`
Dashboard: `http://localhost:8501`

## Project structure

```text
app/                 FastAPI backend + ML inference
scripts/             dataset generation, training, evaluation
dashboard/           Streamlit UI
tests/               automated tests
data/raw/            generated training data
models/              trained models and metrics
.github/workflows/   CI pipeline
```

## Future production upgrades

- Replace synthetic data with a documented public industrial dataset
- Add SHAP-based explanation charts
- Add authentication and role-based access
- Add PostgreSQL for multi-machine deployments
- Connect ESP32/Arduino sensor telemetry
- Add real-time streaming with MQTT/Kafka
- Add model registry and scheduled retraining
- Add cloud deployment and observability
