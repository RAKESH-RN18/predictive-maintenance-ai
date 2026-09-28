from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'raw' / 'machine_sensor_data.csv'

rng = np.random.default_rng(42)
n = 15000

temperature = rng.uniform(35, 105, n)
vibration = rng.uniform(0.05, 0.90, n)
pressure = rng.uniform(1.5, 4.5, n)
rpm = rng.uniform(950, 1600, n)
voltage = rng.uniform(210, 245, n)
current = rng.uniform(3, 14, n)
operating_hours = rng.uniform(100, 15000, n)

temp_risk = np.clip((temperature - 55) / 50, 0, 1)
vibration_risk = np.clip((vibration - 0.15) / 0.70, 0, 1)
pressure_risk = np.clip((pressure - 2.2) / 2.0, 0, 1)
rpm_risk = np.clip((1400 - rpm) / 450, 0, 1)
current_risk = np.clip((current - 6) / 8, 0, 1)
age_risk = np.clip((operating_hours - 4000) / 11000, 0, 1)
health_risk = (
    0.30 * temp_risk + 0.32 * vibration_risk + 0.10 * pressure_risk +
    0.10 * rpm_risk + 0.10 * current_risk + 0.08 * age_risk + rng.normal(0, 0.025, n)
)
labels = np.select([health_risk < 0.30, health_risk < 0.58], [0, 1], default=2)

df = pd.DataFrame({
    'temperature': temperature, 'vibration': vibration, 'pressure': pressure,
    'rpm': rpm, 'voltage': voltage, 'current': current,
    'operating_hours': operating_hours, 'status': labels.astype(int)
})
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f'Saved {len(df):,} rows to {OUT}')
print(df['status'].value_counts(normalize=True).sort_index().rename(index={0:'NORMAL',1:'WARNING',2:'CRITICAL'}))
