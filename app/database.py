import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parents[1] / 'data' / 'predictions.db'

def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with _connect() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            machine_status TEXT NOT NULL,
            status_probability REAL NOT NULL,
            health_score REAL NOT NULL,
            anomaly INTEGER NOT NULL,
            anomaly_severity TEXT NOT NULL,
            likely_fault TEXT NOT NULL,
            maintenance_priority TEXT NOT NULL,
            temperature REAL, vibration REAL, pressure REAL, rpm REAL,
            voltage REAL, current REAL, operating_hours REAL
        )''')
        conn.commit()

def save_prediction(reading, result):
    with _connect() as conn:
        conn.execute('''INSERT INTO predictions (
            created_at, machine_status, status_probability, health_score,
            anomaly, anomaly_severity, likely_fault, maintenance_priority,
            temperature, vibration, pressure, rpm, voltage, current, operating_hours
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            datetime.now(timezone.utc).isoformat(), result['machine_status'],
            result['status_probability'], result['health_score'], int(result['anomaly']),
            result['anomaly_severity'], result['likely_fault'], result['maintenance_priority'],
            reading['temperature'], reading['vibration'], reading['pressure'], reading['rpm'],
            reading['voltage'], reading['current'], reading['operating_hours']))
        conn.commit()

def recent_predictions(limit=30):
    with _connect() as conn:
        return conn.execute('''SELECT id, created_at, machine_status, status_probability,
            health_score, anomaly, anomaly_severity, likely_fault, maintenance_priority
            FROM predictions ORDER BY id DESC LIMIT ?''', (limit,)).fetchall()
