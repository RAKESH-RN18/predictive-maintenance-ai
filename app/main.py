from contextlib import asynccontextmanager
from pathlib import Path
import json
from fastapi import FastAPI, HTTPException, Query
from app.schemas import SensorReading, PredictionResponse, HistoryItem
from app.database import init_db, save_prediction, recent_predictions
from app.ml.predictor import Predictor

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    init_db()
    try:
        predictor = Predictor()
    except FileNotFoundError:
        predictor = None
    yield

app = FastAPI(
    title='AI-Powered Predictive Maintenance API',
    version='3.0.0',
    lifespan=lifespan,
    description='Machine health classification, anomaly detection, fault diagnosis and maintenance recommendations.'
)

@app.get('/')
def root():
    return {'project': 'AI-Powered Predictive Maintenance System', 'status': 'running', 'models_loaded': predictor is not None}

@app.get('/health')
def health():
    return {'status': 'healthy', 'models_loaded': predictor is not None}

@app.get('/model-info')
def model_info():
    path = Path(__file__).resolve().parents[1] / 'models' / 'metrics.json'
    if not path.exists():
        raise HTTPException(503, 'Model metrics are unavailable. Train the models first.')
    data = json.loads(path.read_text(encoding='utf-8'))
    return {'best_model': data['best_model'], 'results': data['results'], 'feature_importance': data['feature_importance']}

@app.post('/predict', response_model=PredictionResponse)
def predict(reading: SensorReading):
    if predictor is None:
        raise HTTPException(503, 'Models not trained. Run python scripts\\generate_data.py and python scripts\\train_model.py first.')
    data = reading.model_dump()
    result = predictor.predict(data)
    save_prediction(data, result)
    return result

@app.get('/history', response_model=list[HistoryItem])
def history(limit: int = Query(default=30, ge=1, le=200)):
    rows = recent_predictions(limit)
    keys = ['id','created_at','machine_status','status_probability','health_score','anomaly','anomaly_severity','likely_fault','maintenance_priority']
    return [dict(zip(keys, row)) for row in rows]
