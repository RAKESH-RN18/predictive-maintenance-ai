import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app

client_context = TestClient(app)

def payload(t=45, v=0.10, p=2.0, rpm=1500, voltage=230, current=5, hours=2000):
    return {'temperature':t,'vibration':v,'pressure':p,'rpm':rpm,'voltage':voltage,'current':current,'operating_hours':hours}

def test_health():
    with client_context as client:
        r = client.get('/health'); assert r.status_code == 200

def test_normal_prediction():
    with client_context as client:
        r = client.post('/predict', json=payload()); assert r.status_code == 200
        body = r.json(); assert body['machine_status'] in {'NORMAL','WARNING','CRITICAL'}
        assert 0 <= body['status_probability'] <= 1
        assert set(body['status_probabilities']) == {'normal','warning','critical'}

def test_warning_prediction():
    with client_context as client:
        r = client.post('/predict', json=payload(72,0.40,3.0,1300,225,9,8000)); assert r.status_code == 200
        assert r.json()['machine_status'] == 'WARNING'

def test_critical_prediction():
    with client_context as client:
        r = client.post('/predict', json=payload(100,0.80,4.0,1000,210,12,12000)); assert r.status_code == 200
        assert r.json()['machine_status'] == 'CRITICAL'

def test_validation():
    data = payload(); data['temperature'] = -1
    with client_context as client:
        assert client.post('/predict', json=data).status_code == 422
