from pathlib import Path
import sys, json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FEATURES = ['temperature','vibration','pressure','rpm','voltage','current','operating_hours']
DATA = ROOT / 'data' / 'raw' / 'machine_sensor_data.csv'
MODEL_DIR = ROOT / 'models'; MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X, y = df[FEATURES], df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'logistic_regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=2000, random_state=42))]),
    'random_forest': RandomForestClassifier(n_estimators=350, class_weight='balanced', random_state=42, n_jobs=-1, min_samples_leaf=2),
    'xgboost': XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42, n_jobs=4)
}

results = {}; reports = {}; matrices = {}; best_name = None; best_f1 = -1
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision_macro': float(precision_score(y_test, pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, pred, average='macro', zero_division=0)),
    }
    results[name] = metrics
    reports[name] = classification_report(y_test, pred, target_names=['NORMAL','WARNING','CRITICAL'], output_dict=True, zero_division=0)
    matrices[name] = confusion_matrix(y_test, pred).tolist()
    if metrics['f1_macro'] > best_f1:
        best_f1, best_name = metrics['f1_macro'], name

best_model = models[best_name]
joblib.dump(best_model, MODEL_DIR / 'failure_model.joblib')
joblib.dump(FEATURES, MODEL_DIR / 'feature_names.joblib')

normal_X = X_train[y_train == 0]
anomaly_model = IsolationForest(n_estimators=250, contamination=0.08, random_state=42)
anomaly_model.fit(normal_X)
joblib.dump(anomaly_model, MODEL_DIR / 'anomaly_model.joblib')

if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
elif hasattr(best_model[-1], 'coef_'):
    importance = np.mean(np.abs(best_model[-1].coef_), axis=0)
else:
    importance = np.zeros(len(FEATURES))
importance = dict(sorted({f: float(v) for f, v in zip(FEATURES, importance)}.items(), key=lambda x: x[1], reverse=True))

payload = {'best_model': best_name, 'results': results, 'classification_reports': reports, 'confusion_matrices': matrices, 'feature_importance': importance}
(MODEL_DIR / 'metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps({'best_model': best_name, 'results': results}, indent=2))
