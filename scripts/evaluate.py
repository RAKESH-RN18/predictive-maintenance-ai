from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
p = ROOT / 'models' / 'metrics.json'
if not p.exists():
    raise SystemExit('Train the model first: python scripts\\train_model.py')
data = json.loads(p.read_text())
print('Best model:', data['best_model'])
for name, m in data['results'].items():
    print(f"{name}: accuracy={m['accuracy']:.4f}, macro_f1={m['f1_macro']:.4f}")
print('\nFeature importance:')
for k,v in data['feature_importance'].items(): print(f'  {k}: {v:.4f}')
