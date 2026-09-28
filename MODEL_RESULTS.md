# Model Evaluation

Run `python scripts/train_model.py` to regenerate the metrics in `models/metrics.json`.

The training pipeline compares Logistic Regression, Random Forest and XGBoost using a stratified test split. The selection metric is **macro F1**, which gives equal weight to NORMAL, WARNING and CRITICAL classes.

The dataset is synthetic, so these scores should be interpreted as a software/ML pipeline demonstration rather than evidence of real-world industrial accuracy.
