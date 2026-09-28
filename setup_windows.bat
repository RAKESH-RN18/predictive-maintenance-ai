@echo off
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\generate_data.py
python scripts\train_model.py
pytest -q
pause
