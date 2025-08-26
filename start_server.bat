@echo off
echo Starting Flask server...
set FLASK_APP=app.py
set FLASK_ENV=development
python -m flask run --port=5000 --host=0.0.0.0
pause
