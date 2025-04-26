@echo off
REM Windows batch file to run simulation
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python main.py
pause
