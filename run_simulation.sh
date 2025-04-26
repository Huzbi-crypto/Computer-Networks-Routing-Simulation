#!/bin/bash
# Linux/Mac shell script to run simulation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
