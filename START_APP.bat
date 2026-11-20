@echo off
title Hinglish Sentiment Analyzer
cd /d "%~dp0"
echo Starting Hinglish Sentiment Analyzer...
echo The app opens at http://localhost:8501. Close this window (or press Ctrl+C) to stop it.
echo NOTE: The first Analyze click can take about a minute while the model loads.
echo.
python -m streamlit run app.py --server.port 8501
if errorlevel 1 (
    echo.
    echo The app did not start. Run:  pip install -r requirements.txt
    pause
)
