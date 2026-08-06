@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No se encontro el entorno .venv.
    echo Cree el entorno e instale requirements.txt siguiendo README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
