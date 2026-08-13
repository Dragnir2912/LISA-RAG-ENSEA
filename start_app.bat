@echo off
setlocal
cd /d "%~dp0"
set "RAG_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%RAG_PYTHON%" (
  echo L'environnement virtuel du projet est introuvable :
  echo %RAG_PYTHON%
  echo Creez ou restaurez .venv avant de lancer l'application.
  pause
  exit /b 1
)

echo Demarrage de RAG ENSEA sur http://127.0.0.1:8000
echo Python utilise : %RAG_PYTHON%
echo Gardez cette fenetre ouverte pendant l'utilisation.
"%RAG_PYTHON%" run_app.py
pause
