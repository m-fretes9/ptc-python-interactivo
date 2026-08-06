#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [[ ! -x ".venv/bin/python" ]]; then
  echo "No se encontró .venv. Cree el entorno e instale requirements.txt según README.md."
  exit 1
fi
.venv/bin/python -m streamlit run app.py
