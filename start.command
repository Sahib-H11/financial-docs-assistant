#!/bin/bash
cd ~/Projects/financial-docs-assistant
source .venv/bin/activate
uvicorn api:app &
streamlit run app.py