.PHONY: train serve mlflow-ui test

train:
	python src/train.py

serve:
	uvicorn app.main:app --reload --port 8000

mlflow-ui:
	mlflow ui

test:
	pip install -r requirements.txt pytest httpx
	pytest tests/ -v
