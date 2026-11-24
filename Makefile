.PHONY: install test evals serve
install:
	pip install -r requirements.txt
test:
	python -m pytest -q
evals:
	python -m evals.run_evals
serve:
	uvicorn memory_service.api:app --reload --port 8000
