.PHONY: help install run-app run-notebooks test clean

help:
	@echo "Available commands:"
	@echo "  make install       Install all dependencies"
	@echo "  make run-app       Run Streamlit application"
	@echo "  make run-notebooks Run Jupyter notebooks"
	@echo "  make test         Run all tests"
	@echo "  make clean        Clean temporary files"

install:
	pip install -r requirements.txt
	pip install -e .

run-app:
	streamlit run app/main.py

run-notebooks:
	jupyter notebook notebooks/

test:
	pytest tests/ -v --cov=src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .coverage