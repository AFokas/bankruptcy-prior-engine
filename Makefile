.PHONY: install download initdb explore build test api html clean

install:
	pip install -e ".[dev]"

download:
	python scripts/download_data.py

initdb:
	python scripts/init_db.py

explore: initdb
	jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_exploration.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_foundation.ipynb

build: initdb
	python -m pytest tests/ -q

test:
	python -m pytest tests/ -q

api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

html:
	mkdir -p results
	for nb in notebooks/*.ipynb; do \
		name=$$(basename $$nb .ipynb); \
		jupyter nbconvert --to html --execute $$nb --output ../results/$$name.html; \
	done

clean:
	rm -rf artefacts/*.joblib artefacts/*.json results/*.html
