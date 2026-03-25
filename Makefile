.PHONY: install test lint format mwe docker-build clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

test:  ## Run all tests
	pytest tests/ -v

lint:  ## Check code style
	python -m py_compile cfa/__init__.py
	python -m py_compile cfa/cli.py
	python -m py_compile cfa/config.py
	python -m py_compile cfa/utils/text_processing.py
	python -m py_compile cfa/utils/scoring.py
	python -m py_compile cfa/utils/io.py

mwe:  ## Run minimal working example
	python -m cfa run-all --config configs/mwe.yaml

docker-build:  ## Build Docker image
	docker build -t cfa .

docker-run:  ## Run pipeline in Docker (pass CONFIG=path and STAGE=name)
	docker run --gpus all --env-file .env -v $(PWD)/outputs:/app/outputs cfa $(STAGE) --config $(CONFIG)

clean:  ## Clean generated files
	rm -rf outputs/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
