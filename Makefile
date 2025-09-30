# Makefile
SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "  setup          : Install dependencies via uv"
	@echo "  test           : Run pytest suite"
	@echo "  eval-retrieval : Run retrieval evaluation with metrics"
	@echo "  eval-generation: Run generation evaluation with summary"
	@echo "  clean          : Clean unnecessary files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build   : Build Docker image"
	@echo "  docker-run     : Run container with docker-compose"
	@echo "  docker-stop    : Stop running container"
	@echo "  docker-logs    : View container logs"
	@echo "  docker-shell   : Open shell in running container"

# Environment
.PHONY: setup
setup:
	uv sync

# Testing & Evaluation
.PHONY: test
test:
	uv run pytest

.PHONY: eval-retrieval
eval-retrieval:
	uv run python -m evals.evals_retrieval --print-metrics

.PHONY: eval-generation
eval-generation:
	uv run python -m evals.evals_generation --print-summary

# Cleaning
.PHONY: clean
clean:
	find . -type f -iname ".DS_Store" -delete
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ipynb_checkpoints" -delete
	rm -f .coverage

# Docker
.PHONY: docker-build
docker-build:
	docker build -t faq-assistant:latest .

.PHONY: docker-run
docker-run:
	docker-compose up -d

.PHONY: docker-stop
docker-stop:
	docker-compose down

.PHONY: docker-logs
docker-logs:
	docker-compose logs -f

.PHONY: docker-shell
docker-shell:
	docker-compose exec faq-assistant /bin/bash