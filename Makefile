.PHONY: help dev up down build test lint format clean setup

# ── Facial Align Development Commands ────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker ───────────────────────────────────────────────────

up: ## Start all services (PostgreSQL, Redis, MinIO, Backend, Frontend)
	docker compose -f infra/docker/docker-compose.yml up -d

down: ## Stop all services
	docker compose -f infra/docker/docker-compose.yml down

up-infra: ## Start only infrastructure (PostgreSQL, Redis, MinIO)
	docker compose -f infra/docker/docker-compose.yml up -d postgres redis minio minio-init

logs: ## Tail logs from all services
	docker compose -f infra/docker/docker-compose.yml logs -f

logs-backend: ## Tail backend logs
	docker compose -f infra/docker/docker-compose.yml logs -f backend celery-worker

build: ## Build all Docker images
	docker compose -f infra/docker/docker-compose.yml build

# ── Backend Development ──────────────────────────────────────

dev-backend: ## Run backend in development mode
	cd apps/backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-worker: ## Run Celery worker in development mode
	cd apps/backend && celery -A app.workers.celery_app worker --loglevel=info

# ── Frontend Development ─────────────────────────────────────

dev-frontend: ## Run frontend in development mode
	cd apps/frontend && npm run dev

install-frontend: ## Install frontend dependencies
	cd apps/frontend && npm install

# ── Testing ──────────────────────────────────────────────────

test: ## Run all tests
	cd apps/backend && pytest tests/ -v --tb=short
	cd apps/frontend && npm test 2>/dev/null || true

test-backend: ## Run backend tests only
	cd apps/backend && pytest tests/ -v --tb=short

test-pipelines: ## Run pipeline tests only
	cd apps/backend && pytest tests/unit/pipelines/ -v --tb=short

test-coverage: ## Run tests with coverage report
	cd apps/backend && pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# ── Code Quality ─────────────────────────────────────────────

lint: ## Run linters (ruff + mypy + eslint)
	cd apps/backend && ruff check . && mypy app/ --ignore-missing-imports
	cd apps/frontend && npx tsc --noEmit

format: ## Auto-format code
	cd apps/backend && ruff format . && ruff check --fix .

# ── Database ─────────────────────────────────────────────────

db-migrate: ## Run database migrations
	cd apps/backend && alembic upgrade head

db-revision: ## Create a new migration (usage: make db-revision msg="description")
	cd apps/backend && alembic revision --autogenerate -m "$(msg)"

db-reset: ## Reset database (WARNING: destroys all data)
	docker compose -f infra/docker/docker-compose.yml exec postgres psql -U facial_align -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	$(MAKE) db-migrate

# ── ML / Models ──────────────────────────────────────────────

download-models: ## Download pre-trained model weights
	@echo "Downloading TotalSegmentator weights..."
	python scripts/download_models.py --model totalsegmentator
	@echo "Downloading DentalSegmentator weights..."
	python scripts/download_models.py --model dental_segmentator

# ── Setup ────────────────────────────────────────────────────

setup: ## First-time project setup
	@echo "Setting up Facial Align development environment..."
	cp -n .env.example .env || true
	$(MAKE) up-infra
	cd apps/backend && pip install -e ".[dev]"
	cd apps/frontend && npm install
	@echo ""
	@echo "Setup complete. Run 'make dev-backend' and 'make dev-frontend' in separate terminals."

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name node_modules -exec rm -rf {} + 2>/dev/null || true
	rm -rf apps/frontend/dist 2>/dev/null || true
