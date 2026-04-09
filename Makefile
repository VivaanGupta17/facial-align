.PHONY: help dev up down build logs logs-backend seed migrate test test-unit test-backend test-pipelines test-coverage lint format fe-dev fe-build dev-backend dev-worker install-frontend up-infra db-migrate db-revision db-reset download-models setup clean

COMPOSE := docker compose -f infra/docker/docker-compose.yml

# ── Facial Align Development Commands ────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker ───────────────────────────────────────────────────

dev: up seed ## Start all services and seed demo data

up: ## Start all services (PostgreSQL, Redis, MinIO, Backend, Frontend)
	$(COMPOSE) up -d

down: ## Stop all services
	$(COMPOSE) down

up-infra: ## Start only infrastructure (PostgreSQL, Redis, MinIO)
	$(COMPOSE) up -d postgres redis minio minio-init

logs: ## Tail logs from all services
	$(COMPOSE) logs -f

logs-backend: ## Tail backend logs
	$(COMPOSE) logs -f backend celery-worker

build: ## Build all Docker images
	$(COMPOSE) build

# ── Backend Development ──────────────────────────────────────

dev-backend: ## Run backend in development mode
	cd apps/backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-worker: ## Run Celery worker in development mode
	cd apps/backend && celery -A app.workers.celery_app worker --loglevel=info

# ── Frontend Development ─────────────────────────────────────

fe-dev: ## Start frontend dev server
	cd apps/frontend && npm run dev

fe-build: ## Build frontend for production
	cd apps/frontend && npm run build

dev-frontend: fe-dev ## Alias for fe-dev

install-frontend: ## Install frontend dependencies
	cd apps/frontend && npm install

# ── Database ─────────────────────────────────────────────────

migrate: ## Run Alembic migrations (via Docker)
	$(COMPOSE) exec backend alembic upgrade head

db-migrate: ## Run Alembic migrations (local)
	cd apps/backend && alembic upgrade head

db-revision: ## Create a new migration (usage: make db-revision msg="description")
	cd apps/backend && alembic revision --autogenerate -m "$(msg)"

db-reset: ## Reset database (WARNING: destroys all data)
	$(COMPOSE) exec postgres psql -U facial_align -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	$(MAKE) db-migrate

seed: ## Seed demo data
	$(COMPOSE) exec backend python -m scripts.seed_demo_data --db

# ── Testing ──────────────────────────────────────────────────

test: ## Run all tests
	cd apps/backend && pytest tests/ -v --tb=short
	cd apps/frontend && npm test 2>/dev/null || true

test-unit: ## Run unit tests only
	cd apps/backend && pytest tests/unit/ -v --tb=short

test-backend: ## Run backend tests only
	cd apps/backend && pytest tests/ -v --tb=short

test-pipelines: ## Run pipeline tests only
	cd apps/backend && pytest tests/unit/pipelines/ -v --tb=short

test-coverage: ## Run tests with coverage report
	cd apps/backend && pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# ── Code Quality ─────────────────────────────────────────────

lint: ## Run linters (ruff + mypy + tsc)
	cd apps/backend && ruff check . && mypy app/ --ignore-missing-imports
	cd apps/frontend && npx tsc --noEmit

format: ## Auto-format code
	cd apps/backend && ruff format . && ruff check --fix .

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
