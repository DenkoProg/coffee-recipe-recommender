
.PHONY: install
install: ## Install dependencies and setup pre-commit hooks
	@echo "🚀 Installing dependencies from lockfile"
	@uv sync --frozen
	@uv run pre-commit install

.PHONY: lint
lint: ## Run ruff linter
	uv run ruff check

.PHONY: format
format: ## Format code and fix linting issues
	uv run ruff format
	uv run ruff check --fix --unsafe-fixes

.PHONY: eval-retrieval
eval-retrieval: ## Evaluate retrieval model on validation set
	uv run python src/scripts/evaluate_retrieval.py \
		--checkpoint runs/retrieval/baseline/retrieval_final.pt \
		--embeddings runs/retrieval/baseline/recipe_embeddings.npy \
		--mode retrieval \
		--eval-split val

.PHONY: eval-retrieval-cold
eval-retrieval-cold: ## Evaluate retrieval model on cold-start users
	uv run python src/scripts/evaluate_retrieval.py \
		--checkpoint runs/retrieval/baseline/retrieval_final.pt \
		--embeddings runs/retrieval/baseline/recipe_embeddings.npy \
		--mode retrieval \
		--eval-split val_cold

.PHONY: eval-hybrid
eval-hybrid: ## Evaluate hybrid model on validation set
	uv run python src/scripts/evaluate_retrieval.py \
		--checkpoint runs/retrieval/baseline/retrieval_final.pt \
		--embeddings runs/retrieval/baseline/recipe_embeddings.npy \
		--ranker-model runs/ranking/improved-features/ranker.pkl \
		--mode hybrid \
		--eval-split val

.PHONY: eval-hybrid-cold
eval-hybrid-cold: ## Evaluate hybrid model on cold-start users
	uv run python src/scripts/evaluate_retrieval.py \
		--checkpoint runs/retrieval/baseline/retrieval_final.pt \
		--embeddings runs/retrieval/baseline/recipe_embeddings.npy \
		--ranker-model runs/ranking/improved-features/ranker.pkl \
		--mode hybrid \
		--eval-split val_cold

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help