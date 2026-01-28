
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

.PHONY: split-data
split-data: ## Split interactions_train.csv into train/val splits to avoid data leakage
	uv run python src/scripts/split_interactions.py

.PHONY: train-retrieval
train-retrieval: ## Train Two-Tower retrieval model
	uv run python src/scripts/train_retrieval.py \
		--data-dir data \
		--output-dir runs/retrieval/baseline \
		--embedding-dim 64 \
		--hidden-dims 256 128 \
		--use-features \
		--epochs 20 \
		--batch-size 512 \
		--lr 1e-3 \
		--device cpu

.PHONY: train-ranker
train-ranker: ## Train LightGBM ranker on retrieval candidates
	uv run python src/scripts/train_ranker.py \
		--data-dir data \
		--output-dir runs/ranking/very-advanced-features \
		--n-candidates 100 \
		--optuna-trials 20

.PHONY: train-cold-start
train-cold-start: ## Train cold-start encoder for zero-history users
	uv run python src/scripts/train_cold_start_encoder.py \
		--checkpoint runs/retrieval/baseline/retrieval_final.pt \
		--users data/users.csv \
		--out runs/retrieval/cold_encoder_baseline/cold_encoder.pt \
		--device cpu \
		--epochs 20 \
		--batch-size 256

.PHONY: train-all
train-all: split-data train-retrieval train-ranker train-cold-start ## Train all models (split → retrieval → ranker → cold-start)
	@echo "✅ All models trained successfully!"

.PHONY: eval-retrieval
eval-retrieval: ## Evaluate retrieval model on validation set (warm users)
	uv run python src/scripts/evaluate_retrieval.py \
		--checkpoint runs/retrieval/baseline/retrieval_final.pt \
		--embeddings runs/retrieval/baseline/recipe_embeddings.npy \
		--mode retrieval \
		--eval-split val

.PHONY: eval-retrieval-cold
eval-retrieval-cold: ## Evaluate retrieval model on cold-start users
	uv run python src/scripts/evaluate_retrieval.py \
		--checkpoint runs/retrieval/baseline/retrieval_final.pt \
		--cold-start-path runs/retrieval/cold_encoder_baseline/cold_encoder.pt \
		--embeddings runs/retrieval/baseline/recipe_embeddings.npy \
		--mode retrieval \
		--eval-split val_cold

.PHONY: eval-hybrid
eval-hybrid: ## Evaluate hybrid model on validation set (warm users)
	uv run python src/scripts/evaluate_retrieval.py \
		--checkpoint runs/retrieval/baseline/retrieval_final.pt \
		--embeddings runs/retrieval/baseline/recipe_embeddings.npy \
		--ranker-model runs/ranking/very-advanced-features/ranker.pkl \
		--mode hybrid \
		--eval-split val

.PHONY: eval-hybrid-cold
eval-hybrid-cold: ## Evaluate hybrid model on cold-start users
	uv run python src/scripts/evaluate_retrieval.py \
		--checkpoint runs/retrieval/baseline/retrieval_final.pt \
		--embeddings runs/retrieval/baseline/recipe_embeddings.npy \
		--cold-start-path runs/retrieval/cold_encoder_baseline/cold_encoder.pt \
		--ranker-model runs/ranking/very-advanced-features/ranker.pkl \
		--mode hybrid \
		--eval-split val_cold

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help