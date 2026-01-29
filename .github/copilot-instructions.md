# Coffee Recipe Recommendation System

Production-ready **hybrid two-stage recommender** for specialty coffee recipes. Achieves NDCG@5 > 0.4 with ~67ms end-to-end inference latency.

## Architecture

**Two-Stage Hybrid Pipeline** (Google/Meta-style):
1. **Retrieval**: Two-Tower neural model + ChromaDB ANN → ~100 candidates (~21ms)
2. **Ranking**: LightGBM ranker with 140+ features → top-N (~46ms)

**Key components**:
- `src/coffee_recipe_recommender/preprocessing/` - Feature engineering (140+ features), embeddings
- `src/coffee_recipe_recommender/models/` - User/Recipe towers (64-dim), LightGBM ranker, hybrid pipeline
- `src/coffee_recipe_recommender/training/` - InfoNCE contrastive loss, datasets, loaders
- `src/coffee_recipe_recommender/inference/` - Main `Recommender` class with SHAP explainability
- `src/coffee_recipe_recommender/evaluation/` - NDCG, Hit Rate, MRR, Coverage metrics
- `src/coffee_recipe_recommender/db/` - VectorStore (ChromaDB), FeatureStore (SQLite)
- `src/client/` - FastAPI app with HTML demo, services (recommend, users, history)
- `src/scripts/` - Training, evaluation, and build scripts

## Domain-Specific Knowledge

**Coffee recommendation challenges:**
- **Equipment constraints**: Hard filter recipes by user's owned equipment (applied in retrieval stage)
- **Taste matching**: 4D taste vectors (bitterness, sweetness, acidity, body) - use cosine similarity
- **Cold-start users**: ColdStartEncoder MLP maps user features → embedding space (seamless fallback)
- **Replayability**: Unlike movies, users remake favorites - recommend already-tried recipes

**Data files** ([docs/README_dataset.md](docs/README_dataset.md)):
- `data/interactions_train.csv` - Train on this (user_id, recipe_id, rating, completed, timestamp)
- `data/interactions_val.csv` - Warm users validation (DO NOT train)
- `data/interactions_val_cold.csv` - Cold-start validation (DO NOT train)
- `data/cold_users.json` - Zero-history users for cold-start testing
- `data/users.csv` - Profiles: owned_equipment (JSON array), taste_pref_* (floats 0-1)
- `data/recipes.csv` - Catalog: required_equipment, taste_* (0-1), difficulty
- `data/feature_store.db` - SQLite database with pre-computed user/recipe stats
- `data/chroma/` - ChromaDB vector store with recipe embeddings

## Developer Workflows

**Setup** (uv-based, not pip):
```bash
make install          # uv sync + pre-commit hooks
source .venv/bin/activate
```

**Code quality**:
```bash
make format           # ruff format + ruff check --fix
make lint             # ruff check only
```

**Training pipeline** (see [Makefile](Makefile)):
```bash
make split-data           # Split interactions into train/val
make train-retrieval      # Two-Tower with InfoNCE loss
make train-cold-start     # Cold-start encoder for new users
make train-ranker         # LightGBM with Optuna tuning
make train-all            # Full pipeline: split → retrieval → ranker → cold-start
```

**Inference stores**:
```bash
make build-feature-store  # Build SQLite feature store from training data
make build-vector-store   # Build ChromaDB vector store from embeddings
```

**Evaluation**:
```bash
make eval-retrieval       # Retrieval-only on warm users
make eval-retrieval-cold  # Retrieval-only on cold-start users
make eval-hybrid          # Full hybrid on warm users
make eval-hybrid-cold     # Full hybrid on cold-start users
```

**Run application**:
```bash
make run-app              # FastAPI with uvicorn --reload
```

## Project-Specific Conventions

**Scripts**: Always use `uv run python src/scripts/<script>.py`

**Negative sampling**: In-batch negatives for contrastive learning (retrieval stage)

**Model serialization**:
- Two-Tower → `.pt` PyTorch checkpoint (ONNX export available)
- Cold-Start Encoder → `.pt` PyTorch checkpoint
- LightGBM → `.pkl` pickle format
- Recipe Embeddings → `.npy` NumPy array

**Feature engineering**: 140+ features in 7 groups:
- **Taste**: cosine similarity, euclidean distance, weighted similarity, diff features
- **Equipment**: match, coverage, missing count, sophistication matching
- **Historical**: user/recipe interaction stats, ratings, completion rates
- **Temporal**: time-of-day patterns, weekend preferences, consistency scores
- **Cross features**: taste×equipment, popularity×match, squared interactions
- **Discovery**: exploration ratio, novelty scores, experience level
- **Dietary**: compatibility flags, vegan matching, dietary restrictions

**Feature presets** (for `FeatureEngineer`):
- `"all"` - All 140+ features
- `"legacy"` - Original ~50 features
- `"fast"` - Minimal set for speed

**Cold-start strategy**: ColdStartEncoder trained to match warm user embeddings from taste preferences

**Explainability**: SHAP TreeExplainer for LightGBM ranker with feature grouping for UI

**Evaluation**: NDCG@5 primary metric, also track HR@10, MRR, Coverage

## Team Structure & Roles

**3-person team** (see [docs/roles.md](docs/roles.md)):
- **Denys** (Data Engineer): Data pipeline + Two-Tower retrieval + Cold-start encoder
- **Dmytro** (ML Engineer): LightGBM ranker + hybrid pipeline + optimization
- **Oleksandr** (Full-Stack): FastAPI + HTML demo + deployment

**Git branching**:
- `main` - stable only
- Feature branches: `denys/*`, `dmytro/*`, `oleksandr/*`
- PR review required before merge to main

## Critical Dependencies

From [pyproject.toml](pyproject.toml):
- **Package manager**: `uv` (NOT pip) - fast Rust-based resolver
- **ML stack**: PyTorch (Two-Tower), LightGBM (ranker), scikit-learn
- **Vector DB**: ChromaDB (ANN search)
- **API/Web**: FastAPI, Pydantic v2, Jinja2
- **Explainability**: SHAP
- **Inference**: ONNX, onnxruntime (optional)
- **Dev tools**: ruff (lint+format), pre-commit

**Code style**:
- Line length: 119 chars (ruff config)
- Quote style: double quotes everywhere

## Key Files to Reference

- [.github/tasks/tasks.md](.github/tasks/tasks.md) - Full implementation plan
- [docs/roles.md](docs/roles.md) - Team responsibilities, workflow, success criteria
- [docs/explanations.md](docs/explanations.md) - Theoretical foundations and pipeline explanation
- [docs/README_dataset.md](docs/README_dataset.md) - Data schema, `recommend()` function signature
- [docs/README_retrieval.md](docs/README_retrieval.md) - Two-Tower model training guide
- [Makefile](Makefile) - All available commands
