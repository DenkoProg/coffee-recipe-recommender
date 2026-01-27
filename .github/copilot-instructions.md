# Coffee Recipe Recommendation System

Production-ready **hybrid two-stage recommender** for specialty coffee recipes. Targets NDCG@5 > 0.4 with <50ms inference latency.

## Architecture

**Two-Stage Hybrid Pipeline** (Google/Meta-style):
1. **Retrieval**: Two-Tower neural model → ~100 candidates (<10ms)
2. **Ranking**: LightGBM ranker with rich features → top-N (<40ms)

**Key components** (planned in [.github/tasks/tasks.md](.github/tasks/tasks.md)):
- `src/coffee_recipe_recommender/data/` - Feature engineering (50+ features), loaders, PyTorch datasets
- `src/coffee_recipe_recommender/models/` - User/Recipe towers (64-dim), LightGBM ranker, hybrid pipeline
- `src/coffee_recipe_recommender/training/` - InfoNCE contrastive loss, training scripts
- `src/coffee_recipe_recommender/inference/` - Main `recommend()` function, explainability (SHAP)
- `src/coffee_recipe_recommender/api/` - FastAPI endpoints (POST /recommend, GET /user/{id}/history)
- `src/client/` - Streamlit demo with taste radar charts, equipment badges

## Domain-Specific Knowledge

**Coffee recommendation challenges:**
- **Equipment constraints**: Hard filter recipes by user's owned equipment (post-retrieval, pre-ranking)
- **Taste matching**: 4D taste vectors (bitterness, sweetness, acidity, body) - use cosine similarity
- **Cold-start users**: Content-based MLP maps user features → embedding space (seamless fallback)
- **Replayability**: Unlike movies, users remake favorites - recommend already-tried recipes

**Data files** ([docs/README_dataset.md](docs/README_dataset.md)):
- `data/interactions_train.csv` - Train on this (user_id, recipe_id, rating, completed, timestamp)
- `data/interactions_val.csv` - Warm users validation (DO NOT train)
- `data/interactions_val_cold.csv` - Cold-start validation (DO NOT train)
- `data/cold_users.json` - Zero-history users for cold-start testing
- `data/users.csv` - Profiles: owned_equipment (JSON array), taste_pref_* (floats 0-1)
- `data/recipes.csv` - Catalog: required_equipment, taste_* (0-1), difficulty

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

**Key commands** (from [Makefile](Makefile)):
- Always use `uv run <cmd>` for Python scripts (e.g., `uv run python scripts/train_retrieval.py`)
- Line length: 119 chars (ruff config)
- Quote style: double quotes everywhere

**Training pipeline** (see [.github/tasks/tasks.md](.github/tasks/tasks.md)):
1. `scripts/train_retrieval.py` - Two-Tower with InfoNCE loss
2. `scripts/train_ranker.py` - Generate candidates from retrieval → train LightGBM
3. `scripts/evaluate.py` - NDCG@5 on val/val_cold splits

## Project-Specific Conventions

**Negative sampling**: In-batch negatives for contrastive learning (retrieval stage)

**Model serialization**:
- Two-Tower → ONNX (fast inference, pre-compute recipe embeddings)
- LightGBM → native format (.txt or pickle)

**Feature engineering**: 50+ features including:
- Taste cosine similarity (user_pref vs recipe_profile)
- User interaction history (count, avg rating, last interaction timestamp)
- Recipe popularity (global completion rate)
- Equipment overlap count

**Cold-start strategy**: Hybrid approach - content MLP trained to match warm user embeddings

**Evaluation**: NDCG@5 primary metric, also track HR@10, MRR, Coverage

## Team Structure & Roles

**3-person team** (see [docs/roles.md](docs/roles.md)):
- **Denys** (Data Engineer): Data pipeline + Two-Tower retrieval model
- **Dmytro** (ML Engineer): LightGBM ranker + hybrid pipeline + optimization
- **Oleksandr** (Full-Stack): FastAPI + Streamlit demo

**Git branching**:
- `main` - stable only
- `denys/data-pipeline`, `dmytro/ranking-model`, `oleksandr/api-webapp`
- PR review required before merge to main

## Critical Dependencies

From [pyproject.toml](pyproject.toml) + planned additions:
- **Package manager**: `uv` (NOT pip) - fast Rust-based resolver
- **ML stack**: PyTorch (Two-Tower), LightGBM (ranker), scikit-learn
- **API/Web**: FastAPI, Streamlit, Pydantic v2
- **Inference**: ONNX, onnxruntime (target <50ms)
- **Dev tools**: ruff (lint+format), mypy, pre-commit

## Key Files to Reference

- [.github/tasks/tasks.md](.github/tasks/tasks.md) - Full implementation plan (11 phases)
- [docs/roles.md](docs/roles.md) - Team responsibilities, workflow, success criteria
- [docs/README_dataset.md](docs/README_dataset.md) - Data schema, `recommend()` function signature
- [docs/README_presentation.md](docs/README_presentation.md) - Project requirements, evaluation criteria
