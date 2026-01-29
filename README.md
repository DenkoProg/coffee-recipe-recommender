# ☕ Coffee Recipe Recommender

A hybrid recommendation system for coffee recipes using **Two-Tower retrieval** + **LightGBM ranking**.

## Key Features

- **Two-stage architecture**: Neural retrieval (~21ms) + gradient boosting ranking (~46ms)
- **Cold-start handling**: Content-based encoder for new users without interaction history
- **140+ engineered features**: Taste similarity, temporal patterns, cross-features
- **SHAP explanations**: Human-readable "why recommended" reasons in the UI

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML/DL** | PyTorch, LightGBM, Optuna, SHAP |
| **Data** | Pandas, NumPy, scikit-learn |
| **Infrastructure** | FastAPI, ChromaDB (vector store) |

## Quick Start

```bash
# Clone & install (requires Git LFS for data/models)
git lfs install
git clone https://github.com/DenkoProg/coffee-recipe-recommender
cd coffee-recipe-recommender
make install

# Run the app (pre-trained models included)
make run-app
```

Open [http://localhost:8000](http://localhost:8000) for the demo UI.

## Results

- **NDCG@5**: 0.80 (warm users)
- **Latency**: ~67ms end-to-end
- **Coverage**: >60%

## Documentation

See [`docs/explanations.md`](docs/explanations.md) for detailed architecture and theory.
