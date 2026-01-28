# Coffee Recipe Recommendation System - Implementation Tasks

> **Goal**: Build a production-ready hybrid recommendation system achieving NDCG@5 > 0.4
> **Primary Approach**: Two-Tower Retrieval + LightGBM Ranking (Industry SOTA)
> **Stack**: Python 3.11+, PyTorch, LightGBM, FastAPI, Streamlit
> **Timeline**: Incremental milestones with testable deliverables

## Architecture Overview

**Hybrid Two-Stage Pipeline** (Production SOTA used by Google, Meta, etc.):
1. **Retrieval Stage**: Two-Tower neural model generates ~100 candidates (<10ms)
2. **Ranking Stage**: LightGBM ranks candidates with rich features (<40ms)
3. **Cold-Start**: Content-based features fed directly to both stages

**Why This Approach?**
- ✅ **Fast**: Two-Tower pre-computes item embeddings, LightGBM is CPU-efficient
- ✅ **Accurate**: Neural retrieval + gradient boosting captures complex patterns
- ✅ **Explainable**: LightGBM provides feature importance out-of-the-box
- ✅ **Production-proven**: Standard architecture at major tech companies

## Phase 0: Project Setup & Data Analysis ✅ COMPLETED

- [x] **0.1** Update `pyproject.toml` with ML dependencies
  - **Core ML**: torch, lightgbm, scikit-learn ✅
  - **Data**: pandas, numpy ✅
  - **API/Web**: fastapi, uvicorn[standard] ✅
  - **Optimization**: onnx, onnxruntime ✅
  - **Vector DB**: chromadb ✅

- [x] **0.2** Create project directory structure
  - ✅ `src/coffee_recipe_recommender/` - Core package
  - ✅ `preprocessing/` - Feature engineering
  - ✅ `training/` - Datasets, loaders, losses
  - ✅ `models/` - Retrieval, Ranking, Hybrid
  - ✅ `evaluation/` - Metrics
  - ✅ `inference/` - Recommender
  - ✅ `db/` - Chroma vector store
  - ✅ `client/` - FastAPI app
  - ✅ `scripts/` - Training scripts

- [x] **0.3** Exploratory Data Analysis
  - ✅ Notebooks created for embeddings visualization
  - ✅ Analyzed interaction patterns
  - ✅ Chroma embeddings exploration

- [x] **0.4** Document data statistics and insights
  - ✅ Documented in README files
  - ✅ Dataset documentation in `docs/README_dataset.md`

---

## Phase 1: Data Pipeline & Feature Engineering ✅ COMPLETED

- [x] **1.1** Implement data loading utilities (`training/loaders.py`) ✅
  - ✅ Load CSV files with proper dtypes
  - ✅ Parse JSON columns (equipment, products, tags)
  - ✅ Handle missing values
  - ✅ Create user/recipe ID mappings

- [x] **1.2** Rich feature engineering (`preprocessing/preprocessing.py`) ✅
  - ✅ **User features**: Taste preferences (4D), equipment, history stats
  - ✅ **Recipe features**: Taste profile (4D), difficulty, equipment, popularity
  - ✅ **Interaction features**: Taste similarity (cosine), equipment match, temporal features
  - ✅ **Cross features**: 50+ features total

- [x] **1.3** Negative sampling implemented ✅
  - ✅ In-batch negatives for contrastive learning
  - ✅ Random candidate sampling for ranker

- [x] **1.4** Create datasets (`training/datasets.py`) ✅
  - ✅ `RetrievalDataset`: Basic user-item pairs
  - ✅ `RetrievalDatasetWithFeatures`: With taste features
  - ✅ Efficient batching with DataLoader

- [x] **1.5** Data validation ✅
  - ✅ Train/val split implemented
  - ✅ Feature generation validated

---

## Phase 2: Retrieval Model (Two-Tower) ✅ MOSTLY COMPLETED

- [x] **2.1** Implement User Tower (`models/retrieval.py`) ✅
  - ✅ User embedding + optional features (taste 4D)
  - ✅ MLP architecture: [256, 128] → 64-dim
  - ✅ Dropout, BatchNorm, ReLU activations
  - ✅ L2 normalization for cosine similarity

- [x] **2.2** Implement Recipe Tower ✅
  - ✅ Recipe embedding + optional features
  - ✅ Same architecture as User Tower
  - ✅ Shared embedding space (64-dim)

- [x] **2.3** Two-Tower model with contrastive loss ✅
  - ✅ Temperature-scaled dot product similarity
  - ✅ InfoNCE loss with in-batch negatives (`training/losses.py`)
  - ✅ Symmetric InfoNCE loss variant
  - ✅ Trained multiple versions (baseline, with features)

- [ ] **2.4** Cold-start user encoder ⚠️ TODO
  - ❌ Need dedicated MLP for feature-only users
  - ❌ Train to match warm user embeddings
  - ⚠️ Current: Falls back to average embedding or fails

## Phase 3: Ranking Model (LightGBM) ✅ COMPLETED

- [x] **3.1** Generate training data for ranker ✅
  - ✅ Sample 50 candidates per user from retrieval
  - ✅ Add true positives (interacted recipes)
  - ✅ Extract 50+ features per (user, recipe) pair
  - ✅ Binary relevance labels

- [x] **3.2** Train LightGBM ranker (`models/ranking.py`) ✅
  - ✅ LambdaRank objective for NDCG optimization
  - ✅ Early stopping on validation set
  - ✅ Trained models saved in `runs/ranking/`

- [x] **3.3** Feature importance analysis ✅
  - ✅ SHAP integration in training script
  - ✅ Top features identified
  - ✅ Visualization in training output

- [x] **3.4** Hyperparameter tuning ✅
  - ✅ Optuna for hyperparameter search
  - ✅ Best params: n_estimators=173, lr=0.1
  - ✅ Cross-validation with NDCG metric

---

## Phase 4: Hybrid Pipeline Integration ✅ COMPLETED

- [x] **4.1** Implement combined pipeline (`models/hybrid.py`) ✅
  - ✅ `HybridRecommenderModel` class
  - ✅ Two-stage pipeline: Retrieval → Ranking
  - ✅ Feature extraction between stages
  - ✅ Generic `Recommender` wrapper in `inference/recommender.py`

- [x] **4.2** Equipment constraint filtering ✅
  - ✅ Implemented in feature engineering
  - ✅ Equipment match score as feature

- [x] **4.3** Training scripts ✅
  - ✅ `scripts/train_retrieval.py` - Complete Two-Tower training
  - ✅ `scripts/train_ranker.py` - LightGBM with Optuna tuning
  - ✅ `scripts/evaluate_retrieval.py` - Evaluation pipeline

- [x] **4.4** Model serialization ✅
  - ✅ Two-Tower saved as PyTorch `.pt` checkpoints
  - ✅ Recipe embeddings saved as `.npy`
  - ✅ LightGBM saved as `.pkl`
  - ✅ ONNX export script available (`scripts/export_onnx.py`)

---

## Phase 5: Evaluation & Metrics ✅ MOSTLY COMPLETED

- [x] **5.1** Implement metrics (`evaluation/metrics.py`) ✅
  - ✅ `ndcg_at_k` - NDCG@5, NDCG@10
  - ✅ `hit_rate_at_k` - HR@5, HR@10
  - ✅ `mrr` - Mean Reciprocal Rank
  - ✅ `coverage` - Catalog coverage

- [x] **5.2** Evaluation pipeline (`scripts/evaluate_retrieval.py`) ✅
  - ✅ Evaluate on `interactions_val.csv` (warm users)
  - ✅ Evaluate on `interactions_val_cold.csv` (cold users)
  - ✅ Retrieval-only evaluation
  - ⚠️ Full hybrid pipeline evaluation TODO

- [ ] **5.3** Baseline comparisons ⚠️ PARTIAL
  - ✅ Handcrafted embeddings baseline (`scripts/create_simple_handcrafted_embeddings.py`)
  - ❌ Popularity baseline
  - ❌ Random baseline
  - ❌ Comparison table

- [ ] **5.4** A/B testing simulation ❌ TODO
  - ❌ Online metrics simulation
  - ❌ Diversity and novelty metrics

---

## Phase 6: Inference Optimization ✅ MOSTLY COMPLETED

- [x] **6.1** Implement main `recommend()` function ✅
  - ✅ `inference/recommender.py` - Generic `Recommender` class
  - ✅ Supports retrieval-only and hybrid modes
  - ✅ `from_retrieval_checkpoint()` and `from_hybrid_checkpoints()` loaders
  - ⚠️ Cold-start handling needs improvement

- [x] **6.2** Pre-computation and caching ✅
  - ✅ Pre-compute ALL recipe embeddings (saved as `.npy`)
  - ✅ Recipe embeddings loaded at startup
  - ⚠️ User embedding caching TODO
  - ⚠️ Equipment compatibility pre-computation TODO

- [x] **6.3** ONNX export ✅
  - ✅ Export script created (`scripts/export_onnx.py`)
  - ⚠️ ONNX inference integration TODO

- [x] **6.4** LightGBM optimization ✅
  - ✅ Native LightGBM inference (fast)
  - ✅ Batch feature extraction (vectorized)

- [x] **6.5** Latency profiling ✅
  - ✅ Timing in API endpoint (`took_ms` field)
  - ✅ Inline timing in recommender

## Phase 7: Explainability ❌ NOT STARTED

- [ ] **7.1** LightGBM feature importance (`inference/explainer.py`) ❌
  - ❌ Dedicated explainer module
  - ❌ SHAP integration for individual predictions
  - ❌ Feature contribution extraction

- [ ] **7.2** Explanation templates ❌
  - ❌ Human-readable explanations
  - ❌ Template system for feature descriptions

- [ ] **7.3** Visualization ❌
  - ❌ SHAP waterfall plots
  - ❌ Taste profile radar charts
  - ❌ Add to API responses

---

## Phase 8: API Development ✅ COMPLETED

- [x] **8.1** FastAPI application (`client/app.py`) ✅
  - ✅ `GET /users` - List users
  - ✅ `GET /recommend/{user_id}` - Get recommendations
  - ✅ `GET /` - Demo HTML UI
  - ✅ FastAPI with static file serving

- [x] **8.2** Pydantic schemas (`client/services/`) ✅
  - ✅ Request/response models in service layer
  - ✅ Error handling with HTTPException

- [x] **8.3** Model loading ✅
  - ✅ Models loaded per request (optimization needed)
  - ✅ Recipe embeddings pre-computed

- [x] **8.4** API documentation ✅
  - ✅ OpenAPI/Swagger auto-generated
  - ✅ FastAPI docs available at `/docs`

---

## Phase 9: Web Application ✅ COMPLETED (HTML/FastAPI)

- [x] **9.1** Web UI (`client/templates/ui.html`) ✅
  - ✅ User selector dropdown
  - ✅ Number of recommendations slider
  - ✅ Display recommendations with images and scores
  - ✅ FastAPI backend serving HTML
  - ⚠️ Not Streamlit, but functional HTML UI

- [x] **9.2** Interactive visualizations ✅
  - ✅ Recipe images displayed
  - ✅ Latency display
  - ❌ Taste profile radar chart TODO
  - ❌ Feature importance TODO

- [ ] **9.3** Advanced features ⚠️ PARTIAL
  - ❌ Filter by difficulty/prep time
  - ❌ Recipe comparison
  - ❌ Explainability features
  - ✅ Real-time latency display

---

## Phase 10: Testing & Documentation ⚠️ PARTIAL

- [ ] **10.1** Unit tests (pytest) ❌
  - ❌ No test suite yet

- [ ] **10.2** Integration tests ❌
  - ❌ End-to-end tests TODO

- [ ] **10.3** Performance benchmarks ⚠️ PARTIAL
  - ✅ Latency tracking in API
  - ❌ Formal benchmark suite TODO

- [x] **10.4** Documentation ✅ GOOD
  - ✅ README files for dataset, retrieval, presentation
  - ✅ Role documentation
  - ✅ Copilot instructions
  - ⚠️ Deployment guide TODO

---

## Phase 11: Optional Experiments ❌ NOT STARTED

_Skipped for MVP - can be explored later_

---

## Phase 12: Production Readiness & Deployment 🚀 PRIORITY

### 12.1 Feature Storage & Caching

- [ ] **12.1.1** Implement SQL feature storage
  - ❌ Set up PostgreSQL/SQLite for feature caching
  - ❌ Schema for user features, recipe features, precomputed similarities
  - ❌ Migration scripts for existing CSV data
  - ❌ ORM models (SQLAlchemy) for features
  - **Why**: Faster feature access, production-ready persistence

- [ ] **12.1.2** Feature caching layer
  - ❌ Redis/in-memory cache for hot features
  - ❌ LRU cache for user embeddings
  - ❌ Cache invalidation strategy
  - **Why**: Sub-10ms feature retrieval

### 12.2 Chroma Integration for Inference

- [x] **12.2.1** Chroma store setup ✅
  - ✅ `db/chroma_store.py` - Chroma wrapper
  - ✅ `scripts/init_chroma.py` - Initialize collections
  - ✅ Recipe embeddings stored in Chroma

- [ ] **12.2.2** Use Chroma in inference pipeline ⚠️ TODO
  - ❌ Replace `.npy` loading with Chroma queries
  - ❌ ANN search for top-K candidates (faster than full scan)
  - ❌ Benchmark: Chroma vs NumPy for retrieval stage
  - **Why**: Scalable to 10K+ recipes, faster ANN search

- [ ] **12.2.3** Chroma metadata filtering
  - ❌ Add equipment constraints as Chroma metadata
  - ❌ Filter at retrieval stage (before ranking)
  - ❌ Use `where` clauses for equipment compatibility
  - **Why**: Reduce invalid candidates, faster pipeline

### 12.3 Improved Cold-Start Handling

- [ ] **12.3.1** Content-based cold-start encoder
  - ❌ MLP: user features → embedding space
  - ❌ Train on warm users (supervised: features → learned embeddings)
  - ❌ Seamless fallback for new users
  - **Why**: Current cold-start fails or uses poor heuristics

- [ ] **12.3.2** Hybrid cold-start strategy
  - ❌ For users with 1-5 interactions: blend collaborative + content
  - ❌ Weight decay from content→collaborative as history grows
  - ❌ Implement weighted ensemble in recommender
  - **Why**: Smooth transition from cold→warm users

- [ ] **12.3.3** Cold-start evaluation
  - ❌ Dedicated evaluation on `interactions_val_cold.csv`
  - ❌ Separate metrics for zero-history vs. few-shot users
  - ❌ Target: NDCG@5 > 0.25 for cold users

### 12.4 Google Cloud Run Deployment

- [ ] **12.4.1** Dockerize application
  - ❌ `Dockerfile` for FastAPI app
  - ❌ Multi-stage build (build + runtime)
  - ❌ Include model artifacts (embeddings, checkpoints)
  - ❌ Optimize image size (<500MB)
  - **Why**: Container required for Cloud Run

- [ ] **12.4.2** Cloud Run setup
  - ❌ Create Cloud Run service
  - ❌ Configure CPU/memory (2 vCPU, 4GB RAM)
  - ❌ Set concurrency and autoscaling
  - ❌ Environment variables for model paths
  - **Why**: Serverless deployment, auto-scaling

- [ ] **12.4.3** Model artifact storage
  - ❌ Upload models to Google Cloud Storage
  - ❌ Download on container startup (or bake into image)
  - ❌ Versioned model paths
  - **Why**: Separate code from large model files

- [ ] **12.4.4** Monitoring and logging
  - ❌ Cloud Logging integration
  - ❌ Latency and error metrics
  - ❌ Alerting on failures
  - **Why**: Production observability

- [ ] **12.4.5** Load testing
  - ❌ Locust/k6 load test script
  - ❌ Test autoscaling behavior
  - ❌ Verify <50ms p95 latency under load
  - **Why**: Validate production readiness

### 12.5 Model Serving Optimization

- [ ] **12.5.1** Model startup optimization
  - ❌ Load models once on container startup (not per request)
  - ❌ FastAPI lifespan events for initialization
  - ❌ Warm-up cache with popular users
  - **Why**: Currently loads model per request (slow)

- [ ] **12.5.2** ONNX inference integration
  - ❌ Use ONNX Runtime instead of PyTorch for retrieval
  - ❌ Quantization (INT8) for faster inference
  - ❌ Benchmark: ONNX vs PyTorch latency
  - **Why**: 2-3x faster inference

- [ ] **12.5.3** Batch inference API
  - ❌ Endpoint: `POST /recommend_batch` (multiple users)
  - ❌ Vectorized feature extraction
  - ❌ Batch LightGBM predictions
  - **Why**: Higher throughput for batch use cases

---

## Critical Path for Deployment

```
1. Fix cold-start (12.3.1) → Test on val_cold
2. Integrate Chroma for inference (12.2.2) → Benchmark
3. Dockerize (12.4.1) → Local testing
4. Deploy to Cloud Run (12.4.2) → Staging environment
5. Load test (12.4.5) → Optimize if needed
6. Feature storage (12.1.1) → Production migration
```

---

## Archived: Phase 11 Optional Experiments

**Alternative Models to Try** (in `experiments/` folder):

- [ ] **11.1** Neural Collaborative Filtering (NCF)
  - Implement GMF + MLP (NeuMF) architecture
  - Compare with Two-Tower retrieval

- [ ] **11.2** Sequential Models (SASRec)
  - Self-attention over user interaction sequences
  - Capture temporal preferences

- [ ] **11.3** Graph Neural Networks (LightGCN)
  - User-recipe bipartite graph
  - Graph convolutions for embeddings
  - Potentially SOTA on collaborative signal

- [ ] **11.4** RecBole Framework Baseline
  - Use RecBole library for quick prototyping
  - Compare built-in models (BPR, LightGCN, etc.)
  - Useful for benchmarking

- [ ] **11.5** XGBoost vs. LightGBM
  - Compare gradient boosting frameworks
  - CatBoost as alternative

- [ ] **11.6** Deep & Cross Network (DCN)
  - Replace LightGBM with neural ranker
  - Cross layers for feature interactions

**Experiment Tracking**: Use W&B to log all experiments with metrics, hyperparameters, and artifacts

---

## Milestone Checklist

| Milestone | Tasks | Target NDCG@5 | Deliverable |
|-----------|-------|---------------|-------------|
| M1: Baseline | 0.1-0.4, 1.1-1.2 | 0.20 | EDA notebook, data pipeline |
| M2: First Model | 1.3-1.5, 2.1, 3.1-3.5 | 0.35 | Two-tower model training |
| M3: Optimized | 2.2-2.4, 4.1-4.4 | 0.42 | Full model with cold-start |
| M4: Production | 5.1-5.5, 7.1-7.4 | 0.42 | API with <50ms inference |
| M5: Demo Ready | 6.1-6.3, 8.1-8.3 | 0.42+ | Web app with explanations |
| M6: Polish | 9.1-9.4, 10.1-10.4 | 0.42+ | Tested, documented, deployable |

---

## Key Design Decisions

1. **Two-Tower as primary architecture**: Enables pre-computation of item embeddings for fast inference
2. **Hybrid cold-start**: Content-based features mapped to embedding space for seamless handling
3. **Equipment constraints as hard filter**: Applied post-scoring to ensure valid recommendations
4. **Contrastive learning**: InfoNCE loss with in-batch negatives for efficient training
5. **ONNX for inference**: Platform-independent, optimized runtime for <50ms target

---

## Dependencies to Add

```toml
dependencies = [
    "torch>=2.0",
    "pandas>=2.0",
    "polars>=0.19",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "streamlit>=1.28",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "onnx>=1.14",
    "onnxruntime>=1.16",
    "wandb>=0.15",
    "tqdm>=4.66",
    "rootutils",
]
```
