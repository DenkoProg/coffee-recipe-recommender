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

## Phase 0: Project Setup & Data Analysis

- [ ] **0.1** Update `pyproject.toml` with ML dependencies
  - **Core ML**: torch, lightgbm, scikit-learn
  - **Data**: pandas, polars, numpy
  - **API/Web**: fastapi, uvicorn[standard], streamlit
  - **Experiment tracking**: wandb
  - **Optimization**: onnx, onnxruntime
  - **Optional frameworks**: recbole, microsoft-recommenders

- [ ] **0.2** Create project directory structure
  ```
  src/
  ├── coffee_recipe_recommender/
  │   ├── __init__.py
  │   ├── data/
  │   │   ├── __init__.py+ feature generation
  │   │   ├── preprocessing.py    # Feature engineering pipeline
  │   │   └── loaders.py          # Data loading utilities
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── retrieval.py        # Two-tower retrieval model
  │   │   ├── ranking.py          # LightGBM ranker
  │   │   ├── hybrid.py           # Combined pipeline
  │   │   └── layers.py           # Custom neural layers
  │   ├── training/
  │   │   ├── __init__.py
  │   │   ├── train_retrieval.py  # Two-tower training
  │   │   ├── train_ranker.py     # LightGBM training
  │   │   └── losses.py           # Custom loss functions
  │   ├── evaluation/
  │   │   ├── __init__.py
  │   │   ├── metrics.py          # NDCG, HR, MRR, Coverage
  │   │   └── evaluator.py        # Evaluation pipeline
  │   ├── inference/
  │   │   ├── __init__.py
  │   │   ├── recommender.py      # Main recommend() function
  │   │   └── explainer.py        # Feature importance & explanations
  │   ├── api/
  │   │   ├── __init__.py
  │   │   ├── main.py             # FastAPI application
  │   │   └── schemas.py          # Pydantic models
  │   └── config/
  │       ├── __init__.py
  │       └── settings.py         # Configuration
  ├── client/
  │   └── app.py                  # Streamlit web app
  ├── experiments/               # Optional: alternative approaches
  │   ├── ncf.py                 # Neural Collaborative Filtering
  │   ├── sasrec.py              # Sequential recommendations
  │   ├── lightgcn.py            # Graph Neural Networks
  │   └── recbole_baseline.py    # Using RecBole framework
  └── scripts/
      ├── train_retrieval.py     # Train retrieval stage
      ├── train_ranker.py        # Train ranking stage
      ├── evaluate.py            # Full evaluation
      └── serve.py               # Production servingscript
      └── export_model.py         # ONNX export script
  ```

- [ ] **0.3** Exploratory Data Analysis notebook
  - Load and profile all CSV files
  - Analyze interaction sparsity (user-item matrix density)
  - Visualize rating distributions
  - Analyze temporal patterns in interactions
  - Profile cold-start users from `cold_users.json`
  - Equipment and product distribution analysis
  - Taste preference clustering

- [ ] **0.4** Document data statistics and insights
  - Number of users, recipes, interactions
  - Sparsity level
  - Average interactions per user/recipe
  - Cold-start user percentage

---

## Phase 1: Data Pipeline & Feature Engineering

- [ ] **1.1** Implement data loading utilities (`src/coffee_recipe_recommender/data/loaders.py`)
  - Load CSV files with proper dtypes
  - Parse JSON columns (equipment, products, tags)
  - Handle missing values in ratings
  - Create user/recipe ID mappings (string → int indices)

- [ ] **1.2** Rich feature engineering for LightGBM (`src/coffee_recipe_recommender/data/preprocessing.py`)
  - **User features** (16+ features):
    - Taste preferences (4D vector: bitterness, sweetness, acidity, body)
    - Preferred strength, portion size (categorical)
    - Equipment count and diversity
    - Account age, interaction history stats
  - **Recipe features** (20+ features):
    - Taste profile (4D vector)
    - Strength, portion size, preparation time, difficulty
    - Equipment requirements, product complexity
    - Tag frequencies, popularity metrics
  - **Interaction features** (15+ features):
    - User-recipe taste similarity (cosine, L2 distance)
    - Equipment compatibility score
    - Recipe popularity among similar users
    - Temporal features (hour, day of week, recency)
    - User engagement patterns (avg rating, completion rate)
  - **Cross features**:
    - Taste preference × recipe profile interactions
    - Equipment overlap features

- [ ] **1.3** Negative sampling for retrieval model
  - Random negatives with 1:4 ratio
  - Hard negatives (popular but not interacted)
  - In-batch negatives for contrastive learning

- [ ] **1.4** Create datasets (`src/coffee_recipe_recommender/data/dataset.py`)
  - `RetrievalDataset`: (user, item, label) for Two-Tower
  - `RankingDataset`: Rich features for LightGBM
  - `ColdStartDataset`: Feature-only for new users
  - Efficient batching with DataLoader

- [ ] **1.5** Data validation
  - Schema validation
  - Train/val split verification
  - Feature distribution checks

---

## Phase 2: Retrieval Model (Two-Tower)

- [ ] **2.1** Implement User Tower (`src/coffee_recipe_recommender/models/retrieval.py`)
  ```python
  class UserTower(nn.Module):
      # Inputs: user_id embedding + user features (taste, equipment, etc.)
      # Architecture: Embedding layer + 3-layer MLP [256, 128, 64]
      # Output: 64-dim user embedding
      # Features: Dropout, BatchNorm, ReLU activations
  ```

- [ ] **2.2** Implement Recipe Tower
  ```python
  class RecipeTower(nn.Module):
      # Inputs: recipe_id embedding + recipe features
      # Architecture: Embedding layer + 3-layer MLP [256, 128, 64]
      # Output: 64-dim recipe embedding (same space as user)
  ```

- [ ] **2.3** Two-Tower model with contrastive loss
  - Temperature-scaled dot product similarity
  - InfoNCE loss with in-batch negatives
  - Optional: Add supervised head for rating prediction

- [ ] **2.4** Cold-start user encoder
  - MLP mapping user features → user embedding space
  - Train to match warm user embeddings from features
  - Seamless fallback during inference

## Phase 3: Ranking Model (LightGBM)

- [ ] **3.1** Generate training data for ranker
  - For each user: sample 100 candidates from retrieval model
  - Add true positives (interacted recipes)
  - Extract all 50+ features per (user, recipe) pair
  - Create LightGBM dataset with label (rating/clicked)

- [ ] **3.2** Train LightGBM ranker (`src/coffee_recipe_recommender/models/ranking.py`)
  ```python
  import lightgbm as lgb

  params = {
      'objective': 'lambdarank',  # or 'regression' for rating prediction
      'metric': 'ndcg',
      'ndcg_eval_at': [5],
      'num_leaves': 31,
      'learning_rate': 0.05,
      'feature_fraction': 0.8,
  }
  ```

- [ ] **3.3** Feature importance analysis
  - Extract SHAP values from LightGBM
  - Identify top features for recommendations
  - Use for explainability

- [ ] **3.4** Hyperparameter tuning
  - Optuna for LightGBM hyperparameter search
  - Cross-validation with NDCG@5
  - Early stopping on validation set

---

## Phase 4: Hybrid Pipeline Integration

- [ ] **4.1** Implement combined pipeline (`src/coffee_recipe_recommender/models/hybrid.py`)
  ```python
  class HybridRecommender:
      def __init__(self, retrieval_model, ranker_model):
          self.retrieval = retrieval_model  # Two-Tower
          self.ranker = ranker_model        # LightGBM

      def recommend(self, user_id, n=5, candidate_size=100):
          # 1. Retrieval: Get top-100 candidates
          # 2. Feature extraction for (user, candidate) pairs
          # 3. Ranking: LightGBM scores top-100
          # 4. Return top-n
  ```

- [ ] **4.2** Equipment constraint filtering
  - Apply after retrieval, before ranking (reduce LightGBM load)
  - Pre-compute equipment compatibility matrix
  - Bitwise operations for fast filtering

- [ ] **4.3** Training scripts
  - `scripts/train_retrieval.py`: Train Two-Tower model
  - `scripts/train_ranker.py`: Generate candidates → train LightGBM
  - Proper train/val splitting throughout

- [ ] **4.4** Model serialization
  - Save Two-Tower as ONNX for fast inference
  - Save LightGBM in native format
  - Save feature extractors and preprocessors

---

## Phase 5: Evaluation & Metrics

- [ ] **5.1** Implement metrics (`src/coffee_recipe_recommender/evaluation/metrics.py`)
  ```python
  def ndcg_at_k(relevances: List[float], k: int) -> float
  def hit_rate_at_k(recommendations, ground_truth, k: int) -> float
  def mrr(recommendations, ground_truth) -> float
  def coverage(all_recommendations, catalog_size) -> float
  ```

- [ ] **5.2** Full evaluation pipeline (`scripts/evaluate.py`)
  - Evaluate on `interactions_val.csv` (warm users)
  - Evaluate on `interactions_val_cold.csv` (cold users)
  - Separate metrics for retrieval vs. full pipeline
  - Generate detailed report (NDCG@5, HR@10, Coverage, etc.)

- [ ] **5.3** Baseline comparisons
  - Random baseline
  - Popularity baseline
  - Content-based (cosine similarity on taste features)
  - Pure collaborative filtering (user/item similarity)
  - Report all in comparison table

- [ ] **5.4** A/B testing simulation
  - Online metrics: CTR, conversion rate simulation
  - Diversity and novelty metrics
  - Temporal validation (time-based split)

---
6: Inference Optimization (<50ms target)

- [ ] **6.1** Implement main `recommend()` function (`src/coffee_recipe_recommender/inference/recommender.py`)
  ```python
  def recommend(
      user_id: str,
      users_df: pd.DataFrame,
      recipes_df: pd.DataFrame,
      train_df: pd.DataFrame,
      n: int = 5
  ) -> List[Tuple[str, float]]:
      # 1. Check if cold-start user
      # 2. Retrieval: Get 100 candidates (<10ms)
      # 3. Filter by equipment
      # 4. Extract features
      # 5. Ranking: LightGBM scores (<30ms)
      # 6. Return top-n
  ```

- [ ] **6.2** Pre-computation and caching
  - Pre-compute ALL recipe embeddings at startup (ONNX inference)
  - Build FAISS index for fast ANN search (optional, if >1000 recipes)
  - Cache user embeddings with LRU cache (TTL: 1 hour)
  - Pre-compute equipment compatibility matrix

- [ ] **6.3** ONNX export for retrieval model
  - Export User/Recipe towers to ONNX
  - Optimize with ONNX Runtime (graph optimization, quantization)
  - Benchmark: should be <5ms per user embedding

- [ ] **6.4** LightGBM optimization
  - Use native LightGBM inference (already fast)
  - Batch feature extraction (vectorized operations)
  - Optional: Convert to Treelite for faster prediction

- [ ] **6.5** Latency profiling
  - Instrument each stage with timing
  - Ident7: Explainability (Bonus)

- [ ] **7.1** LightGBM feature importance (`src/coffee_recipe_recommender/inference/explainer.py`)
  ```python
  def explain_recommendation(user_id: str, recipe_id: str) -> Dict:
      # 1. Get feature values for this (user, recipe) pair
      # 2. Compute SHAP values using LightGBM TreeExplainer
      # 3. Identify top contributing features
      # 4. Generate human-readable explanation
  ```

- [ ] **7.2** Explanation templates
  - "High taste match (0.92): You prefer bitter coffee, this recipe scores 0.9 on bitterness"
  - "Equipment compatible: You own all required equipment"
  - "Similar to your favorites: Users who liked X also enjoyed this"
  - "Quick preparation (3 min) matches your preference"

- [ ] **7.3** Visualization
  - SHAP waterfall plots for feature contributions
  - Taste profile comparison (user vs. recipe radar chart)
  - Add to API responses and web UIrecipe_005"],
          "top_features": ["bitterness_match", "quick_preparation"],
         8: API Development

- [ ] **8.1** FastAPI application (`src/coffee_recipe_recommender/api/main.py`)
  ```python
  @app.post("/recommend")
  async def get_recommendations(
      user_id: str,
      n: int = 5,
      include_explanations: bool = False,
      candidate_size: int = 100
  ) -> RecommendationResponse

  @app.get("/user/{user_id}")
  async def get_user_profile(user_id: str) -> UserProfile

  @app.get("/user/{user_id}/history")
  async def get_user_history(user_id: str, limit: int = 50) -> List[Interaction]

  @app.get("/recipe/{recipe_id}")
  async def get_recipe_details(recipe_id: str) -> Recipe

  @app.get("/health")
  async def health_check() -> Dict[str, str]

  @app.get("/metrics")
  async def get_metrics() -> Dict[str, float]  # Model performance stats
  ```

- [ ] **8.2** Pydantic schemas (`src/coffee_recipe_recommender/api/schemas.py`)
  - Request/response validation
  - Proper error handling

- [ ] **8.3** Model loading and warm-up
  - Load models on startup (lifespan event)
  - Pre-compute recipe embeddings
  - Warm-up cache with popular users

- [ ] **8.4** API documentation
  - OpenAPI/Swagger auto-docs
  - Example requests and curl command

- [ ] **7.3** Model loading and caching
  - Load model on startup
  - Warm up cache with popular users

- [ ] **7.4** API documentation
  - OpenAPI/Swagger auto-generated
  - Example requests/responses

---

## Phase 9: Web Application

- [ ] **9.1** Streamlit app (`client/app.py`)
  - User selector dropdown (with search)
  - Number of recommendations slider (1-20)
  - Display user profile: taste preferences, equipment, stats
  - Show interaction history with ratings and timestamps
  - Display recommendations with scores and explanations

- [ ] **9.2** Interactive visualizations
  - Taste profile radar chart (user vs. recipe comparison)
  - Rating distribution histogram
  - Equipment compatibility badges
  - Feature importance bar chart (from SHAP)

- [ ] **9.3** Advanced features
  - Filter recommendations by difficulty/preparation time
  - Compare multiple recipes side-by-side
  - "Why not recommended?" for specific recipes
  - Real-time latency display

---

## Phase 10: Testing & Documentation

- [ ] **10.1** Unit tests (pytest)
  - Data loading and preprocessing
  - Feature engineering correctness
  - Model forward pass (retrieval + ranking)
  - Metric calculations (NDCG, HR, etc.)
  - Equipment filtering logic

- [ ] **10.2** Integration tests
  - End-to-end recommendation pipeline
  - API endpoint tests (FastAPI TestClient)
  - Cold-start user handling

- [ ] **10.3** Performance benchmarks
  - Inference latency (target: <50ms)
  - Memory usage profiling
  - Throughput tests (requests/second)

- [ ] **10.4** Documentation
  - Update README with architecture diagram
  - Model card (training data, performance, limitations)
  - API usage examples
  - Deployment guide (Docker, requirements)

---

## Phase 11: Optional Experiments

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
