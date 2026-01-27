# Coffee Recipe Recommendation System - Implementation Tasks

> **Goal**: Build a state-of-the-art deep learning recommendation system achieving NDCG@5 > 0.4
> **Stack**: Python 3.11+, PyTorch, FastAPI, Streamlit
> **Timeline**: Incremental milestones with testable deliverables

---

## Phase 0: Project Setup & Data Analysis

- [ ] **0.1** Update `pyproject.toml` with ML dependencies
  - PyTorch, pandas, numpy, scikit-learn
  - FastAPI, uvicorn, streamlit
  - polars (for fast data processing)
  - wandb or mlflow (experiment tracking)
  - onnx, onnxruntime (inference optimization)

- [ ] **0.2** Create project directory structure
  ```
  src/
  ├── coffee_rec/
  │   ├── __init__.py
  │   ├── data/
  │   │   ├── __init__.py
  │   │   ├── dataset.py          # PyTorch Dataset classes
  │   │   ├── preprocessing.py    # Feature engineering
  │   │   └── loaders.py          # Data loading utilities
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── base.py             # Abstract recommender base
  │   │   ├── two_tower.py        # Two-tower retrieval model
  │   │   ├── ncf.py              # Neural Collaborative Filtering
  │   │   ├── sasrec.py           # Self-Attentive Sequential Rec
  │   │   └── layers.py           # Custom layers (MLP, attention)
  │   ├── training/
  │   │   ├── __init__.py
  │   │   ├── trainer.py          # Training loop
  │   │   ├── losses.py           # Custom loss functions
  │   │   └── callbacks.py        # Early stopping, checkpointing
  │   ├── evaluation/
  │   │   ├── __init__.py
  │   │   ├── metrics.py          # NDCG, HR, MRR implementations
  │   │   └── evaluator.py        # Evaluation pipeline
  │   ├── inference/
  │   │   ├── __init__.py
  │   │   ├── recommender.py      # Main recommend() function
  │   │   └── serving.py          # ONNX export, optimization
  │   ├── api/
  │   │   ├── __init__.py
  │   │   ├── main.py             # FastAPI application
  │   │   └── schemas.py          # Pydantic models
  │   └── config/
  │       ├── __init__.py
  │       └── settings.py         # Pydantic settings
  ├── client/
  │   └── app.py                  # Streamlit application
  └── scripts/
      ├── train.py                # Training entrypoint
      ├── evaluate.py             # Evaluation script
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

- [ ] **1.1** Implement data loading utilities (`src/coffee_rec/data/loaders.py`)
  - Load CSV files with proper dtypes
  - Parse JSON columns (equipment, products, tags)
  - Handle missing values in ratings
  - Create user/recipe ID mappings (string → int indices)

- [ ] **1.2** Feature engineering (`src/coffee_rec/data/preprocessing.py`)
  - **User features**:
    - Taste preferences (4D vector: bitterness, sweetness, acidity, body)
    - Preferred strength (categorical → embedding or one-hot)
    - Preferred portion size (categorical)
    - Equipment count and types (multi-hot encoding)
    - Account age
  - **Recipe features**:
    - Taste profile (4D vector)
    - Strength, portion size, preparation time, difficulty
    - Equipment requirements (multi-hot)
    - Tags (multi-hot or TF-IDF)
  - **Interaction features**:
    - Rating (when available)
    - Completion status
    - Time-based features (hour, day of week)
    - Interaction recency

- [ ] **1.3** Implement negative sampling strategy
  - Random negative sampling with configurable ratio (1:4 recommended)
  - Hard negative mining (recipes with similar features but no interaction)
  - Ensure negatives respect equipment constraints

- [ ] **1.4** Create PyTorch Dataset classes (`src/coffee_rec/data/dataset.py`)
  - `InteractionDataset`: For training with (user, item, label) triplets
  - `SequentialDataset`: For sequence-aware models
  - `ColdStartDataset`: Feature-only dataset for cold users
  - Implement efficient batching with DataLoader

- [ ] **1.5** Build data validation pipeline
  - Schema validation for loaded data
  - Train/val split verification (no leakage)
  - Equipment constraint validation

---

## Phase 2: Model Architecture (SOTA Deep Learning)

### 2.1 Two-Tower Retrieval Model (Primary)

- [ ] **2.1.1** Implement User Tower (`src/coffee_rec/models/two_tower.py`)
  ```python
  class UserTower(nn.Module):
      # Inputs: user_id (embedding), taste_prefs, equipment, strength, etc.
      # Architecture: Embedding + MLP with residual connections
      # Output: 128-dim user representation
  ```

- [ ] **2.1.2** Implement Recipe Tower
  ```python
  class RecipeTower(nn.Module):
      # Inputs: recipe_id (embedding), taste_profile, equipment, tags
      # Architecture: Embedding + MLP with residual connections
      # Output: 128-dim recipe representation
  ```

- [ ] **2.1.3** Two-Tower model with temperature-scaled dot product
  - Learnable temperature parameter
  - Contrastive loss (InfoNCE / NT-Xent)
  - In-batch negatives for efficient training

### 2.2 Neural Collaborative Filtering (NCF)

- [ ] **2.2.1** Implement GMF (Generalized Matrix Factorization)
  - User/item embeddings with element-wise product

- [ ] **2.2.2** Implement MLP pathway
  - Concatenated embeddings through deep MLP

- [ ] **2.2.3** Combine into NeuMF
  - Fusion of GMF and MLP pathways
  - Pre-training strategy for embeddings

### 2.3 Sequential Recommendation (SASRec)

- [ ] **2.3.1** Implement self-attention layers
  - Multi-head self-attention for sequence modeling
  - Positional embeddings (learnable)
  - Causal masking for autoregressive prediction

- [ ] **2.3.2** Sequence feature extraction
  - User interaction history as sequence
  - Include timestamps for time-aware attention

### 2.4 Cold-Start Handler

- [ ] **2.4.1** Content-based fallback model
  - MLP mapping user features → embedding space
  - Train to predict warm user embeddings from features

- [ ] **2.4.2** Hybrid switching logic
  - Detect cold-start users
  - Route to content-based model when needed
  - Smooth transition as user gains interactions

---

## Phase 3: Training Pipeline

- [ ] **3.1** Implement loss functions (`src/coffee_rec/training/losses.py`)
  - Binary Cross-Entropy for implicit feedback
  - BPR (Bayesian Personalized Ranking) loss
  - InfoNCE contrastive loss for two-tower
  - MSE for explicit ratings (optional auxiliary task)

- [ ] **3.2** Implement trainer class (`src/coffee_rec/training/trainer.py`)
  - Mixed precision training (AMP)
  - Gradient accumulation
  - Learning rate scheduling (cosine annealing with warmup)
  - Early stopping based on validation NDCG

- [ ] **3.3** Implement callbacks (`src/coffee_rec/training/callbacks.py`)
  - Model checkpointing (best + last)
  - Logging to W&B/MLflow
  - Learning rate logging

- [ ] **3.4** Hyperparameter configuration
  - Use Hydra or simple YAML configs
  - Key hyperparameters:
    - Embedding dimension: 64-256
    - MLP hidden dims: [256, 128, 64]
    - Learning rate: 1e-4 to 1e-3
    - Batch size: 512-2048
    - Negative sampling ratio: 4
    - Dropout: 0.1-0.3
    - Temperature: 0.05-0.1

- [ ] **3.5** Training script (`scripts/train.py`)
  - CLI with argument parsing
  - Reproducibility (seed setting)
  - Multi-GPU support (optional)

---

## Phase 4: Evaluation & Metrics

- [ ] **4.1** Implement metrics (`src/coffee_rec/evaluation/metrics.py`)
  ```python
  def ndcg_at_k(relevances: List[float], k: int) -> float
  def hit_rate_at_k(recommendations: List, ground_truth: Set, k: int) -> float
  def mrr(recommendations: List, ground_truth: Set) -> float
  def coverage(recommendations: List[List], catalog_size: int) -> float
  ```

- [ ] **4.2** Implement evaluator (`src/coffee_rec/evaluation/evaluator.py`)
  - Batch evaluation for efficiency
  - Separate metrics for warm vs cold users
  - Equipment constraint validation in evaluation

- [ ] **4.3** Create evaluation script (`scripts/evaluate.py`)
  - Load trained model
  - Evaluate on `interactions_val.csv` (warm users)
  - Evaluate on `interactions_val_cold.csv` (cold users)
  - Generate detailed report with per-user breakdown

- [ ] **4.4** Baseline comparisons
  - Random baseline
  - Popularity baseline
  - Content-based (cosine similarity on taste)
  - Collaborative filtering (ALS/SVD)

---

## Phase 5: Inference Optimization (<50ms target)

- [ ] **5.1** Implement main `recommend()` function (`src/coffee_rec/inference/recommender.py`)
  ```python
  def recommend(
      user_id: str,
      users_df: pd.DataFrame,
      recipes_df: pd.DataFrame,
      train_df: pd.DataFrame,
      n: int = 5
  ) -> List[Tuple[str, float]]:
      # 1. Load/cache user features
      # 2. Get user embedding (or compute for cold-start)
      # 3. Score all valid recipes (equipment filter)
      # 4. Return top-n with scores
  ```

- [ ] **5.2** Pre-compute and cache embeddings
  - Pre-compute all recipe embeddings at startup
  - Cache user embeddings with LRU cache
  - Use FAISS for approximate nearest neighbor (optional)

- [ ] **5.3** ONNX export (`src/coffee_rec/inference/serving.py`)
  - Export PyTorch models to ONNX
  - Optimize with ONNX Runtime
  - Benchmark inference time

- [ ] **5.4** Equipment filtering optimization
  - Pre-compute equipment compatibility matrix
  - Use bitwise operations for fast filtering

- [ ] **5.5** Batch inference support
  - Vectorized scoring for multiple users
  - GPU inference option

---

## Phase 6: Explainability (Bonus)

- [ ] **6.1** Feature contribution analysis
  - SHAP values for feature importance
  - Attention weights visualization (for SASRec)

- [ ] **6.2** Explanation generation
  ```python
  def explain_recommendation(user_id: str, recipe_id: str) -> Dict:
      return {
          "taste_match": 0.85,  # Cosine similarity
          "equipment_compatible": True,
          "similar_to_liked": ["recipe_001", "recipe_005"],
          "top_features": ["bitterness_match", "quick_preparation"],
          "explanation_text": "Recommended because you enjoy bitter coffee..."
      }
  ```

- [ ] **6.3** Add explanations to API response

---

## Phase 7: API Development

- [ ] **7.1** FastAPI application (`src/coffee_rec/api/main.py`)
  ```python
  @app.post("/recommend")
  async def get_recommendations(
      user_id: str,
      n: int = 5,
      include_explanations: bool = False
  ) -> RecommendationResponse

  @app.get("/user/{user_id}")
  async def get_user_profile(user_id: str) -> UserProfile

  @app.get("/user/{user_id}/history")
  async def get_user_history(user_id: str) -> List[Interaction]

  @app.get("/recipe/{recipe_id}")
  async def get_recipe(recipe_id: str) -> Recipe

  @app.get("/health")
  async def health_check() -> HealthStatus
  ```

- [ ] **7.2** Pydantic schemas (`src/coffee_rec/api/schemas.py`)
  - Request/response models
  - Validation

- [ ] **7.3** Model loading and caching
  - Load model on startup
  - Warm up cache with popular users

- [ ] **7.4** API documentation
  - OpenAPI/Swagger auto-generated
  - Example requests/responses

---

## Phase 8: Web Application

- [ ] **8.1** Streamlit app (`client/app.py`)
  - User selector dropdown
  - Number of recommendations slider
  - Display user profile and preferences
  - Show interaction history with ratings
  - Display recommendations with scores
  - Recipe detail cards with taste profiles

- [ ] **8.2** Visualization components
  - Taste profile radar charts
  - Rating distribution histograms
  - Equipment compatibility indicators

- [ ] **8.3** Interactive features
  - Filter recommendations by difficulty
  - Sort by different criteria
  - Compare multiple recipes

---

## Phase 9: Testing & Quality

- [ ] **9.1** Unit tests
  - Data loading and preprocessing
  - Model forward pass
  - Metric calculations
  - Equipment filtering logic

- [ ] **9.2** Integration tests
  - End-to-end recommendation pipeline
  - API endpoint tests
  - Cold-start handling

- [ ] **9.3** Performance tests
  - Inference latency benchmarks
  - Memory usage profiling
  - Load testing for API

- [ ] **9.4** Code quality
  - Type hints throughout
  - Docstrings for public APIs
  - Pre-commit hooks (ruff, mypy)

---

## Phase 10: Documentation & Deployment

- [ ] **10.1** Update README.md
  - Project overview
  - Installation instructions
  - Quick start guide
  - Model architecture description

- [ ] **10.2** Model card
  - Training data description
  - Performance metrics
  - Limitations and biases
  - Intended use cases

- [ ] **10.3** Docker setup
  - Dockerfile for API
  - docker-compose for full stack
  - Environment configuration

- [ ] **10.4** Demo preparation
  - Sample queries
  - Performance benchmarks
  - Comparison with baselines

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
