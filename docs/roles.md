# Розподіл ролей та завдань
---

## 👨‍💻 Денис - Data Engineer & Retrieval Model Owner

### Зона відповідальності
**Data Pipeline + Two-Tower Retrieval Model + Vector DB**

### ✅ Завершені завдання (MVP)

#### Milestone 1: Data Pipeline ✅
- ✅ **Phase 0**: Налаштування проєкту, EDA
  - Проєкт структура, dependencies (pyproject.toml)
  - Notebooks для аналізу embeddings
  - Chroma exploration
- ✅ **Phase 1**: Data Pipeline
  - `training/loaders.py` - CSV loading, ID mappings ✅
  - `preprocessing/preprocessing.py` - 50+ features ✅
  - `training/datasets.py` - PyTorch datasets (Retrieval, RetrievalWithFeatures) ✅
  - Negative sampling (in-batch, random candidates) ✅

#### Milestone 2: Retrieval Model ✅
- ✅ **Phase 2**: Two-Tower Neural Model
  - `models/retrieval.py` - UserTower + RecipeTower ✅
  - InfoNCE + Symmetric InfoNCE losses (`training/losses.py`) ✅
  - Multiple training runs (baseline, with features) ✅
  - Recipe embeddings pre-computation ✅
- ✅ `scripts/train_retrieval.py` - повний training pipeline ✅
- ✅ `scripts/evaluate_retrieval.py` - метрики (NDCG, HR, MRR) ✅
- ✅ `db/chroma_store.py` - Chroma wrapper ✅
- ✅ `scripts/init_chroma.py` - Chroma initialization ✅

### 🚀 Майбутні завдання (Production)

#### Priority 1: Improved Cold-Start ⚠️
- [ ] **Content-based encoder for new users**
  - MLP: user features (taste, equipment) → 64-dim embedding
  - Train supervised: features of warm users → their learned embeddings
  - Add to `models/retrieval.py` as `ColdStartEncoder`
  - Integrate seamless fallback in `inference/recommender.py`
  - **Goal**: NDCG@5 > 0.25 on `interactions_val_cold.csv`

#### Priority 2: Chroma Integration for Inference ⚠️
- [ ] **Replace .npy with Chroma queries**
  - Modify `inference/recommender.py` to use Chroma instead of NumPy
  - ANN search for top-K candidates (faster than full cosine scan)
  - Add equipment metadata filtering in Chroma
  - Benchmark: Chroma vs NumPy latency
  - **Goal**: <5ms retrieval stage, scalable to 10K+ recipes

#### Priority 3: Feature Storage in SQL 💾
- [ ] **PostgreSQL/SQLite schema for features**
  - Design schema: `user_features`, `recipe_features`, `interaction_stats`
  - SQLAlchemy ORM models
  - Migration script: CSV → SQL
  - Update `preprocessing/preprocessing.py` to read from SQL
  - **Goal**: Production-ready feature store, faster access

#### Supporting: Training Improvements
- [ ] Experiment with harder negative sampling
- [ ] Ablation study: features vs no-features
- [ ] Hyperparameter tuning for Two-Tower

---

## 👨‍💻 Дмитро - ML Engineer & Ranking Model Owner

### Зона відповідальності
**LightGBM Ranker + Hybrid Pipeline + Optimization**

### ✅ Завершені завдання (MVP)

#### Milestone 2: Ranking Model ✅
- ✅ **Phase 3**: LightGBM Ranker
  - Trained retrieval model від Дениса → candidate generation ✅
  - `models/ranking.py` - LightGBM lambdarank ✅
  - `scripts/train_ranker.py` - training + Optuna hyperparameter tuning ✅
  - SHAP feature importance visualization ✅
  - Best params: n_estimators=173, lr=0.1 ✅

#### Milestone 3: Hybrid Pipeline ✅
- ✅ **Phase 4**: Integration
  - `models/hybrid.py` - HybridRecommenderModel ✅
  - `inference/recommender.py` - Generic Recommender wrapper ✅
  - Equipment filtering in features ✅
  - Model serialization (.pt, .npy, .pkl) ✅

#### Evaluation ✅
- ✅ **Phase 5**: Metrics
  - `evaluation/metrics.py` - NDCG, HR, MRR, Coverage ✅
  - Retrieval evaluation pipeline ✅
  - Handcrafted baseline (`scripts/create_simple_handcrafted_embeddings.py`) ✅

#### Optimization ✅
- ✅ **Phase 6**: Inference
  - ONNX export script (`scripts/export_onnx.py`) ✅
  - Recipe embeddings pre-computation ✅
  - Latency tracking in API ✅

### 🚀 Майбутні завдання (Production)

#### Priority 1: End-to-End Hybrid Evaluation 📊ufe0f
- [ ] **Full pipeline benchmarks**
  - Evaluate hybrid model on `interactions_val.csv`
  - Separate retrieval vs ranking contribution
  - Ablation study: retrieval-only vs hybrid
  - Compare with baselines (popularity, random)
  - **Goal**: Document NDCG@5 improvements

#### Priority 2: Feature Engineering Improvements 🛠
- [ ] **SQL-based feature generation**
  - Migrate `preprocessing/preprocessing.py` to use SQL
  - Add caching for expensive features (user history stats)
  - Pre-compute similarity features offline
  - **Goal**: <20ms feature extraction per user

#### Priority 3: Model Serving Optimization ⚡
- [ ] **ONNX Runtime integration**
  - Replace PyTorch with ONNX for retrieval inference
  - Benchmark latency improvements
  - INT8 quantization experiments
  - **Goal**: <3ms per user embedding

- [ ] **LightGBM optimization**
  - Treelite compilation (optional)
  - Batch prediction for multiple users
  - Profile feature extraction bottlenecks

#### Supporting: Experiments
- [ ] Try XGBoost/CatBoost instead of LightGBM
- [ ] Neural ranker (DCN, DeepFM) comparison
- [ ] A/B testing simulation framework

---

## 👨‍💻 Олександр - Full-Stack Developer & API/UI Owner

### Зона відповідальності
**FastAPI Backend + Web UI + Deployment**

### ✅ Завершені завдання (MVP)

#### Milestone 3-4: API Development ✅
- ✅ **Phase 8**: FastAPI Backend
  - `client/app.py` - FastAPI app ✅
    - `GET /users` - list users ✅
    - `GET /recommend/{user_id}` - recommendations with latency tracking ✅
    - `GET /` - HTML demo page ✅
    - Static file serving for images ✅
  - `client/services/` - Service layer (recommend_service, users_service) ✅
  - Pydantic models for validation ✅
  - Error handling (HTTPException) ✅
  - OpenAPI docs at `/docs` ✅

#### Milestone 4-5: Web Application ✅
- ✅ **Phase 9**: HTML/JS UI
  - `client/templates/ui.html` - Interactive demo ✅
  - User dropdown selector ✅
  - Recommendations slider (1-50) ✅
  - Recipe cards with images ✅
  - Latency display ✅
  - **Note**: HTML/FastAPI instead of Streamlit

#### Integration ✅
- ✅ **Phase 6.1**: Inference API
  - Generic `Recommender` class integration ✅
  - Hybrid model loading from checkpoints ✅
  - Per-request model initialization (needs optimization) ✅

### 🚀 Майбутні завдання (Production)

#### Priority 1: Google Cloud Run Deployment ☁️
- [ ] **Dockerize FastAPI app**
  - Write `Dockerfile` (multi-stage build)
  - Include model artifacts (or download from GCS)
  - Optimize image size (<500MB)
  - Test locally with `docker run`
  - **Deliverable**: Working Docker image

- [ ] **Deploy to Cloud Run**
  - Set up Cloud Run service (GCP console or gcloud CLI)
  - Configure: 2 vCPU, 4GB RAM, concurrency=10
  - Upload models to Google Cloud Storage
  - Environment variables for model paths
  - HTTPS endpoint with custom domain (optional)
  - **Deliverable**: Live production URL

- [ ] **CI/CD pipeline**
  - GitHub Actions: build → push to GCR → deploy to Cloud Run
  - Automated deployment on merge to `main`
  - Rollback strategy

- [ ] **Monitoring & Logging**
  - Cloud Logging integration
  - Structured logging (JSON format)
  - Latency metrics, error rates
  - Alerting on failures (Email/Slack)
  - **Goal**: Observability for production

#### Priority 2: API Optimization ⚡
- [ ] **Model loading refactor**
  - Load models ONCE on startup (FastAPI lifespan events)
  - Global singleton for `Recommender`
  - Warm-up cache with popular users
  - **Goal**: Eliminate per-request model loading

- [ ] **Async endpoints**
  - Make `recommend` endpoint async
  - Use `asyncio` for concurrent user requests
  - Batch inference for multiple users
  - **Goal**: Higher throughput

- [ ] **Additional endpoints**
  - `GET /user/{user_id}/profile` - user details
  - `GET /user/{user_id}/history` - interaction history
  - `GET /recipe/{recipe_id}` - recipe details
  - `POST /recommend_batch` - batch recommendations
  - `GET /health` - health check for Cloud Run
  - `GET /metrics` - Prometheus-style metrics

#### Priority 3: UI Improvements 🎨
- [ ] **Explainability features**
  - Show top features for each recommendation
  - SHAP waterfall plot (via endpoint)
  - "Why this recipe?" tooltip

- [ ] **Taste profile visualization**
  - Radar chart: user taste vs recipe taste
  - Use Chart.js or similar

- [ ] **Advanced filters**
  - Filter by equipment, difficulty, prep time
  - Multi-select equipment filter
  - Difficulty slider

- [ ] **Alternative: Migrate to Streamlit** (optional)
  - If HTML becomes too complex
  - Streamlit has built-in charts, faster prototyping

#### Supporting: Load Testing
- [ ] **Locust/k6 load test**
  - Simulate 100 concurrent users
  - Measure p50, p95, p99 latency
  - Test autoscaling on Cloud Run
  - **Goal**: Validate <50ms p95 latency

---

## 🤝 Спільні завдання та синхронізація

### ✅ MVP Completed (Phases 0-9)
- ✅ Data pipeline, feature engineering
- ✅ Two-Tower retrieval model
- ✅ LightGBM ranker
- ✅ Hybrid pipeline
- ✅ FastAPI + HTML UI
- ✅ Basic evaluation (retrieval-only)

### 🚀 Production Priorities (Phase 12)

#### Critical Path:
```
1. Денис: Cold-start encoder (12.3.1) → Evaluate on val_cold
2. Денис: Chroma inference (12.2.2) → Benchmark vs NumPy
3. Олександр: Docker + Cloud Run (12.4.1-12.4.2) → Deploy staging
4. Олександр: Load testing (12.4.5) → Validate latency
5. Дмитро: Full hybrid evaluation (12.3.3) → Report metrics
6. Денис: SQL feature store (12.1.1) → Production migration
```

#### Parallel Work:
- **Денис**: Cold-start encoder + Chroma integration
- **Дмитро**: Hybrid evaluation + ONNX optimization + Feature engineering
- **Олександр**: Dockerization + Cloud Run + API optimization

#### Weekly Sync Points:
1. **Week 1**: Cold-start working + Docker ready
2. **Week 2**: Cloud Run staging deployed + Chroma integrated
3. **Week 3**: SQL features + Load testing passed
4. **Week 4**: Production deployment + monitoring

### Phase 7: Explainability (All Together)
- ❌ Postponed to post-deployment
- Low priority vs cold-start + deployment

### Phase 10: Testing (All Together)
- [ ] Unit tests (pytest) - each member for their modules
- [ ] Integration tests - Олександр (API-level)
- [ ] Performance benchmarks - Дмитро (model) + Олександр (API)

---

## 🔄 Workflow & Dependencies

### ✅ MVP Critical Path (Completed):
```
Денис: Data Pipeline → Two-Tower Model ✅
           ↓
Дмитро: LightGBM Ranker → Hybrid Pipeline ✅
           ↓
Олександр: API Integration → Web App ✅
```

### 🚀 Production Critical Path:
```
Денис: Cold-start encoder (blocking hybrid eval)
           ↓
Дмитро: Full hybrid evaluation (blocking deployment decision)
           ↓
Олександр: Cloud Run deployment (blocking production)
           ↓
Денис: SQL features (post-deployment optimization)
```

### Паралельна робота:
- **Денис**: Chroma integration (independent)
- **Дмитро**: ONNX optimization (independent)
- **Олександр**: Docker + API optimization (independent)

### Точки синхронізації:
1. ✅ **MVP Done**: Data + Models + API working
2. ⚠️ **Cold-start ready**: Денис → Дмитро can evaluate
3. ⚠️ **Evaluation complete**: Дмитро → Олександр can deploy
4. ⚠️ **Staging deployed**: Олександр → Team load tests
5. ⚠️ **Production live**: All → Monitoring + iteration

---

## 🛠 Інструменти комунікації

### Git Strategy
- **Branches**:
  - `main` - stable production (MVP done, deploy from here)
  - `denys/cold-start` - Cold-start encoder work
  - `denys/chroma-inference` - Chroma integration
  - `dmytro/hybrid-eval` - Full evaluation pipeline
  - `dmytro/onnx-optimize` - ONNX optimization
  - `oleksandr/docker-deploy` - Dockerization + Cloud Run
  - `oleksandr/api-optimize` - API improvements
- **Pull Requests**: Code review від іншого member
- **Merge**: До `main` після review + CI passing

### Project Board (GitHub Projects)
- **Backlog**: Phase 12 tasks
- **In Progress**: Current sprint tasks
- **Done**: Completed MVP phases

### Weekly Standups (Async)
- What did I complete?
- What am I working on?
- Any blockers?
- ETA for current task
---

## 🎯 Success Criteria

### ✅ MVP KPIs (Completed):
- ✅ Data pipeline без помилок
- ✅ Two-Tower retrieval working (NDCG@5 ~0.30)
- ✅ LightGBM ranker trained
- ✅ Hybrid pipeline functional
- ✅ FastAPI + HTML UI
- ✅ Latency <100ms (per-request model loading)

### 🚀 Production KPIs:

**Денис (Cold-Start + Chroma + Features)**:
- [ ] Cold-start NDCG@5 > 0.25 on `interactions_val_cold.csv`
- [ ] Chroma integration: <5ms retrieval latency
- [ ] SQL feature store: <20ms feature extraction
- [ ] Documentation: Cold-start model card, Chroma setup guide

**Дмитро (Evaluation + Optimization)**:
- [ ] Full hybrid NDCG@5 > 0.40 on `interactions_val.csv`
- [ ] Ablation study: retrieval vs hybrid improvement
- [ ] ONNX latency: <3ms per user embedding
- [ ] Feature importance report (top 10 features)

**Олександр (Deployment + API)**:
- [ ] Docker image built (<500MB)
- [ ] Cloud Run deployed (staging + production)
- [ ] API latency: <50ms p95 under load
- [ ] Load test: 100 concurrent users, no errors
- [ ] Monitoring: Cloud Logging + alerting setup
- [ ] Uptime: 99%+ after 1 week

### Team KPIs:
- [ ] 🎯 **Cold-start NDCG@5 > 0.25**
- [ ] 🎯 **Warm-user NDCG@5 > 0.40**
- [ ] 🎯 **Inference latency < 50ms p95**
- [ ] 🎯 **Production deployed on Cloud Run**
- [ ] 🎯 **SQL feature store operational**
- [ ] 🎯 **Chroma for ANN search**
- [ ] Monitoring and alerting
- [ ] Load test passed (100 users)
