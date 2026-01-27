# Розподіл ролей та завдань
---

## 👨‍💻 Денис - Data Engineer & Retrieval Model Owner

### Зона відповідальності
**Data Pipeline + Two-Tower Retrieval Model**

### Основні завдання

#### Milestone 1: Data Pipeline
- ✅ **Phase 0**: Налаштування проєкту, EDA notebook
  - Аналіз даних, візуалізація, статистика
  - Документація insights
- ✅ **Phase 1**: Data Pipeline
  - `src/coffee_recipe_recommender/data/loaders.py` - завантаження CSV
  - `src/coffee_recipe_recommender/data/preprocessing.py` - feature engineering (50+ ознак!)
  - `src/coffee_recipe_recommender/data/dataset.py` - PyTorch datasets
  - Negative sampling стратегія

#### Milestone 2: Retrieval Model
- ✅ **Phase 2**: Two-Tower Neural Model
  - `src/coffee_recipe_recommender/models/retrieval.py` - User Tower + Recipe Tower
  - `src/coffee_recipe_recommender/models/layers.py` - MLP layers з dropout/batchnorm
  - `src/coffee_recipe_recommender/training/losses.py` - InfoNCE contrastive loss
  - Cold-start encoder для нових юзерів
- ✅ `scripts/train_retrieval.py` - тренування Two-Tower
- ✅ Збереження моделі для передачі Дмитру

#### Інтеграційні завдання (спільно)
- 🤝 Допомога Олександру з API endpoints для data loading
- 🤝 Передача trained retrieval model Дмитру для ranking pipeline

---

## 👨‍💻 Олександр - ML Engineer & Ranking Model Owner

### Зона відповідальності
**LightGBM Ranker + Hybrid Pipeline Integration**

### Основні завдання

#### Milestone 2: Ranking Model
- ✅ **Phase 3**: LightGBM Ranker
  - Чекати retrieval model від Дениса
  - Генерація training data (100 кандидатів per user)
  - `src/coffee_recipe_recommender/models/ranking.py` - LightGBM lambdarank
  - Feature importance extraction (SHAP values)
  - Hyperparameter tuning з Optuna
- ✅ `scripts/train_ranker.py` - тренування LightGBM

#### Milestone 3: Hybrid Pipeline
- ✅ **Phase 4**: Інтеграція Retrieval + Ranking
  - `src/coffee_recipe_recommender/models/hybrid.py` - HybridRecommender class
  - Equipment filtering logic (після retrieval, перед ranking)
  - Model serialization (ONNX для Two-Tower, native для LightGBM)
  - `scripts/train_ranker.py` - фінальний pipeline

#### Evaluation (спільно з Денисом)
- ✅ **Phase 5**: Metrics & Evaluation
  - `src/coffee_recipe_recommender/evaluation/metrics.py` - NDCG@5, HR@10, MRR
  - `src/coffee_recipe_recommender/evaluation/evaluator.py` - evaluation pipeline
  - `scripts/evaluate.py` - benchmark на validation sets
  - Baseline comparisons (random, popularity, content-based)

#### Optimization
- ✅ **Phase 6**: Inference Optimization
  - ONNX export для retrieval model
  - LightGBM inference optimization
  - Caching strategy (recipe embeddings, user embeddings)
  - Latency profiling (<50ms target)

---

## 👨‍💻 Дмитро - Full-Stack Developer & API/UI Owner

### Зона відповідальності
**FastAPI Backend + Streamlit Frontend + Explainability**

### Основні завдання

#### Milestone 3-4: API Development
- ✅ **Phase 7**: Explainability
  - `src/coffee_recipe_recommender/inference/explainer.py` - SHAP-based explanations
  - Explanation templates (human-readable)
  - Visualization utilities (radar charts, SHAP plots)

- ✅ **Phase 8**: FastAPI Backend
  - `src/coffee_recipe_recommender/api/main.py` - REST API endpoints
    - `POST /recommend` - головний endpoint
    - `GET /user/{user_id}` - user profile
    - `GET /user/{user_id}/history` - interaction history
    - `GET /recipe/{recipe_id}` - recipe details
    - `GET /health` - health check
    - `GET /metrics` - model performance stats
  - `src/coffee_recipe_recommender/api/schemas.py` - Pydantic models
  - Model loading на startup
  - OpenAPI documentation

#### Milestone 4-5: Web Application
- ✅ **Phase 9**: Streamlit Web App
  - `client/app.py` - інтерактивний UI
  - User selector + recommendations slider
  - Візуалізації:
    - Taste profile radar charts (user vs recipe)
    - Rating distributions
    - Equipment compatibility badges
    - SHAP feature importance
  - Advanced features:
    - Recipe filtering (difficulty, prep time)
    - Side-by-side comparison
    - Real-time latency display

#### Integration & Serving
- ✅ **Phase 6.1**: Inference API
  - `src/coffee_recipe_recommender/inference/recommender.py` - main `recommend()` function
  - Інтеграція з hybrid model від Дмитра
  - Cold-start handling
  - Error handling & validation

### Deliverables
1. **День 2-3**: Working FastAPI з усіма endpoints + документація
2. **День 4**: Streamlit demo app з візуалізаціями, готове до деплою (потенційно), презентація

---

## 🤝 Спільні завдання та синхронізація

### Phase 5: Evaluation (Денис + Дмитро)
- Metrics implementation
- Benchmark на validation sets (warm + cold users)
- Baseline comparisons
- A/B testing simulation

### Phase 6: Optimization (Дмитро + Олександр)
- ONNX export та integration в API
- Caching strategy
- Latency profiling
- Performance testing

### Phase 10: Testing & Documentation (Всі троє)
- Unit tests для своїх модулів
- Integration tests
- README updates
- Model card
- Docker setup

---

## 🔄 Workflow & Dependencies

### Критичний шлях (блокуючі залежності):
```
Денис: Data Pipeline → Two-Tower Model
           ↓
Дмитро: LightGBM Ranker → Hybrid Pipeline → Optimization
           ↓
Олександр: API Integration → Web App
```

### Паралельна робота (незалежні):
- Денис: EDA, feature engineering
- Дмитро: Архітектура ranking pipeline, підготовка training scripts
- Олександр: API schemas, UI mockups, explainability templates

### Точки синхронізації:
1. Денис передає data pipeline → Дмитро починає ranker, Олександр тестує data loading
2. Денис передає retrieval model → Дмитро інтегрує в hybrid pipeline
3. Дмитро передає hybrid pipeline → Олександр інтегрує в API
4. Code freeze → Спільне тестування та debugging
5. Фінальна презентація

---

## 🛠 Інструменти комунікації

### Git Strategy
- **Branches**:
  - `main` - stable production
  - `denys/data-pipeline` - Денис
  - `dmytro/ranking-model` - Дмитро
  - `oleksandr/api-webapp` - Олександр
- **Pull Requests**: Обов'язковий code review від іншого member
- **Merge**: До `main` після review, реліз тільки stable milestones
---

## 🎯 Success Criteria

### Individual KPIs:
- **Денис**:
  - ✅ Data pipeline без помилок
  - ✅ Retrieval NDCG@5 > 0.30
  - ✅ Feature engineering документація

- **Дмитро**:
  - ✅ Full pipeline NDCG@5 > 0.42
  - ✅ Inference latency < 50ms
  - ✅ Evaluation report з baselines

- **Олександр**:
  - ✅ Working API (всі endpoints)
  - ✅ Streamlit demo з візуалізаціями
  - ✅ Deployment-ready (Docker)

### Team KPIs:
- ✅ NDCG@5 > 0.4 на validation
- ✅ Холодний старт handled
- ✅ Пояснення (explainability)
- ✅ Working demo
- ✅ < 50ms inference
- ✅ Повна документація
