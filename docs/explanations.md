# Теоретичні основи та пайплайн рекомендаційної системи кавових рецептів

## Зміст

1. [Вступ та постановка задачі](#1-вступ-та-постановка-задачі)
2. [Архітектура системи](#2-архітектура-системи)
3. [Етап 1: Retrieval (Пошук кандидатів)](#3-етап-1-retrieval-пошук-кандидатів)
4. [Етап 2: Ranking (Ранжування)](#4-етап-2-ranking-ранжування)
5. [Feature Engineering (Інженерія ознак)](#5-feature-engineering-інженерія-ознак)
6. [Метрики оцінки](#6-метрики-оцінки)
7. [Cold-Start проблема](#7-cold-start-проблема)
8. [Повний пайплайн](#8-повний-пайплайн)
9. [Оптимізація для продакшену](#9-оптимізація-для-продакшену)
10. [Explainability (Пояснюваність)](#10-explainability-пояснюваність)

---

## 1. Вступ та постановка задачі

### Мета проекту

Побудувати **production-ready гібридну рекомендаційну систему** для кавових рецептів, яка:
- Досягає **NDCG@5 > 0.4** на валідаційних даних
- Забезпечує **латентність < 50мс** при інференсі
- Працює як для "теплих" користувачів (з історією), так і для "холодних" (без історії)

### Особливості домену кави

Рекомендація кавових рецептів має унікальні виклики:

1. **Обмеження обладнання**: Не можна рекомендувати рецепт еспресо користувачу без еспресо-машини
2. **Смакові профілі**: 4D вектори смаку (гіркота, солодкість, кислотність, тіло) — ключові для персоналізації
3. **Повторюваність**: На відміну від фільмів, користувачі готують улюблені рецепти багаторазово
4. **Cold-start**: ~300 користувачів мають нуль взаємодій у тренувальних даних

### Дані

| Файл | Опис |
|------|------|
| `recipes.csv` | Каталог рецептів: смак, складність, обладнання, теги |
| `users.csv` | Профілі користувачів: обладнання, смакові вподобання |
| `interactions_train.csv` | Тренувальні взаємодії (user_id, recipe_id, rating, completed) |
| `interactions_val.csv` | Валідація для "теплих" користувачів |
| `interactions_val_cold.csv` | Валідація для "холодних" користувачів |

---

## 2. Архітектура системи

### Чому гібридний двоетапний підхід?

Ми використовуємо **Two-Stage Retrieval + Ranking** архітектуру — індустріальний стандарт у Google, Meta, Netflix:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🔄 ГІБРИДНА РЕКОМЕНДАЦІЙНА СИСТЕМА                        ║
║                        Двоетапна архітектура                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                              ┌───────┴───────┐
                              │   user_id     │
                              └───────┬───────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ╔════════════════════════════════════════════════════════════════════════╗  │
│  ║  📡 ЕТАП 1: RETRIEVAL                                    ⏱️ ~21мс      ║  │
│  ║  ══════════════════════════════════════════════════════════════════════║  │
│  ║                                                                        ║  │
│  ║    ┌─────────────────┐           ┌─────────────────┐                   ║  │
│  ║    │   👤 User       │           │   ☕ Recipe      │                   ║  │
│  ║    │     Tower       │           │     Tower       │                   ║  │
│  ║    │  ═══════════    │           │  ═══════════    │                   ║  │
│  ║    │  user_id → 64D  │           │ recipe_id → 64D │                   ║  │
│  ║    │  + taste_prefs  │           │ + taste_profile │                   ║  │
│  ║    └────────┬────────┘           └────────┬────────┘                   ║  │
│  ║             │                             │                            ║  │
│  ║             └──────────┬──────────────────┘                            ║  │
│  ║                        │                                               ║  │
│  ║                        ▼                                               ║  │
│  ║             ┌──────────────────────┐                                   ║  │
│  ║             │  cos(u_emb, r_emb)   │                                   ║  │
│  ║             │  ─────────────────── │                                   ║  │
│  ║             │   ANN-індекс Chroma  │                                   ║  │
│  ║             └──────────┬───────────┘                                   ║  │
│  ║                        │                                               ║  │
│  ║                        ▼                                               ║  │
│  ║              📦 ~50 кандидатів                                         ║  │
│  ╚════════════════════════════════════════════════════════════════════════╝  │
│                                      │                                       │
│                                      ▼                                       │
│  ╔════════════════════════════════════════════════════════════════════════╗  │
│  ║  📊 ЕТАП 2: RANKING                                      ⏱️ ~46мс      ║  │
│  ║  ══════════════════════════════════════════════════════════════════════║  │
│  ║                                                                        ║  │
│  ║   ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║   │                   Feature Engineering                           │  ║  │
│  ║   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │  ║  │
│  ║   │  │ 🎨 Смак │ │ ⚙️ Облад│ │ 📈 Іст  │ │ 🕐 Час   │ │ ⚡ Cross │    │  ║  │
│  ║   │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │  ║  │
│  ║   │       └───────────┴───────────┼───────────┴───────────┘         │  ║  │
│  ║   │                               ▼                                 │  ║  │
│  ║   │                    Feature Matrix (140+ ознак)                  │  ║  │
│  ║   └─────────────────────────────────────────────────────────────────┘  ║  │
│  ║                               │                                        ║  │
│  ║                               ▼                                        ║  │
│  ║   ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║   │           🌲 LightGBM + LambdaRank                              │  ║  │
│  ║   │           ════════════════════════                              │  ║  │
│  ║   │           Оптимізація NDCG напряму                              │  ║  │
│  ║   └─────────────────────────────────────────────────────────────────┘  ║  │
│  ║                               │                                        ║  │
│  ║                               ▼                                        ║  │
│  ║                     🎯 top-N рекомендацій                              ║  │
│  ╚════════════════════════════════════════════════════════════════════════╝  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         📊 Розподіл часу по етапах                           │
│  ════════════════════════════════════════════════════════════════════════    │
│                                                                              │
│   Retrieval (Two-Tower + ANN)   :  20.82мс ( 30.9%)                          │
│   Ranking (Features + LightGBM) :  46.48мс ( 69.1%)                          │
│   ────────────────────────────────────────────────                           │
│   ВСЬОГО                        :  67.30мс                                   │
│                                                                              │
│   [████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]                                  │
│    █ = Retrieval, ▒ = Ranking                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Переваги архітектури

| Аспект | Перевага |
|--------|----------|
| **Швидкість** | Two-Tower попередньо обчислює ембедінги рецептів; LightGBM ефективний на CPU |
| **Точність** | Нейронний retrieval + gradient boosting захоплює складні патерни |
| **Інтерпретованість** | LightGBM дає feature importance "з коробки" |
| **Масштабованість** | Retrieval — O(1) з ANN індексом; Ranking — лінійний по кількості кандидатів |

---

## 3. Етап 1: Retrieval (Пошук кандидатів)

### Two-Tower Model (Модель з двома вежами)

Це архітектура для **dense retrieval**, де користувачі та рецепти відображаються в спільний векторний простір.

#### Математична формалізація

Нехай:
- $u \in \mathcal{U}$ — користувач
- $r \in \mathcal{R}$ — рецепт
- $d$ — розмірність ембедінгу (64)

**User Tower** (вежа користувача):
$$f_u: \mathcal{U} \times \mathbb{R}^4 \rightarrow \mathbb{R}^d$$

**Recipe Tower** (вежа рецепту):
$$f_r: \mathcal{R} \times \mathbb{R}^4 \rightarrow \mathbb{R}^d$$

**Similarity** (подібність):
$$\text{sim}(u, r) = \frac{f_u(u)^\top f_r(r)}{\|f_u(u)\| \cdot \|f_r(r)\|}$$

#### Архітектура веж

```
User Tower:                              Recipe Tower:
┌─────────────────────────┐              ┌─────────────────────────┐
│ user_id (int)           │              │ recipe_id (int)         │
│         ↓               │              │         ↓               │
│ Embedding(N_users, 64)  │              │ Embedding(N_recipes, 64)│
│         ↓               │              │         ↓               │
│ [Concat taste_prefs_4D] │   optional   │ [Concat taste_profile]  │
│         ↓               │              │         ↓               │
│ Linear(68 → 256)        │              │ Linear(68 → 256)        │
│ BatchNorm → ReLU        │              │ BatchNorm → ReLU        │
│ Dropout(0.2)            │              │ Dropout(0.2)            │
│         ↓               │              │         ↓               │
│ Linear(256 → 128)       │              │ Linear(256 → 128)       │
│ BatchNorm → ReLU        │              │ BatchNorm → ReLU        │
│ Dropout(0.2)            │              │ Dropout(0.2)            │
│         ↓               │              │         ↓               │
│ Linear(128 → 64)        │              │ Linear(128 → 64)        │
│         ↓               │              │         ↓               │
│ L2 Normalize            │              │ L2 Normalize            │
└─────────────────────────┘              └─────────────────────────┘
         ↓                                        ↓
    user_emb ∈ ℝ⁶⁴                         recipe_emb ∈ ℝ⁶⁴
```

### InfoNCE Loss (Contrastive Loss)

**Ідея**: Максимізувати схожість позитивних пар (user, recipe) та мінімізувати — негативних.

Для батча розміру $B$ з парами $(u_i, r_i)$:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\text{sim}(u_i, r_i) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(u_i, r_j) / \tau)}$$

Де:
- $\tau$ — температура (0.07 за замовчуванням)
- Позитивна пара: $(u_i, r_i)$ — реальна взаємодія
- Негативні пари: $(u_i, r_j)$ для $j \neq i$ — **in-batch negatives**

#### In-Batch Negative Sampling

Замість явного семплювання негативів, ми використовуємо **всі інші рецепти в батчі** як негативи:

```
Батч:
  (user_0, recipe_0) ← позитив
  (user_1, recipe_1) ← позитив
  (user_2, recipe_2) ← позитив
  ...

Для user_0:
  + recipe_0 (позитив)
  - recipe_1 (негатив — in-batch)
  - recipe_2 (негатив — in-batch)
  ...
```

**Переваги**:
- Ефективність: $B$ позитивів + $B(B-1)$ негативів без додаткових обчислень
- Складні негативи: інші популярні рецепти з батча — релевантні негативи

#### Симетричний InfoNCE

Ми також використовуємо **bidirectional loss** — оптимізуємо і user→recipe, і recipe→user:

$$\mathcal{L}_{\text{symmetric}} = \frac{1}{2}(\mathcal{L}_{u \rightarrow r} + \mathcal{L}_{r \rightarrow u})$$

### Гіперпараметри Retrieval

| Параметр | Значення | Коментар |
|----------|----------|----------|
| `embedding_dim` | 64 | Розмірність ембедінгу |
| `hidden_dims` | [256, 128] | Приховані шари MLP |
| `temperature` | 0.07 | Температура для softmax |
| `batch_size` | 512 | Більший батч = більше негативів |
| `learning_rate` | 1e-3 | AdamW optimizer |
| `epochs` | 20-50 | З early stopping |

---

## 4. Етап 2: Ranking (Ранжування)

### LightGBM з LambdaRank

Після retrieval ми маємо ~100 кандидатів. **LightGBM Ranker** ранжує їх за допомогою багатих ознак.

#### Чому LightGBM?

1. **LambdaRank objective**: Безпосередньо оптимізує NDCG
2. **Швидкість**: CPU-ефективний, <40мс на 100 кандидатів
3. **Інтерпретованість**: Feature importance, SHAP values
4. **Обробка ознак**: Автоматично працює з категоріальними та числовими

#### LambdaRank

Замість оптимізації pointwise (окремо для кожного елемента) чи pairwise (порівняння пар), **LambdaRank** оптимізує **listwise** — весь ранжований список:

$$\lambda_{ij} = \frac{\partial \mathcal{L}}{\partial s_i} - \frac{\partial \mathcal{L}}{\partial s_j} = -\frac{1}{1 + e^{s_i - s_j}} \cdot |\Delta \text{NDCG}_{ij}|$$

Де $|\Delta \text{NDCG}_{ij}|$ — зміна NDCG при перестановці елементів $i$ та $j$.

#### Формат даних для ранкера

```python
# Для кожного user_id генеруємо:
# - K позитивних рецептів (реальні взаємодії)
# - (100-K) негативних рецептів (випадкові)

# Кожна пара (user, recipe) має:
# - 50+ ознак
# - relevance label (rating/5.0 або 0)

# Групування за user_id для LambdaRank
groups = [n_candidates_per_user_1, n_candidates_per_user_2, ...]
```

### Конфігурація LightGBM

```python
params = {
    "objective": "lambdarank",     # Оптимізація NDCG
    "metric": "ndcg",
    "boosting_type": "gbdt",       # Gradient Boosting Decision Trees
    "n_estimators": 173,           # Оптимізовано Optuna
    "learning_rate": 0.1,
    "early_stopping_rounds": 10,
}
```

---

## 5. Feature Engineering (Інженерія ознак)

Ми генеруємо **50+ ознак** для кожної пари (user, recipe). Вони критичні для якості ранжування.

### Категорії ознак

#### 5.1 Смакові ознаки (Taste Features)

```python
# Базові смакові профілі рецепту
taste_features = [
    "taste_bitterness",      # Гіркота (0-1)
    "taste_sweetness",       # Солодкість (0-1)
    "taste_acidity",         # Кислотність (0-1)
    "taste_body",            # Тіло/насиченість (0-1)
    "strength",              # Міцність (1-5)
]

# Різниця між вподобаннями користувача та рецептом
taste_diff_features = [
    "diff_bitterness",       # user_pref - recipe_taste
    "diff_sweetness",
    "diff_acidity",
    "diff_body",
]
```

#### 5.2 Similarity Features (Ознаки подібності)

```python
# Косинусна подібність смакових векторів
taste_cosine_similarity = cos_sim(user_taste_4d, recipe_taste_4d)

# Евклідова відстань
taste_euclidean_distance = ||user_taste - recipe_taste||_2

# Манхеттенська відстань
taste_manhattan_distance = ||user_taste - recipe_taste||_1

# Зважена подібність (важливіші компоненти мають більшу вагу)
taste_weighted_similarity = Σ(w_i * |user_i - recipe_i|)
```

**Формула косинусної подібності**:
$$\text{cosine\_similarity} = \frac{\mathbf{u} \cdot \mathbf{r}}{\|\mathbf{u}\| \|\mathbf{r}\|}$$

#### 5.3 Equipment Features (Ознаки обладнання)

```python
equipment_features = [
    "owned_equipment_count",       # Скільки обладнання має користувач
    "required_equipment_count",    # Скільки потребує рецепт
    "equipment_match",             # required ⊆ owned ? 1 : 0
    "equipment_coverage",          # |owned ∩ required| / |required|
    "equipment_missing_count",     # |required - owned|

    # One-hot для ключового обладнання
    "has_espresso_machine",
    "has_grinder",
    "has_milk_frother",
    "requires_espresso_machine",
    "requires_milk_frother",
]
```

#### 5.4 Historical Features (Історичні ознаки)

```python
# Статистика користувача
user_features = [
    "user_total_interactions",    # Кількість взаємодій
    "user_avg_rating",            # Середня оцінка
    "user_rating_std",            # Стандартне відхилення оцінок
    "user_completion_rate",       # % завершених рецептів
]

# Статистика рецепту
recipe_features = [
    "recipe_total_interactions",  # Глобальна популярність
    "recipe_avg_rating",          # Середня оцінка
    "recipe_completion_rate",     # % успішних приготувань
    "recipe_popularity_score",    # Нормалізований скор популярності
]
```

#### 5.5 Temporal Features (Часові ознаки)

```python
temporal_features = [
    "user_pct_morning",           # % взаємодій вранці
    "user_pct_afternoon",         # % вдень
    "user_pct_evening",           # % ввечері
    "user_pct_weekend",           # % у вихідні
    "user_time_consistency",      # Наскільки стабільний час активності
    "user_avg_hour_sin",          # sin(hour) — циклічна ознака
    "user_avg_hour_cos",          # cos(hour) — циклічна ознака
]
```

**Циклічне кодування часу**:
$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
$$\text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

#### 5.6 Cross Features (Перехресні ознаки)

```python
cross_features = [
    # Смак × Обладнання
    "taste_match_x_equipment",    # cosine_sim * equipment_match

    # Смак × Популярність
    "taste_match_x_popularity",   # cosine_sim * popularity_score

    # Квадратичні ознаки (high impact!)
    "taste_match_squared",        # cosine_sim²
    "equipment_taste_squared",    # (equipment_match * cosine_sim)²

    # Часові × Поведінкові
    "morning_strength_score",     # pct_morning * strength_match
    "weekend_exploration_score",  # pct_weekend * exploration_ratio
]
```

### Як генеруються ознаки

```python
class FeatureEngineer:
    def fit(self, users_df, recipes_df, train_df):
        """Попередньо обчислює статистики"""
        self._compute_user_stats(users_df, train_df)
        self._compute_recipe_stats(recipes_df, train_df)
        self._compute_temporal_behavioral_stats(...)

    def generate(self, pairs_df, users_df, recipes_df):
        """Генерує ознаки для пар (user, recipe)"""
        # Merge user features
        # Merge recipe features
        # Compute similarity features
        # Compute cross features
        return features_df
```

---

## 6. Метрики оцінки

### NDCG@k (Normalized Discounted Cumulative Gain)

**Основна метрика проекту**: NDCG@5

#### Інтуїція

NDCG вимірює якість ранжування, враховуючи:
1. **Релевантність**: Вищі оцінки — краще
2. **Позицію**: Релевантні елементи на вищих позиціях — краще

#### Формула

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i + 1)}$$

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

Де IDCG — ідеальний DCG (якби елементи були відсортовані за релевантністю).

#### Приклад

```
Рекомендації: [A, B, C, D, E]
Релевантність: [5, 3, 0, 4, 1]  (рейтинги у валідації)

DCG@5 = 5/log₂(2) + 3/log₂(3) + 0/log₂(4) + 4/log₂(5) + 1/log₂(6)
      = 5 + 1.89 + 0 + 1.72 + 0.39
      = 9.0

Ідеальний порядок: [5, 4, 3, 1, 0]
IDCG@5 = 5/log₂(2) + 4/log₂(3) + 3/log₂(4) + 1/log₂(5) + 0/log₂(6)
       = 5 + 2.52 + 1.5 + 0.43 + 0
       = 9.45

NDCG@5 = 9.0 / 9.45 = 0.952
```

### Інші метрики

| Метрика | Формула | Інтерпретація |
|---------|---------|---------------|
| **Hit Rate@k** | 1 якщо хоч один релевантний у top-k | Чи є "влучання" |
| **MRR** | 1/rank першого релевантного | Швидкість знаходження |
| **Precision@k** | \|relevant ∩ top-k\| / k | Точність top-k |
| **Coverage** | \|унікальні рекомендації\| / \|каталог\| | Різноманітність |

---

## 7. Cold-Start проблема

### Визначення

**Cold-start користувач** — користувач без історії взаємодій у тренувальних даних (300 таких у нашому датасеті).

### Проблема для Two-Tower

Two-Tower модель вивчає ембедінги для відомих user_id. Для нових користувачів:
- Немає вивченого ембедінгу
- Модель не може обчислити user representation

### Рішення: Cold-Start Encoder

Ми тренуємо додатковий MLP, який відображає **тільки ознаки користувача** (без ID) в embedding простір:

```
┌─────────────────────────┐
│ Cold-Start Encoder      │
│ ─────────────────────── │
│ Input: taste_prefs (4D) │
│         ↓               │
│ Linear(4 → 128)         │
│ BatchNorm → ReLU        │
│ Dropout(0.3)            │
│         ↓               │
│ Linear(128 → 64)        │
│ BatchNorm → ReLU        │
│ Dropout(0.3)            │
│         ↓               │
│ Linear(64 → 64)         │
│ L2 Normalize            │
│         ↓               │
│ user_emb ∈ ℝ⁶⁴          │
└─────────────────────────┘
```

#### Тренування Cold-Start Encoder

1. Беремо "теплих" користувачів з вивченими ембедінгами
2. Для кожного отримуємо: (taste_features, learned_embedding)
3. Тренуємо MLP мінімізувати MSE:

$$\mathcal{L}_{\text{cold}} = \|f_{\text{cold}}(\text{features}_u) - f_u(u)\|_2^2$$

#### Інференс

```python
if user_id in known_users:
    # Використовуємо вивчений ембедінг
    user_emb = user_tower(user_idx, features)
else:
    # Cold-start: використовуємо encoder
    user_emb = cold_start_encoder(features_only)
```

---

## 8. Повний пайплайн

### 8.1 Тренування

```bash
# 1. Тренуємо Two-Tower Retrieval
uv run python src/scripts/train_retrieval.py \
    --data-dir data \
    --output-dir runs/retrieval/baseline \
    --embedding-dim 64 \
    --use-features \
    --epochs 20 \
    --batch-size 512

# 2. Тренуємо Cold-Start Encoder (опційно)
uv run python src/scripts/train_cold_start_encoder.py \
    --retrieval-checkpoint runs/retrieval/baseline/retrieval_final.pt \
    --output-dir runs/retrieval/cold_encoder_baseline

# 3. Тренуємо LightGBM Ranker
uv run python src/scripts/train_ranker.py \
    --retrieval-checkpoint runs/retrieval/baseline/retrieval_final.pt \
    --embeddings-path runs/retrieval/baseline/recipe_embeddings.npy \
    --output-dir runs/ranking/improved-features \
    --use-optuna  # Hyperparameter search
```

### 8.2 Інференс

```python
from coffee_recipe_recommender.inference.recommender import Recommender

# Завантажуємо гібридну модель
recommender = Recommender.from_hybrid_checkpoints(
    retrieval_checkpoint_path="runs/retrieval/baseline/retrieval_final.pt",
    ranker_model_path="runs/ranking/improved-features/ranker.pkl",
    vector_store_path="data/chroma",
    feature_store_path="data/feature_store.db",
    users_df=users_df,
    recipes_df=recipes_df,
    candidate_size=100,
)

# Генеруємо рекомендації
recommendations = recommender.recommend(
    user_id="user_00001",
    users_df=users_df,
    recipes_df=recipes_df,
    train_df=train_df,
    n=5,
)
# [("recipe_espresso_001", 4.85), ("recipe_latte_003", 4.72), ...]
```

### 8.3 Детальний flow рекомендації

```
Вхід: user_id = "user_00001", n = 5

1. RETRIEVAL
   ├─ Перевірка: user_id в user_to_idx?
   │   ├─ Так → Обчислюємо user_emb через User Tower
   │   └─ Ні → Використовуємо Cold-Start Encoder
   ├─ Similarity: user_emb @ recipe_embeddings.T
   ├─ Equipment filtering: mask несумісних рецептів
   └─ Top-K candidates (K=100)

2. FEATURE EXTRACTION
   ├─ Для кожного (user, candidate):
   │   ├─ Taste similarity features
   │   ├─ Equipment features
   │   ├─ Historical features
   │   ├─ Temporal features
   │   └─ Cross features
   └─ Feature matrix: (100, 50+)

3. RANKING
   ├─ LightGBM.predict(feature_matrix)
   ├─ Sort by scores
   └─ Return top-N

Вихід: [(recipe_id_1, score_1), ..., (recipe_id_5, score_5)]
```

---

## 9. Оптимізація для продакшену

### 9.1 Pre-computed Embeddings

Рецепти статичні → обчислюємо ембедінги **один раз** при старті:

```python
# При завантаженні моделі
recipe_embeddings = model.get_recipe_embeddings(all_recipe_ids)
np.save("recipe_embeddings.npy", recipe_embeddings)

# При інференсі — тільки user embedding
user_emb = model.get_user_embeddings(user_id)
similarities = user_emb @ recipe_embeddings.T  # O(d * N_recipes)
```

### 9.2 ONNX Export

Для швидшого інференсу експортуємо PyTorch → ONNX:

```python
torch.onnx.export(
    model.user_tower,
    (user_idx, user_features),
    "user_tower.onnx",
    input_names=["user_idx", "features"],
    output_names=["embedding"],
)
```

### 9.3 Approximate Nearest Neighbors (ANN)

Для великих каталогів (>100K рецептів) використовуємо HNSW індекс через ChromaDB:

```python
# Зберігаємо ембедінги в Chroma через VectorStore
from coffee_recipe_recommender.db.vector_store import VectorStore

store = VectorStore(persist_dir="data/chroma")
store.save(embeddings=recipe_embeddings, ids=recipe_ids, reset=True)

# Швидкий пошук
similar_ids, distances = store.query(user_embedding, n_results=100)
```

### 9.4 Latency Budget

```
📊 Виміряний розподіл часу (Гібридний режим)
════════════════════════════════════════════════════════════════════════
  Retrieval (Two-Tower + ANN)   :  20.82мс ( 30.9%)
  Ranking (Features + LightGBM) :  46.48мс ( 69.1%)
  ──────────────────────────────────────────────────────────────────────
  ВСЬОГО                        :  67.30мс
════════════════════════════════════════════════════════════════════════

Візуалізація:
  [████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
   █ = Retrieval, ▒ = Ranking
```

| Етап | Компоненти | Час |
|------|------------|-----|
| **Retrieval** | User embedding + ANN пошук | ~21мс |
| **Ranking** | Feature extraction + LightGBM | ~46мс |
| **Всього** | End-to-end | **~67мс** |

---

## 10. Explainability (Пояснюваність)

### Чому пояснюваність важлива?

Пояснення рекомендацій допомагає:
- **Користувачам** — зрозуміти, чому саме цей рецепт
- **Розробникам** — відлагоджувати та покращувати модель
- **Бізнесу** — відповідати вимогам прозорості AI

### SHAP (SHapley Additive exPlanations)

Ми використовуємо **SHAP** для пояснення рішень LightGBM ранкера:

- **TreeExplainer** — спеціалізований пояснювач для tree-based моделей
- **Shapley values** — теоретично обґрунтований метод розподілу "внеску" кожної ознаки

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

Де:
- $\phi_i$ — SHAP value для ознаки $i$
- $f(S)$ — передбачення моделі з ознаками $S$
- $N$ — множина всіх ознак

### Інтеграція в систему

```python
from coffee_recipe_recommender.inference.recommender import Recommender

# Завантажуємо гібридну модель
recommender = Recommender.from_hybrid_checkpoints(...)

# Отримуємо рекомендації з SHAP поясненнями
top_recipes, X_features, shap_values, explainer = recommender.recommend_with_shap(
    user_id="user_00001",
    users_df=users_df,
    recipes_df=recipes_df,
    train_df=train_df,
    n=5,
)

# shap_values — матриця (n_recipes, n_features) з внесками кожної ознаки
```

### Групування ознак для UI

Для зручного відображення в інтерфейсі, ознаки згруповані за категоріями:

| Група | Ознаки | Пояснення для користувача |
|-------|--------|---------------------------|
| **Taste match** | taste_cosine_similarity, taste_diff_*, ... | "Відповідає вашим смаковим вподобанням" |
| **Equipment fit** | equipment_match, equipment_coverage, ... | "Підходить для вашого обладнання" |
| **Fits your habits** | user_avg_rating, user_completion_rate, ... | "Підходить під ваші звички" |
| **Time & context** | morning_combo_score, weekend_exploration_score, ... | "Вдалий вибір для цього часу" |
| **Discovery & novelty** | exploration_novelty_score, user_exploration_ratio, ... | "Баланс нового та знайомого" |
| **Quality & popularity** | recipe_avg_rating, recipe_popularity_score, ... | "Популярний/перевірений вибір" |
| **Dietary compatibility** | dietary_compatible, vegan_compatible_taste, ... | "Відповідає вашим дієтичним вподобанням" |

### API для пояснень

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  🎯 ПОЯСНЕННЯ РЕКОМЕНДАЦІЇ                                                    │
│  ════════════════════════════════════════════════════════════════════════    │
│                                                                              │
│   Рецепт: "Cappuccino"                                                       │
│   Score: 4.85                                                                │
│                                                                              │
│   ✅ Причини рекомендації (top-3 positive):                                   │
│   ├─ Taste match: +0.42 — "Відповідає вашим смаковим вподобанням"             │
│   ├─ Equipment fit: +0.28 — "Підходить для вашого обладнання"                 │
│   └─ Quality: +0.15 — "Популярний/перевірений вибір"                          │
│                                                                              │
│   ⚠️ Можливі компроміси (top-1 negative):                                     │
│   └─ Time context: -0.08 — "Не оптимальний час для цього напою"               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Функція пояснення для UI

У `src/client/app.py` реалізована функція `explain_for_ui()`:

```python
def explain_for_ui(
    *,
    recipe_id: str,
    shap_row: np.ndarray,       # SHAP values для одного рецепту
    feature_names: list[str],   # Назви ознак
    base: float,                # Базове значення (середнє)
    pred: float,                # Фінальне передбачення
    max_reasons: int = 3,       # Кількість позитивних причин
    max_tradeoffs: int = 1,     # Кількість негативних пунктів
) -> dict:
    """
    Returns:
        {
            "recipe_id": "recipe_cappuccino_001",
            "score": 4.85,
            "reasons": [
                {"group": "Taste match", "impact": 0.42, "text": "..."}
            ],
            "tradeoffs": [
                {"group": "Time context", "impact": -0.08, "text": "..."}
            ]
        }
    """
```

### Візуалізація SHAP

```python
import shap
import matplotlib.pyplot as plt

# Waterfall plot для одного рецепту
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    feature_names=X_features.columns.tolist(),
))

# Beeswarm plot для всіх top-N рецептів
shap.plots.beeswarm(shap.Explanation(
    values=shap_values,
    feature_names=X_features.columns.tolist(),
))
```

### Feature Importance (глобальний рівень)

LightGBM надає `feature_importance` "з коробки":

```python
# З моделі
importance = ranker.model.feature_importance(importance_type='gain')

# Топ-10 ознак
top_features = sorted(
    zip(feature_names, importance),
    key=lambda x: x[1],
    reverse=True
)[:10]
```

Типові топ-ознаки:
1. `taste_cosine_similarity` — збіг смакових профілів
2. `equipment_match` — відповідність обладнання
3. `recipe_avg_rating` — середня оцінка рецепту
4. `user_avg_rating` — середня оцінка користувача
5. `taste_match_squared` — квадратичний ефект збігу смаку

---

## Висновки

Наша система реалізує **state-of-the-art гібридний підхід**:

1. **Two-Tower Retrieval** — ефективний пошук у спільному embedding просторі з InfoNCE loss
2. **LightGBM Ranking** — точне ранжування з 140+ ознаками та LambdaRank
3. **Cold-Start Encoder** — обробка нових користувачів через content-based MLP
4. **Production optimizations** — pre-computed embeddings, ONNX, ANN індекси (ChromaDB)
5. **Explainability** — SHAP пояснення з групуванням ознак для UI

**Ключові метрики**:
- NDCG@5 > 0.4 (target)
- Latency ~67мс end-to-end
- Coverage > 30%

Система готова до продакшену і може масштабуватися для великих каталогів через ANN індексацію та надає прозорі пояснення для кожної рекомендації.
