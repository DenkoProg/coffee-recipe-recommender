# Data Split Strategy (Fixed Data Leakage)

## Problem
Previously, the training pipeline had data leakage because:
- `interactions_train.csv` was used for **both training AND validation** in training scripts
- This violated ML best practices and inflated model performance metrics

## Solution
Created proper 80/20 split from `interactions_train.csv`:

### Data Files

| File | Purpose | Count | Users |
|------|---------|-------|-------|
| `interactions_train_split.csv` | Training neural models (Two-Tower, Cold-start encoder) | 60,716 | 1,470 |
| `interactions_test_split.csv` | Development/tuning validation | 15,179 | 1,449 |
| `interactions_val.csv` | **Final** warm-user evaluation (held-out) | 25,300 | 1,606 |
| `interactions_val_cold.csv` | **Final** cold-start evaluation (held-out) | 4,098 | 287 |

### Updated Scripts

**Training Scripts** (now use proper splits):
- `train_retrieval.py` → uses `interactions_train_split.csv` for training
- `train_ranker.py` → uses `interactions_train_split.csv` for training
- `train_cold_start_encoder.py` → uses `interactions_train_split.csv` for training

**Evaluation Scripts** (now use held-out sets):
- `evaluate_retrieval.py` → uses proper held-out splits:
  - `val` split → `interactions_test_split.csv` for model tuning/debugging
  - `val_cold` split → `interactions_val_cold.csv` for cold-start evaluation
  - For production eval, also keep `interactions_val.csv` as final test set

**Inference**:
- `recommend_service.py` → loads `interactions_train_split.csv` for feature generation only

### How to Generate Splits

```bash
# Create the splits from original interactions_train.csv
uv run python src/scripts/split_interactions.py
```

This creates:
- `data/interactions_train_split.csv` (80% of original)
- `data/interactions_test_split.csv` (20% of original)

---

**Note**: Do NOT merge splits back. Always train on `train_split` and evaluate on held-out `val_split` / `val_cold` for honest metrics.
