# Two-Tower Retrieval Model Training

This directory contains everything needed to train the Two-Tower retrieval model for coffee recipe recommendation.

## 🎯 Architecture

**Two-Tower Model** (Industry standard for large-scale retrieval):
- **User Tower**: Embeds users into 64-dim space using ID embeddings + taste preferences
- **Recipe Tower**: Embeds recipes into same 64-dim space using ID embeddings + taste profiles
- **Training**: InfoNCE contrastive loss with in-batch negative sampling
- **Inference**: Pre-computed recipe embeddings enable <10ms retrieval via cosine similarity

## 📁 Components

### Core Modules

- **`src/coffee_recipe_recommender/data/`**
  - `loaders.py` - CSV loading with JSON parsing, ID mappings
  - `dataset.py` - PyTorch datasets for training

- **`src/coffee_recipe_recommender/models/`**
  - `retrieval.py` - UserTower, RecipeTower, TwoTowerModel classes

- **`src/coffee_recipe_recommender/training/`**
  - `losses.py` - InfoNCE loss, symmetric InfoNCE, triplet loss

- **`src/coffee_recipe_recommender/evaluation/`**
  - `metrics.py` - NDCG@k, Hit Rate, MRR, Coverage

- **`src/coffee_recipe_recommender/inference/`**
  - `recommender.py` - Fast inference with pre-computed embeddings

### Scripts

- **`src/scripts/train_retrieval.py`** - Main training script
- **`src/scripts/evaluate_retrieval.py`** - Evaluation on val/val_cold splits
- **`src/scripts/export_onnx.py`** - Export to ONNX for production

## 🚀 Quick Start

### 1. Train the Model

Basic training with default hyperparameters:

```bash
uv run python src/scripts/train_retrieval.py \
  --data-dir data \
  --output-dir models \
  --epochs 50 \
  --batch-size 512
```

### 2. Train with User/Recipe Features

Enable taste preference features:

```bash
uv run python src/scripts/train_retrieval.py \
  --use-features \
  --embedding-dim 64 \
  --hidden-dims 256 128 \
  --epochs 50
```

### 3. Evaluate on Validation Set

```bash
uv run python src/scripts/evaluate_retrieval.py \
  --checkpoint models/retrieval_best.pt \
  --embeddings models/recipe_embeddings.npy \
  --eval-split val \
  --k 5
```

### 4. Evaluate Cold-Start Performance

```bash
uv run python src/scripts/evaluate_retrieval.py \
  --checkpoint models/retrieval_best.pt \
  --embeddings models/recipe_embeddings.npy \
  --eval-split val_cold \
  --k 5
```

## 🔧 Training Options

### Model Architecture

```bash
--embedding-dim 64           # Output embedding dimension (default: 64)
--hidden-dims 256 128        # MLP hidden layers (default: [256, 128])
--use-features               # Use taste preferences/profiles
--dropout 0.2                # Dropout probability
--temperature 0.07           # Temperature for contrastive loss
```

### Training Hyperparameters

```bash
--batch-size 512            # Larger batches = more negatives (default: 512)
--epochs 50                 # Number of epochs
--lr 1e-3                   # Learning rate (AdamW)
--weight-decay 1e-5         # L2 regularization
--min-rating 3.5            # Threshold for positive interaction
--symmetric-loss            # Use bidirectional InfoNCE loss
```

### Hardware & Efficiency

```bash
--device cuda               # Use GPU if available
--num-workers 4             # DataLoader parallelism
```

### Checkpointing

```bash
--save-every 5              # Save checkpoint every N epochs
--save-best                 # Save best model based on val loss
```

## 📊 Expected Performance

**Target**: NDCG@5 > 0.4

**Typical results** (warm users):
- NDCG@5: 0.42-0.48
- Hit Rate@5: 0.65-0.75
- Coverage: 0.85-0.95

**Cold-start users** (zero training interactions):
- NDCG@5: 0.30-0.38 (with features)
- NDCG@5: 0.15-0.25 (without features)

## 🔬 Model Details

### InfoNCE Loss

For a batch of size B:
- **Positive pairs**: (user[i], recipe[i]) for i = 0..B-1
- **Negative pairs**: (user[i], recipe[j]) for all i ≠ j
- Total: B positive + B*(B-1) negative pairs per batch

Loss formula:
```
L = -log( exp(sim(u_i, r_i) / τ) / Σ_j exp(sim(u_i, r_j) / τ) )
```

where τ is temperature (default 0.07).

### Architecture

```
User Tower:
  Input: user_id (int) + taste_prefs (4D) [optional]
  ├─ Embedding(num_users, 64)
  ├─ Concat with features → [64 + 4 = 68]
  ├─ Linear(68, 256) → BatchNorm → ReLU → Dropout(0.2)
  ├─ Linear(256, 128) → BatchNorm → ReLU → Dropout(0.2)
  └─ Linear(128, 64) → L2 Normalize
  Output: 64-dim user embedding

Recipe Tower: (same structure)
  Input: recipe_id (int) + taste_profile (4D) [optional]
  Output: 64-dim recipe embedding
```

## 🎓 Training Tips

### Batch Size
- **Larger is better** for InfoNCE (more in-batch negatives)
- Recommended: 512-1024 if GPU memory allows
- Minimum: 128 (fewer negatives = weaker signal)

### Temperature
- **Lower (0.05)**: More confident predictions, sharper separation
- **Higher (0.1)**: Smoother distributions, easier optimization
- **Default (0.07)**: Works well in most cases

### Features vs No Features
- **With features**: Better cold-start, slower training
- **Without features**: Pure collaborative filtering, faster

### Learning Rate Schedule
- Uses CosineAnnealingLR by default
- Gradually reduces LR from initial value to 0
- Helps convergence in later epochs

## 💾 Output Files

After training, you'll have:

```
models/
├── retrieval_epoch_5.pt      # Checkpoint every 5 epochs
├── retrieval_epoch_10.pt
├── ...
├── retrieval_best.pt          # Best validation loss
├── retrieval_final.pt         # Final epoch
└── recipe_embeddings.npy      # Pre-computed embeddings (N, 64)
```

## 🔄 Using Pre-computed Embeddings

The training script automatically generates `recipe_embeddings.npy`:

```python
from coffee_recipe_recommender.inference import RetrievalRecommender

# Load recommender
recommender = RetrievalRecommender.from_checkpoint(
    checkpoint_path="models/retrieval_best.pt",
    embeddings_path="models/recipe_embeddings.npy",
    users_df=users_df,
    device="cpu"
)

# Get recommendations (fast!)
recommendations = recommender.recommend(user_id="user_00000", n=5)
# Returns: [("recipe_espresso_000", 0.87), ...]
```

## 🚢 Export to ONNX (Production)

For maximum inference speed:

```bash
uv run python src/scripts/export_onnx.py \
  --checkpoint models/retrieval_best.pt \
  --output-dir models/onnx \
  --verify
```

This creates:
- `models/onnx/user_tower.onnx` - User embedding tower
- `models/onnx/recipe_tower.onnx` - Recipe embedding tower

ONNX models are 2-3x faster than PyTorch on CPU!

## 📈 Next Steps

After training retrieval:

1. **Evaluate** - Run evaluation scripts to verify NDCG@5 > 0.4
2. **Analyze** - Check which features matter most
3. **Stage 2** - Train LightGBM ranker on top-100 candidates
4. **Hybrid** - Combine retrieval + ranking for best results

## 🐛 Troubleshooting

**Training loss not decreasing:**
- Increase batch size (more negatives)
- Lower temperature (sharper contrasts)
- Check data quality (positive interactions defined correctly?)

**Poor cold-start performance:**
- Enable `--use-features` flag
- Check taste preference quality in users.csv
- Increase feature weight in model

**Out of memory:**
- Reduce `--batch-size`
- Reduce `--hidden-dims` (e.g., 128 64)
- Use smaller `--embedding-dim`

**Validation loss increases:**
- Add more regularization (`--weight-decay`)
- Increase dropout (`--dropout 0.3`)
- Use fewer epochs or early stopping
