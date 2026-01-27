import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from coffee_recipe_recommender.models.retrieval import TwoTowerModel
from coffee_recipe_recommender.training.datasets import RetrievalDataset, RetrievalDatasetWithFeatures
from coffee_recipe_recommender.training.loaders import create_id_mappings, load_interactions, load_recipes, load_users
from coffee_recipe_recommender.training.losses import InfoNCELoss, InfoNCELossWithSymmetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Two-Tower retrieval model for recipe recommendation.")

    # Data paths
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to save trained models")

    # Model architecture
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[256, 128], help="Hidden layer dimensions (e.g., 256 128)"
    )
    parser.add_argument("--use-features", action="store_true", help="Use user/recipe features in towers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument("--min-rating", type=float, default=3.5, help="Minimum rating for positive interaction")
    parser.add_argument("--use-completed-only", action="store_true", help="Only use completed interactions")
    parser.add_argument("--symmetric-loss", action="store_true", help="Use symmetric InfoNCE loss")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")

    # Checkpointing
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--save-best", action="store_true", help="Save best model based on validation loss")

    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_features: bool,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        if use_features:
            user_idx = batch["user_idx"].to(device)
            recipe_idx = batch["recipe_idx"].to(device)
            user_features = batch["user_features"].to(device)
            recipe_features = batch["recipe_features"].to(device)
        else:
            user_idx, recipe_idx = batch
            user_idx = user_idx.to(device)
            recipe_idx = recipe_idx.to(device)
            user_features = None
            recipe_features = None

        # Forward pass
        user_emb, recipe_emb = model(user_idx, recipe_idx, user_features, recipe_features)

        # Compute loss
        loss = criterion(user_emb, recipe_emb)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str, use_features: bool) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            if use_features:
                user_idx = batch["user_idx"].to(device)
                recipe_idx = batch["recipe_idx"].to(device)
                user_features = batch["user_features"].to(device)
                recipe_features = batch["recipe_features"].to(device)
            else:
                user_idx, recipe_idx = batch
                user_idx = user_idx.to(device)
                recipe_idx = recipe_idx.to(device)
                user_features = None
                recipe_features = None

            user_emb, recipe_emb = model(user_idx, recipe_idx, user_features, recipe_features)
            loss = criterion(user_emb, recipe_emb)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main() -> None:
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_dir}...")

    # Load data
    users_df = load_users(args.data_dir / "users.csv")
    recipes_df = load_recipes(args.data_dir / "recipes.csv")
    train_df = load_interactions(args.data_dir / "interactions_train.csv")
    val_df = load_interactions(args.data_dir / "interactions_val.csv")

    # Create ID mappings
    user_to_idx, recipe_to_idx, idx_to_user, idx_to_recipe = create_id_mappings(users_df, recipes_df)

    print(f"Number of users: {len(user_to_idx)}")
    print(f"Number of recipes: {len(recipe_to_idx)}")
    print(f"Training interactions: {len(train_df)}")
    print(f"Validation interactions: {len(val_df)}")

    # Create datasets
    if args.use_features:
        train_dataset = RetrievalDatasetWithFeatures(
            train_df,
            users_df,
            recipes_df,
            user_to_idx,
            recipe_to_idx,
            args.min_rating,
            args.use_completed_only,
        )
        val_dataset = RetrievalDatasetWithFeatures(
            val_df, users_df, recipes_df, user_to_idx, recipe_to_idx, args.min_rating, args.use_completed_only
        )
    else:
        train_dataset = RetrievalDataset(
            train_df, user_to_idx, recipe_to_idx, args.min_rating, args.use_completed_only
        )
        val_dataset = RetrievalDataset(val_df, user_to_idx, recipe_to_idx, args.min_rating, args.use_completed_only)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize model
    print("\nInitializing Two-Tower model...")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Use features: {args.use_features}")
    print(f"  Temperature: {args.temperature}")

    model = TwoTowerModel(
        num_users=len(user_to_idx),
        num_recipes=len(recipe_to_idx),
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        use_features=args.use_features,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(args.device)

    # Loss function
    if args.symmetric_loss:
        criterion = InfoNCELossWithSymmetry(temperature=args.temperature)
        print("Using symmetric InfoNCE loss")
    else:
        criterion = InfoNCELoss(temperature=args.temperature)
        print("Using standard InfoNCE loss")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device, args.use_features)
        print(f"  Train loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, criterion, args.device, args.use_features)
        print(f"  Val loss: {val_loss:.4f}")

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Learning rate: {current_lr:.6f}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = args.output_dir / f"retrieval_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "args": vars(args),
                    "user_to_idx": user_to_idx,
                    "recipe_to_idx": recipe_to_idx,
                    "idx_to_user": idx_to_user,
                    "idx_to_recipe": idx_to_recipe,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if args.save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = args.output_dir / "retrieval_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "args": vars(args),
                    "user_to_idx": user_to_idx,
                    "recipe_to_idx": recipe_to_idx,
                    "idx_to_user": idx_to_user,
                    "idx_to_recipe": idx_to_recipe,
                },
                best_path,
            )
            print(f"  New best model saved: {best_path} (val_loss: {val_loss:.4f})")

    # Save final model
    final_path = args.output_dir / "retrieval_final.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "user_to_idx": user_to_idx,
            "recipe_to_idx": recipe_to_idx,
            "idx_to_user": idx_to_user,
            "idx_to_recipe": idx_to_recipe,
        },
        final_path,
    )
    print(f"\nTraining complete! Final model saved: {final_path}")

    # Pre-compute and save recipe embeddings for fast inference
    print("\nPre-computing recipe embeddings...")
    model.eval()
    all_recipe_embeddings = []

    with torch.no_grad():
        for recipe_idx in tqdm(range(len(recipe_to_idx)), desc="Computing embeddings"):
            recipe_tensor = torch.tensor([recipe_idx], dtype=torch.long, device=args.device)

            if args.use_features:
                # Get recipe features from dataset
                recipe_id = idx_to_recipe[recipe_idx]
                recipe_row = recipes_df[recipes_df["recipe_id"] == recipe_id].iloc[0]
                recipe_features = torch.tensor(
                    [
                        [
                            recipe_row["taste_bitterness"],
                            recipe_row["taste_sweetness"],
                            recipe_row["taste_acidity"],
                            recipe_row["taste_body"],
                        ]
                    ],
                    dtype=torch.float32,
                    device=args.device,
                )
            else:
                recipe_features = None

            emb = model.get_recipe_embeddings(recipe_tensor, recipe_features)
            all_recipe_embeddings.append(emb.cpu().numpy())

    recipe_embeddings = np.concatenate(all_recipe_embeddings, axis=0)
    embeddings_path = args.output_dir / "recipe_embeddings.npy"
    np.save(embeddings_path, recipe_embeddings)
    print(f"Saved recipe embeddings: {embeddings_path} (shape: {recipe_embeddings.shape})")


if __name__ == "__main__":
    main()
