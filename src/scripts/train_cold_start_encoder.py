import argparse
import pathlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from coffee_recipe_recommender.models.retrieval import ColdStartEncoder, TwoTowerModel
from coffee_recipe_recommender.training.loaders import load_users


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--users", type=Path, default=Path("data/users.csv"))
    p.add_argument("--out", type=Path, default=Path("runs/retrieval/cold_encoder_baseline/cold_encoder.pt"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    with torch.serialization.safe_globals([pathlib.PosixPath]):
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})

    model = TwoTowerModel(
        num_users=len(checkpoint["user_to_idx"]),
        num_recipes=len(checkpoint["recipe_to_idx"]),
        embedding_dim=model_args.get("embedding_dim", 64),
        hidden_dims=model_args.get("hidden_dims", [256, 128]),
        use_features=model_args.get("use_features", False),
        dropout=model_args.get("dropout", 0.2),
        temperature=model_args.get("temperature", 0.07),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # user ordering from checkpoint
    user_to_idx = checkpoint["user_to_idx"]
    # sort by idx to get stable order
    sorted_users = sorted(user_to_idx.items(), key=lambda x: x[1])
    user_ids = [uid for uid, _ in sorted_users]

    users_df = load_users(args.users)
    users_df = users_df.set_index("user_id")

    # Build feature matrix (default: taste 4D)
    taste_cols = [
        "taste_pref_bitterness",
        "taste_pref_sweetness",
        "taste_pref_acidity",
        "taste_pref_body",
    ]

    missing = [c for c in taste_cols if c not in users_df.columns]
    if missing:
        raise RuntimeError(f"Missing taste columns in users.csv: {missing}")

    # Align users
    user_features = users_df.loc[user_ids, taste_cols].fillna(0.5).values.astype(np.float32)
    user_features = torch.from_numpy(user_features).to(device)

    # Get target embeddings from trained user tower
    with torch.no_grad():
        user_indices = torch.arange(len(user_ids), dtype=torch.long, device=device)
        if model.user_tower.use_features:
            # pass taste features as user_features to user_tower
            target_embeddings = model.get_user_embeddings(user_indices, user_features)
        else:
            target_embeddings = model.get_user_embeddings(user_indices, None)

    # Train encoder (features -> target_embeddings)
    encoder = ColdStartEncoder(feature_dim=user_features.shape[1], embedding_dim=model.embedding_dim)
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    dataset = TensorDataset(user_features, target_embeddings)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = encoder(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "feature_dim": user_features.shape[1],
            "embedding_dim": model.embedding_dim,
        },
        args.out,
    )
    print(f"Saved encoder to {args.out}")


if __name__ == "__main__":
    main()
