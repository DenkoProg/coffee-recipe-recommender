import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from coffee_recipe_recommender.models.retrieval import TwoTowerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Two-Tower model to ONNX format.")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/onnx"), help="Output directory for ONNX models"
    )
    parser.add_argument("--opset-version", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX model outputs match PyTorch")

    return parser.parse_args()


def export_user_tower(
    model: TwoTowerModel,
    output_path: Path,
    use_features: bool,
    opset_version: int = 14,
) -> None:
    """
    Export user tower to ONNX.

    Args:
        model: Trained TwoTowerModel
        output_path: Output path for ONNX file
        use_features: Whether model uses features
        opset_version: ONNX opset version
    """
    model.eval()

    # Create dummy inputs
    batch_size = 1
    user_idx = torch.zeros(batch_size, dtype=torch.long)

    if use_features:
        user_features = torch.zeros(batch_size, 4, dtype=torch.float32)
        dummy_input = (user_idx, user_features)
        input_names = ["user_idx", "user_features"]
        dynamic_axes = {
            "user_idx": {0: "batch_size"},
            "user_features": {0: "batch_size"},
            "user_embedding": {0: "batch_size"},
        }
    else:
        dummy_input = (user_idx, None)
        input_names = ["user_idx"]
        dynamic_axes = {
            "user_idx": {0: "batch_size"},
            "user_embedding": {0: "batch_size"},
        }

    # Export
    torch.onnx.export(
        model.user_tower,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=["user_embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"User tower exported to: {output_path}")


def export_recipe_tower(
    model: TwoTowerModel,
    output_path: Path,
    use_features: bool,
    opset_version: int = 14,
) -> None:
    """
    Export recipe tower to ONNX.

    Args:
        model: Trained TwoTowerModel
        output_path: Output path for ONNX file
        use_features: Whether model uses features
        opset_version: ONNX opset version
    """
    model.eval()

    batch_size = 1
    recipe_idx = torch.zeros(batch_size, dtype=torch.long)

    if use_features:
        recipe_features = torch.zeros(batch_size, 4, dtype=torch.float32)
        dummy_input = (recipe_idx, recipe_features)
        input_names = ["recipe_idx", "recipe_features"]
        dynamic_axes = {
            "recipe_idx": {0: "batch_size"},
            "recipe_features": {0: "batch_size"},
            "recipe_embedding": {0: "batch_size"},
        }
    else:
        dummy_input = (recipe_idx, None)
        input_names = ["recipe_idx"]
        dynamic_axes = {
            "recipe_idx": {0: "batch_size"},
            "recipe_embedding": {0: "batch_size"},
        }

    torch.onnx.export(
        model.recipe_tower,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=["recipe_embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"Recipe tower exported to: {output_path}")


def verify_onnx_model(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    use_features: bool,
    tower_type: str,
) -> None:
    """
    Verify ONNX model outputs match PyTorch.

    Args:
        pytorch_model: PyTorch model (user_tower or recipe_tower)
        onnx_path: Path to ONNX model
        use_features: Whether model uses features
        tower_type: "user" or "recipe"
    """
    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))

    # Create test inputs
    batch_size = 4
    idx = torch.randint(0, 10, (batch_size,), dtype=torch.long)

    if use_features:
        features = torch.randn(batch_size, 4, dtype=torch.float32)
        pytorch_input = (idx, features)
        onnx_input = {
            f"{tower_type}_idx": idx.numpy(),
            f"{tower_type}_features": features.numpy(),
        }
    else:
        pytorch_input = (idx, None)
        onnx_input = {f"{tower_type}_idx": idx.numpy()}

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(*pytorch_input).numpy()

    # ONNX inference
    onnx_output = ort_session.run(None, onnx_input)[0]

    # Compare
    max_diff = np.abs(pytorch_output - onnx_output).max()
    print(f"\n{tower_type.capitalize()} tower verification:")
    print(f"  Max difference: {max_diff:.6f}")

    if max_diff < 1e-5:
        print("  ✅ ONNX model matches PyTorch!")
    else:
        print("  ⚠️  Warning: ONNX outputs differ from PyTorch")


def main() -> None:
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Extract model arguments
    model_args = checkpoint["args"]
    print("\nModel configuration:")
    print(f"  Embedding dim: {model_args['embedding_dim']}")
    print(f"  Hidden dims: {model_args['hidden_dims']}")
    print(f"  Use features: {model_args['use_features']}")

    # Initialize model
    model = TwoTowerModel(
        num_users=len(checkpoint["user_to_idx"]),
        num_recipes=len(checkpoint["recipe_to_idx"]),
        embedding_dim=model_args["embedding_dim"],
        hidden_dims=model_args["hidden_dims"],
        use_features=model_args["use_features"],
        dropout=model_args.get("dropout", 0.2),
        temperature=model_args.get("temperature", 0.07),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nExporting to ONNX...")

    # Export user tower
    user_tower_path = args.output_dir / "user_tower.onnx"
    export_user_tower(model, user_tower_path, model_args["use_features"], args.opset_version)

    # Export recipe tower
    recipe_tower_path = args.output_dir / "recipe_tower.onnx"
    export_recipe_tower(model, recipe_tower_path, model_args["use_features"], args.opset_version)

    # Verify ONNX models
    if args.verify:
        print("\nVerifying ONNX models...")
        verify_onnx_model(model.user_tower, user_tower_path, model_args["use_features"], "user")
        verify_onnx_model(model.recipe_tower, recipe_tower_path, model_args["use_features"], "recipe")

    # Check ONNX models
    print("\nChecking ONNX models...")
    user_model = onnx.load(str(user_tower_path))
    onnx.checker.check_model(user_model)
    print("  ✅ User tower ONNX model is valid")

    recipe_model = onnx.load(str(recipe_tower_path))
    onnx.checker.check_model(recipe_model)
    print("  ✅ Recipe tower ONNX model is valid")

    print(f"\nExport complete! ONNX models saved to: {args.output_dir}")
    print("\nUsage:")
    print(f"  User tower: {user_tower_path}")
    print(f"  Recipe tower: {recipe_tower_path}")


if __name__ == "__main__":
    main()
