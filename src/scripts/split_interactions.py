from pathlib import Path

import pandas as pd


def main():
    data_dir = Path("data")

    # Load current train file
    train_df = pd.read_csv(data_dir / "interactions_train.csv")

    print(f"Total interactions in interactions_train.csv: {len(train_df)}")

    # Split 80/20: 80% train, 20% validation
    # Use stratified split by user to ensure each user has data in both splits
    train_split = train_df.sample(frac=0.8, random_state=42)
    test_split = train_df.drop(train_split.index)

    print(f"Train split: {len(train_split)} interactions ({len(train_split.user_id.unique())} users)")
    print(f"Test split:   {len(test_split)} interactions ({len(test_split.user_id.unique())} users)")

    # Save splits
    train_split.to_csv(data_dir / "interactions_train_split.csv", index=False)
    test_split.to_csv(data_dir / "interactions_test_split.csv", index=False)
    print("\n✅ Created:")
    print(f"   - {data_dir / 'interactions_train_split.csv'}")
    print(f"   - {data_dir / 'interactions_test_split.csv'}")
    print("\nNote: interactions_val.csv and interactions_val_cold.csv remain for final evaluation")


if __name__ == "__main__":
    main()
