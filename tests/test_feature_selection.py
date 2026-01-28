import numpy as np
import pandas as pd

from coffee_recipe_recommender.preprocessing.features import FEATURE_GROUP_MAP, PRESETS, FeatureEngineer


def test_feature_selection():
    print("Testing Feature Selection...")

    # 1. Test 'all' preset (default)
    fe_all = FeatureEngineer()
    print(f"Default features count: {len(fe_all.feature_cols)}")
    assert len(fe_all.feature_cols) > 80

    # 2. Test 'legacy' preset
    fe_legacy = FeatureEngineer(preset="legacy")
    print(f"Legacy features count: {len(fe_legacy.feature_cols)}")
    assert len(fe_legacy.feature_cols) == 80
    # Check some specific features are present
    assert "taste_bitterness" in fe_legacy.feature_cols
    assert "preference_mismatch" in fe_legacy.feature_cols
    # Check advanced features are NOT present
    assert "morning_strength_score" not in fe_legacy.feature_cols

    # 3. Test explicit groups
    fe_custom = FeatureEngineer(enabled_groups=["taste", "equipment"])
    expected_count = len(FEATURE_GROUP_MAP["taste"]) + len(FEATURE_GROUP_MAP["equipment"])
    print(f"Custom features count: {len(fe_custom.feature_cols)} (expected {expected_count})")
    assert len(fe_custom.feature_cols) == expected_count

    # 4. Test generate() with legacy
    candidates_df = pd.DataFrame({"user_id": ["1", "1"], "recipe_id": ["r1", "r2"]})
    # Mock dataframes with minimum required columns
    users_df = pd.DataFrame(
        {
            "user_id": ["1"],
            "owned_equipment": ["[]"],
            "available_products": ["[]"],
            "dietary_restrictions": ["[]"],
            "preferred_portion_size": ["medium"],
            "taste_pref_bitterness": [3],
            "taste_pref_sweetness": [3],
            "taste_pref_acidity": [3],
            "taste_pref_body": [3],
            "preferred_strength": [3],
        }
    )
    recipes_df = pd.DataFrame(
        {
            "recipe_id": ["r1", "r2"],
            "required_equipment": ["[]", "[]"],
            "required_products": ["{}", "{}"],
            "tags": ["[]", "[]"],
            "difficulty": ["beginner", "beginner"],
            "taste_bitterness": [3, 4],
            "taste_sweetness": [3, 4],
            "taste_acidity": [3, 4],
            "taste_body": [3, 4],
            "strength": [3, 4],
            "portion_size_ml": [250, 250],
            "preparation_time_minutes": [10, 15],
        }
    )

    features = fe_legacy.generate(candidates_df, users_df, recipes_df)
    print(f"Generated features shape: {features.shape}")
    assert features.shape[1] == 80
    assert list(features.columns) == fe_legacy.feature_cols

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_feature_selection()
