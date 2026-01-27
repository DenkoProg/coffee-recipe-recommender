import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity


def generate_training_data(interactions_df, all_recipe_ids, n_candidates=100, random_state=42):
    np.random.seed(random_state)

    real_pairs = set(zip(interactions_df["user_id"], interactions_df["recipe_id"], strict=False))
    all_recipes_arr = np.array(all_recipe_ids)

    new_rows = []

    print(f"Generating negative samples to reach ~{n_candidates} ")

    for user_id, group in interactions_df.groupby("user_id"):
        n_positives = len(group)
        n_negatives = n_candidates - n_positives

        if n_negatives <= 0:
            continue

        potential_negatives = np.random.choice(all_recipes_arr, size=n_negatives * 2, replace=True)

        count = 0
        for rid in potential_negatives:
            if (user_id, rid) not in real_pairs:
                new_rows.append(
                    {"user_id": user_id, "recipe_id": rid, "rating": 0, "completed": False, "relevance": 0}
                )
                real_pairs.add((user_id, rid))
                count += 1
                if count >= n_negatives:
                    break

    negatives_df = pd.DataFrame(new_rows)

    positives_df = interactions_df.copy()

    positives_df["relevance"] = positives_df["rating"].fillna(0)

    mask_implicit = positives_df["rating"].isna() & (positives_df["completed"])
    positives_df.loc[mask_implicit, "relevance"] = 3.0

    mask_click = positives_df["rating"].isna() & (~positives_df["completed"])
    positives_df.loc[mask_click, "relevance"] = 1.0

    full_train = pd.concat([positives_df, negatives_df], ignore_index=True)

    full_train = full_train.sort_values(by="user_id").reset_index(drop=True)

    return full_train


class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineer with comprehensive feature list"""

        # Basic taste features
        self.taste_features = [
            "taste_bitterness",
            "taste_sweetness",
            "taste_acidity",
            "taste_body",
            "strength",
            "preparation_time_minutes",
            "difficulty_numeric",
            "portion_size_ml",
        ]

        # Taste difference features
        self.taste_diff_features = [
            "diff_bitterness",
            "diff_sweetness",
            "diff_acidity",
            "diff_body",
            "diff_strength",
            "is_strength_match",
        ]

        # Advanced taste match features
        self.taste_match_features = [
            "taste_cosine_similarity",
            "taste_euclidean_distance",
            "taste_manhattan_distance",
            "taste_diff_mean",
            "taste_weighted_similarity",
        ]

        # User aggregate features
        self.user_agg_features = [
            "user_taste_pref_sum",
            "user_taste_pref_mean",
            "user_taste_pref_std",
            "user_taste_pref_max",
            "user_taste_pref_min",
            "user_dominant_taste_value",
        ]

        # Recipe aggregate features
        self.recipe_agg_features = [
            "recipe_taste_sum",
            "recipe_taste_mean",
            "recipe_taste_std",
            "recipe_taste_complexity",
            "recipe_taste_balance",
        ]

        # Equipment features
        self.equipment_features = [
            "owned_equipment_count",
            "required_equipment_count",
            "equipment_match",
            "equipment_coverage",
            "equipment_missing_count",
            "has_espresso_machine",
            "has_grinder",
            "has_milk_frother",
            "requires_espresso_machine",
            "requires_milk_frother",
            "equipment_sophistication_user",
            "equipment_sophistication_recipe",
        ]

        # Technical match features
        self.technical_features = [
            "portion_size_match",
            "portion_size_diff",
            "portion_size_ratio",
            "prep_time_acceptable",
            "prep_time_ratio",
        ]

        # Tag and category features
        self.tag_features = [
            "tags_count",
            "is_hot",
            "is_cold",
            "is_iced",
            "is_classic",
            "is_quick",
            "is_specialty",
            "is_strong",
            "is_sweet",
        ]

        # Dietary features
        self.dietary_features = [
            "dietary_restrictions_count",
            "is_vegan",
            "is_lactose_intolerant",
            "dietary_compatible",
            "requires_milk",
        ]

        # Historical features (for warm users)
        self.historical_features = [
            "user_total_interactions",
            "user_avg_rating",
            "user_rating_std",
            "user_completion_rate",
            "user_rated_count",
            "recipe_total_interactions",
            "recipe_avg_rating",
            "recipe_rating_std",
            "recipe_completion_rate",
            "recipe_popularity_score",
            "user_tried_similar_count",
            "user_avg_rating_similar",
            "similar_recipe_completion_rate",
        ]

        # Cold start features
        self.cold_start_features = ["recipe_global_popularity", "recipe_category_popularity"]

        # Feature interactions
        self.interaction_features = [
            "taste_match_x_equipment",
            "taste_match_x_popularity",
            "strength_match_x_completion",
        ]

        # All features combined
        self.feature_cols = (
            self.taste_features
            + self.taste_diff_features
            + self.taste_match_features
            + self.user_agg_features
            + self.recipe_agg_features
            + self.equipment_features
            + self.technical_features
            + self.tag_features
            + self.dietary_features
            + self.historical_features
            + self.cold_start_features
            + self.interaction_features
            + ["preference_mismatch"]  # from original code
        )

        # Precomputed stats (will be filled in fit())
        self.user_stats = None
        self.recipe_stats = None
        self.global_stats = None

    def fit(self, users_df, recipes_df, train_interactions_df):
        """
        Precompute statistics from training data

        Args:
            users_df: Users dataframe
            recipes_df: Recipes dataframe
            train_interactions_df: Training interactions
        """
        print("🔧 Computing user statistics...")
        self._compute_user_stats(users_df, train_interactions_df)

        print("🔧 Computing recipe statistics...")
        self._compute_recipe_stats(recipes_df, train_interactions_df)

        print("🔧 Computing global statistics...")
        self._compute_global_stats(users_df, recipes_df, train_interactions_df)

        print("✅ Feature engineering fit complete!")

    def _parse_json_columns(self, df, json_columns):
        """Helper to parse JSON columns safely"""
        for col in json_columns:
            if col in df.columns:
                df[f"{col}_list"] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
                )
        return df

    def _compute_user_stats(self, users_df, train_interactions_df):
        """Compute aggregate statistics for each user"""

        # User interaction stats
        user_stats = (
            train_interactions_df.groupby("user_id")
            .agg({"interaction_id": "count", "rating": ["mean", "std", "count"], "completed": "mean"})
            .reset_index()
        )

        user_stats.columns = [
            "user_id",
            "user_total_interactions",
            "user_avg_rating",
            "user_rating_std",
            "user_rated_count",
            "user_completion_rate",
        ]

        # Fill NaN for users with no ratings
        user_stats["user_avg_rating"] = user_stats["user_avg_rating"].fillna(3.0)
        user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0.0)

        self.user_stats = user_stats

    def _compute_recipe_stats(self, recipes_df, train_interactions_df):
        """Compute aggregate statistics for each recipe"""

        recipe_stats = (
            train_interactions_df.groupby("recipe_id")
            .agg({"interaction_id": "count", "rating": ["mean", "std", "count"], "completed": "mean"})
            .reset_index()
        )

        recipe_stats.columns = [
            "recipe_id",
            "recipe_total_interactions",
            "recipe_avg_rating",
            "recipe_rating_std",
            "recipe_rated_count",
            "recipe_completion_rate",
        ]

        # Fill NaN
        recipe_stats["recipe_avg_rating"] = recipe_stats["recipe_avg_rating"].fillna(3.0)
        recipe_stats["recipe_rating_std"] = recipe_stats["recipe_rating_std"].fillna(0.0)

        # Popularity score (normalized)
        max_interactions = recipe_stats["recipe_total_interactions"].max()
        recipe_stats["recipe_popularity_score"] = (
            recipe_stats["recipe_total_interactions"] / max_interactions if max_interactions > 0 else 0
        )

        # Global popularity for cold start
        recipe_stats["recipe_global_popularity"] = recipe_stats["recipe_popularity_score"]

        self.recipe_stats = recipe_stats

    def _compute_global_stats(self, users_df, recipes_df, train_interactions_df):
        """Compute global statistics"""

        self.global_stats = {
            "global_avg_rating": train_interactions_df["rating"].mean(),
            "global_completion_rate": train_interactions_df["completed"].mean(),
            "total_users": users_df["user_id"].nunique(),
            "total_recipes": recipes_df["recipe_id"].nunique(),
        }

    def _add_user_features(self, df, users_df):
        """Add user-level features"""

        # Parse JSON columns
        users_df = self._parse_json_columns(
            users_df, ["owned_equipment", "available_products", "dietary_restrictions"]
        )

        # Equipment counts
        users_df["owned_equipment_count"] = users_df["owned_equipment_list"].apply(len)
        users_df["available_products_count"] = users_df["available_products_list"].apply(len)
        users_df["dietary_restrictions_count"] = users_df["dietary_restrictions_list"].apply(len)

        # Specific equipment
        users_df["has_espresso_machine"] = users_df["owned_equipment_list"].apply(
            lambda x: int("espresso_machine" in x)
        )
        users_df["has_grinder"] = users_df["owned_equipment_list"].apply(lambda x: int("grinder" in x))
        users_df["has_milk_frother"] = users_df["owned_equipment_list"].apply(lambda x: int("milk_frother" in x))

        # Equipment sophistication
        equipment_sophistication = {
            "espresso_machine": 3,
            "grinder": 2,
            "milk_frother": 2,
            "pour_over": 1,
            "french_press": 1,
            "kettle": 1,
            "moka_pot": 2,
            "aeropress": 2,
            "cold_brew_maker": 2,
        }
        users_df["equipment_sophistication_user"] = users_df["owned_equipment_list"].apply(
            lambda equip_list: sum(equipment_sophistication.get(e, 1) for e in equip_list)
        )

        # Dietary restrictions
        users_df["is_vegan"] = users_df["dietary_restrictions_list"].apply(lambda x: int("vegan" in x))
        users_df["is_lactose_intolerant"] = users_df["dietary_restrictions_list"].apply(
            lambda x: int("lactose_intolerant" in x)
        )

        # Portion size numeric
        portion_size_mapping = {"small": 150, "medium": 250, "large": 350}
        users_df["preferred_portion_size_ml"] = users_df["preferred_portion_size"].map(portion_size_mapping)

        # Taste preference aggregates
        taste_cols = ["taste_pref_bitterness", "taste_pref_sweetness", "taste_pref_acidity", "taste_pref_body"]
        users_df["user_taste_pref_sum"] = users_df[taste_cols].sum(axis=1)
        users_df["user_taste_pref_mean"] = users_df[taste_cols].mean(axis=1)
        users_df["user_taste_pref_std"] = users_df[taste_cols].std(axis=1)
        users_df["user_taste_pref_max"] = users_df[taste_cols].max(axis=1)
        users_df["user_taste_pref_min"] = users_df[taste_cols].min(axis=1)
        users_df["user_dominant_taste_value"] = users_df[taste_cols].max(axis=1)

        return df.merge(users_df, on="user_id", how="left", suffixes=("", "_user"))

    def _add_recipe_features(self, df, recipes_df):
        """Add recipe-level features"""

        # Parse JSON columns
        recipes_df = self._parse_json_columns(recipes_df, ["required_equipment", "required_products", "tags"])

        # Equipment and products
        recipes_df["required_equipment_count"] = recipes_df["required_equipment_list"].apply(len)
        recipes_df["required_products_count"] = recipes_df["required_products_list"].apply(
            lambda x: len(x) if isinstance(x, dict) else 0
        )

        # Specific requirements
        recipes_df["requires_espresso_machine"] = recipes_df["required_equipment_list"].apply(
            lambda x: int("espresso_machine" in x)
        )
        recipes_df["requires_milk_frother"] = recipes_df["required_equipment_list"].apply(
            lambda x: int("milk_frother" in x)
        )

        # Check if requires milk
        recipes_df["requires_milk"] = recipes_df["required_products_list"].apply(
            lambda x: int(any("milk" in str(k).lower() for k in x.keys())) if isinstance(x, dict) else 0
        )

        # Equipment sophistication
        equipment_sophistication = {
            "espresso_machine": 3,
            "grinder": 2,
            "milk_frother": 2,
            "pour_over": 1,
            "french_press": 1,
            "kettle": 1,
            "moka_pot": 2,
            "aeropress": 2,
            "cold_brew_maker": 2,
        }
        recipes_df["equipment_sophistication_recipe"] = recipes_df["required_equipment_list"].apply(
            lambda equip_list: sum(equipment_sophistication.get(e, 1) for e in equip_list)
        )

        # Difficulty numeric
        difficulty_mapping = {"beginner": 1, "intermediate": 2, "advanced": 3}
        recipes_df["difficulty_numeric"] = recipes_df["difficulty"].map(difficulty_mapping)

        # Tags
        recipes_df["tags_count"] = recipes_df["tags_list"].apply(len)
        recipes_df["is_hot"] = recipes_df["tags_list"].apply(lambda x: int("hot" in x))
        recipes_df["is_cold"] = recipes_df["tags_list"].apply(lambda x: int("cold" in x))
        recipes_df["is_iced"] = recipes_df["tags_list"].apply(lambda x: int("iced" in x))
        recipes_df["is_classic"] = recipes_df["tags_list"].apply(lambda x: int("classic" in x))
        recipes_df["is_quick"] = recipes_df["tags_list"].apply(lambda x: int("quick" in x))
        recipes_df["is_specialty"] = recipes_df["tags_list"].apply(lambda x: int("specialty" in x))
        recipes_df["is_strong"] = recipes_df["tags_list"].apply(lambda x: int("strong" in x))
        recipes_df["is_sweet"] = recipes_df["tags_list"].apply(lambda x: int("sweet" in x))

        # Taste aggregates
        taste_cols = ["taste_bitterness", "taste_sweetness", "taste_acidity", "taste_body"]
        recipes_df["recipe_taste_sum"] = recipes_df[taste_cols].sum(axis=1)
        recipes_df["recipe_taste_mean"] = recipes_df[taste_cols].mean(axis=1)
        recipes_df["recipe_taste_std"] = recipes_df[taste_cols].std(axis=1)

        # Taste complexity (entropy)
        recipes_df["recipe_taste_complexity"] = recipes_df[taste_cols].apply(lambda row: entropy(row + 0.01), axis=1)

        # Taste balance
        recipes_df["recipe_taste_balance"] = 1 - recipes_df["recipe_taste_std"]

        return df.merge(recipes_df, on="recipe_id", how="left", suffixes=("", "_recipe"))

    def _add_interaction_features(self, df):
        """Add user-recipe interaction features"""

        # === TASTE MATCHING FEATURES ===

        # Basic differences
        for attr in ["bitterness", "sweetness", "acidity", "body"]:
            u_col = f"taste_pref_{attr}"
            r_col = f"taste_{attr}"
            df[f"diff_{attr}"] = abs(df[u_col] - df[r_col])

        # Mean taste difference
        df["taste_diff_mean"] = df[["diff_bitterness", "diff_sweetness", "diff_acidity", "diff_body"]].mean(axis=1)

        # Cosine similarity
        def compute_taste_cosine_similarity(row):
            user_vector = np.array(
                [
                    row["taste_pref_bitterness"],
                    row["taste_pref_sweetness"],
                    row["taste_pref_acidity"],
                    row["taste_pref_body"],
                ]
            )
            recipe_vector = np.array(
                [row["taste_bitterness"], row["taste_sweetness"], row["taste_acidity"], row["taste_body"]]
            )
            # Avoid division by zero
            if np.linalg.norm(user_vector) == 0 or np.linalg.norm(recipe_vector) == 0:
                return 0
            return 1 - cosine(user_vector, recipe_vector)

        df["taste_cosine_similarity"] = df.apply(compute_taste_cosine_similarity, axis=1)

        # Euclidean distance
        def compute_euclidean(row):
            user_vector = np.array(
                [
                    row["taste_pref_bitterness"],
                    row["taste_pref_sweetness"],
                    row["taste_pref_acidity"],
                    row["taste_pref_body"],
                ]
            )
            recipe_vector = np.array(
                [row["taste_bitterness"], row["taste_sweetness"], row["taste_acidity"], row["taste_body"]]
            )
            return euclidean(user_vector, recipe_vector)

        df["taste_euclidean_distance"] = df.apply(compute_euclidean, axis=1)

        # Manhattan distance
        df["taste_manhattan_distance"] = (
            df["diff_bitterness"] + df["diff_sweetness"] + df["diff_acidity"] + df["diff_body"]
        )

        # Weighted similarity (prioritize bitterness and sweetness)
        df["taste_weighted_similarity"] = (
            0.3 * (1 - df["diff_bitterness"])
            + 0.3 * (1 - df["diff_sweetness"])
            + 0.2 * (1 - df["diff_acidity"])
            + 0.2 * (1 - df["diff_body"])
        )

        # === TECHNICAL MATCHING FEATURES ===

        # Strength
        df["diff_strength"] = abs(df["preferred_strength"] - df["strength"])
        df["is_strength_match"] = (df["preferred_strength"] == df["strength"]).astype(int)

        # Equipment match (CRITICAL!)
        def check_equipment_match(row):
            user_equipment = set(row.get("owned_equipment_list", []))
            required_equipment = set(row.get("required_equipment_list", []))
            return int(required_equipment.issubset(user_equipment))

        df["equipment_match"] = df.apply(check_equipment_match, axis=1)

        # Equipment coverage
        def compute_equipment_coverage(row):
            user_equipment = set(row.get("owned_equipment_list", []))
            required_equipment = set(row.get("required_equipment_list", []))
            if not required_equipment:
                return 1.0
            return len(required_equipment.intersection(user_equipment)) / len(required_equipment)

        df["equipment_coverage"] = df.apply(compute_equipment_coverage, axis=1)

        # Equipment missing count
        def compute_missing_equipment(row):
            user_equipment = set(row.get("owned_equipment_list", []))
            required_equipment = set(row.get("required_equipment_list", []))
            return len(required_equipment - user_equipment)

        df["equipment_missing_count"] = df.apply(compute_missing_equipment, axis=1)

        # Portion size
        df["portion_size_match"] = (df["preferred_portion_size_ml"] == df["portion_size_ml"]).astype(int)
        df["portion_size_diff"] = abs(df["preferred_portion_size_ml"] - df["portion_size_ml"])
        df["portion_size_ratio"] = df.apply(
            lambda row: min(
                row["preferred_portion_size_ml"] / (row["portion_size_ml"] + 1),
                row["portion_size_ml"] / (row["preferred_portion_size_ml"] + 1),
            ),
            axis=1,
        )

        # Preparation time (assume acceptable is mean + 50%)
        # For now, use a simple threshold
        df["prep_time_acceptable"] = (df["preparation_time_minutes"] <= 20).astype(int)
        df["prep_time_ratio"] = df["preparation_time_minutes"] / 20.0

        # Dietary compatibility
        def check_dietary_compatible(row):
            restrictions = row.get("dietary_restrictions_list", [])
            if not restrictions:
                return 1

            # Check vegan/dairy restrictions
            if any(r in ["vegan", "dairy_free", "lactose_intolerant"] for r in restrictions):
                if row.get("requires_milk", 0) == 1:
                    return 0

            return 1

        df["dietary_compatible"] = df.apply(check_dietary_compatible, axis=1)

        return df

    def _add_historical_features(self, df):
        """Add historical statistics features"""

        # Merge user stats
        if self.user_stats is not None:
            df = df.merge(self.user_stats, on="user_id", how="left")

            # Fill NaN for cold users
            df["user_total_interactions"] = df["user_total_interactions"].fillna(0)
            df["user_avg_rating"] = df["user_avg_rating"].fillna(self.global_stats["global_avg_rating"])
            df["user_rating_std"] = df["user_rating_std"].fillna(0)
            df["user_rated_count"] = df["user_rated_count"].fillna(0)
            df["user_completion_rate"] = df["user_completion_rate"].fillna(self.global_stats["global_completion_rate"])

        # Merge recipe stats
        if self.recipe_stats is not None:
            df = df.merge(self.recipe_stats, on="recipe_id", how="left")

            # Fill NaN for new recipes
            df["recipe_total_interactions"] = df["recipe_total_interactions"].fillna(0)
            df["recipe_avg_rating"] = df["recipe_avg_rating"].fillna(self.global_stats["global_avg_rating"])
            df["recipe_rating_std"] = df["recipe_rating_std"].fillna(0)
            df["recipe_rated_count"] = df["recipe_rated_count"].fillna(0)
            df["recipe_completion_rate"] = df["recipe_completion_rate"].fillna(
                self.global_stats["global_completion_rate"]
            )
            df["recipe_popularity_score"] = df["recipe_popularity_score"].fillna(0)
            df["recipe_global_popularity"] = df["recipe_global_popularity"].fillna(0)

        # Placeholder for user tried similar features (would need more complex logic)
        df["user_tried_similar_count"] = 0
        df["user_avg_rating_similar"] = df["user_avg_rating"]
        df["similar_recipe_completion_rate"] = df["recipe_completion_rate"]

        # Category popularity (placeholder)
        df["recipe_category_popularity"] = df["recipe_popularity_score"]

        return df

    def _add_feature_interactions(self, df):
        """Add feature interaction terms"""

        # Taste match * Equipment match
        df["taste_match_x_equipment"] = df["taste_cosine_similarity"] * df["equipment_match"]

        # Taste match * Popularity
        df["taste_match_x_popularity"] = df["taste_cosine_similarity"] * df["recipe_popularity_score"]

        # Strength match * Completion rate
        df["strength_match_x_completion"] = df["is_strength_match"] * df["recipe_completion_rate"]

        return df

    def generate(self, candidates_df, users_df, recipes_df, train_interactions_df=None):
        """
        Generate all features for candidate user-recipe pairs

        Args:
            candidates_df: DataFrame with user_id and recipe_id columns
            users_df: Users dataframe
            recipes_df: Recipes dataframe
            train_interactions_df: Training interactions (optional, for historical features)

        Returns:
            DataFrame with all features
        """

        # 1. Clean column names
        users_df.columns = users_df.columns.str.strip()
        recipes_df.columns = recipes_df.columns.str.strip()
        candidates_df.columns = candidates_df.columns.str.strip()

        # 2. Ensure consistent types
        candidates_df["user_id"] = candidates_df["user_id"].astype(str)
        users_df["user_id"] = users_df["user_id"].astype(str)
        candidates_df["recipe_id"] = candidates_df["recipe_id"].astype(str)
        recipes_df["recipe_id"] = recipes_df["recipe_id"].astype(str)

        # 3. Add user features
        df = self._add_user_features(candidates_df.copy(), users_df.copy())

        # 4. Add recipe features
        df = self._add_recipe_features(df, recipes_df.copy())

        # 5. Add interaction features
        df = self._add_interaction_features(df)

        # 6. Add historical features (if fit was called)
        if self.user_stats is not None and self.recipe_stats is not None:
            df = self._add_historical_features(df)
        else:
            # Add placeholder columns
            for col in self.historical_features:
                df[col] = 0

        # 7. Add feature interactions
        df = self._add_feature_interactions(df)

        # 8. Add preference mismatch (from original code)
        df["preference_mismatch"] = 0  # Placeholder, can be computed with _calculate_user_calibration

        # 9. Ensure all feature columns exist
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            for c in missing_cols:
                df[c] = 0.0

        # 10. Fill NaN values
        df[self.feature_cols] = df[self.feature_cols].fillna(0)
        print(df.keys())

        # X is pandas DataFrame
        X = df[self.feature_cols+["user_id", "recipe_id"]]
        X.to_parquet("artifacts/train_features.parquet")

        # also save column order
        df[self.feature_cols].columns.to_series().to_csv(
            "artifacts/feature_names.csv",
            index=False,
            header=False
        )

        return df[self.feature_cols]


# Example usage:
if __name__ == "__main__":
    # Load data
    # users_df = pd.read_csv("users.csv")
    # recipes_df = pd.read_csv("recipes.csv")
    # train_df = pd.read_csv("interactions_train.csv")

    # Initialize and fit
    # fe = FeatureEngineer()
    # fe.fit(users_df, recipes_df, train_df)

    # Generate features for candidates
    # candidates_df = pd.DataFrame({
    #     'user_id': ['user_00001', 'user_00001'],
    #     'recipe_id': ['recipe_espresso_001', 'recipe_latte_002']
    # })

    # features = fe.generate(candidates_df, users_df, recipes_df, train_df)
    # print(features.head())

    print("Feature Engineer ready to use!")
