import numpy as np
import pandas as pd

def generate_training_data(interactions_df, all_recipe_ids, n_candidates=100, random_state=42):
    """
    Додає до реальних інтеракцій негативні приклади (випадкові рецепти),
    щоб у кожного юзера було n_candidates (або менше, якщо рецептів всього мало).
    """
    np.random.seed(random_state)
    
    real_pairs = set(zip(interactions_df['user_id'], interactions_df['recipe_id']))
    all_recipes_arr = np.array(all_recipe_ids)
    
    new_rows = []
    
    print(f"Generating negative samples to reach ~{n_candidates} ")
    
    for user_id, group in interactions_df.groupby('user_id'):
        n_positives = len(group)
        n_negatives = n_candidates - n_positives
        
        if n_negatives <= 0:
            continue
            
        potential_negatives = np.random.choice(all_recipes_arr, size=n_negatives * 2, replace=True)
        
        count = 0
        for rid in potential_negatives:
            if (user_id, rid) not in real_pairs:
                new_rows.append({
                    'user_id': user_id,
                    'recipe_id': rid,
                    'rating': 0,
                    'completed': False,
                    'relevance': 0
                })
                real_pairs.add((user_id, rid))
                count += 1
                if count >= n_negatives:
                    break
                    
    negatives_df = pd.DataFrame(new_rows)
    
    positives_df = interactions_df.copy()
    
    positives_df['relevance'] = positives_df['rating'].fillna(0)
    
    mask_implicit = positives_df['rating'].isna() & (positives_df['completed'] == True)
    positives_df.loc[mask_implicit, 'relevance'] = 3.0
    
    mask_click = positives_df['rating'].isna() & (positives_df['completed'] == False)
    positives_df.loc[mask_click, 'relevance'] = 1.0

    full_train = pd.concat([positives_df, negatives_df], ignore_index=True)
    
    full_train = full_train.sort_values(by='user_id').reset_index(drop=True)
    
    return full_train