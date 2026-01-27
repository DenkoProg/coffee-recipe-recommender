import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import json

class FeatureEngineer:
    def __init__(self):
        self.feature_cols = [
            'taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body',
            'strength', 'preparation_time_minutes',
            'diff_bitterness', 'diff_sweetness', 'diff_acidity', 'diff_body',
            'diff_strength', 'is_strength_match',
            'preference_mismatch',
            'diff_real_bitterness',
            'diff_real_sweetness',
            'diff_real_acidity',
            'diff_real_body'
        ]

    def _calculate_user_calibration(self, interactions_df, recipes_df, users_df):
        """
        Твій код, адаптований для генерації мапи реальних смаків.
        Повертає DataFrame з user_id та 'real_' колонками.
        """
        full_data = interactions_df.merge(recipes_df, on='recipe_id', how='left')
        
        good_interactions = full_data[full_data['rating'] >= 4.0]
        
        if good_interactions.empty:
            return pd.DataFrame()
        real_profile = good_interactions.groupby('user_id')[
            ['taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body']
        ].mean()
        
        real_profile.columns = [f"real_{col.replace('taste_', '')}" for col in real_profile.columns]
        
        comparison = real_profile.join(users_df.set_index('user_id'), how='left')
        
        def get_dist(row):
            v_stated = [
                row.get('taste_pref_bitterness', 0.5), 
                row.get('taste_pref_sweetness', 0.5), 
                row.get('taste_pref_acidity', 0.5), 
                row.get('taste_pref_body', 0.5)
            ]
            v_real = [
                row['real_bitterness'], 
                row['real_sweetness'], 
                row['real_acidity'], 
                row['real_body']
            ]
            return euclidean(v_stated, v_real) / 2.0

        comparison['preference_mismatch'] = comparison.apply(get_dist, axis=1)
        
        return comparison.reset_index()

    def generate(self, candidates_df, users_df, recipes_df, train_interactions_df=None):
        # 1. Примусова чистка вхідних таблиць перед мерджем
        # Це видаляє пробіли з початку і кінця назв колонок
        users_df.columns = users_df.columns.str.strip()
        recipes_df.columns = recipes_df.columns.str.strip()
        candidates_df.columns = candidates_df.columns.str.strip()

        # 2. Перевірка типів для мерджу (щоб уникнути пустого результату)
        candidates_df['user_id'] = candidates_df['user_id'].astype(str)
        users_df['user_id'] = users_df['user_id'].astype(str)
        candidates_df['recipe_id'] = candidates_df['recipe_id'].astype(str)
        recipes_df['recipe_id'] = recipes_df['recipe_id'].astype(str)

        # 3. Мердж
        df = candidates_df.merge(users_df, on='user_id', how='left')
        df = df.merge(recipes_df, on='recipe_id', how='left')

        # --- NUCLEAR CLEAN (Вирішення твоєї помилки) ---
        # Ми перейменовуємо колонки, видаляючи будь-яке сміття
        # Якщо там був '\ufefftaste_pref...', він стане 'taste_pref...'
        df.columns = df.columns.str.strip()
        
        # Додаткова страховка: Debug print, якщо колонка все одно не знайдена
        if 'taste_pref_bitterness' not in df.columns:
            print("\n🛑 CRITICAL ERROR: Columns mismatch after cleanup!")
            print("Real columns in DataFrame (repr):")
            # repr() покаже невидимі символи, як \r, \t або подвійні пробіли
            print([repr(c) for c in df.columns if 'bitterness' in c])
            raise KeyError("Column 'taste_pref_bitterness' not found even after strip!")
        # -----------------------------------------------

        # 4. Обчислення фічей (твій код)
        # 2. Taste Features (Math)
        for attr in ['bitterness', 'sweetness', 'acidity', 'body']:
            # Використовуємо .get() або fillna, щоб код не падав, якщо є NaN
            u_col = f'taste_pref_{attr}'
            r_col = f'taste_{attr}'
            
            # Якщо раптом якоїсь колонки все ж нема - кине зрозумілу помилку
            if u_col not in df.columns: raise KeyError(f"Missing user column: {u_col}")
            if r_col not in df.columns: raise KeyError(f"Missing recipe column: {r_col}")

            df[f'diff_{attr}'] = abs(df[u_col] - df[r_col])
            
        # 3. Strength Features
        df['diff_strength'] = abs(df['preferred_strength'] - df['strength'])
        df['is_strength_match'] = (df['preferred_strength'] == df['strength']).astype(int)
        
        # 4. Fill NaNs final
        # Переконуємось, що потрібні колонки існують перед поверненням
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            # Якщо якихось фічей немає, створимо їх нулями (наприклад, diff_real_...)
            for c in missing_cols:
                df[c] = 0.0
        
        df[self.feature_cols] = df[self.feature_cols].fillna(0)
        
        return df[self.feature_cols]