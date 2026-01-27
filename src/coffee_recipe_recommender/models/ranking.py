import joblib
import lightgbm as lgb


class LightGBMRankerModel:
    def __init__(self, params=None):
        if params is None:
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "boosting_type": "gbdt",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbosity": -1,
            }
        self.model = lgb.LGBMRanker(**params)

    def fit(self, X, y, groups, eval_set=None):
        """
        X, y, groups: Training data
        eval_set: Tuple (X_val, y_val, groups_val) <-- Ми передаємо це з train_ranker.py
        """
        # --- FIX: Правильна підготовка валідаційних даних ---
        real_eval_set = None
        real_eval_group = None
        callbacks = None

        if eval_set:
            # Розпаковуємо наш кастомний кортеж (X, y, qids)
            X_val, y_val, groups_val = eval_set

            # LightGBM вимагає список кортежів [(X, y)]
            real_eval_set = [(X_val, y_val)]

            # LightGBM вимагає список груп [groups]
            real_eval_group = [groups_val]

            # Early stopping активуємо тільки якщо є валідація
            # (Цифра 10 означає: якщо за 10 ітерацій скор не покращився - стоп)
            callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]

        # Запуск тренування
        self.model.fit(
            X,
            y,
            group=groups,
            eval_set=real_eval_set,
            eval_group=real_eval_group,
            eval_metric="ndcg",
            callbacks=callbacks,
        )

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
