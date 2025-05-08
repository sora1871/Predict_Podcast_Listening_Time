# scripts/train_runner.py

import pandas as pd
from scripts.train import train_lgb_regression

def run_baseline_training(x_train, y_train, id_train, save_dir="../models"):
    y_train = y_train.astype(float)
    # ハイパーパラメータ（ベースライン）
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'n_estimators': 1000,
        'random_state': 42
    }

    # 学習関数を呼び出す
    return train_lgb_regression(
        x_train,
        y_train,
        id_train,
        params=params,
        list_nfold=[0, 1, 2, 3, 4],
        n_splits=5,
        save_dir=save_dir
    )
