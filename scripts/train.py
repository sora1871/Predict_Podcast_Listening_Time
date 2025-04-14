# scripts/train.py

import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def train_lgb_regression(input_x,
                         input_y,
                         input_id,
                         params,
                         list_nfold=[0,1,2,3,4],
                         n_splits=5,
                         save_dir="models"):
    train_oof = np.zeros(len(input_x))
    metrics = []
    imp = pd.DataFrame()

    os.makedirs(save_dir, exist_ok=True)
    cv = list(KFold(n_splits=n_splits, shuffle=True, random_state=123).split(input_x))
    
    for nfold in list_nfold:
        print("-" * 20, f"Fold {nfold}", "-" * 20)
        idx_tr, idx_va = cv[nfold]
        x_tr, y_tr, id_tr = input_x.loc[idx_tr, :], input_y[idx_tr], input_id.loc[idx_tr, :]
        x_va, y_va, id_va = input_x.loc[idx_va, :], input_y[idx_va], input_id.loc[idx_va, :]
        print(f"Train shape: {x_tr.shape}, Validation shape: {x_va.shape}")
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            x_tr, y_tr,
            eval_set=[(x_tr, y_tr), (x_va, y_va)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        fname_lgb = os.path.join(save_dir, f"model_lgb_fold{nfold}.joblib")
        joblib.dump(model, fname_lgb)
        
        y_tr_pred = model.predict(x_tr)
        y_va_pred = model.predict(x_va)
        
        metric_tr = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
        metric_va = np.sqrt(mean_squared_error(y_va, y_va_pred))
        metrics.append([nfold, metric_tr, metric_va])
        print(f"[RMSE] Train: {metric_tr:.4f}, Validation: {metric_va:.4f}")
        
        train_oof[idx_va] = y_va_pred
        
        _imp = pd.DataFrame({"col": input_x.columns, "imp": model.feature_importances_, "nfold": nfold})
        imp = pd.concat([imp, _imp])
      
    print("-" * 20, "Training Results", "-" * 20)
    
    metrics = np.array(metrics)
    print("[CV] Train RMSE: {:.4f}±{:.4f}, Validation RMSE: {:.4f}±{:.4f}".format(
        metrics[:, 1].mean(), metrics[:, 1].std(),
        metrics[:, 2].mean(), metrics[:, 2].std(),
    ))
    print("[OOF RMSE] {:.4f}".format(np.sqrt(mean_squared_error(input_y, train_oof))))
    
    train_oof = pd.concat([
        input_id.reset_index(drop=True),
        pd.DataFrame({"pred": train_oof})
    ], axis=1)
    
    imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index()
    imp.columns = ["col", "imp", "imp_std"]
    
    print("Training completed. Models saved to:", save_dir)
    
    return train_oof, imp, metrics
