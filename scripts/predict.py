import joblib
import pandas as pd
import numpy as np
import os

def predict_lgb_regression(input_x,
                            input_id,
                            list_nfold=[0, 1, 2, 3, 4],
                            model_dir="../models"):
    """
    複数のfoldで学習したLightGBMモデルを使って推論を行う関数

    Parameters:
    - input_x (pd.DataFrame): 推論に使用する特徴量
    - input_id (pd.DataFrame): ID列（推論結果と結合するため）
    - list_nfold (list): 使用するfold番号のリスト（例: [0,1,2,3,4]）
    - model_dir (str): モデルが格納されたディレクトリのパス

    Returns:
    - pd.DataFrame: IDと予測値を含むデータフレーム
    """
    # 推論結果を格納する配列
    pred = np.zeros((len(input_x), len(list_nfold)))
    
    # 各foldのモデルで推論
    for nfold in list_nfold:
        print("-" * 20, f"Fold {nfold}", "-" * 20)
        fname_lgb = os.path.join(model_dir, f"model_lgb_fold{nfold}.joblib")
        
        if not os.path.exists(fname_lgb):
            raise FileNotFoundError(f"Model file not found: {fname_lgb}")
        
        model = joblib.load(fname_lgb)
        pred[:, nfold] = model.predict(input_x)
    
    # 平均予測を計算し、IDと結合
    pred_df = pd.concat([
        input_id.reset_index(drop=True),
        pd.DataFrame({"pred": pred.mean(axis=1)})
    ], axis=1)
    
    print("Inference completed.")
    
    return pred_df
