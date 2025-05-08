import joblib
import pandas as pd
from pathlib import Path
import sys
# -------------------------------
# scripts ディレクトリを import パスに追加
# -----------

# basic_feature.py の前処理関数を import
from scripts.basic_feature import preprocess_features
from scripts.feature_isna import handle_missing_values  # ← 新しく追加

# -------------------------------
# モデル（5fold）をすべて読み込む
# -------------------------------
models = []
model_dir = Path(__file__).resolve().parent.parent / "models"

for i in range(5):
    model_path = model_dir / f"model_lgb_fold{i}.joblib"
    models.append(joblib.load(model_path))

EXPECTED_COLUMNS = [
    "Podcast_Name",
    "Episode_Title",
    "Episode_Length_minutes",
    "Genre",
    "Host_Popularity_percentage",
    "Publication_Day",
    "Publication_Time",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
    "Episode_Sentiment",
    "Episode_Length_minutes_raw",
    "Episode_Length_minutes_was_missing",
    "Guest_Popularity_percentage_raw",
    "Guest_Popularity_percetage_was_missing"
]


# -------------------------------
# 前処理関数（API用に再ラップ）
# -------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_features(df)
    df = handle_missing_values(df)  # 欠損処理をここで追加
    df = df[EXPECTED_COLUMNS]  # ← モデルが期待する列に揃える（順番も一致）
    return df

# -------------------------------
# API から呼び出される予測関数（1件分）
# -------------------------------
def predict_single(input_dict: dict) -> float:
    df = pd.DataFrame([input_dict])        # JSON → DataFrame
    df = preprocess(df)                    # 前処理を適用
    preds = [model.predict(df)[0] for model in models]  # 5モデルで予測
    return float(sum(preds) / len(preds))  # 平均を返す
