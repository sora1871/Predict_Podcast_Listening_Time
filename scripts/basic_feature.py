# scripts/basic_feature.py

import pandas as pd
from scripts.utils import reduce_mem_usage
from scripts.feature_isna import handle_missing_values

def preprocess_features(df):
    # object型 → category型に変換
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
    return df


def create_train_data(path="../data/train.csv"):
    """
    学習データの読み込み・前処理・分割をまとめて行う関数。
    
    Returns:
        x_train: 特徴量データ（DataFrame）
        y_train: 目的変数（Series）
        id_train: ID列（DataFrame）
    """
    df = pd.read_csv(path)
    df = reduce_mem_usage(df)
    df = preprocess_features(df)
    df = handle_missing_values(df)

    # データセット作成
    x_train = df.drop(["Listening_Time_minutes", "id"], axis=1)
    y_train = df["Listening_Time_minutes"]
    id_train = df[["id"]]

    return x_train, y_train, id_train
