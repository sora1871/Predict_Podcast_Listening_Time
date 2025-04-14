def preprocess_features(df):
     # object型 → category型に変換
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")

    return df
