import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scripts.predict import predict_lgb_regression
from scripts.basic_feature import preprocess_features
from scripts.feature_isna import handle_missing_values

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans JP', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)


# モジュール検索パスにプロジェクトルートを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -----------------------------
# 🎧 アプリタイトル
# -----------------------------
st.title("🎧 ポッドキャストのリスニング時間予測アプリ")
st.markdown("ジャンルや出演者の人気度などを入力すると、予測されるリスニング時間（分）を表示します。")

# -----------------------------
# 📝 ユーザー入力（4項目）
# -----------------------------
genre = st.selectbox("ジャンル", ["Technology", "Education", "Comedy", "Society & Culture"])
host_popularity = st.slider("ホストの人気度（%）", 0, 100, 50)
guest_popularity = st.slider("ゲストの人気度（%）", 0, 100, 50)
ads = st.number_input("広告の数", min_value=0, max_value=10, value=1)

# -----------------------------
# 🔮 予測ボタン
# -----------------------------
if st.button("予測する"):
    # --- 入力値＋デフォルト値をまとめてDataFrame化 ---
    base_df = pd.DataFrame([{
        "Podcast_Name": "Default Podcast",
        "Episode_Title": "Episode X",
        "Episode_Length_minutes": np.nan,
        "Genre": genre,
        "Host_Popularity_percentage": host_popularity,
        "Publication_Day": "Monday",
        "Publication_Time": "08:00",
        "Guest_Popularity_percentage": guest_popularity,
        "Number_of_Ads": ads,
        "Episode_Sentiment": "Neutral"
    }])

    # --- 前処理 ---
    base_df = handle_missing_values(base_df)
    base_df = preprocess_features(base_df)

    expected_columns = [
        "Podcast_Name", "Episode_Title", "Episode_Length_minutes", "Genre",
        "Host_Popularity_percentage", "Publication_Day", "Publication_Time",
        "Guest_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment",
        "Episode_Length_minutes_raw", "Episode_Length_minutes_was_missing",
        "Guest_Popularity_percentage_raw", "Guest_Popularity_percetage_was_missing"
    ]
    base_df = base_df[expected_columns]
    input_id = pd.DataFrame({"id": [0]})

    # --- 予測 ---
    result = predict_lgb_regression(base_df, input_id, model_dir="models")
    pred_minutes = round(result["pred"].iloc[0], 2)
    st.success(f"📈 予測リスニング時間は **{pred_minutes} 分** です。")

    # ====================
    # 🎯 感度分析グラフ
    # ====================

    # 1. ホスト人気度
    st.subheader("📊 ホスト人気度による変化")
    vals = list(range(0, 101, 5))
    preds = []
    for v in vals:
        df = base_df.copy()
        df["Host_Popularity_percentage"] = v
        preds.append(predict_lgb_regression(df, input_id, model_dir="models")["pred"].iloc[0])
    fig, ax = plt.subplots()
    ax.plot(vals, preds)
    ax.set_xlabel("ホスト人気度（%）")
    ax.set_ylabel("リスニング時間（分）")
    ax.set_title("ホスト人気度 vs リスニング時間")
    st.pyplot(fig)

    # 2. ゲスト人気度
    st.subheader("📊 ゲスト人気度による変化")
    vals = list(range(0, 101, 5))
    preds = []
    for v in vals:
        df = base_df.copy()
        df["Guest_Popularity_percentage"] = v
        df = handle_missing_values(df)
        df = preprocess_features(df)
        df = df[expected_columns]
        preds.append(predict_lgb_regression(df, input_id, model_dir="models")["pred"].iloc[0])
    fig, ax = plt.subplots()
    ax.plot(vals, preds)
    ax.set_xlabel("ゲスト人気度（%）")
    ax.set_ylabel("リスニング時間（分）")
    ax.set_title("ゲスト人気度 vs リスニング時間")
    st.pyplot(fig)

    # 3. 広告数
    st.subheader("📊 広告数による変化")
    vals = list(range(0, 4))
    preds = []
    for v in vals:
        df = base_df.copy()
        df["Number_of_Ads"] = v
        preds.append(predict_lgb_regression(df, input_id, model_dir="models")["pred"].iloc[0])
    fig, ax = plt.subplots()
    ax.bar(vals, preds)
    ax.set_xlabel("広告数")
    ax.set_ylabel("リスニング時間（分）")
    ax.set_title("広告数 vs リスニング時間")
    st.pyplot(fig)

    # 4. ジャンル
    st.subheader("📊 ジャンルによる変化")
    genre_list = ["Technology", "Education", "Comedy", "Sports"]
    preds = []
    for g in genre_list:
        df = base_df.copy()
        df["Genre"] = g
        df = preprocess_features(df)
        df = df[expected_columns]
        preds.append(predict_lgb_regression(df, input_id, model_dir="models")["pred"].iloc[0])
    fig, ax = plt.subplots()
    ax.bar(genre_list, preds)
    ax.set_xlabel("ジャンル")
    ax.set_ylabel("リスニング時間（分）")
    ax.set_title("ジャンル別リスニング時間予測")
    st.pyplot(fig)
