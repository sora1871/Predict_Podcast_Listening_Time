import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import requests

# FastAPIのURL（環境変数がなければローカル用）
API_URL = os.getenv("API_URL", "http://localhost:8000")

# フォント設定
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans JP', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)


from scripts.basic_feature import preprocess_features
from scripts.feature_isna import handle_missing_values

# 🎧 アプリのタイトル
st.title("🎧 Podcast Listening Time Prediction App")
st.markdown("Enter the genre, popularity scores, and number of ads to predict expected listening time (in minutes).")

# 🔗 FastAPIに予測リクエストを送る関数
def call_fastapi_api(data: pd.DataFrame) -> float:
    payload = data.iloc[0].to_dict()
    response = requests.post(f"{API_URL}/predict", json=payload)
    response.raise_for_status()
    return response.json()["predicted_listening_time"]

# 📝 ユーザー入力
genre = st.selectbox("Genre", ["Technology", "Education", "Comedy", "Society & Culture"])
host_popularity = st.slider("Host Popularity (%)", 0, 100, 50)
guest_popularity = st.slider("Guest Popularity (%)", 0, 100, 50)
ads = st.number_input("Number of Ads", min_value=0, max_value=10, value=1)

# 🔮 予測ボタンが押されたとき
if st.button("Predict"):
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

    # 前処理（FastAPIが期待する形式に変換）
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

    # FastAPI へリクエスト送信
    try:
        pred_minutes = round(call_fastapi_api(base_df), 2)
        st.success(f"📈 Predicted Listening Time: **{pred_minutes} minutes**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
