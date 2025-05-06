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

# Add project root to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -----------------------------
# 🎧 App Title
# -----------------------------
st.title("🎧 Podcast Listening Time Prediction App")
st.markdown("Enter the genre, popularity scores, and number of ads to predict expected listening time (in minutes).")

# -----------------------------
# 📝 User Inputs
# -----------------------------
genre = st.selectbox("Genre", ["Technology", "Education", "Comedy", "Society & Culture"])
host_popularity = st.slider("Host Popularity (%)", 0, 100, 50)
guest_popularity = st.slider("Guest Popularity (%)", 0, 100, 50)
ads = st.number_input("Number of Ads", min_value=0, max_value=10, value=1)

# -----------------------------
# 🔮 Prediction Button
# -----------------------------
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

    result = predict_lgb_regression(base_df, input_id, model_dir="models")
    pred_minutes = round(result["pred"].iloc[0], 2)
    st.success(f"📈 Predicted Listening Time: **{pred_minutes} minutes**")

    # ====================
    # 🎯 Sensitivity Analysis Charts
    # ====================

    # 1. Host Popularity
    st.subheader("📊 Effect of Host Popularity")
    vals = list(range(0, 101, 5))
    preds = []
    for v in vals:
        df = base_df.copy()
        df["Host_Popularity_percentage"] = v
        preds.append(predict_lgb_regression(df, input_id, model_dir="models")["pred"].iloc[0])
    fig, ax = plt.subplots()
    ax.plot(vals, preds)
    ax.set_xlabel("Host Popularity (%)")
    ax.set_ylabel("Listening Time (minutes)")
    ax.set_title("Host Popularity vs. Listening Time")
    st.pyplot(fig)

    # 2. Guest Popularity
    st.subheader("📊 Effect of Guest Popularity")
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
    ax.set_xlabel("Guest Popularity (%)")
    ax.set_ylabel("Listening Time (minutes)")
    ax.set_title("Guest Popularity vs. Listening Time")
    st.pyplot(fig)

    # 3. Number of Ads
    st.subheader("📊 Effect of Number of Ads")
    vals = list(range(0, 4))
    preds = []
    for v in vals:
        df = base_df.copy()
        df["Number_of_Ads"] = v
        preds.append(predict_lgb_regression(df, input_id, model_dir="models")["pred"].iloc[0])
    fig, ax = plt.subplots()
    ax.bar(vals, preds)
    ax.set_xlabel("Number of Ads")
    ax.set_ylabel("Listening Time (minutes)")
    ax.set_title("Number of Ads vs. Listening Time")
    st.pyplot(fig)

    # 4. Genre
    st.subheader("📊 Effect of Genre")
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
    ax.set_xlabel("Genre")
    ax.set_ylabel("Listening Time (minutes)")
    ax.set_title("Listening Time by Genre")
    st.pyplot(fig)
