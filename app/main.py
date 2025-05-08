from fastapi import FastAPI
from pydantic import BaseModel
from app.predict_api import predict_single
import matplotlib
import math

matplotlib.rcParams["font.family"] = "IPAexGothic"

app = FastAPI(
    title="Podcast Listening Time Prediction API",
    description="API for predicting podcast listening time using LightGBM 5-fold model",
    version="1.0.0"
)

# 入力データの構造
class PodcastInput(BaseModel):
    Podcast_Name: str
    Episode_Title: str
    Episode_Length_minutes: float
    Genre: str
    Host_Popularity_percentage: float
    Publication_Day: str
    Publication_Time: str
    Guest_Popularity_percentage: float
    Number_of_Ads: int
    Episode_Sentiment: str

from typing import Optional  # 追加

def sanitize_prediction(value: float) -> Optional[float]:
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


# ヘルスチェック用
@app.get("/")
def read_root():
    return {"message": "Podcast Listening Time Prediction API is running!"}

# 予測エンドポイント
@app.post("/predict")
def predict(data: PodcastInput):
    input_data = data.dict()
    
    # 予測値を取得
    prediction = predict_single(input_data)
    
    # JSONに安全な値に変換
    clean_prediction = sanitize_prediction(prediction)

    return {"predicted_listening_time": clean_prediction}
