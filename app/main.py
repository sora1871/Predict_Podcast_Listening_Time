from fastapi import FastAPI
from pydantic import BaseModel
from app.predict_api import predict_single  # predict.py の予測関数を呼び出す
import matplotlib
matplotlib.rcParams["font.family"] = "IPAexGothic"  # または "Meiryo", "MS Gothic" など

app = FastAPI(
    title="Podcast Listening Time Prediction API",
    description="API for predicting podcast listening time using LightGBM 5-fold model",
    version="1.0.0"
)

# -------------------------------
# 入力データのスキーマ（前処理前の「生の」データ形式）
# -------------------------------
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

# -------------------------------
# 動作確認用（GETリクエスト）
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Podcast Listening Time Prediction API is running!"}

# -------------------------------
# 予測エンドポイント（POSTリクエスト）
# -------------------------------
@app.post("/predict")
def predict(data: PodcastInput):
    input_data = data.dict()                  # 入力データ（BaseModel）→ dict に変換
    prediction = predict_single(input_data)   # predict.py の関数で予測
    return {"predicted_listening_time": prediction}
