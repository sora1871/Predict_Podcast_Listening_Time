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

@app.get("/")
def read_root():
    return {"message": "Podcast Listening Time Prediction API is running!"}

@app.post("/predict")
def predict(data: PodcastInput):
    input_data = data.dict()
    prediction = predict_single(input_data)

    # JSONにできない値は None に変換
    if isinstance(prediction, float) and not math.isfinite(prediction):
        prediction = None

    return {"predicted_listening_time": prediction}
