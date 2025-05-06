#  Predict Podcast Listening Time

このプロジェクトは、ポッドキャストのメタデータ（ジャンル・出演者の人気度・広告数など）をもとに、  
ユーザーのリスニング時間（何分聞かれるか）を予測する機械学習アプリケーションです。

LightGBMで構築したモデルを用い、StreamlitでWebアプリ化。  
さらにユーザー入力に対する感度分析（グラフ表示）も可能にしています。

---

##  アプリURL（Streamlit Cloudで公開済み）

👉 [アプリを見る](https://predictpodcastlisteningtime-bmhp29rl8pneytab34x29h.streamlit.app/)  

---

##  使用した特徴量（全14項目）

| 特徴量名 | 説明 |
|----------|------|
| `Podcast_Name` | 番組名（カテゴリ） |
| `Episode_Title` | エピソードのタイトル |
| `Episode_Length_minutes` | エピソードの長さ（分） |
| `Genre` | ジャンル（カテゴリ） |
| `Host_Popularity_percentage` | ホストの人気度（%） |
| `Publication_Day` | 公開された曜日 |
| `Publication_Time` | 公開された時間（HH:MM） |
| `Guest_Popularity_percentage` | ゲストの人気度（%） |
| `Number_of_Ads` | 広告の数 |
| `Episode_Sentiment` | 内容のトーン（例: Positive） |
| `Episode_Length_minutes_raw` | 補完前の長さデータ |
| `Episode_Length_minutes_was_missing` | 長さの欠損フラグ |
| `Guest_Popularity_percentage_raw` | ゲスト人気度の生データ |
| `Guest_Popularity_percetage_was_missing` | ゲスト人気度の欠損フラグ |

---

##  アプリの主な機能

- ユーザーが指定した条件からリスニング時間を予測
- 5-Fold LightGBMモデルによる平均予測
- 入力特徴量ごとの感度分析をグラフで表示
- FastAPIによるAPI構築済み（別途利用可）

---

##  ディレクトリ構成
```
Predict_Podcast_Listening_Time/
├── app/
│ ├── main.py # FastAPI アプリ
│ └── predict_api.py
├── streamlit_app/
│ └── main.py # Streamlit UI
├── scripts/
│ ├── train.py
│ ├── predict.py
│ ├── train_runner.py
│ ├── basic_feature.py
│ ├── feature_isna.py
│ └── utils.py
├── models/ # LightGBMモデル (.joblib or .txt)
├── notebooks/
│ └── base.ipynb # EDA & 検証
├── requirements.txt
└── README.md
```

---

## ローカルでの実行方法

### 1. 環境構築

```bash
pip install -r requirements.txt

2. Streamlitアプリの起動

streamlit run streamlit_app/main.py

→ http://localhost:8501 が開きます。
 感度分析グラフ例

    ホスト人気度（0〜100%）による予測変化

    ゲスト人気度（0〜100%）

    広告数（0〜10個）

    ジャンルごとの影響
