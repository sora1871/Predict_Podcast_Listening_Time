import numpy as np
import pandas as pd
import gc
import pickle
import os
import datetime as dt

#plot
import matplotlib.pyplot as plt

#LightGBM
import lightgbm as lgb

#sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

#表示桁数の指定
pd.options.display.float_format = '{:.4f}'.format

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --- scripts パス追加 ---
import sys
if "../scripts" not in sys.path:
    sys.path.append("../scripts")

# --- utils ---
import utils
import importlib
importlib.reload(utils)
from utils import reduce_mem_usage

# --- basic_feature ---
import basic_feature
importlib.reload(basic_feature)
from basic_feature import preprocess_features

# --- train_runner ---
import train_runner
importlib.reload(train_runner)
from train_runner import run_baseline_training

# --- train ---
import train
importlib.reload(train)
from train import train_lgb_regression


# テストデータ読み込み、メモリ削減、オブジェクトに変換
df = pd.read_csv("../data/test.csv")
df = reduce_mem_usage(df)
df = preprocess_features(df)

#特徴量エンジニアリング1
# 欠損フラグを追加（1: 欠損あり、0: 欠損なし）
# 補完前のオリジナル列（欠損あり）をコピー
df['Episode_Length_minutes_raw'] = df['Episode_Length_minutes']
df['Episode_Length_minutes_was_missing'] = df['Episode_Length_minutes'].isna().astype(int)
# 欠損を中央値で補完
df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())

#上記と同様の操作を行う
df['Guest_Popularity_percentage_raw'] = df['Guest_Popularity_percentage']
df['Guest_Popularity_percetage_was_missing'] = df['Guest_Popularity_percentage'].isna().astype(int)
df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())


#データセット作成
x_train = df.drop(["id",'Episode_Sentiment','Publication_Time'], axis=1)
id_train = df[["id"]]

train_oof, imp, metrics = run_baseline_training(x_train, id_train)











