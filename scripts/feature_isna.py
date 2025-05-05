import pandas as pd

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Episode_Length_minutes の欠損処理
    df['Episode_Length_minutes_raw'] = df['Episode_Length_minutes']
    df['Episode_Length_minutes_was_missing'] = df['Episode_Length_minutes'].isna().astype(int)
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())

    # Guest_Popularity_percentage の欠損処理
    df['Guest_Popularity_percentage_raw'] = df['Guest_Popularity_percentage']
    df['Guest_Popularity_percetage_was_missing'] = df['Guest_Popularity_percentage'].isna().astype(int)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())

    return df
