import pandas as pd
import numpy as np
import ast
import os
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer


def load_and_clean_data():
    # Use the absolute path logic
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'movies.csv')
    df = pd.read_csv(file_path)
    
    # Clean missing values identified in audit
    df = df.dropna(subset=['overview', 'runtime', 'release_date'])
    df['tagline'] = df['tagline'].fillna('No Tagline')
    df = df.drop(columns=['homepage'])
    
    return df


def engineer_features(df):
    # Parse nested JSON-like strings safely
    def parse_json(val):
        return [i['name'] for i in ast.literal_eval(val)]

    # Apply to both Genres and Keywords to increase feature count
    df['genres'] = df['genres'].apply(parse_json)
    df['keywords'] = df['keywords'].apply(parse_json)
    
    # Filter keywords by frequency (10+ occurrences)
    all_keywords = [k for sublist in df['keywords'] for k in sublist]
    keyword_counts = Counter(all_keywords)
    
    # Keep only keywords that appear in at least 10 movies
    frequent_keywords = {k for k, count in keyword_counts.items() if count >= 10}
    df['keywords'] = df['keywords'].apply(lambda x: [k for k in x if k in frequent_keywords])
    
    # 3. Encoding for both text columns
    mlb_genres = MultiLabelBinarizer()
    genre_df = pd.DataFrame(mlb_genres.fit_transform(df['genres']), columns=mlb_genres.classes_, index=df.index)
    
    mlb_keywords = MultiLabelBinarizer()
    keyword_df = pd.DataFrame(mlb_keywords.fit_transform(df['keywords']), columns=mlb_keywords.classes_, index=df.index)
    
    # In-line Imputation for Numerical Gaps
    # Isolate the continuous numerical columns to patch missing values
    num_cols = ['budget', 'runtime']
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # Numerical Scaling
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Combine into Final Design Matrix (X)
    final_df = pd.concat([df[num_cols], genre_df, keyword_df], axis=1)
    
    return final_df

    
if __name__ == "__main__":
    raw_df = load_and_clean_data()
    X = engineer_features(raw_df)
    print("Feature Matrix (X) successfully created.")
    print(f"Matrix Shape: {X.shape}")
    print(X.head())

    # Check for proper datatypes
    print(X.dtypes.value_counts())