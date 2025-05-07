#!/usr/bin/env python3
"""
adhd_recommendor.py

Usage:
    python adhd_recommendor.py "I need help focusing and staying productive."
"""

import sys
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import traceback

DEBUG_MODE = False  # Toggle for Debugging (True/False)

# -------------------------------------------
# Error Handling Function
# -------------------------------------------
def log_error(e):
    error_message = "".join(traceback.format_exception(None, e, e.__traceback__))
    print("❌ Error:", error_message)
    if DEBUG_MODE:
        print(error_message)

# -------------------------------------------
# Load Data and Model
# -------------------------------------------
def load_data_and_model():
    try:
        df = pd.read_pickle("thriving_adhd_posts_with_embeddings.pkl")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = np.vstack(df['embedding'].values).astype("float32")
        
        nn = NearestNeighbors(n_neighbors=2, metric='cosine')
        nn.fit(embeddings)
        print(f"✅ Loaded {len(df)} thriving posts.")
        return df, model, nn
    except FileNotFoundError:
        print("❌ Error: Data file 'thriving_adhd_posts_with_embeddings.pkl' not found.")
        return None, None, None
    except Exception as e:
        log_error(e)
        return None, None, None

# -------------------------------------------
# Query and Recommend
# -------------------------------------------
def recommend_stories(user_input, df, model, nn):
    try:
        query_emb = model.encode([user_input], convert_to_numpy=True).astype('float32')
        distances, indices = nn.kneighbors(query_emb)

        print('\nTop 2 Thriving ADHD Stories:\n')
        for i in indices[0]:
            title = df.iloc[i]['title']
            excerpt = df.iloc[i]['selftext'][:300] + '...'
            link = df.iloc[i]['url']
            print(f"🔹 {title}\n{excerpt}\n[Read full post]({link})\n")
    except Exception as e:
        log_error(e)

# -------------------------------------------
# Main Execution
# -------------------------------------------
def main():
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = "I need help focusing and staying productive."

    print("\n🚀 Thriving ADHD Stories Recommender\n")
    df, model, nn = load_data_and_model()

    if df is not None and model is not None and nn is not None:
        print(f"\nYour Challenge: {user_input}\n")
        recommend_stories(user_input, df, model, nn)

if __name__ == '__main__':
    main()
