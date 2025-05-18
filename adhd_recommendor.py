#!/usr/bin/env python3
"""
adhd_recommendor.py

Usage:
    python adhd_recommendor.py "I need help focusing and staying productive."
"""

import praw
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time
import colorama
from colorama import Fore, Style
import os
from dotenv import load_dotenv

nltk.download("vader_lexicon", quiet=True)

# Load environment variables from .env file
load_dotenv()

# Securely initialize Reddit API using environment variables
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# List of subreddits to scrape
subreddits = [
    "ADHD", "Anxiety", "Depression", "GetMotivated",
    "DecidingToBeBetter", "selfimprovement", "MentalHealth"
]

def recommend_stories(query, num_recommendations=5):
    df = pd.read_pickle("thriving_adhd_posts_with_embeddings.pkl")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Encode user query
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    embeddings = np.vstack(df['embedding'].values).astype("float32")
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()
    
    # Convert to percentages (0.1% precision)
    df["similarity"] = cosine_similarities
    df["similarity_percent"] = (df["similarity"] * 100).round(1)
    
    # Sorting by similarity
    df_sorted = df.sort_values(by=["similarity_percent"], ascending=False)
    top_posts = df_sorted.head(num_recommendations)

    print("\nTop Recommendations:\n")
    for index, row in top_posts.iterrows():
        color = Fore.GREEN if row['similarity_percent'] > 50 else Fore.RED
        print(f"{color}{row['title']} - {row['similarity_percent']}% Match{Style.RESET_ALL}")
        print(row["selftext"][:300] + "...\n")
        print(f"Read full post: {row['url']}\n")
        print("---")
        
def collect_reddit_posts(limit=1000):
    all_posts = []
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            count = 0
            print(f"Scraping subreddit: {subreddit_name}")

            for post in subreddit.new(limit=None):
                all_posts.append({
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'selftext': post.selftext or '',
                    'score': post.score,
                    'created_utc': post.created_utc,
                    'num_comments': post.num_comments,
                    'url': f"https://www.reddit.com{post.permalink}"
                })
                count += 1
                if count >= (limit // len(subreddits)):
                    break
                time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ Could not access subreddit: {subreddit_name}. Error: {str(e)}")

    df = pd.DataFrame(all_posts)
    print(f"✅ Collected {len(df)} Reddit posts across multiple subreddits")
    df.to_csv("thriving_adhd_posts.csv", index=False)
    return df

def process_and_filter(df):
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["selftext"].apply(lambda t: sia.polarity_scores(t)['compound'])

    # Filtering only positive posts (sentiment > 0.2)
    positive_df = df[df['sentiment'] > 0.2].copy()
    
    # Embedding the posts using SBERT
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = (positive_df['title'].fillna('') + ' ' + positive_df['selftext'].fillna('')).tolist()
    embeddings = model.encode(corpus, show_progress_bar=True)
    positive_df['embedding'] = embeddings.tolist()

    # Saving the positive posts with embeddings
    positive_df.to_pickle("thriving_adhd_posts_with_embeddings.pkl")
    print(f"✅ Processed and saved {len(positive_df)} positive thriving posts")
    return positive_df

# Collecting posts (up to 1,000 across all subreddits)
df = collect_reddit_posts(1000)
positive_df = process_and_filter(df)
