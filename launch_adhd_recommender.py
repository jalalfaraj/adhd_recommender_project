#!/usr/bin/env python3
"""
launch_adhd_recommender.py

A single-click launcher to run either:
1. The CLI recommender (adhd_recommendor.py).
2. The full Streamlit Web App (app.py).
3. Scrape fresh data directly from Reddit (10,000 posts).
"""

import os
import subprocess
import sys

def main():
    print("\n📌 Welcome to the Thriving ADHD Stories Recommender")
    print("Choose an option:\n")
    print("1. Run the CLI Recommender (adhd_recommendor.py)")
    print("2. Launch the Streamlit Web App (app.py)")
    print("3. Scrape Fresh Data (10,000 Posts)")
    print("4. Exit\n")

    choice = input("Enter your choice (1/2/3/4): ").strip()
    
    if choice == "1":
        print("\n🚀 Running CLI Recommender...")
        subprocess.run([sys.executable, "adhd_recommendor.py"])
        
    elif choice == "1":
        print("\n🚀 Running CLI Recommender...")
        user_input = input("Enter your ADHD challenge: ")
        subprocess.run([sys.executable, "adhd_recommendor.py", user_input])
    
    elif choice == "2":
        print("\n🚀 Launching Streamlit Web App...")
        subprocess.run(["streamlit", "run", "app.py"])
    
    elif choice == "3":
        print("\n🔄 Scraping Fresh Data (10,000 posts)...")
        try:
            import praw, nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            from sentence_transformers import SentenceTransformer
            import pandas as pd
            import numpy as np

            # Ensure VADER is available
            nltk.download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()

            reddit = praw.Reddit(
                client_id="UJoTpwzPJnh-kRvDGnFtoA",
                client_secret="jn_m05OJLKIcJsk4VOgRhRTsAgSmdA",
                user_agent="adhd research script by /u/adhd_scraper"
            )

            print("🔍 Scraping r/ADHD...")
            posts = []
            for post in reddit.subreddit("ADHD").hot(limit=10000):
                posts.append({
                    'title': post.title,
                    'selftext': post.selftext or '',
                    'score': post.score,
                    'created_utc': post.created_utc,
                    'num_comments': post.num_comments,
                    'url': f"https://www.reddit.com{post.permalink}"
                })

            if not posts:
                print("❌ Error: Failed to scrape any posts.")
                return

            df = pd.DataFrame(posts)
            keywords = [
                'I managed to', 'I finally', 'helped me', 'making progress', 
                'I overcame', 'I succeeded', 'I learned', 'I found a way',
                'feeling better', 'improved my', 'this worked for me',
                'positive change', 'got better', 'achieved'
            ]
            df['sentiment'] = df['selftext'].apply(lambda t: sia.polarity_scores(t)['compound'])
            df['thriving'] = df.apply(
                lambda r: any(kw.lower() in r['selftext'].lower() for kw in keywords) 
                          and r['sentiment'] > 0.15, 
                axis=1
            )

            thriving_df = df[df['thriving']]
            thriving_df.to_csv('thriving_adhd_posts.csv', index=False)
            print(f"✅ Found {len(thriving_df)} thriving posts.")

            model = SentenceTransformer('all-MiniLM-L6-v2')
            corpus = (thriving_df['title'] + ' ' + thriving_df['selftext']).tolist()
            embeddings = model.encode(corpus, show_progress_bar=True)
            thriving_df['embedding'] = embeddings.tolist()
            thriving_df.to_pickle('thriving_adhd_posts_with_embeddings.pkl')

            print("✅ Data scraped and saved as 'thriving_adhd_posts_with_embeddings.pkl'")
        except Exception as e:
            print(f"❌ Error while scraping: {e}")

    elif choice == "4":
        print("\n✅ Exiting. Have a great day!")
    
    else:
        print("\n❌ Invalid choice. Please try again.")
        main()  # Restart the choice menu

if __name__ == "__main__":
    main()
