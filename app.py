# app.py (Final Stable Version with Updated Streamlit Caching)
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import traceback

DEBUG_MODE = False  # Toggle for Spyder Debugging (True/False)

# --- Error Handling Function ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

def log_error(e):
    error_message = "".join(traceback.format_exception(None, e, e.__traceback__))
    print("❌ Error:", error_message)
    st.error("An error occurred. Please check the console for details.")
    if DEBUG_MODE:
        print(error_message)

# --- 1. Load Data and Embeddings ---
@st.cache_data
def load_data():
    try:
        df = pd.read_pickle("thriving_adhd_posts_with_embeddings.pkl")
        embeddings = np.vstack(df["embedding"].values).astype("float32")
        if len(df) == 0 or len(embeddings) == 0:
            st.error("Data file exists but is empty. Try scraping fresh data.")
            return None, None
        st.success(f"Loaded {len(df)} thriving posts with {embeddings.shape[0]} embeddings.")
        return df, embeddings
    except FileNotFoundError:
        st.error("Data file 'thriving_adhd_posts_with_embeddings.pkl' not found.")
        if DEBUG_MODE:
            print("❌ FileNotFoundError: Data file not found.")
        return None, None
    except Exception as e:
        log_error(e)
        return None, None

# --- 2. Scrape Fresh Data (Up to 10,000 Posts) ---
def scrape_and_process_data(limit=10000):
    try:
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()

        reddit = praw.Reddit(
            client_id="UJoTpwzPJnh-kRvDGnFtoA",
            client_secret="jn_m05OJLKIcJsk4VOgRhRTsAgSmdA",
            user_agent="adhd research script by /u/adhd_scraper"
        )

        posts = []
        for post in reddit.subreddit("ADHD").hot(limit=limit):
            posts.append({
                'title': post.title,
                'selftext': post.selftext or '',
                'score': post.score,
                'created_utc': post.created_utc,
                'num_comments': post.num_comments,
                'url': f"https://www.reddit.com{post.permalink}"
            })

        if not posts:
            st.error("Failed to scrape any posts from Reddit.")
            return None, None

        df = pd.DataFrame(posts)
        df['sentiment'] = df['selftext'].apply(lambda t: sia.polarity_scores(t)['compound'])

        # Extended positive phrases
        keywords = [
            'I managed to', 'I finally', 'helped me', 'making progress', 
            'I overcame', 'I succeeded', 'I learned', 'I found a way',
            'feeling better', 'improved my', 'this worked for me',
            'positive change', 'got better', 'achieved', 'I conquered', 'overcame my fear'
        ]
        df['thriving'] = df.apply(
            lambda r: any(kw.lower() in r['selftext'].lower() for kw in keywords) 
                      and r['sentiment'] > 0.15, 
            axis=1
        )

        thriving_df = df[df['thriving']]
        if thriving_df.empty:
            st.error("No thriving posts found. Try scraping more posts.")
            return None, None

        model = SentenceTransformer('all-MiniLM-L6-v2')
        corpus = (thriving_df['title'] + ' ' + thriving_df['selftext']).tolist()
        embeddings = model.encode(corpus, show_progress_bar=True)
        thriving_df['embedding'] = embeddings.tolist()
        thriving_df.to_pickle('thriving_adhd_posts_with_embeddings.pkl')

        st.success(f"Scraped and processed {len(thriving_df)} thriving posts.")
        return thriving_df, embeddings
    except Exception as e:
        log_error(e)
        return None, None

# --- 3. Main Streamlit UI ---
st.title("Thriving ADHD Stories Recommender")
st.write("Paste a brief description of your current ADHD challenge, and get inspired by success stories.")

if st.button("Scrape Fresh Data"):
    with st.spinner("Scraping fresh data..."):
        df, embeddings = scrape_and_process_data(limit=10000)
else:
    df, embeddings = load_data()

if df is None or embeddings is None:
    st.warning("Data not loaded. Try scraping fresh data.")
else:
    st.success(f"Loaded {len(df)} thriving posts.")

# --- Build NearestNeighbors Index ---
@st.cache_resource
def build_index(embeddings):
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(embeddings)
    return nn

if df is not None and embeddings is not None:
    nn = build_index(embeddings)

# --- Query Section ---
user_input = st.text_area("Your challenge", height=100)
if st.button("Find Stories"):
    if not user_input.strip():
        st.warning("Please enter a sentence or two about your challenge.")
    elif df is None or embeddings is None:
        st.error("No data loaded. Please scrape fresh data.")
    else:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_vec = model.encode([user_input], convert_to_numpy=True).astype("float32")
            distances, indices = nn.kneighbors(query_vec)

            st.subheader("Top 2 Related Thriving Stories")
            for idx in indices[0]:
                title = df.iloc[idx]["title"]
                excerpt = df.iloc[idx]["selftext"][:300] + "..."
                link = df.iloc[idx]["url"]

                st.markdown(f"### {title}")
                st.write(excerpt)
                st.write(f"[Read full post on Reddit]({link})")
                st.markdown("---")
        except Exception as e:
            log_error(e)
