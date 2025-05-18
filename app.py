# app.py - Streamlit Recommender with Enhanced Similarity and Sentiment Filtering
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Downloading VADER for sentiment analysis
nltk.download("vader_lexicon", quiet=True)

# Displaying Subreddits Being Used
st.title("Thriving ADHD Stories Recommender")
st.markdown("#### Currently Scraping and Analyzing Stories From:")
st.markdown("- r/ADHD\n- r/Anxiety\n- r/Depression\n- r/GetMotivated")
st.markdown("- r/DecidingToBeBetter\n- r/selfimprovement\n- r/ADHD_Parents\n- r/MentalHealth")

# Load data and embeddings
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_pickle("thriving_adhd_posts_with_embeddings.pkl")
    embeddings = np.vstack(df['embedding'].values).astype("float32")
    return df, embeddings

df, embeddings = load_data()

# Loading the SBERT Model
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# User Input
st.write("Enter your ADHD challenge below to receive similar thriving stories.")
user_input = st.text_area("Your ADHD Challenge", height=100)
num_recommendations = st.selectbox("Select number of stories:", [3, 5, 10], index=1)

if st.button("Find Stories"):
    if not user_input.strip():
        st.warning("Please enter a sentence or two about your challenge.")
    else:
        # Encode user input
        query_embedding = model.encode([user_input], convert_to_numpy=True).astype("float32")
        
        # Calculate cosine similarity
        query_embedding = query_embedding.reshape(1, -1)
        cosine_similarities = cosine_similarity(query_embedding, embeddings).flatten()
        
        # Convert cosine similarities to percentages rounded to the nearest tenth (0.1%)
        df["similarity"] = cosine_similarities
        df["similarity_percent"] = (df["similarity"] * 100).round(1)
        
        # Sort posts by cosine similarity (descending)
        df_sorted = df.sort_values(by=["similarity_percent", "sentiment"], ascending=[False, False])
        top_posts = df_sorted.head(num_recommendations)

        st.write(f"### Top {num_recommendations} Most Similar Thriving Stories:")
        for index, row in top_posts.iterrows():
            # Determine color (Green if >50%, Red if <=50%)
            color = "green" if row['similarity_percent'] > 50 else "red"
            st.markdown(f"#### <span style='color:{color};'>{row['title']} - {row['similarity_percent']}% Match</span>", unsafe_allow_html=True)
            st.write(row["selftext"][:300] + "...")
            st.markdown(f"[Read full post]({row['url']})")
            st.markdown("---")
