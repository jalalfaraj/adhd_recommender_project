
# 🌟 ADHD Stories Recommender

A machine learning-powered application that helps users discover "thriving" ADHD stories from Reddit. This project leverages NLP and sentiment analysis to identify positive and inspiring stories, making them easily accessible to users who may benefit from them.

## 🚀 Project Overview
This project is a comprehensive solution for discovering and recommending "thriving" ADHD stories from the r/ADHD subreddit. It features:
- A **CLI Recommender**: Quickly find inspiring stories directly in your terminal.
- A **Streamlit Web App**: Interactive and user-friendly for a broader audience.
- An efficient **Data Scraper and Processor**: Collects Reddit posts, analyzes sentiment, and identifies thriving stories.
- Embedding and Similarity Matching: Finds the most relevant stories using Sentence Transformers.

## ✨ Features
- **Interactive Streamlit Web App**:
  - Enter an ADHD challenge and get top 2 most similar thriving stories.
  - Links directly to Reddit for full reading.
- **Command-Line Interface (CLI)**:
  - Query ADHD stories directly from the terminal.
- **Efficient Data Scraper**:
  - Collects up to 1,000 Reddit posts.
  - Analyzes sentiment and filters for thriving stories.
- **Dynamic Embedding and Recommendation**:
  - Uses Sentence Transformers for semantic matching.
  - Recommends stories based on similarity.

## 📁 Project Structure
```
📂 ADHD Stories Recommender Project
├── adhd_recommendor.py       # CLI Recommender
├── app.py                    # Streamlit Web App
├── launch_adhd_recommender.py # Unified Launcher (CLI + Streamlit + Scraping)
├── thriving_adhd_posts.csv   # Scraped and filtered data (CSV)
├── thriving_adhd_posts_with_embeddings.pkl # Data with embeddings (Pickle)
├── thriving_adhd_embeddings.npy # Embeddings (Numpy)
└── README.md                 # Project Documentation
```

## ⚡️ Installation
### Prerequisites:
- Python 3.8 or higher
- pip (Python package installer)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/adhd-stories-recommender.git
cd adhd-stories-recommender
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Reddit API Credentials
- Go to [Reddit API Developer Portal](https://www.reddit.com/prefs/apps).
- Create an app and get your:
  - `client_id`
  - `client_secret`
  - `user_agent`
- Replace these values in both `app.py` and `adhd_recommendor.py`.

## 🚀 Usage
### 1. Running the Unified Launcher
```bash
python launch_adhd_recommender.py
```
- Choose between CLI Recommender, Streamlit Web App, and Data Scraper.

### 2. Running the Streamlit Web App Directly
```bash
streamlit run app.py
```

### 3. Using the CLI Recommender
```bash
python adhd_recommendor.py "I need help focusing and staying productive."
```

### 4. Scraping Fresh Data
- Use the launcher or run the scraper manually.

## 🔧 Technical Details
- **NLP**: SentenceTransformer (all-MiniLM-L6-v2), Sentiment Analysis (VADER).
- **ML**: NearestNeighbors (Scikit-Learn).
- **Web Scraping**: Reddit API with PRAW.

## 🌐 Contact
For any questions or feedback, feel free to reach out to me by email:
farajj7@gmail.com
