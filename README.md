ADHD Stories Recommender
Welcome to the ADHD Stories Recommender, an advanced natural language processing (NLP) project designed to provide users with the most inspiring "thriving" stories from ADHD and mental health communities on Reddit.

🚀 Project Features
Multi-Subreddit Scraping: Collects thriving stories from r/ADHD, r/Anxiety, r/Depression, r/GetMotivated, and more.

Sentiment Analysis: Filters out positive stories using NLTK's VADER sentiment analysis.

Semantic Search (Cosine Similarity): Matches user queries to the most similar thriving stories using Sentence Transformers (SBERT).

Percentage Match Display: Shows how similar each recommended story is to the user’s challenge (color-coded).

✅ Green for matches above 50%.

🔴 Red for matches below 50%.

Two Access Modes:

Streamlit Web App (Interactive, User-Friendly).

Command-Line Interface (CLI) with color-coded recommendations.

📌 How to Use
Clone this Repository:

bash
Copy
Edit
git clone https://github.com/jalalfaraj/adhd_recommender_project.git
cd adhd_recommender_project
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Launch the Project:

Run the launcher:

bash
Copy
Edit
python launch_adhd_recommender.py
Choose your mode:

Run CLI Recommender (Command-Line).

Run Streamlit Web App.

Scrape Fresh Data (Multi-Subreddit).

🖥️ Streamlit Web App Features
Enter your ADHD challenge to receive the most similar "thriving" stories.

Percentage Match (Cosine Similarity) displayed for each recommended story.

Top stories sorted by similarity (Top 3, 5, or 10).

Color-coded percentage:

✅ Green (Above 50% Match)

🔴 Red (Below 50% Match)

🚀 How It Works (Technical Overview)
Data Collection: Scrapes multiple subreddits (up to 10,000 posts) using Reddit API (PRAW).

Sentiment Filtering: Analyzes sentiment with NLTK VADER and keeps only positive stories.

Semantic Search: Embeds stories using SBERT (all-MiniLM-L6-v2).

Similarity Matching: Recommends stories using Cosine Similarity + Sentiment.

Color-Coded Recommendations: Green (High Match), Red (Low Match).

✅ Future Improvements
Expand to more mental health subreddits.

Allow users to customize subreddits and sentiment thresholds.

Add user authentication for personalized recommendations.

🌟 Created by Jalal Faraj
For any inquiries, feel free to reach out via LinkedIn or GitHub.
