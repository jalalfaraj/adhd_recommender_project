# launch_adhd_recommender.py - Unified Launcher (CLI + Streamlit + Scraping)
import os
import subprocess

def main():
    print("\n🚀 ADHD Stories Recommender - Unified Launcher")
    print("1. Run the CLI Recommender")
    print("2. Run the Streamlit Web App")
    print("3. Scrape Fresh Data (Multi-Subreddit)")
    print("4. Quit")

    choice = input("Choose an option (1-4): ").strip()

    if choice == "1":
        print("\nRunning CLI Recommender...")
        query = input("\nEnter your ADHD challenge: ")
        from adhd_recommendor import recommend_stories
        recommend_stories(query)

    elif choice == "2":
        print("\nLaunching Streamlit Web App...")
        os.system("streamlit run app.py")

    elif choice == "3":
        print("\nStarting Fresh Data Scraping...")
        from adhd_recommendor import collect_reddit_posts, process_and_filter
        
        limit = input("Enter the number of posts to scrape (recommended: 10000): ").strip()
        limit = int(limit) if limit.isdigit() else 10000
        df = collect_reddit_posts(limit)
        process_and_filter(df)
        print("\n✅ Fresh data scraped and processed. Ready to use.")

    elif choice == "4":
        print("\nGoodbye!")
        return

    else:
        print("\nInvalid choice. Please restart the launcher.")

if __name__ == "__main__":
    main()
