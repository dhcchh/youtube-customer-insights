"""
Just run: python run_analysis.py
"""

import os
from dotenv import load_dotenv
from src.pipeline import analyze_video

load_dotenv()

def main():
    print("YouTube Comment Analysis")
    print("=" * 40)
    
    api_key = os.environ.get('API_KEY')
    if not api_key:
        print("Error: YOUTUBE_API_KEY not found in .env file")
        print("Create a .env file with: YOUTUBE_API_KEY=your_key_here")
        return
    
    video_url = "https://www.youtube.com/watch?v=5oJq8CVoHBw"
    
    print(f"📹 Analyzing: {video_url}")
    print()
    
    try:
        results = analyze_video(video_url)
        
        print("\n" + "=" * 40)
        print("ANALYSIS COMPLETE")
        print("=" * 40)
        print(f"Video ID: {results['video_id']}")
        print(f"Comments: {results['total_comments']:,}")
        print(f"Negative: {results['negative_pct']:.1f}%")
        print(f"Positive: {results['positive_pct']:.1f}%")
        print(f"Top Issue: {results['top_issue']}")
        
        print(f"\n Next steps:")
        print(f"1. Run dashboard: streamlit run app.py")
        print(f"2. Check results in data/exports/")
        print(f"3. Upload data/st_dashboard_ready/ to Apache Superset")
        
        if results['negative_pct'] > 50:
            print(f"\n WARNING: High negative sentiment ({results['negative_pct']:.1f}%)")
        
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    main()