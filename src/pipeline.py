import os
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

from src.load_comments import YouTubeDataLoader
from src.clean_comments import BERTYouTubeCommentCleaner
from src.sentiment_analysis_BERT import BERTSentimentAnalyzer, quick_sentiment_summary
from src.feature_extractor import FeatureExtractor


class Pipeline:
    """
    Simple YouTube comment analysis pipeline.
    Does one thing well: analyze YouTube comments for business insights.
    """
    
    def __init__(self, api_key=None):
        """Initialize with API key"""
        self.api_key = api_key or os.environ.get('API_KEY')
        if not self.api_key:
            raise ValueError("Need YouTube API key: set API_KEY env var or pass api_key")
        
        self.folders = ['data/raw', 'data/processed', 'data/st_dashboard_ready', 'data/exports']
        for folder in self.folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        self.loader = YouTubeDataLoader(self.api_key)
        self.cleaner = BERTYouTubeCommentCleaner()
        self.analyzer = None  # Load when needed
        self.extractor = FeatureExtractor()
        
        # Simple logging
        # level=logging.INFO: Only shows INFO level and above (INFO, WARNING, ERROR, CRITICAL)
    
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)  # Creates a logger instance for this specific module

    def run(self, video_url):
        """
        Run complete analysis pipeline.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            dict: Results summary
        """
        video_id = self.loader.extract_video_id(video_url)
        
        self.logger.info(f"Starting analysis for video: {video_id}")
        
        self.logger.info("Step 1: Loading data...")
        comments_df, video_stats = self._load_data(video_url, video_id)
        
        self.logger.info("Step 2: Cleaning data...")
        cleaned_df = self._clean_data(comments_df, video_id)
        
        self.logger.info("Step 3: Analyzing sentiment...")
        sentiment_df = self._analyze_sentiment(cleaned_df, video_id)
        
        self.logger.info("Step 4: Extracting insights...")
        chart_data = self._extract_insights(sentiment_df, video_id)
        
        self.logger.info("Step 5: Preparing dashboard data...")
        self._prepare_dashboard_data(sentiment_df, chart_data, video_stats, video_id)
        
        self.logger.info("Step 6: Exporting results...")
        self._export_results(sentiment_df, video_id)
        
        results = {
            'video_id': video_id,
            'total_comments': len(sentiment_df),
            'negative_pct': (sentiment_df['sentiment_label'] == 'Negative').mean() * 100,
            'positive_pct': (sentiment_df['sentiment_label'] == 'Positive').mean() * 100,
            'top_issue': chart_data.iloc[0]['issue'] if not chart_data.empty else 'None',
            'dashboard_ready': True
        }
        
        self.logger.info("Analysis complete!")
        return results
    
    def _load_data(self, video_url, video_id):
        """Load or download YouTube data"""
        comments_file = f"data/raw/raw_{video_id}.csv"
        stats_file = f"data/raw/video_statistics_{video_id}.csv"
        
        # Check if data exists
        if os.path.exists(comments_file) and os.path.exists(stats_file):
            self.logger.info("Loading existing data...")
            comments_df = pd.read_csv(comments_file)
            video_stats = pd.read_csv(stats_file).iloc[0].to_dict()
        else:
            self.logger.info("Downloading fresh data...")
            video_stats = self.loader.get_video_stats(video_id, save_to_dir="data/raw", dislike_count=280)
            comments_df = self.loader.get_all_comments(
                video_url, 
                save_to_file=f"data/raw/raw_{video_id}"
            )
        
        self.logger.info(f"Loaded {len(comments_df)} comments")
        return comments_df, video_stats
    
    def _clean_data(self, comments_df, video_id):
        """Clean comment data"""
        cleaned_df = self.cleaner.clean_dataframe(comments_df)
        cleaned_df.to_csv(f"data/processed/cleaned_{video_id}.csv", index=False)
        
        removed = len(comments_df) - len(cleaned_df)
        self.logger.info(f"Cleaned {len(cleaned_df)} comments (removed {removed})")
        return cleaned_df
    
    def _analyze_sentiment(self, cleaned_df, video_id):
        """Run sentiment analysis"""
        if self.analyzer is None:
            self.analyzer = BERTSentimentAnalyzer()
        
        sentiment_df = self.analyzer.analyze_dataframe(cleaned_df)
        sentiment_df.to_csv(f"data/processed/sentiment_{video_id}.csv", index=False)
        quick_sentiment_summary(sentiment_df)
        return sentiment_df
    
    def _extract_insights(self, sentiment_df, video_id):
        """Extract business insights"""
        insights = self.extractor.extract_insights(sentiment_df)
        chart_data = self.extractor.get_chart_data(insights)
        
        if not chart_data.empty:
            chart_data.to_csv(f"data/processed/features_{video_id}.csv", index=False)
            self.logger.info(f"Found {len(chart_data)} issue categories")
        
        return chart_data
    
    def _prepare_dashboard_data(self, sentiment_df, chart_data, video_stats, video_id):
        """Prepare data for Streamlit dashboard"""
        
        sentiment_summary = []
        total = len(sentiment_df)
        for sentiment in ['Negative', 'Neutral', 'Positive']:
            subset = sentiment_df[sentiment_df['sentiment_label'] == sentiment]
            count = len(subset)
            sentiment_summary.append({
                'sentiment_type': sentiment,
                'count': count,
                'percentage': (count / total) * 100,
                'total_likes': subset['like_count'].sum(),
                'avg_likes': subset['like_count'].mean() if count > 0 else 0
            })
        
        pd.DataFrame(sentiment_summary).to_csv("data/st_dashboard_ready/sentiment_for_st.csv", index=False)
        
        if not chart_data.empty:
            chart_data['impact_category'] = chart_data['weighted_score'].apply(
                lambda x: 'High Impact' if x > chart_data['weighted_score'].quantile(0.7) 
                         else 'Medium Impact' if x > chart_data['weighted_score'].quantile(0.3)
                         else 'Low Impact'
            )
            chart_data.to_csv("data/st_dashboard_ready/features_for_st.csv", index=False)
        
        negative_pct = (sentiment_df['sentiment_label'] == 'Negative').mean() * 100
        summary_metrics = [
            {'metric': 'Total Comments', 'value': len(sentiment_df)},
            {'metric': 'Negative Sentiment %', 'value': f"{negative_pct:.1f}"},
            {'metric': 'Top Issue', 'value': chart_data.iloc[0]['issue'] if not chart_data.empty else 'None'},
            {'metric': 'High Impact Issues', 'value': (chart_data['impact_category'] == 'High Impact').sum() if not chart_data.empty else 0}
        ]
        pd.DataFrame(summary_metrics).to_csv("data/st_dashboard_ready/summary_for_st.csv", index=False)
        
        pd.DataFrame([video_stats]).to_csv(f"data/st_dashboard_ready/video_statistics_{video_id}.csv", index=False)
    
    def _export_results(self, sentiment_df, video_id):
        """Export final results"""
        sentiment_df.to_csv(f"data/exports/final_sentiment_analysis_result_{video_id}.csv", index=False)


def analyze_video(video_url, api_key=None):
    """
    Convenience function to analyze a single video.
    
    Args:
        video_url: YouTube video URL
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        dict: Analysis results
    """
    pipeline = Pipeline(api_key)
    return pipeline.run(video_url)