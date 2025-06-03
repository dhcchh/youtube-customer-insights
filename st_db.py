import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Configure page
st.set_page_config(
    page_title="YouTube Comment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

def load_data():
    """Load the transformed data"""
    try:
        # Try different possible paths
        paths_to_try = [
            '../data/superset_ready/sentiment_for_superset.csv',
            'data/superset_ready/sentiment_for_superset.csv',
            '../data/superset_ready/sentiment_for_superset.csv'
        ]
        
        sentiment_df = None
        for path in paths_to_try:
            try:
                sentiment_df = pd.read_csv(path.replace('sentiment_for_superset.csv', 'sentiment_for_superset.csv'))
                # If successful, use this path base for other files
                base_path = path.replace('sentiment_for_superset.csv', '')
                feature_df = pd.read_csv(f'{base_path}features_for_superset.csv')
                summary_df = pd.read_csv(f'{base_path}summary_for_superset.csv')
                st.success(f"✅ Data loaded from: {base_path}")
                return sentiment_df, feature_df, summary_df
            except FileNotFoundError:
                continue
        
        # If none of the paths worked, show helpful error
        st.error("❌ Data files not found. Please check:")
        st.write("Current working directory:", os.getcwd())
        st.write("Looking for files in:")
        for path in paths_to_try:
            st.write(f"  - {path}")
        
        return None, None, None
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def create_sentiment_pie_chart(sentiment_df):
    """Create sentiment distribution pie chart"""
    fig = px.pie(
        sentiment_df, 
        values='count', 
        names='sentiment_type',
        title="Comment Sentiment Distribution",
        color_discrete_map={
            'Negative': '#ff6b6b',
            'Neutral': '#ffd93d', 
            'Positive': '#6bcf7f'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_engagement_bar_chart(sentiment_df):
    """Create engagement by sentiment bar chart"""
    fig = px.bar(
        sentiment_df,
        x='sentiment_type',
        y='total_likes',
        title="Total Likes by Sentiment<br><sub>Sum of all likes received by comments in each sentiment category</sub>",
        color='sentiment_type',
        color_discrete_map={
            'Negative': '#ff6b6b',
            'Neutral': '#ffd93d', 
            'Positive': '#6bcf7f'
        }
    )
    fig.update_layout(showlegend=False)
    return fig

def create_issue_bar_chart(feature_df):
    """Create top issues horizontal bar chart"""
    # Sort by count for better visualization
    df_sorted = feature_df.sort_values('count', ascending=True)
    
    fig = px.bar(
        df_sorted,
        x='count',
        y='issue',
        title="Customer Issues by Frequency",
        color='impact_category',
        color_discrete_map={
            'High Impact': '#ff6b6b',
            'Medium Impact': '#ffd93d',
            'Low Impact': '#6bcf7f'
        },
        orientation='h'
    )
    fig.update_layout(height=400)
    return fig

def create_impact_scatter(feature_df):
    """Create issue impact vs engagement scatter plot"""
    fig = px.scatter(
        feature_df,
        x='count',
        y='total_likes',
        size='weighted_score',
        color='impact_category',
        hover_name='issue',
        title="Issue Impact Matrix: Frequency vs Engagement<br><sub>Bubble size = weighted_score (count × total_likes). Position shows frequency vs total likes received.</sub>",
        color_discrete_map={
            'High Impact': '#ff6b6b',
            'Medium Impact': '#ffd93d',
            'Low Impact': '#6bcf7f'
        }
    )
    fig.update_layout(
        xaxis_title="Number of Mentions",
        yaxis_title="Total Likes"
    )
    return fig

def main():
    st.title("🎬 YouTube Comment Analysis Dashboard")
    st.subheader("Nespresso Descaling Video Analysis")
    
    # Load data
    sentiment_df, feature_df, summary_df, video_df = load_data()
    
    if sentiment_df is None:
        st.stop()
    
    # Surface Level Metrics
    st.markdown("### 📊 Surface Level Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get video statistics or use defaults
    if video_df is not None and len(video_df) > 0:
        video_stats = video_df.iloc[0]
        total_views = video_stats['view_count']
        video_likes = video_stats['like_count']
        video_title = video_stats['title']
    else:
        total_views = 'N/A'
        video_likes = 'N/A' 
        video_title = 'Video data not available'
    
    with col1:
        st.metric("Total Views", f"{total_views:,}" if isinstance(total_views, int) else total_views)
    
    with col2:
        st.metric("Video Likes", f"{video_likes:,}" if isinstance(video_likes, int) else video_likes)
    
    with col3:
        st.metric("Video Dislikes", "Hidden*")
        st.caption("*YouTube removed public dislike counts")
    
    with col4:
        # Calculate engagement rate if we have the data
        if isinstance(total_views, int) and isinstance(video_likes, int):
            engagement_rate = (video_likes / total_views * 100)
            st.metric("Like Rate", f"{engagement_rate:.2f}%")
        else:
            st.metric("Like Rate", "N/A")
    
    st.markdown("---")
    
    # Comment Analysis KPIs
    st.markdown("### 💬 Comment Analysis Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Extract summary metrics safely - convert types as needed
    summary_dict = dict(zip(summary_df['metric'], summary_df['value']))
    
    with col1:
        total_comments = summary_dict.get('Total Comments', 0)
        st.metric("Total Comments", int(float(total_comments)))
    
    with col2:
        negative_pct = summary_dict.get('Negative Sentiment %', 0)
        try:
            negative_pct_num = float(negative_pct)  # Convert string to float
            st.metric(
                "Negative Sentiment", 
                f"{negative_pct_num}%",
                delta=f"{negative_pct_num - 33.3:.1f}% vs balanced" if negative_pct_num > 33.3 else None,
                delta_color="inverse"
            )
        except (ValueError, TypeError):
            st.metric("Negative Sentiment", str(negative_pct))
    
    with col3:
        st.metric("Top Issue", str(summary_dict.get('Top Issue', 'N/A')))
    
    with col4:
        high_impact = summary_dict.get('High Impact Issues', 0)
        st.metric("High Impact Issues", int(float(high_impact)))
    
    # Main charts
    st.markdown("---")
    
    # Row 1: Sentiment analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = create_sentiment_pie_chart(sentiment_df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = create_engagement_bar_chart(sentiment_df)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Row 2: Issue analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_issues = create_issue_bar_chart(feature_df)
        st.plotly_chart(fig_issues, use_container_width=True)
    
    with col2:
        fig_scatter = create_impact_scatter(feature_df)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Data tables
    st.markdown("---")
    st.subheader("📋 Detailed Data")
    
    tab1, tab2 = st.tabs(["Sentiment Analysis", "Issue Analysis"])
    
    with tab1:
        st.dataframe(sentiment_df, use_container_width=True)
    
    with tab2:
        # Sort by priority for better readability
        feature_display = feature_df.sort_values('weighted_score', ascending=False)
        st.dataframe(feature_display, use_container_width=True)
    
    # Business insights
    st.markdown("---")
    st.subheader("🎯 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Sentiment Analysis:**
        - 65.7% negative sentiment indicates major customer dissatisfaction
        - Only 10.2% positive sentiment - significant room for improvement
        - Negative comments receive high engagement (10.48 avg likes)
        """)
    
    with col2:
        st.markdown("""
        **Top Issues to Address:**
        1. **Descaling Button Request** (19 mentions, 720 likes) - High priority
        2. **Complexity Complaints** (15 mentions, 678 likes) - High priority  
        3. **Simplification Requests** (3 mentions, 143 likes) - Medium priority
        """)
    
    # Action items
    st.markdown("---")
    st.subheader("📝 Recommended Actions")
    
    # Convert negative percentage for comparison
    try:
        negative_pct_for_check = float(summary_dict.get('Negative Sentiment %', 0))
        if negative_pct_for_check > 50:
            st.error("🚨 **URGENT**: Negative sentiment > 50% - immediate action required")
    except (ValueError, TypeError):
        st.warning("Could not parse negative sentiment percentage")
    
    st.markdown("""
    **Immediate Actions:**
    1. **Add dedicated descaling button** - most requested feature
    2. **Simplify descaling process** - reduce complexity complaints
    3. **Improve instructions** - clearer step-by-step guidance
    4. **Create follow-up video** addressing top concerns
    
    **Success Metrics to Track:**
    - Reduce negative sentiment below 30%
    - Increase positive sentiment above 40%
    - Decrease complexity-related issues
    """)

if __name__ == "__main__":
    main()