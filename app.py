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
        base_path = 'data/st_dashboard_ready/'
        
        # Load required files
        sentiment_df = pd.read_csv(f'{base_path}sentiment_for_st.csv')
        feature_df = pd.read_csv(f'{base_path}features_for_st.csv')
        summary_df = pd.read_csv(f'{base_path}summary_for_st.csv')
        video_df = pd.read_csv(f'{base_path}video_statistics_5oJq8CVoHBw.csv')
        
        st.success(f"✅ Data loaded from: {base_path}")
        return sentiment_df, feature_df, summary_df, video_df
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None

def create_sentiment_pie_chart(sentiment_df):
    """Create sentiment distribution pie chart"""
    # Ensure the color mapping works by sorting the data
    sentiment_df_sorted = sentiment_df.sort_values('sentiment_type')
    
    fig = px.pie(
        sentiment_df_sorted, 
        values='count', 
        names='sentiment_type',
        title="Comment Sentiment Distribution",
        color='sentiment_type',
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

def safe_get_metric(summary_df, metric_name, default_value=0):
    """Safely extract metric from summary dataframe"""
    try:
        summary_dict = dict(zip(summary_df['metric'], summary_df['value']))
        value = summary_dict.get(metric_name, default_value)
        return float(value) if value != 'N/A' else default_value
    except (ValueError, TypeError, KeyError):
        return default_value

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
        total_views = video_stats.get('view_count', 'N/A')
        video_likes = video_stats.get('like_count', 'N/A') 
        video_dislikes = video_stats.get('dislike_count', 'N/A')
        video_title = video_stats.get('title', 'Video data not available')
    else:
        total_views = 'N/A'
        video_likes = 'N/A'
        video_dislikes = 'N/A'
        video_title = 'Video data not available'
    
    with col1:
        if isinstance(total_views, (int, float)):
            st.metric("Total Views", f"{int(total_views):,}")
        else:
            st.metric("Total Views", str(total_views))
    
    with col2:
        if isinstance(video_likes, (int, float)):
            st.metric("Video Likes", f"{int(video_likes):,}")
        else:
            st.metric("Video Likes", str(video_likes))
    
    with col3:
        if video_dislikes != 'N/A' and pd.notna(video_dislikes):
            st.metric("Video Dislikes", f"{int(video_dislikes):,}")
        else:
            st.metric("Video Dislikes", "Hidden*")
        st.caption("*Hardcoded: YouTube removed public dislike counts")
    
    with col4:
        # Calculate like to dislike ratio if we have the data
        if (video_likes != 'N/A' and pd.notna(video_likes) and 
            video_dislikes != 'N/A' and pd.notna(video_dislikes) and video_dislikes > 0):
            like_dislike_ratio = int(video_likes) // int(video_dislikes)
            st.metric("Like:Dislike Ratio", f"{like_dislike_ratio}:1")
        elif video_likes != 'N/A' and pd.notna(video_likes) and video_dislikes == 0:
            st.metric("Like:Dislike Ratio", f"{int(video_likes)}:0")
        else:
            st.metric("Like:Dislike Ratio", "N/A")
    
    st.markdown("---")
    
    # Comment Analysis KPIs
    st.markdown("### 💬 Comment Analysis Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_comments = safe_get_metric(summary_df, 'Total Comments', 0)
        st.metric("Total Comments", int(total_comments))
        st.caption("*Excluding replies")
    
    with col2:
        # Get negative sentiment from sentiment_df
        negative_row = sentiment_df[sentiment_df['sentiment_type'] == 'Negative']
        if len(negative_row) > 0:
            negative_pct = negative_row['percentage'].iloc[0]
            st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
        else:
            st.metric("Negative Sentiment", "N/A")
    
    with col3:
        top_issue = summary_df[summary_df['metric'] == 'Top Issue']['value'].iloc[0] if len(summary_df[summary_df['metric'] == 'Top Issue']) > 0 else 'N/A'
        st.metric("Top Issue", str(top_issue))
    
    with col4:
        high_impact = safe_get_metric(summary_df, 'High Impact Issues', 0)
        st.metric("High Impact Issues", int(high_impact))
    
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
    
    tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Issue Analysis", "Summary Metrics"])
    
    with tab1:
        st.dataframe(sentiment_df, use_container_width=True)
    
    with tab2:
        # Sort by priority for better readability
        if 'weighted_score' in feature_df.columns:
            feature_display = feature_df.sort_values('weighted_score', ascending=False)
        else:
            feature_display = feature_df.sort_values('count', ascending=False)
        st.dataframe(feature_display, use_container_width=True)
    
    with tab3:
        st.dataframe(summary_df, use_container_width=True)
    
    # Business insights
    st.markdown("---")
    st.subheader("🎯 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sentiment Analysis:**")
        
        # Get sentiment percentages from sentiment_df instead of summary_df
        negative_row = sentiment_df[sentiment_df['sentiment_type'] == 'Negative']
        positive_row = sentiment_df[sentiment_df['sentiment_type'] == 'Positive']
        neutral_row = sentiment_df[sentiment_df['sentiment_type'] == 'Neutral']
        
        negative_pct = negative_row['percentage'].iloc[0] if len(negative_row) > 0 else 0
        positive_pct = positive_row['percentage'].iloc[0] if len(positive_row) > 0 else 0
        neutral_pct = neutral_row['percentage'].iloc[0] if len(neutral_row) > 0 else 0
        
        # Color-coded negative sentiment
        if negative_pct > 50:
            st.markdown(f"- <span style='color: #ff4444;'>{negative_pct:.1f}% negative sentiment - indicates major customer dissatisfaction</span>", unsafe_allow_html=True)
        elif negative_pct > 25:
            st.markdown(f"- <span style='color: #ff8800;'>{negative_pct:.1f}% negative sentiment - moderate concern</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"- <span style='color: #00cc44;'>{negative_pct:.1f}% negative sentiment - acceptable level</span>", unsafe_allow_html=True)
        
        # Color-coded positive sentiment
        if positive_pct < 20:
            st.markdown(f"- <span style='color: #ff4444;'>{positive_pct:.1f}% positive sentiment - significant room for improvement</span>", unsafe_allow_html=True)
        elif positive_pct < 40:
            st.markdown(f"- <span style='color: #ff8800;'>{positive_pct:.1f}% positive sentiment - below average</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"- <span style='color: #00cc44;'>{positive_pct:.1f}% positive sentiment - good customer satisfaction</span>", unsafe_allow_html=True)
        
        st.write(f"- {neutral_pct:.1f}% neutral sentiment")
        
        # Get average likes for negative comments
        if len(negative_row) > 0:
            avg_neg_likes = negative_row['avg_likes'].iloc[0]
            if avg_neg_likes > 8:
                st.markdown(f"- <span style='color: #ff4444;'>Negative comments receive {avg_neg_likes:.1f} avg likes - high engagement on complaints</span>", unsafe_allow_html=True)
            else:
                st.write(f"- Negative comments receive {avg_neg_likes:.1f} avg likes")
    
    with col2:
        st.markdown("**Top Issues to Address:**")
        
        # Get top 3 issues by count
        if 'count' in feature_df.columns and 'issue' in feature_df.columns:
            top_issues = feature_df.nlargest(3, 'count')
            for i, (_, row) in enumerate(top_issues.iterrows(), 1):
                issue_name = row['issue']
                count = row['count']
                likes = row.get('total_likes', 0)
                impact = row.get('impact_category', 'Unknown')
                st.write(f"{i}. **{issue_name}** ({count} mentions, {likes} likes) - {impact}")
        else:
            st.write("Issue data not available")
    
    # Action items
    st.markdown("---")
    st.subheader("📝 Recommended Actions")
    
    # Check negative sentiment threshold
    negative_pct = safe_get_metric(summary_df, 'Negative Sentiment %', 0)
    if negative_pct > 50:
        st.error("🚨 **URGENT**: Negative sentiment > 50% - immediate action required")
    elif negative_pct > 35:
        st.warning("⚠️ **ATTENTION**: Elevated negative sentiment - monitor closely")
    else:
        st.success("✅ Sentiment levels within acceptable range")
    
    st.markdown("""
    **Immediate Actions:**
    1. Address top customer issues identified in the analysis
    2. Create follow-up content addressing common concerns
    3. Improve video description with clearer instructions
    4. Monitor sentiment trends over time
    
    **Success Metrics to Track:**
    - Reduce negative sentiment below 30%
    - Increase positive sentiment above 40%
    - Decrease frequency of top issues in future videos
    """)

if __name__ == "__main__":
    main()