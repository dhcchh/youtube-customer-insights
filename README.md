# Nespresso Tutorial Video NLP Project

## YouTube Comment Analysis for Business Insights

A Python-based NLP (Natural Language Processing) project for analysing YouTube comments, inspired by my own experience with Nespresso products. Built to analyze customer feedback on product demonstration videos and extract business insights.

## 🎯 Project Overview

This project demonstrates end-to-end NLP pipeline development:
- **Data Collection**: YouTube API integration for comment extraction
- **Text Preprocessing**: BERT-optimized cleaning pipeline
- **Sentiment Analysis**: Transformer-based sentiment classification (RoBERTa)
- **Feature Extraction**: Business-specific insight extraction using regex patterns
- **Visualization**: Interactive Streamlit dashboard for business stakeholders

## 🏗️ Architecture

```
├── src/
│   ├── load_comments.py           # YouTube API client for data collection
│   ├── clean_comments.py          # BERT-optimized text preprocessing
│   ├── sentiment_analysis_BERT.py # Transformer-based sentiment analysis
│   ├── feature_extractor.py       # Business insight extraction
│   └── pipeline.py               # End-to-end analysis pipeline
├── app.py                        # Streamlit dashboard
├── run_analysis.py              # Simple script to run analysis
├── data/
│   ├── exports/                 # Final analysis results & reports
│   ├── processed/               # Cleaned data (where necessary) & sentiment analysis
│   ├── raw/                     # Raw YouTube comments & video stats
│   └── st_dashboard_ready/      # Data files transformed to a form suitable for visualisation
├── notebooks/ # Independent testing by myself while developing this project
└── .env                         # API key (create this file on your own)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested with Python 3.13)
- YouTube Data API v3 key
- ~2GB RAM for BERT model inference
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/dhcchh/youtube-customer-insights
cd nespresso-vid-proj
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up YouTube API credentials**

You need to obtain a YouTube Data API v3 key:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the YouTube Data API v3
4. Create credentials (API Key)
5. Create a `.env` file in the project root:

```bash
API_KEY=your_api_key_here
```

**Note**: The `.env` file is not included in this repository for security reasons. You must create your own API key.

### Basic Usage

#### Simple Analysis
```bash
python run_analysis.py
```
## 📊 Dashboard Features

Launch the dashboard with: `streamlit run app.py`

The dashboard provides:
- **Surface Metrics**: Video views, likes, dislikes, engagement ratios
- **Sentiment Distribution**: Pie chart showing positive/negative/neutral breakdown
- **Engagement Analysis**: How sentiment correlates with comment likes
- **Issue Identification**: Top customer pain points with frequency metrics
- **Impact Matrix**: Scatter plot showing issue frequency vs engagement
- **Actionable Insights**: Business recommendations based on analysis

## 🔧 Technical Implementation

### Text Preprocessing (BERT-Optimized)
- HTML entity decoding (`&amp;` → `&`)
- URL/mention anonymization (`[URL]`, `[USER]`)
- Emoji removal while preserving punctuation
- Minimal preprocessing to maintain BERT's natural language understanding
- Case preservation (BERT is case-sensitive)

### Sentiment Analysis
- Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa)
- Batch processing for GPU efficiency
- Confidence scoring for prediction quality assessment
- Manual validation framework included (`SentimentValidator`)

### Feature Extraction (Nespresso-Specific)
Business issue categories detected via Regex:
- **Descaling Button Requests**: "descaling.*button", "add.*descaling.*button"
- **App Improvements**: "app.*should.*button", "app.*descaling.*mode"
- **Simplification Requests**: "simpler.*process", "one.*button"
- **Complexity Complaints**: "too.*complicated", "complex.*process"
- **Instruction Problems**: "instructions.*unclear", "confusing.*process"
- **Technical Issues**: "machine.*not working", "error.*light"

### Pipeline Design
- **Modular Architecture**: Each component is independently testable
- **Data Persistence**: Results saved at each stage for debugging
- **Error Handling**: API rate limiting and quota management
- **Scalability**: Batch processing for large comment datasets

## 📈 Key Metrics Tracked

- **Sentiment Distribution**: Percentage breakdown (Positive/Negative/Neutral)
- **Engagement by Sentiment**: Total likes received by sentiment category  
- **Issue Frequency**: Count of mentions for each business issue
- **Impact Score**: Weighted metric (frequency × engagement)
- **Confidence Analysis**: Model prediction quality assessment
- **Risk Assessment**: Automated alerts for high negative sentiment

## 🎓 Learning Objectives

This project demonstrates:
- **Business Intelligence**: Converting unstructured text to actionable insights
- **NLP Pipeline Development**: End-to-end text processing workflow
- **API Integration**: YouTube Data API usage with pagination and rate limiting
- **Transformer Models**: BERT/RoBERTa for sentiment classification
- **Model Validation**: Confidence analysis and manual validation frameworks
- **Dashboard Development**: Streamlit for stakeholder communication

## 📦 Dependencies

`requirements.txt`:
```
streamlit>=1.28.0
pandas>=1.5.0
torch>=1.13.0
transformers>=4.21.0
plotly>=5.15.0
google-api-python-client>=2.0.0
python-dotenv>=0.19.0
tqdm>=4.64.0
numpy>=1.21.0
scikit-learn>=1.1.0
```

## 🔄 Data Flow

1. **Collection**: YouTube API → Raw comments & video stats → `data/raw/`
2. **Preprocessing**: Text cleaning → BERT-ready format → `data/processed/`
3. **Analysis**: RoBERTa sentiment classification → Labeled dataset → `data/processed/`
4. **Extraction**: Pattern matching → Business insights → `data/processed/`
5. **Dashboard**: Streamlit transformation → `data/st_dashboard_ready/`
6. **Export**: Final analysis results → `data/exports/`

### Data Directory Details
- **`data/raw/`**: Original YouTube data (comments CSV, video statistics)
- **`data/processed/`**: Cleaned comments, sentiment analysis results, feature extraction
- **`data/st_dashboard_ready/`**: Aggregated data ready for Streamlit visualization
- **`data/exports/`**: Final business reports and analysis summaries
