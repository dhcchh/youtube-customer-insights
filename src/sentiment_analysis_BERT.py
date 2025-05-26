import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BERTSentimentAnalyzer:
    """
    Simple BERT-based sentiment analyzer for YouTube comments.
    
    Takes your preprocessed comments and adds sentiment analysis columns.
    """
    
    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment-latest', 
                 device=None, batch_size=16):
        """
        Initialize BERT sentiment analyzer.
        
        Args:
            model_name: Pre-trained BERT model for sentiment analysis
            device: 'cuda' for GPU, 'cpu' for CPU, None for auto-detect
            batch_size: Number of comments to process at once
        """
        self.batch_size = batch_size
        
        # Set device (GPU vs CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load BERT model and tokenizer
        print("Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")
    
    def predict_sentiment(self, texts):
        """
        Predict sentiment for a list of texts.
        
        Args:
            texts: List of comment strings
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Analyzing sentiment"):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get predictions
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Store results
                batch_probs = probs.cpu().numpy()
                batch_predictions = np.argmax(batch_probs, axis=1)
                
                all_predictions.extend(batch_predictions.tolist())
                all_probabilities.extend(batch_probs.tolist())
        
        # Convert to sentiment labels
        sentiment_labels = self._convert_to_labels(all_predictions)
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'sentiment_labels': sentiment_labels,
            'confidence_scores': [max(probs) for probs in all_probabilities]
        }
    
    def _convert_to_labels(self, predictions):
        """
        Convert numeric predictions to sentiment labels.
        
        Twitter RoBERTa model outputs:
        0 = Negative
        1 = Neutral  
        2 = Positive
        """
        label_mapping = {
            0: 'Negative',
            1: 'Neutral', 
            2: 'Positive'
        }
        return [label_mapping[pred] for pred in predictions]
    
    def analyze_dataframe(self, df, text_column='cleaned_text'):
        """
        Add sentiment analysis to your DataFrame.
        
        Args:
            df: Your preprocessed comments DataFrame
            text_column: Column with cleaned text
            
        Returns:
            DataFrame with sentiment columns added
        """
        print(f"Analyzing {len(df)} comments...")
        
        # Get sentiment predictions
        results = self.predict_sentiment(df[text_column].tolist())
        
        # Add results to DataFrame
        df_results = df.copy()
        df_results['sentiment_label'] = results['sentiment_labels']
        df_results['sentiment_confidence'] = results['confidence_scores']
        df_results['sentiment_prediction'] = results['predictions']
        
        # Add probability columns (3 classes for Twitter model)
        prob_array = np.array(results['probabilities'])
        df_results['prob_negative'] = prob_array[:, 0]
        df_results['prob_neutral'] = prob_array[:, 1]
        df_results['prob_positive'] = prob_array[:, 2]
        
        return df_results


def quick_sentiment_summary(df):
    """
    Generate a quick summary of sentiment results.
    
    Args:
        df: DataFrame with sentiment analysis results
    """
    total = len(df)
    sentiment_counts = df['sentiment_label'].value_counts()
    
    print("\nSentiment Summary:")
    print("-" * 50)
    print(f"{'Sentiment':<15} {'Count':<6} {'%':<6} {'Total Likes':<12}")
    print("-" * 50)
    
    for sentiment, count in sentiment_counts.items():
        pct = (count / total) * 100
        # Calculate total likes for this sentiment category
        sentiment_likes = df[df['sentiment_label'] == sentiment]['like_count'].sum()
        print(f"{sentiment:<15}: {count:>4} ({pct:>5.1f}%) {sentiment_likes:>10} likes")
    
    print("-" * 50)
    
    # Overall sentiment with like counts
    negative_mask = df['sentiment_label'] == 'Negative'
    positive_mask = df['sentiment_label'] == 'Positive'
    neutral_mask = df['sentiment_label'] == 'Neutral'
    
    negative_count = negative_mask.sum()
    positive_count = positive_mask.sum()
    neutral_count = neutral_mask.sum()
    
    negative_likes = df[negative_mask]['like_count'].sum()
    positive_likes = df[positive_mask]['like_count'].sum()
    neutral_likes = df[neutral_mask]['like_count'].sum()
    total_likes = df['like_count'].sum()
    
    negative_pct = (negative_count / total) * 100
    positive_pct = (positive_count / total) * 100
    neutral_pct = (neutral_count / total) * 100
    
    print("\nOverall Summary:")
    print("-" * 30)
    print(f"Negative: {negative_pct:>5.1f}% ({negative_count:>4} comments, {negative_likes:>6} likes)")
    print(f"Neutral:  {neutral_pct:>5.1f}% ({neutral_count:>4} comments, {neutral_likes:>6} likes)")
    print(f"Positive: {positive_pct:>5.1f}% ({positive_count:>4} comments, {positive_likes:>6} likes)")
    print(f"Total:    100.0% ({total:>4} comments, {total_likes:>6} likes)")
    
    # Engagement insights
    if total_likes > 0:
        negative_like_pct = (negative_likes / total_likes) * 100
        positive_like_pct = (positive_likes / total_likes) * 100
        
        print(f"\nEngagement Insights:")
        print(f"Negative comments received {negative_like_pct:.1f}% of total likes")
        print(f"Positive comments received {positive_like_pct:.1f}% of total likes")
        
        # Average likes per comment by sentiment
        avg_negative_likes = negative_likes / negative_count if negative_count > 0 else 0
        avg_positive_likes = positive_likes / positive_count if positive_count > 0 else 0
        avg_neutral_likes = neutral_likes / neutral_count if neutral_count > 0 else 0
        
        print(f"\nAverage likes per comment:")
        print(f"Negative: {avg_negative_likes:.1f} likes/comment")
        print(f"Neutral:  {avg_neutral_likes:.1f} likes/comment") 
        print(f"Positive: {avg_positive_likes:.1f} likes/comment")
    
    # Risk assessment
    print(f"\nRisk Assessment:")
    if negative_pct > 40:
        print("⚠️  HIGH NEGATIVE SENTIMENT - Requires immediate attention")
    elif negative_pct > 25:
        print("⚡ MODERATE NEGATIVE SENTIMENT - Monitor closely")
    else:
        print("✅ LOW NEGATIVE SENTIMENT - Good customer satisfaction")



class SentimentValidator:
    """
    Validate BERT sentiment analysis results through various methods.
    
    Provides tools for manual validation, confidence analysis, and statistical checks
    to ensure sentiment analysis quality.
    """
    
    def __init__(self, sentiment_df):
        """
        Initialize validator with sentiment analysis results.
        
        Args:
            sentiment_df: DataFrame with BERT sentiment analysis results
        """
        self.df = sentiment_df
        
        # Validate required columns exist
        required_cols = ['text', 'sentiment_label', 'sentiment_confidence']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def create_manual_validation_sample(self, n=50, strategy='random', save_path=None):
        """
        Create a sample for manual validation.
        
        Args:
            n: Number of samples to create
            strategy: 'random', 'low_confidence', 'high_confidence', or 'mixed'
            save_path: Optional path to save validation CSV
            
        Returns:
            DataFrame ready for manual labeling
        """
        if strategy == 'random':
            sample = self.df.sample(n)
        elif strategy == 'low_confidence':
            # Focus on predictions model is unsure about
            low_conf = self.df[self.df['sentiment_confidence'] < 0.7]
            sample = low_conf.sample(min(n, len(low_conf)))
        elif strategy == 'high_confidence':
            # Check if high confidence predictions are actually correct
            high_conf = self.df[self.df['sentiment_confidence'] > 0.9]
            sample = high_conf.sample(min(n, len(high_conf)))
        elif strategy == 'mixed':
            # Balanced sample across confidence levels
            n_per_group = n // 3
            low_conf = self.df[self.df['sentiment_confidence'] < 0.6].sample(min(n_per_group, len(self.df[self.df['sentiment_confidence'] < 0.6])))
            mid_conf = self.df[(self.df['sentiment_confidence'] >= 0.6) & (self.df['sentiment_confidence'] <= 0.8)].sample(min(n_per_group, len(self.df[(self.df['sentiment_confidence'] >= 0.6) & (self.df['sentiment_confidence'] <= 0.8)])))
            high_conf = self.df[self.df['sentiment_confidence'] > 0.8].sample(min(n_per_group, len(self.df[self.df['sentiment_confidence'] > 0.8])))
            sample = pd.concat([low_conf, mid_conf, high_conf])
        else:
            raise ValueError("Strategy must be 'random', 'low_confidence', 'high_confidence', or 'mixed'")
        
        # Create validation DataFrame
        validation_df = sample[['text', 'sentiment_label', 'sentiment_confidence']].copy()
        validation_df = validation_df.sort_values('sentiment_confidence', ascending=False)
        validation_df['manual_label'] = ''  # For human to fill
        validation_df['correct'] = ''       # For marking True/False
        validation_df['notes'] = ''         # For additional comments
        
        if save_path:
            validation_df.to_csv(save_path, index=False)
            print(f"Manual validation sample saved to {save_path}")
            print(f"Instructions: Fill 'manual_label' column with your assessment")
            print(f"Mark 'correct' as True/False, add 'notes' if needed")
        
        return validation_df
    
    def calculate_validation_accuracy(self, validation_df):
        """
        Calculate accuracy from completed manual validation.
        
        Args:
            validation_df: Completed validation DataFrame with 'correct' column filled
            
        Returns:
            Dictionary with accuracy metrics
        """
        if 'correct' not in validation_df.columns:
            raise ValueError("Validation DataFrame missing 'correct' column")
        
        # Filter out entries without manual validation
        completed = validation_df[validation_df['correct'].isin([True, False, 'True', 'False', 'true', 'false'])]
        
        if len(completed) == 0:
            print("No completed validations found")
            return None
        
        # Convert to boolean if string
        completed = completed.copy()
        completed['correct'] = completed['correct'].astype(str).str.lower().map({'true': True, 'false': False})
        
        total_validated = len(completed)
        correct_predictions = completed['correct'].sum()
        accuracy = (correct_predictions / total_validated) * 100
        
        # Accuracy by confidence level
        high_conf_acc = completed[completed['sentiment_confidence'] > 0.8]['correct'].mean() * 100 if len(completed[completed['sentiment_confidence'] > 0.8]) > 0 else 0
        low_conf_acc = completed[completed['sentiment_confidence'] < 0.6]['correct'].mean() * 100 if len(completed[completed['sentiment_confidence'] < 0.6]) > 0 else 0
        
        # Accuracy by sentiment
        sentiment_accuracy = completed.groupby('sentiment_label')['correct'].agg(['count', 'sum', 'mean']).round(3)
        
        results = {
            'overall_accuracy': accuracy,
            'total_validated': total_validated,
            'correct_predictions': correct_predictions,
            'high_confidence_accuracy': high_conf_acc,
            'low_confidence_accuracy': low_conf_acc,
            'sentiment_breakdown': sentiment_accuracy
        }
        
        print(f"\nValidation Results:")
        print(f"Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_validated})")
        print(f"High Confidence (>0.8): {high_conf_acc:.1f}% accuracy")
        print(f"Low Confidence (<0.6): {low_conf_acc:.1f}% accuracy")
        print(f"\nAccuracy by Sentiment:")
        print(sentiment_accuracy)
        
        return results
    
    def analyze_confidence_distribution(self):
        """
        Analyze confidence score patterns to identify potential issues.
        
        Returns:
            Dictionary with confidence analysis results
        """
        conf_stats = self.df['sentiment_confidence'].describe()
        
        # Confidence level buckets
        very_low = self.df[self.df['sentiment_confidence'] < 0.5]
        low = self.df[(self.df['sentiment_confidence'] >= 0.5) & (self.df['sentiment_confidence'] < 0.7)]
        medium = self.df[(self.df['sentiment_confidence'] >= 0.7) & (self.df['sentiment_confidence'] < 0.9)]
        high = self.df[self.df['sentiment_confidence'] >= 0.9]
        
        print("\nConfidence Distribution Analysis:")
        print("-" * 40)
        print(f"Very Low (<0.5): {len(very_low):>4} ({len(very_low)/len(self.df)*100:>5.1f}%)")
        print(f"Low (0.5-0.7):  {len(low):>4} ({len(low)/len(self.df)*100:>5.1f}%)")
        print(f"Medium (0.7-0.9):{len(medium):>4} ({len(medium)/len(self.df)*100:>5.1f}%)")
        print(f"High (>0.9):     {len(high):>4} ({len(high)/len(self.df)*100:>5.1f}%)")
        
        print(f"\nConfidence Statistics:")
        print(f"Mean: {conf_stats['mean']:.3f}")
        print(f"Std:  {conf_stats['std']:.3f}")
        print(f"Min:  {conf_stats['min']:.3f}")
        print(f"Max:  {conf_stats['max']:.3f}")
        
        # Flag potential issues
        warnings = []
        if len(very_low) / len(self.df) > 0.2:  # >20% very low confidence
            warnings.append("⚠️  High proportion of very low confidence predictions")
        
        if conf_stats['mean'] < 0.6:
            warnings.append("⚠️  Low average confidence - model may be struggling")
        
        if conf_stats['std'] < 0.1:
            warnings.append("⚠️  Very low confidence variance - check for bias")
        
        if warnings:
            print(f"\nWarnings:")
            for warning in warnings:
                print(warning)
        
        # Return low confidence samples for manual review
        low_confidence_samples = very_low.nsmallest(10, 'sentiment_confidence')[['text', 'sentiment_label', 'sentiment_confidence']]
        
        return {
            'statistics': conf_stats,
            'distribution': {
                'very_low': len(very_low),
                'low': len(low), 
                'medium': len(medium),
                'high': len(high)
            },
            'warnings': warnings,
            'low_confidence_samples': low_confidence_samples
        }
    
    def run_statistical_validation(self):
        """
        Run statistical checks on sentiment distribution and scores.
        
        Returns:
            Dictionary with statistical validation results
        """
        # Sentiment distribution analysis
        sentiment_dist = self.df['sentiment_label'].value_counts(normalize=True) * 100
        
        print("\nStatistical Validation:")
        print("-" * 30)
        print("Sentiment Distribution:")
        for sentiment, pct in sentiment_dist.items():
            print(f"{sentiment:<15}: {pct:>5.1f}%")
        
        # Check for unrealistic distributions
        warnings = []
        
        # Check for extreme distributions
        very_negative_pct = sentiment_dist.get('Very Negative', 0)
        very_positive_pct = sentiment_dist.get('Very Positive', 0)
        negative_pct = sentiment_dist.get('Negative', 0) + very_negative_pct
        positive_pct = sentiment_dist.get('Positive', 0) + very_positive_pct
        
        if very_negative_pct > 40:
            warnings.append("⚠️  Extremely high 'Very Negative' sentiment - validate manually")
        
        if very_positive_pct > 60:
            warnings.append("⚠️  Suspiciously high 'Very Positive' sentiment - check for bias")
        
        if negative_pct > 70:
            warnings.append("⚠️  Overwhelming negative sentiment - verify data quality")
        
        if positive_pct > 80:
            warnings.append("⚠️  Unrealistically positive - possible data filtering issue")
        
        # Check confidence score distribution
        conf_mean = self.df['sentiment_confidence'].mean()
        conf_std = self.df['sentiment_confidence'].std()
        
        if conf_mean > 0.95:
            warnings.append("⚠️  Suspiciously high average confidence - check for overfitting")
        
        if conf_std < 0.05:
            warnings.append("⚠️  Very low confidence variance - possible model issues")
        
        # Check for correlation between likes and sentiment
        if 'like_count' in self.df.columns:
            sentiment_likes = self.df.groupby('sentiment_label')['like_count'].agg(['count', 'sum', 'mean']).round(2)
            print(f"\nLikes by Sentiment:")
            print(sentiment_likes)
            
            # Check if negative comments get disproportionate likes
            total_likes = self.df['like_count'].sum()
            if total_likes > 0:
                negative_like_share = self.df[self.df['sentiment_label'].isin(['Negative', 'Very Negative'])]['like_count'].sum() / total_likes
                if negative_like_share > 0.4:
                    warnings.append("⚠️  Negative comments receiving high engagement - major issue")
        
        if warnings:
            print(f"\nStatistical Warnings:")
            for warning in warnings:
                print(warning)
        else:
            print(f"\n✅ Statistical validation passed - distribution looks reasonable")
        
        return {
            'sentiment_distribution': sentiment_dist.to_dict(),
            'confidence_stats': {
                'mean': conf_mean,
                'std': conf_std
            },
            'warnings': warnings
        }
    
    def full_validation_report(self, save_manual_sample=True, sample_size=50):
        """
        Run complete validation analysis.
        
        Args:
            save_manual_sample: Whether to create manual validation sample
            sample_size: Size of manual validation sample
            
        Returns:
            Dictionary with all validation results
        """
        print("="*60)
        print("SENTIMENT ANALYSIS VALIDATION REPORT")
        print("="*60)
        
        # 1. Statistical validation
        statistical_results = self.run_statistical_validation()
        
        # 2. Confidence analysis  
        confidence_results = self.analyze_confidence_distribution()
        
        # 3. Create manual validation sample
        manual_sample = None
        if save_manual_sample:
            print(f"\n" + "="*40)
            print("MANUAL VALIDATION SAMPLE")
            print("="*40)
            manual_sample = self.create_manual_validation_sample(
                n=sample_size, 
                strategy='mixed',
                save_path='manual_validation_sample.csv'
            )
        
        return {
            'statistical_validation': statistical_results,
            'confidence_analysis': confidence_results,
            'manual_validation_sample': manual_sample
        }
