import re
import pandas as pd
import unicodedata


class YouTubeCommentCleaner:
    """
    Minimal text cleaner optimized for BERT-based sentiment analysis.
    Preserves natural language patterns that BERT handles well.
    """
    
    def __init__(self, remove_urls=True, remove_mentions=True, min_length=3):
        """
        Initialize cleaner with minimal BERT-optimized settings.
        
        Args:
            remove_urls: Remove HTTP/HTTPS links (they're noise for sentiment)
            remove_mentions: Remove @username mentions (usually irrelevant)
            min_length: Minimum comment length after cleaning
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.min_length = min_length
        
    def clean_text(self, text: str) -> str:
        """
        Clean text with minimal preprocessing for BERT.
        
        BERT-optimized approach:
        - Preserves case (BERT is case-sensitive)
        - Preserves punctuation patterns (sentiment indicators)
        - Preserves contractions (BERT understands them naturally)
        - Minimal processing to avoid artifacts
        
        Args:
            text: Raw comment text
            
        Returns:
            Cleaned text optimized for BERT
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        cleaned = text.strip()
        
        # 1. Decode HTML entities
        html_entities = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>', 
            '&quot;': '"', '&#39;': "'", '&nbsp;': ' ',
            '&apos;': "'", '&copy;': '©', '&reg;': '®'
        }
        for entity, replacement in html_entities.items():
            cleaned = cleaned.replace(entity, replacement)
        
        # 2. Remove URLs (noise for sentiment analysis)
        if self.remove_urls:
            cleaned = re.sub(r'https?://\S+|www\.\S+', '[URL]', cleaned)
        
        # 3. Remove @mentions (usually not sentiment-relevant)
        if self.remove_mentions:
            cleaned = re.sub(r'@\w+', '[USER]', cleaned)
            
        # 4. Remove emojis (BERT can't process them reliably)
        # Remove all characters that are symbols/pictographs
        cleaned = ''.join(char for char in cleaned 
                         if not (0x1F000 <= ord(char) <= 0x1FFFF or 
                                0x2600 <= ord(char) <= 0x27BF or
                                unicodedata.category(char) in ['So', 'Sm', 'Sk']))
        
        # 5. Normalize whitespace only
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 6. Check minimum length
        if len(cleaned) < self.min_length:
            return ""
            
        return cleaned
    
    def clean_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Clean all comments in DataFrame.
        
        Args:
            df: DataFrame with comment data
            text_column: Name of column containing comment text
            
        Returns:
            DataFrame with cleaned text and basic metadata
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        df_cleaned = df.copy()
        
        # Apply cleaning
        df_cleaned['cleaned_text'] = df_cleaned[text_column].apply(self.clean_text)
        
        # Add basic metadata
        df_cleaned['original_length'] = df_cleaned[text_column].str.len()
        df_cleaned['cleaned_length'] = df_cleaned['cleaned_text'].str.len()
        
        # Remove empty comments after cleaning
        initial_count = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned['cleaned_text'].str.len() >= self.min_length].copy()
        df_cleaned.reset_index(drop=True, inplace=True)
        
        removed_count = initial_count - len(df_cleaned)
        if removed_count > 0:
            print(f"Removed {removed_count} comments that were too short after cleaning")
        
        return df_cleaned
    
    def preview_cleaning(self, df: pd.DataFrame, text_column: str = 'text', 
                        n_samples: int = 5) -> pd.DataFrame:
        """
        Preview cleaning results with before/after comparison.
        
        Args:
            df: DataFrame with comments
            text_column: Name of text column
            n_samples: Number of samples to show
            
        Returns:
            DataFrame showing original vs cleaned text
        """
        sample_df = df.head(n_samples).copy()
        sample_df['cleaned_text'] = sample_df[text_column].apply(self.clean_text)
        
        preview_df = pd.DataFrame({
            'original': sample_df[text_column],
            'cleaned': sample_df['cleaned_text'],
            'char_reduction': ((sample_df[text_column].str.len() - sample_df['cleaned_text'].str.len()) /
                             sample_df[text_column].str.len() * 100).round(1)
        })
        
        return preview_df