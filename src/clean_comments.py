import re
import pandas as pd
import unicodedata


class YouTubeCommentCleaner:
    """
    Simple text cleaner for YouTube comments. 
    Currently has methods only for BERT, will extend for other models/tasks
    in the future.
    """
    
    def __init__(self, remove_urls=True, remove_mentions=True, lowercase=False):
        """
        Initialize cleaner with basic options.
        
        Args:
            remove_urls: Remove HTTP/HTTPS links
            remove_mentions: Remove @username mentions  
            lowercase: Convert to lowercase
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.lowercase = lowercase
        
    def clean_text(self, text: str) -> str:
        """
        Clean a single comment text.
        
        What it does:
        - Decodes HTML entities (&amp; → &)
        - Removes URLs and @mentions 
        - Removes emojis
        - Expands contractions (can't → cannot)
        - Normalizes whitespace
        - Converts to lowercase
        
        Args:
            text: Raw comment text
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        cleaned = text
        
        # Decode HTML entities
        html_entities = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>', 
            '&quot;': '"', '&#39;': "'", '&nbsp;': ' '
        }
        for entity, replacement in html_entities.items():
            cleaned = cleaned.replace(entity, replacement)
        
        # Remove URLs
        if self.remove_urls:
            cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
        
        # Remove @mentions
        if self.remove_mentions:
            cleaned = re.sub(r'@\w+', '', cleaned)
            
        # Remove emojis
        cleaned = ''.join(char for char in cleaned if unicodedata.category(char) != 'So')
        
        # Address contractions based on the most common cases - avoid cases like b'day
        # Note: This is a simplified version and may not cover all cases
        contractions = {
            r"\bwon't\b": "will not",
            r"\bcan't\b": "cannot", 
            r"\bwouldn't\b": "would not",
            r"\bcouldn't\b": "could not",
            r"\bshouldn't\b": "should not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bdon't\b": "do not",
            r"\bisn't\b": "is not",
            r"\baren't\b": "are not",
            r"\bwasn't\b": "was not",
            r"\bweren't\b": "were not",
            r"\bhaven't\b": "have not",
            r"\bhasn't\b": "has not",
            r"\bhadn't\b": "had not",
            r"\byou're\b": "you are",
            r"\bwe're\b": "we are", 
            r"\bthey're\b": "they are",
            r"\bi'm\b": "i am",
            r"\byou've\b": "you have",
            r"\bwe've\b": "we have",
            r"\bthey've\b": "they have",
            r"\bi've\b": "i have",
            r"\byou'll\b": "you will",
            r"\bwe'll\b": "we will",
            r"\bthey'll\b": "they will",
            r"\bi'll\b": "i will"
        }
        for pattern, replacement in contractions.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Clean up excessive punctuation but preserve emphasis
        cleaned = re.sub(r'([.!?]){4,}', r'\1\1\1', cleaned)  # Keep up to 3 for VADER
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Convert to lowercase (disabled by default for VADER)
        if self.lowercase:
            cleaned = cleaned.lower()
            
        return cleaned
    
    def clean_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Clean all comments in a DataFrame.
        
        What it does:
        - Applies clean_text() to all comments
        - Removes empty comments after cleaning
        - Resets DataFrame index
        
        Args:
            df: DataFrame with comment data
            text_column: Name of column containing comment text
            
        Returns:
            DataFrame with cleaned text in 'cleaned_text' column
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        df_cleaned = df.copy()
        df_cleaned['cleaned_text'] = df_cleaned[text_column].apply(self.clean_text)
        
        # Remove empty comments
        df_cleaned = df_cleaned[df_cleaned['cleaned_text'].str.len() > 0].copy()
        df_cleaned.reset_index(drop=True, inplace=True)
        
        return df_cleaned
    
    def preview_cleaning(self, df: pd.DataFrame, text_column: str = 'text', 
                        n_samples: int = 3) -> pd.DataFrame:
        """
        Preview cleaning results on sample comments.
        
        What it does:
        - Shows before/after cleaning for first n comments
        - Useful for checking if cleaning works as expected
        
        Args:
            df: DataFrame with comments
            text_column: Name of text column
            n_samples: Number of samples to show
            
        Returns:
            DataFrame showing original vs cleaned text
        """
        sample_df = df.head(n_samples).copy()
        sample_df['cleaned_text'] = sample_df[text_column].apply(self.clean_text)
        
        return pd.DataFrame({
            'original': sample_df[text_column],
            'cleaned': sample_df['cleaned_text']
        })