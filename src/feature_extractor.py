import re
import pandas as pd
from typing import Dict, List


class FeatureExtractor:
    """
    Extract business insights from Nespresso descaling video comments.
    """
    
    def __init__(self):
        """Initialize with Nespresso-specific patterns"""
        self.categories = {
            'descaling_button_request': [
                r'descaling.*button',
                r'button.*descal',
                r'just put.*button.*machine',
                r'add.*descaling.*button'
            ],
            
            'app_improvements': [
                r'app.*(?:should|could|need).*button',
                r'app.*(?:start|begin).*descaling',
                r'button.*app',
                r'app.*descaling.*mode'
            ],
            
            'simplification_requests': [
                r'simpler.*(?:process|way|method)',
                r'easier.*(?:process|way|method)',
                r'one.*button.*(?:process|descaling)',
                r'single.*button',
                r'automatic.*descaling',
                r'auto.*descaling'
            ],
            
            'complexity_complaints': [
                r'(?:overly|too|so).*complicated',
                r'most complicated.*process',
                r'complicated.*button.*pressing',
                r'complex.*process',
                r'(?:hard|difficult).*follow'
            ],
            
            'instruction_problems': [
                r'instructions?.*(?:unclear|confusing|bad)',
                r'confusing.*(?:instructions?|process|video)',
                r'don\'t understand.*(?:process|instructions?)',
                r'explain.*better',
                r'poorly explained'
            ],
            
            'technical_issues': [
                r'machine.*(?:not working|broken|error)',
                r'error.*light',
                r'won\'t.*(?:work|start|descale)',
                r'buttons?.*not.*(?:working|responding)',
                r'flashing.*(?:light|button).*not.*work'
            ]
        }
    
    def extract_insights(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> Dict:
        """
        Extract business insights from comments DataFrame.
        
        Args:
            df: Comments DataFrame with sentiment analysis
            text_column: Column containing cleaned text
            
        Returns:
            Dictionary with results for each category
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")
        
        results = {}
        
        for category, patterns in self.categories.items():
            matches = self._find_category_matches(df, patterns, text_column)
            
            results[category] = {
                'count': len(matches),
                'comments': matches,
                'total_likes': sum(c.get('like_count', 0) for c in matches),
                'negative_count': len([c for c in matches if c.get('sentiment_label') == 'Negative'])
            }
        
        return results
    
    def _find_category_matches(self, df: pd.DataFrame, patterns: List[str], text_column: str) -> List[Dict]:
        """Find all comments matching any pattern in the category"""
        matches = []
        seen_indices = set()
        
        for idx, row in df.iterrows():
            if idx in seen_indices:
                continue
                
            text = str(row[text_column]).lower()
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches.append({
                        'index': idx,
                        'text': row.get('text', ''),
                        'sentiment_label': row.get('sentiment_label', ''),
                        'like_count': row.get('like_count', 0),
                        'matched_pattern': pattern
                    })
                    seen_indices.add(idx)
                    break
        
        return matches
    
    def get_chart_data(self, results: Dict) -> pd.DataFrame:
        """
        Extract data for bar chart visualization.
        
        Args:
            results: Results from extract_insights()
            
        Returns:
            DataFrame with columns: issue, count, total_likes, weighted_score
        """
        chart_data = []
        
        for category, data in results.items():
            if data['count'] > 0:  # Only include categories with matches
                issue_name = category.replace('_', ' ').title()
                count = data['count']
                total_likes = data['total_likes']
                
                # If either is zero, use sum instead of multiplication
                if count == 0 or total_likes == 0:
                    weighted_score = count + total_likes
                else:
                    weighted_score = count * total_likes
                
                chart_data.append({
                    'issue': issue_name,
                    'count': count,
                    'total_likes': total_likes,
                    'weighted_score': weighted_score
                })
        
        # Sort by weighted score descending
        df = pd.DataFrame(chart_data)
        return df.sort_values('weighted_score', ascending=False) if not df.empty else df