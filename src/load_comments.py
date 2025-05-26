import os
import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class YouTubeCommentLoader:
    """
    Class for retrieving YouTube comments from videos using the YouTube Data API.
    Handles pagination and rate limiting to fetch all comments from a video.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the YouTube API client.
        
        Args:
            api_key: Your YouTube API key. If None, looks for YOUTUBE_API_KEY env variable.
        """
        self.api_key = api_key or os.environ.get('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key is required. Either pass as parameter or set YOUTUBE_API_KEY environment variable.")
        
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
    
    def extract_video_id(self, url):
        """
        Extract video ID from various YouTube URL formats.
        
        Args:
            url: YouTube URL in any standard format
            
        Returns:
            Video ID as string
        """
        if "youtube.com/watch" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif "youtube.com/embed/" in url:
            return url.split("embed/")[1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL format")
    
    def get_video_comments(self, video_id, max_results=None, batch_size=100, 
                          verbose=True, sleep_time=0.1):
        """
        Fetch comments from a YouTube video with batching and progress reporting.
        
        Args:
            video_id: The YouTube video ID
            max_results: Maximum number of comments to fetch (None for all)
            batch_size: Number of comments to fetch per API call (max 100)
            verbose: Whether to print progress updates
            sleep_time: Time to sleep between API calls to avoid rate limits
            
        Returns:
            DataFrame with comments data
        """
        if batch_size > 100:
            batch_size = 100  # API limit
            
        comments_data = []
        next_page_token = None
        total_comments = 0
        
        if verbose:
            print(f"Fetching comments for video: {video_id}")
        
        try:
            while True:
                # Check if we've reached the max_results
                if max_results is not None and total_comments >= max_results:
                    break
                
                # Adjust batch size if needed
                current_batch = batch_size
                if max_results is not None:
                    current_batch = min(batch_size, max_results - total_comments)
                
                # Prepare API request
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=current_batch,
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                
                # Execute request
                response = request.execute()
                
                # Process comment threads
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    
                    comment_data = {
                        'comment_id': item['id'],
                        'author': comment['authorDisplayName'],
                        'author_channel_id': comment.get('authorChannelId', {}).get('value', ''),
                        'text': comment['textDisplay'],
                        'like_count': comment['likeCount'],
                        'published_at': comment['publishedAt'],
                        'updated_at': comment['updatedAt']
                    }
                    
                    comments_data.append(comment_data)
                
                # Update counts
                batch_comments = len(response['items'])
                total_comments += batch_comments
                
                if verbose:
                    print(f"Fetched batch of {batch_comments} comments. Total: {total_comments}")
                
                # Check if there are more comments
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    if verbose:
                        print("No more comments available.")
                    break
                
                # Sleep to avoid rate limiting
                time.sleep(sleep_time)
        
        except HttpError as e:
            error_details = e._get_reason()
            if "quotaExceeded" in str(error_details):
                print("YouTube API quota exceeded! Try again tomorrow or use a different API key.")
            else:
                print(f"An HTTP error occurred: {e}")
        
        df = pd.DataFrame(comments_data) if comments_data else pd.DataFrame()
        
        if verbose:
            print(f"Retrieved {len(df)} comments in total for video {video_id}")
            
        return df
    
    def get_all_comments(self, video_url, save_to_file=None, chunk_size=1000, 
                         verbose=True, sleep_between_chunks=1):
        """
        Convenience method to get ALL comments from a video URL with progress saving.
        Handles extremely large comment sections by saving in chunks.
        
        Args:
            video_url: YouTube video URL
            save_to_file: Base filename to save chunks (None to disable saving)
            chunk_size: Save to disk after this many comments
            verbose: Whether to print progress updates
            sleep_between_chunks: Seconds to sleep between chunks to avoid rate limits
            
        Returns:
            Complete DataFrame with all comments
        """
        video_id = self.extract_video_id(video_url)
        
        all_comments = []
        chunk_num = 1
        next_page_token = None
        
        if verbose:
            print(f"Retrieving ALL comments for video: {video_url}")
            print(f"Video ID: {video_id}")
            if save_to_file:
                print(f"Will save in chunks of {chunk_size} comments to {save_to_file}_chunk*.csv")
        
        try:
            while True:
                # Prepare API request
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,  # Max allowed by API
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                
                # Execute request
                response = request.execute()
                
                # Process comment threads
                batch_comments = []
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    
                    comment_data = {
                        'comment_id': item['id'],
                        'author': comment['authorDisplayName'],
                        'author_channel_id': comment.get('authorChannelId', {}).get('value', ''),
                        'text': comment['textDisplay'],
                        'like_count': comment['likeCount'],
                        'published_at': comment['publishedAt'],
                        'updated_at': comment['updatedAt']
                    }
                    
                    batch_comments.append(comment_data)
                
                # Add to our collection
                all_comments.extend(batch_comments)
                
                if verbose:
                    print(f"Fetched batch of {len(batch_comments)} comments. Total: {len(all_comments)}")
                
                # Save chunk if needed
                if save_to_file and len(all_comments) >= chunk_size * chunk_num:
                    chunk_df = pd.DataFrame(all_comments)
                    chunk_filename = f"{save_to_file}_chunk{chunk_num}.csv"
                    chunk_df.to_csv(chunk_filename, index=False)
                    if verbose:
                        print(f"Saved chunk {chunk_num} with {len(chunk_df)} comments to {chunk_filename}")
                    chunk_num += 1
                
                # Check if there are more comments
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    if verbose:
                        print("No more comments available.")
                    break
                
                # Sleep to avoid rate limiting
                time.sleep(sleep_between_chunks)
        
        except HttpError as e:
            error_details = e._get_reason()
            if "quotaExceeded" in str(error_details):
                print("YouTube API quota exceeded! Try again tomorrow or use a different API key.")
            else:
                print(f"An HTTP error occurred: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_comments)
        
        # Save final result if requested
        if save_to_file and not df.empty:
            final_filename = f"{save_to_file}_all_comments.csv"
            df.to_csv(final_filename, index=False)
            if verbose:
                print(f"Saved all {len(df)} comments to {final_filename}")
        
        if verbose:
            print(f"Retrieved {len(df)} comments in total for video {video_id}")
            
        return df
    
    def save_to_csv(self, df, filename="youtube_comments.csv"):
        """
        Save the comments DataFrame to a CSV file.
        
        Args:
            df: DataFrame containing comments
            filename: Output filename
        """
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} comments to {filename}")