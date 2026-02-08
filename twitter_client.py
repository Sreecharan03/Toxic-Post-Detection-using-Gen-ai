import tweepy
import os
import time
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class TweetData:
    """Data class for structured tweet information"""
    id: str
    text: str
    author_id: str
    author_username: str
    created_at: datetime
    public_metrics: Dict[str, int]
    context_annotations: List[Dict] = None
    referenced_tweets: List[Dict] = None
    reply_count: int = 0

class TwitterAPIClient:
    """
    Twitter API v2 client for fetching tweet data
    Handles authentication, rate limiting, and data extraction
    """
    
    def __init__(self, env_file_path: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        self._load_credentials(env_file_path)
        
        # Initialize Twitter API client
        self.client = None
        self._authenticate()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
    def _load_credentials(self, env_file_path: str = None):
        """Load Twitter API credentials from environment"""
        if env_file_path:
            from dotenv import load_dotenv
            load_dotenv(env_file_path)
        
        # OAuth 2.0 Credentials (Primary)
        self.client_id = os.getenv('TWITTER_CLIENT_ID')
        self.client_secret = os.getenv('TWITTER_CLIENT_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        # OAuth 1.0a Credentials (Fallback)
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        # Validate credentials
        if not self.bearer_token:
            self.logger.warning("Bearer Token missing - Limited API access")
    
    def _authenticate(self):
        """Authenticate with Twitter API using OAuth 2.0"""
        try:
            # Primary: OAuth 2.0 with Bearer Token (recommended for API v2)
            if self.bearer_token:
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    wait_on_rate_limit=True
                )
                self.logger.info("✅ Authenticated with OAuth 2.0 Bearer Token")
                return
            
            # Secondary: OAuth 2.0 with Client Credentials
            elif self.client_id and self.client_secret:
                self.client = tweepy.Client(
                    consumer_key=self.client_id,
                    consumer_secret=self.client_secret,
                    wait_on_rate_limit=True
                )
                self.logger.info("✅ Authenticated with OAuth 2.0 Client Credentials")
                return
            
            # Fallback: OAuth 1.0a (for user context endpoints)
            elif all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
                self.client = tweepy.Client(
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_token_secret,
                    wait_on_rate_limit=True
                )
                self.logger.info("✅ Authenticated with OAuth 1.0a")
                return
            
            else:
                self.logger.error("❌ No valid Twitter API credentials found")
                self.logger.error("Need Bearer Token OR Client ID/Secret OR OAuth 1.0a credentials")
                raise ValueError("Twitter API authentication failed")
                
        except Exception as e:
            self.logger.error(f"Twitter API authentication error: {str(e)}")
            raise
    
    def _rate_limit_wait(self):
        """Implement basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_tweet_by_id(self, tweet_id: str) -> Optional[TweetData]:
        """
        Fetch a single tweet by ID using direct HTTP requests
        
        Args:
            tweet_id (str): Tweet ID to fetch
            
        Returns:
            Optional[TweetData]: Tweet data if successful, None otherwise
        """
        import requests
        import time
        
        try:
            self._rate_limit_wait()
            
            # Direct API call like the working GPT code
            url = f"https://api.x.com/2/tweets/{tweet_id}"
            params = {
                "tweet.fields": "created_at,public_metrics,lang,conversation_id,entities,possibly_sensitive",
                "expansions": "author_id,attachments.media_keys",
                "user.fields": "name,username,profile_image_url,verified",
                "media.fields": "type,url,preview_image_url,alt_text",
            }
            
            # Clean bearer token (remove 'Bearer ' prefix if present)
            bearer_token = self.bearer_token
            if bearer_token.lower().startswith("bearer "):
                bearer_token = bearer_token.split(" ", 1)[1].strip()
            
            headers = {"Authorization": f"Bearer {bearer_token}"}
            
            self.logger.info(f"Fetching tweet {tweet_id} via HTTP API")
            
            # Make request with timeout
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            # Handle HTTP errors
            if response.status_code == 401:
                self.logger.error("401 Unauthorized: Bearer token invalid or wrong permissions")
                return None
            elif response.status_code == 403:
                self.logger.error("403 Forbidden: Tweet not accessible or app suspended")
                return None
            elif response.status_code == 429:
                self.logger.error("429 Rate Limited: Too many requests")
                return None
            elif response.status_code == 404:
                self.logger.warning(f"Tweet {tweet_id} not found or deleted")
                return None
            elif response.status_code >= 400:
                self.logger.error(f"HTTP {response.status_code} error: {response.text[:200]}")
                return None
            
            # Parse JSON response
            data = response.json()
            tweet = data.get("data", {})
            includes = data.get("includes", {})
            
            if not tweet:
                self.logger.warning(f"No tweet data returned for {tweet_id}")
                return None
            
            # Extract author information
            users = {u["id"]: u for u in includes.get("users", [])}
            author = users.get(tweet.get("author_id"), {})
            author_username = author.get("username", "unknown")
            
            # Create datetime object
            created_at_str = tweet.get("created_at")
            if created_at_str:
                from datetime import datetime
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            else:
                created_at = datetime.now()
            
            # Create TweetData object
            tweet_data = TweetData(
                id=tweet.get("id"),
                text=tweet.get("text", ""),
                author_id=tweet.get("author_id", ""),
                author_username=author_username,
                created_at=created_at,
                public_metrics=tweet.get("public_metrics", {}),
                context_annotations=tweet.get("entities", {}).get("annotations", []),
                referenced_tweets=[],
                reply_count=tweet.get("public_metrics", {}).get("reply_count", 0)
            )
            
            self.logger.info(f"✅ Successfully fetched tweet {tweet_id}")
            self.logger.info(f"📝 Text: {tweet_data.text[:100]}...")
            self.logger.info(f"👤 Author: @{tweet_data.author_username}")
            
            return tweet_data
            
        except requests.RequestException as e:
            self.logger.error(f"Network error fetching tweet {tweet_id}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching tweet {tweet_id}: {str(e)}")
            return None
    
    def get_tweet_replies(self, tweet_id: str, max_results: int = 10) -> List[TweetData]:
        """
        Fetch replies to a tweet
        
        Args:
            tweet_id (str): Original tweet ID
            max_results (int): Maximum number of replies to fetch
            
        Returns:
            List[TweetData]: List of reply tweet data
        """
        try:
            self._rate_limit_wait()
            
            # Search for replies using conversation_id
            query = f"conversation_id:{tweet_id} is:reply"
            
            tweet_fields = ['created_at', 'author_id', 'public_metrics', 'referenced_tweets']
            user_fields = ['username']
            expansions = ['author_id']
            
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=tweet_fields,
                user_fields=user_fields,
                expansions=expansions
            )
            
            replies = []
            
            if response.data:
                # Create user mapping for usernames
                user_map = {}
                if response.includes and 'users' in response.includes:
                    for user in response.includes['users']:
                        user_map[user.id] = user.username
                
                for tweet in response.data:
                    reply_data = TweetData(
                        id=tweet.id,
                        text=tweet.text,
                        author_id=tweet.author_id,
                        author_username=user_map.get(tweet.author_id, "unknown"),
                        created_at=tweet.created_at,
                        public_metrics=tweet.public_metrics or {},
                        referenced_tweets=tweet.referenced_tweets or []
                    )
                    replies.append(reply_data)
                
                self.logger.info(f"Fetched {len(replies)} replies for tweet {tweet_id}")
            
            return replies
            
        except Exception as e:
            self.logger.error(f"Error fetching replies for tweet {tweet_id}: {str(e)}")
            return []
    
    def get_tweet_thread(self, tweet_id: str) -> List[TweetData]:
        """
        Fetch entire tweet thread including original tweet and replies
        
        Args:
            tweet_id (str): Tweet ID to get thread for
            
        Returns:
            List[TweetData]: List containing original tweet and replies
        """
        thread = []
        
        # Get original tweet
        original_tweet = self.get_tweet_by_id(tweet_id)
        if original_tweet:
            thread.append(original_tweet)
            
            # Get replies
            replies = self.get_tweet_replies(tweet_id, max_results=50)
            thread.extend(replies)
        
        return thread
    
    def batch_get_tweets(self, tweet_ids: List[str]) -> Dict[str, TweetData]:
        """
        Fetch multiple tweets efficiently
        
        Args:
            tweet_ids (List[str]): List of tweet IDs to fetch
            
        Returns:
            Dict[str, TweetData]: Dictionary mapping tweet ID to TweetData
        """
        results = {}
        
        # Process in batches of 100 (Twitter API limit)
        batch_size = 100
        
        for i in range(0, len(tweet_ids), batch_size):
            batch = tweet_ids[i:i + batch_size]
            
            try:
                self._rate_limit_wait()
                
                response = self.client.get_tweets(
                    ids=batch,
                    tweet_fields=['created_at', 'author_id', 'public_metrics'],
                    user_fields=['username'],
                    expansions=['author_id']
                )
                
                if response.data:
                    # Create user mapping
                    user_map = {}
                    if response.includes and 'users' in response.includes:
                        for user in response.includes['users']:
                            user_map[user.id] = user.username
                    
                    # Process tweets
                    for tweet in response.data:
                        tweet_data = TweetData(
                            id=tweet.id,
                            text=tweet.text,
                            author_id=tweet.author_id,
                            author_username=user_map.get(tweet.author_id, "unknown"),
                            created_at=tweet.created_at,
                            public_metrics=tweet.public_metrics or {}
                        )
                        results[tweet.id] = tweet_data
                
                self.logger.info(f"Batch fetched {len(batch)} tweets")
                
            except Exception as e:
                self.logger.error(f"Error in batch fetch: {str(e)}")
                continue
        
        return results
    
    def export_tweet_data(self, tweet_data: TweetData, filepath: str):
        """
        Export tweet data to JSON file
        
        Args:
            tweet_data (TweetData): Tweet data to export
            filepath (str): File path for export
        """
        try:
            export_dict = {
                'id': tweet_data.id,
                'text': tweet_data.text,
                'author_id': tweet_data.author_id,
                'author_username': tweet_data.author_username,
                'created_at': tweet_data.created_at.isoformat(),
                'public_metrics': tweet_data.public_metrics,
                'context_annotations': tweet_data.context_annotations,
                'referenced_tweets': tweet_data.referenced_tweets,
                'reply_count': tweet_data.reply_count
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_dict, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Tweet data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting tweet data: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize client (will need proper credentials)
    try:
        client = TwitterAPIClient()
        print("Twitter API Client initialized successfully")
        
        # Example: Fetch a tweet (this will fail without proper credentials)
        # tweet_data = client.get_tweet_by_id("1234567890123456789")
        # if tweet_data:
        #     print(f"Tweet: {tweet_data.text}")
        #     print(f"Author: @{tweet_data.author_username}")
        
    except Exception as e:
        print(f"Client initialization failed: {str(e)}")
        print("This is expected without proper Twitter API credentials")