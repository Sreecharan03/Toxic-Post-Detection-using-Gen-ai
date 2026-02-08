import os
import time
import logging
import requests
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import json
from dotenv import load_dotenv

@dataclass
class TweetData:
    """Data class for structured tweet information"""
    id: str
    text: str
    author_id: str
    author_username: str
    created_at: datetime
    public_metrics: dict
    context_annotations: list = None
    referenced_tweets: list = None
    reply_count: int = 0

class SimpleTwitterClient:
    """
    Simplified Twitter client using direct HTTP requests
    Based on the working GPT code approach
    """
    
    def __init__(self, env_file_path: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        if env_file_path:
            load_dotenv(env_file_path)
        
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        if not self.bearer_token:
            raise ValueError("TWITTER_BEARER_TOKEN missing in environment")
        
        # Clean bearer token (remove 'Bearer ' prefix if present)
        if self.bearer_token.lower().startswith("bearer "):
            self.bearer_token = self.bearer_token.split(" ", 1)[1].strip()
            self.logger.warning("Bearer token had 'Bearer ' prefix; normalized it")
        
        self.logger.info("✅ HTTP Twitter Client initialized with Bearer Token")
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0
    
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
        try:
            self._rate_limit_wait()
            
            # Use the same API endpoint and parameters as working GPT code
            url = f"https://api.x.com/2/tweets/{tweet_id}"
            params = {
                "tweet.fields": "created_at,public_metrics,lang,conversation_id,entities,possibly_sensitive",
                "expansions": "author_id,attachments.media_keys",
                "user.fields": "name,username,profile_image_url,verified",
                "media.fields": "type,url,preview_image_url,alt_text",
            }
            
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            
            self.logger.info(f"Fetching tweet {tweet_id} via HTTP API")
            
            # Make request with timeout
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            # Handle HTTP errors with detailed logging
            if response.status_code == 401:
                self.logger.error("401 Unauthorized: Bearer token invalid/expired OR wrong app permissions")
                return None
            elif response.status_code == 403:
                self.logger.error("403 Forbidden: Tweet not accessible or app suspended")
                return None
            elif response.status_code == 429:
                reset = response.headers.get("x-rate-limit-reset")
                self.logger.error(f"429 Rate Limited: Too many requests. Reset: {reset}")
                return None
            elif response.status_code == 404:
                self.logger.warning(f"404 Not Found: Tweet {tweet_id} not found or deleted")
                return None
            elif response.status_code >= 400:
                try:
                    err = response.json()
                except:
                    err = response.text[:500]
                self.logger.error(f"HTTP {response.status_code} error: {err}")
                return None
            
            # Success - parse JSON response
            try:
                data = response.json()
            except Exception:
                self.logger.error(f"Response was not JSON. Raw: {response.text[:500]}")
                return None
            
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
            self.logger.info(f"📝 Text: {tweet_data.text}")
            self.logger.info(f"👤 Author: @{tweet_data.author_username}")
            
            # Log engagement metrics
            metrics = tweet_data.public_metrics
            if metrics:
                self.logger.info(f"📊 Engagement: {metrics.get('like_count', 0)} likes, "
                               f"{metrics.get('retweet_count', 0)} retweets, "
                               f"{metrics.get('reply_count', 0)} replies")
            
            return tweet_data
            
        except requests.RequestException as e:
            self.logger.error(f"Network error fetching tweet {tweet_id}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching tweet {tweet_id}: {str(e)}")
            return None

# Test script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        client = SimpleTwitterClient("/teamspace/studios/this_studio/.env")
        
        # Test with the working tweet ID
        tweet_data = client.get_tweet_by_id("2001161855340003510")
        
        if tweet_data:
            print("\n=== TWEET SUCCESSFULLY FETCHED ===")
            print(f"Author: @{tweet_data.author_username}")
            print(f"Text: {tweet_data.text}")
            print(f"Created: {tweet_data.created_at}")
            print(f"Metrics: {tweet_data.public_metrics}")
        else:
            print("Failed to fetch tweet")
            
    except Exception as e:
        print(f"Error: {str(e)}")