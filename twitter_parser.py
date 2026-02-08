import re
import urllib.parse
from typing import Optional, Dict, Any
import logging

class TwitterURLParser:
    """
    Parses Twitter URLs and extracts tweet IDs from various formats
    Supports twitter.com, x.com, mobile links, and shortened URLs
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Twitter URL patterns
        self.url_patterns = [
            # Standard twitter.com URLs
            r'(?:https?://)?(?:www\.)?twitter\.com/[^/]+/status/(\d+)',
            # X.com URLs 
            r'(?:https?://)?(?:www\.)?x\.com/[^/]+/status/(\d+)',
            # Mobile URLs
            r'(?:https?://)?(?:www\.)?mobile\.twitter\.com/[^/]+/status/(\d+)',
            # Direct status URLs
            r'(?:https?://)?(?:www\.)?twitter\.com/i/status/(\d+)',
            r'(?:https?://)?(?:www\.)?x\.com/i/status/(\d+)',
        ]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.url_patterns]
    
    def extract_tweet_id(self, url: str) -> Optional[str]:
        """
        Extract tweet ID from Twitter URL
        
        Args:
            url (str): Twitter URL to parse
            
        Returns:
            Optional[str]: Tweet ID if found, None otherwise
        """
        if not url:
            return None
            
        # Clean and normalize URL
        url = url.strip()
        
        # Try each pattern
        for pattern in self.compiled_patterns:
            match = pattern.search(url)
            if match:
                tweet_id = match.group(1)
                if self.validate_tweet_id(tweet_id):
                    self.logger.info(f"Successfully extracted tweet ID: {tweet_id}")
                    return tweet_id
        
        self.logger.warning(f"Could not extract tweet ID from URL: {url}")
        return None
    
    def validate_tweet_id(self, tweet_id: str) -> bool:
        """
        Validate tweet ID format
        
        Args:
            tweet_id (str): Tweet ID to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not tweet_id:
            return False
            
        # Tweet IDs should be numeric and within reasonable length
        if not tweet_id.isdigit():
            return False
            
        # Twitter snowflake IDs are typically 15-19 digits
        if len(tweet_id) < 10 or len(tweet_id) > 20:
            return False
            
        return True
    
    def parse_url_components(self, url: str) -> Dict[str, Any]:
        """
        Parse URL and extract all components
        
        Args:
            url (str): URL to parse
            
        Returns:
            Dict[str, Any]: Dictionary with URL components
        """
        try:
            parsed = urllib.parse.urlparse(url)
            tweet_id = self.extract_tweet_id(url)
            
            return {
                'original_url': url,
                'domain': parsed.netloc,
                'path': parsed.path,
                'query': parsed.query,
                'tweet_id': tweet_id,
                'is_valid': tweet_id is not None,
                'platform': self._detect_platform(parsed.netloc)
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing URL {url}: {str(e)}")
            return {
                'original_url': url,
                'error': str(e),
                'is_valid': False
            }
    
    def _detect_platform(self, domain: str) -> str:
        """
        Detect which Twitter platform is being used
        
        Args:
            domain (str): Domain name
            
        Returns:
            str: Platform name
        """
        domain = domain.lower()
        
        if 'x.com' in domain:
            return 'x.com'
        elif 'twitter.com' in domain:
            if 'mobile' in domain:
                return 'mobile.twitter.com'
            else:
                return 'twitter.com'
        else:
            return 'unknown'
    
    def batch_extract(self, urls: list) -> list:
        """
        Extract tweet IDs from multiple URLs
        
        Args:
            urls (list): List of URLs to process
            
        Returns:
            list: List of dictionaries with extraction results
        """
        results = []
        
        for url in urls:
            result = self.parse_url_components(url)
            results.append(result)
            
        return results


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize parser
    parser = TwitterURLParser()
    
    # Test URLs
    test_urls = [
        "https://twitter.com/elonmusk/status/1234567890123456789",
        "https://x.com/user/status/9876543210987654321", 
        "https://mobile.twitter.com/someone/status/1111111111111111111",
        "https://twitter.com/i/status/2222222222222222222",
        "invalid-url",
        ""
    ]
    
    # Test individual extraction
    print("=== Individual Tweet ID Extraction ===")
    for url in test_urls:
        tweet_id = parser.extract_tweet_id(url)
        print(f"URL: {url}")
        print(f"Tweet ID: {tweet_id}\n")
    
    # Test batch processing
    print("=== Batch Processing ===")
    batch_results = parser.batch_extract(test_urls)
    for result in batch_results:
        print(f"URL: {result['original_url']}")
        print(f"Valid: {result['is_valid']}")
        print(f"Tweet ID: {result.get('tweet_id', 'None')}")
        print(f"Platform: {result.get('platform', 'Unknown')}")
        print("---")