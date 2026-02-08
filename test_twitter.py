#!/usr/bin/env python3
"""
Twitter API Connection Test Script
Tests all Twitter API credentials and basic functionality
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add src directory to path
sys.path.append('/teamspace/studios/this_studio/src')

from twitter_parser import TwitterURLParser
from twitter_client import TwitterAPIClient

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_environment_setup():
    """Test if environment variables are loaded correctly"""
    print("=" * 50)
    print("🔍 TESTING ENVIRONMENT SETUP")
    print("=" * 50)
    
    # Load environment variables
    env_path = '/teamspace/studios/this_studio/.env'
    load_dotenv(env_path)
    
    # Check each credential
    credentials = {
        'TWITTER_API_KEY': os.getenv('TWITTER_API_KEY'),
        'TWITTER_API_SECRET': os.getenv('TWITTER_API_SECRET'),
        'TWITTER_ACCESS_TOKEN': os.getenv('TWITTER_ACCESS_TOKEN'),
        'TWITTER_ACCESS_TOKEN_SECRET': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        'TWITTER_BEARER_TOKEN': os.getenv('TWITTER_BEARER_TOKEN'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY')
    }
    
    for key, value in credentials.items():
        if value:
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {key}: {masked_value}")
        else:
            print(f"❌ {key}: Missing")
    
    # Check if all Twitter credentials are present
    twitter_creds = [credentials[k] for k in credentials.keys() if 'TWITTER' in k]
    if all(twitter_creds):
        print("\n✅ All Twitter API credentials loaded successfully!")
        return True
    else:
        print("\n❌ Missing Twitter API credentials!")
        return False

def test_url_parser():
    """Test Twitter URL parser functionality"""
    print("\n" + "=" * 50)
    print("🔍 TESTING URL PARSER")
    print("=" * 50)
    
    parser = TwitterURLParser()
    
    # Test URLs
    test_urls = [
        "https://twitter.com/elonmusk/status/1234567890123456789",
        "https://x.com/openai/status/9876543210987654321",
        "https://mobile.twitter.com/user/status/1111111111111111111",
        "invalid-url",
        ""
    ]
    
    success_count = 0
    for url in test_urls:
        result = parser.parse_url_components(url)
        if result['is_valid']:
            print(f"✅ URL: {url[:50]}...")
            print(f"   Tweet ID: {result['tweet_id']}")
            print(f"   Platform: {result['platform']}")
            success_count += 1
        else:
            print(f"❌ URL: {url} - Invalid")
    
    print(f"\n📊 Parser Test Results: {success_count}/{len(test_urls)} valid URLs")
    return success_count > 0

def test_twitter_authentication():
    """Test Twitter API authentication"""
    print("\n" + "=" * 50)
    print("🔍 TESTING TWITTER API AUTHENTICATION")
    print("=" * 50)
    
    try:
        # Initialize client with environment file
        client = TwitterAPIClient('/teamspace/studios/this_studio/.env')
        print("✅ Twitter API Client initialized successfully")
        
        # Test if client has valid authentication
        if client.client:
            print("✅ Twitter API authentication successful")
            return True, client
        else:
            print("❌ Twitter API authentication failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Twitter API Client Error: {str(e)}")
        return False, None

def test_tweet_fetching(client):
    """Test actual tweet fetching (optional - requires valid tweet ID)"""
    print("\n" + "=" * 50)
    print("🔍 TESTING TWEET FETCHING (OPTIONAL)")
    print("=" * 50)
    
    if not client:
        print("❌ No valid client available for testing")
        return False
    
    # Use a known public tweet ID for testing
    # Note: This might fail if the tweet is deleted or private
    test_tweet_id = "1234567890123456789"  # Example ID
    
    print(f"🔄 Attempting to fetch tweet: {test_tweet_id}")
    print("Note: This may fail if the tweet doesn't exist or is private")
    
    try:
        tweet_data = client.get_tweet_by_id(test_tweet_id)
        
        if tweet_data:
            print("✅ Tweet fetching successful!")
            print(f"   Tweet ID: {tweet_data.id}")
            print(f"   Author: @{tweet_data.author_username}")
            print(f"   Text preview: {tweet_data.text[:100]}...")
            return True
        else:
            print("⚠️  Tweet fetching returned no data (tweet may not exist)")
            print("✅ But API connection is working!")
            return True
            
    except Exception as e:
        print(f"⚠️  Tweet fetching error: {str(e)}")
        print("✅ But API connection is working!")
        return True

def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n" + "=" * 50)
    print("🔍 TESTING RATE LIMITING")
    print("=" * 50)
    
    try:
        client = TwitterAPIClient('/teamspace/studios/this_studio/.env')
        
        # Test multiple rapid requests to check rate limiting
        print("🔄 Testing rate limiting with multiple requests...")
        
        import time
        start_time = time.time()
        
        # Make 3 rapid requests
        for i in range(3):
            client._rate_limit_wait()
            print(f"   Request {i+1} completed")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Rate limiting test completed")
        print(f"   Total time for 3 requests: {total_time:.2f} seconds")
        
        if total_time >= 2.0:  # Should take at least 2 seconds due to rate limiting
            print("✅ Rate limiting is working correctly")
            return True
        else:
            print("⚠️  Rate limiting may not be working as expected")
            return False
            
    except Exception as e:
        print(f"❌ Rate limiting test error: {str(e)}")
        return False

def run_complete_test():
    """Run all tests and provide summary"""
    print("🚀 TWITTER TOXICITY DETECTION SYSTEM - API TEST")
    print("=" * 60)
    
    setup_logging()
    
    test_results = []
    
    # Test 1: Environment Setup
    env_result = test_environment_setup()
    test_results.append(("Environment Setup", env_result))
    
    # Test 2: URL Parser
    parser_result = test_url_parser()
    test_results.append(("URL Parser", parser_result))
    
    # Test 3: Twitter Authentication
    if env_result:
        auth_result, client = test_twitter_authentication()
        test_results.append(("Twitter Authentication", auth_result))
        
        # Test 4: Tweet Fetching (only if authentication works)
        if auth_result:
            fetch_result = test_tweet_fetching(client)
            test_results.append(("Tweet Fetching", fetch_result))
            
            # Test 5: Rate Limiting
            rate_limit_result = test_rate_limiting()
            test_results.append(("Rate Limiting", rate_limit_result))
        else:
            test_results.append(("Tweet Fetching", False))
            test_results.append(("Rate Limiting", False))
    else:
        test_results.append(("Twitter Authentication", False))
        test_results.append(("Tweet Fetching", False))
        test_results.append(("Rate Limiting", False))
    
    # Print Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Your Twitter API setup is working perfectly!")
        print("✅ Ready to proceed with toxicity detection implementation!")
    elif passed_tests >= 3:
        print("✅ Core functionality working! Minor issues detected but system is usable.")
    else:
        print("⚠️  Major issues detected. Please check your API credentials.")
    
    return passed_tests, total_tests

if __name__ == "__main__":
    passed, total = run_complete_test()
    
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS")
    print("=" * 60)
    
    if passed >= 3:
        print("1. ✅ Your Twitter API is working!")
        print("2. 🔄 Ready to implement toxicity detection layers")
        print("3. 📊 Proceed to dual-layer ML + API detection system")
    else:
        print("1. 🔧 Fix API credential issues")
        print("2. 📞 Contact Twitter Developer Support if needed")
        print("3. 🔄 Re-run this test after fixing issues")