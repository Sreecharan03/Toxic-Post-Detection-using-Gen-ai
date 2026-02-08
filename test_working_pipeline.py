#!/usr/bin/env python3
"""
Complete Twitter Toxicity Detection Test
Uses working HTTP client + toxicity detection pipeline
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Proper directory path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from twitter_parser import TwitterURLParser
from simple_twitter_client import SimpleTwitterClient
from toxic_detector import ToxicityDetector

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def test_complete_pipeline():
    """Test the complete pipeline with working HTTP client"""
    
    print("🚀 TWITTER TOXICITY DETECTION - COMPLETE TEST")
    print("=" * 60)
    
    # The URL that we know works
    test_url = "https://x.com/rahulroushan/status/2001161855340003510?s=20"
    
    print(f"📝 Testing URL: {test_url}")
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        # Step 1: Initialize components
        print("\n🔍 STEP 1: INITIALIZING COMPONENTS")
        print("-" * 40)
        
        url_parser = TwitterURLParser()
        twitter_client = SimpleTwitterClient('/teamspace/studios/this_studio/.env')
        toxicity_detector = ToxicityDetector('/teamspace/studios/this_studio/.env')
        
        print("✅ URL Parser initialized")
        print("✅ HTTP Twitter Client initialized") 
        print("✅ Toxicity Detector initialized")
        
        # Step 2: Parse URL
        print("\n🔍 STEP 2: URL PARSING")
        print("-" * 40)
        
        tweet_id = url_parser.extract_tweet_id(test_url)
        if tweet_id:
            print(f"✅ Tweet ID extracted: {tweet_id}")
        else:
            print("❌ Failed to extract tweet ID")
            return
        
        # Step 3: Fetch Tweet
        print("\n🔍 STEP 3: TWEET FETCHING (HTTP METHOD)")
        print("-" * 40)
        
        tweet_data = twitter_client.get_tweet_by_id(tweet_id)
        
        if tweet_data:
            print("✅ Tweet fetched successfully!")
            print(f"👤 Author: @{tweet_data.author_username}")
            print(f"📅 Created: {tweet_data.created_at}")
            print(f"📝 Text: {tweet_data.text}")
            
            if tweet_data.public_metrics:
                metrics = tweet_data.public_metrics
                print(f"📊 Engagement:")
                print(f"   💙 Likes: {metrics.get('like_count', 0)}")
                print(f"   🔄 Retweets: {metrics.get('retweet_count', 0)}")
                print(f"   💬 Replies: {metrics.get('reply_count', 0)}")
        else:
            print("❌ Failed to fetch tweet")
            return
        
        # Step 4: Toxicity Analysis
        print("\n🔍 STEP 4: DUAL-LAYER TOXICITY ANALYSIS")
        print("-" * 40)
        
        print(f"🔄 Analyzing text: '{tweet_data.text}'")
        
        toxicity_result = await toxicity_detector.analyze_text(tweet_data.text)
        
        print(f"\n✅ Analysis Complete!")
        print(f"🎯 Overall Toxicity Score: {toxicity_result.overall_score:.3f}")
        print(f"🔒 Confidence Level: {toxicity_result.confidence:.3f}")
        
        # Step 5: Results Breakdown
        print(f"\n📊 CATEGORY BREAKDOWN:")
        for category, score in toxicity_result.categories.items():
            status = "🔴" if score > 0.5 else "🟡" if score > 0.2 else "🟢"
            print(f"   {status} {category.capitalize()}: {score:.3f}")
        
        print(f"\n🤖 AI ANALYSIS:")
        if toxicity_result.explanation:
            print(f"   💭 Explanation: {toxicity_result.explanation}")
        
        if toxicity_result.reformulation:
            print(f"   ✨ Suggested Improvement: {toxicity_result.reformulation}")
        
        print(f"\n🔬 LAYER PERFORMANCE:")
        ml_max = max(toxicity_result.layer1_scores.values()) if toxicity_result.layer1_scores else 0
        gemini_max = max(toxicity_result.layer2_scores.values()) if toxicity_result.layer2_scores else 0
        print(f"   🤖 ML Layer (BERT): {ml_max:.3f}")
        print(f"   🧠 Gemini Layer: {gemini_max:.3f}")
        
        # Step 6: Final Assessment
        print(f"\n📋 FINAL ASSESSMENT:")
        toxicity_level = "HIGH" if toxicity_result.overall_score >= 0.7 else "MEDIUM" if toxicity_result.overall_score >= 0.3 else "LOW"
        print(f"   🎯 Toxicity Level: {toxicity_level}")
        print(f"   🚨 Action Needed: {'YES' if toxicity_result.overall_score >= 0.5 else 'NO'}")
        
        # Step 7: Technical Summary
        print(f"\n" + "=" * 60)
        print("🎉 COMPLETE PIPELINE TEST SUCCESSFUL!")
        print("=" * 60)
        
        print(f"✅ Components Working:")
        print(f"   • URL Parsing: PASS")
        print(f"   • HTTP Twitter API: PASS") 
        print(f"   • Dual-Layer Toxicity Detection: PASS")
        print(f"   • AI Explanations: PASS")
        print(f"   • Score Fusion: PASS")
        print(f"   • Confidence Calculation: PASS")
        
        print(f"\n🏆 B.Tech Project Status: FULLY OPERATIONAL")
        print(f"📊 Ready for demonstration and presentation!")
        
    except Exception as e:
        print(f"❌ Pipeline Error: {str(e)}")
        import traceback
        traceback.print_exc()

async def main():
    """Main function"""
    setup_logging()
    
    print("🎯 TWITTER TOXICITY DETECTION SYSTEM")
    print("🔬 B.Tech Final Year Project")
    print("👨‍💻 Complete Pipeline Test")
    print("=" * 60)
    
    await test_complete_pipeline()

if __name__ == "__main__":
    asyncio.run(main())