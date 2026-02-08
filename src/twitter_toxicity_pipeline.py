import os
import sys
import asyncio
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv

# Add src directory to path
sys.path.append('/teamspace/studios/this_studio/src')

from twitter_parser import TwitterURLParser
from simple_twitter_client import SimpleTwitterClient, TweetData  # Use working HTTP client
from toxicity_detector import ToxicityDetector, ToxicityResult

@dataclass
class TwitterToxicityAnalysis:
    """Complete analysis result for a tweet"""
    tweet_data: TweetData
    toxicity_result: ToxicityResult
    analysis_timestamp: datetime = None
    tweet_url: str = ""
    
    def __post_init__(self):
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now()

@dataclass
class ThreadAnalysis:
    """Analysis results for an entire thread"""
    original_tweet_analysis: TwitterToxicityAnalysis
    replies_analysis: List[TwitterToxicityAnalysis]
    thread_summary: Dict[str, Any]
    total_tweets: int
    toxic_count: int
    avg_toxicity_score: float

class TwitterToxicityPipeline:
    """
    Complete pipeline for analyzing Twitter content toxicity
    Combines URL parsing, tweet fetching, and toxicity detection
    """
    
    def __init__(self, env_file_path: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components with working HTTP client
        self.url_parser = TwitterURLParser()
        self.twitter_client = SimpleTwitterClient(env_file_path)  # Use working HTTP client
        self.toxicity_detector = ToxicityDetector(env_file_path)
        
        # Analysis settings
        self.toxicity_threshold = 0.5  # Threshold for flagging toxic content
        self.max_replies_to_analyze = 50  # Limit for thread analysis
        
        self.logger.info("🚀 Twitter Toxicity Pipeline initialized")
    
    async def analyze_tweet_from_url(self, tweet_url: str) -> Optional[TwitterToxicityAnalysis]:
        """
        Analyze a single tweet from its URL
        
        Args:
            tweet_url (str): Twitter URL to analyze
            
        Returns:
            Optional[TwitterToxicityAnalysis]: Complete analysis result
        """
        self.logger.info(f"Starting analysis for URL: {tweet_url}")
        
        # Step 1: Extract tweet ID from URL
        tweet_id = self.url_parser.extract_tweet_id(tweet_url)
        if not tweet_id:
            self.logger.error(f"Could not extract tweet ID from URL: {tweet_url}")
            return None
        
        # Step 2: Fetch tweet data
        tweet_data = self.twitter_client.get_tweet_by_id(tweet_id)
        if not tweet_data:
            self.logger.error(f"Could not fetch tweet data for ID: {tweet_id}")
            return None
        
        # Step 3: Analyze toxicity
        toxicity_result = await self.toxicity_detector.analyze_text(tweet_data.text)
        
        # Step 4: Create complete analysis
        analysis = TwitterToxicityAnalysis(
            tweet_data=tweet_data,
            toxicity_result=toxicity_result,
            tweet_url=tweet_url
        )
        
        self.logger.info(f"Analysis complete for tweet {tweet_id} - Toxicity: {toxicity_result.overall_score:.2f}")
        return analysis
    
    async def analyze_thread_from_url(self, tweet_url: str) -> Optional[ThreadAnalysis]:
        """
        Analyze an entire thread starting from a tweet URL
        
        Args:
            tweet_url (str): URL of the original tweet in the thread
            
        Returns:
            Optional[ThreadAnalysis]: Complete thread analysis
        """
        self.logger.info(f"Starting thread analysis for URL: {tweet_url}")
        
        # Step 1: Extract tweet ID
        tweet_id = self.url_parser.extract_tweet_id(tweet_url)
        if not tweet_id:
            self.logger.error(f"Could not extract tweet ID from URL: {tweet_url}")
            return None
        
        # Step 2: Fetch thread data
        thread_tweets = self.twitter_client.get_tweet_thread(tweet_id)
        if not thread_tweets:
            self.logger.error(f"Could not fetch thread for tweet ID: {tweet_id}")
            return None
        
        # Step 3: Analyze original tweet
        original_tweet = thread_tweets[0]
        original_toxicity = await self.toxicity_detector.analyze_text(original_tweet.text)
        original_analysis = TwitterToxicityAnalysis(
            tweet_data=original_tweet,
            toxicity_result=original_toxicity,
            tweet_url=tweet_url
        )
        
        # Step 4: Analyze replies (limit to avoid rate limits)
        replies_analysis = []
        reply_tweets = thread_tweets[1:self.max_replies_to_analyze + 1]
        
        if reply_tweets:
            self.logger.info(f"Analyzing {len(reply_tweets)} replies...")
            
            # Batch analyze replies for efficiency
            reply_texts = [tweet.text for tweet in reply_tweets]
            reply_toxicity_results = self.toxicity_detector.batch_analyze(reply_texts)
            
            for tweet_data, toxicity_result in zip(reply_tweets, reply_toxicity_results):
                reply_url = f"https://twitter.com/{tweet_data.author_username}/status/{tweet_data.id}"
                reply_analysis = TwitterToxicityAnalysis(
                    tweet_data=tweet_data,
                    toxicity_result=toxicity_result,
                    tweet_url=reply_url
                )
                replies_analysis.append(reply_analysis)
        
        # Step 5: Generate thread summary
        all_analyses = [original_analysis] + replies_analysis
        toxic_count = sum(1 for analysis in all_analyses 
                         if analysis.toxicity_result.overall_score >= self.toxicity_threshold)
        
        avg_toxicity = sum(analysis.toxicity_result.overall_score for analysis in all_analyses) / len(all_analyses)
        
        thread_summary = {
            'total_tweets': len(all_analyses),
            'toxic_tweets': toxic_count,
            'toxic_percentage': (toxic_count / len(all_analyses)) * 100,
            'avg_toxicity_score': avg_toxicity,
            'max_toxicity_score': max(analysis.toxicity_result.overall_score for analysis in all_analyses),
            'most_toxic_categories': self._get_most_common_categories(all_analyses),
            'thread_sentiment': 'toxic' if avg_toxicity >= self.toxicity_threshold else 'clean'
        }
        
        thread_analysis = ThreadAnalysis(
            original_tweet_analysis=original_analysis,
            replies_analysis=replies_analysis,
            thread_summary=thread_summary,
            total_tweets=len(all_analyses),
            toxic_count=toxic_count,
            avg_toxicity_score=avg_toxicity
        )
        
        self.logger.info(f"Thread analysis complete - {toxic_count}/{len(all_analyses)} toxic tweets")
        return thread_analysis
    
    def _get_most_common_categories(self, analyses: List[TwitterToxicityAnalysis]) -> Dict[str, float]:
        """Get the most common toxicity categories in a set of analyses"""
        category_sums = {}
        
        for analysis in analyses:
            for category, score in analysis.toxicity_result.categories.items():
                if category not in category_sums:
                    category_sums[category] = 0.0
                category_sums[category] += score
        
        # Average scores across all analyses
        category_averages = {
            category: total_score / len(analyses)
            for category, total_score in category_sums.items()
        }
        
        return category_averages
    
    async def batch_analyze_urls(self, tweet_urls: List[str]) -> List[TwitterToxicityAnalysis]:
        """
        Analyze multiple tweet URLs efficiently
        
        Args:
            tweet_urls (List[str]): List of Twitter URLs to analyze
            
        Returns:
            List[TwitterToxicityAnalysis]: Analysis results for each URL
        """
        self.logger.info(f"Starting batch analysis for {len(tweet_urls)} URLs")
        
        results = []
        
        # Process URLs in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(tweet_urls), batch_size):
            batch = tweet_urls[i:i + batch_size]
            
            # Create analysis tasks for this batch
            tasks = [self.analyze_tweet_from_url(url) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_results = [result for result in batch_results 
                           if isinstance(result, TwitterToxicityAnalysis)]
            results.extend(valid_results)
            
            self.logger.info(f"Batch {i//batch_size + 1} complete: {len(valid_results)} successful")
            
            # Small delay between batches to be respectful to APIs
            if i + batch_size < len(tweet_urls):
                await asyncio.sleep(2)
        
        self.logger.info(f"Batch analysis complete: {len(results)} successful analyses")
        return results
    
    def generate_report(self, analyses: List[TwitterToxicityAnalysis], 
                       output_format: str = 'json') -> Dict[str, Any]:
        """
        Generate a comprehensive report from analyses
        
        Args:
            analyses (List[TwitterToxicityAnalysis]): Analysis results
            output_format (str): Report format ('json', 'summary')
            
        Returns:
            Dict[str, Any]: Generated report
        """
        if not analyses:
            return {'error': 'No analyses provided'}
        
        # Calculate summary statistics
        total_tweets = len(analyses)
        toxic_tweets = sum(1 for analysis in analyses 
                          if analysis.toxicity_result.overall_score >= self.toxicity_threshold)
        
        avg_toxicity = sum(analysis.toxicity_result.overall_score for analysis in analyses) / total_tweets
        max_toxicity = max(analysis.toxicity_result.overall_score for analysis in analyses)
        
        # Get category breakdown
        category_breakdown = self._get_most_common_categories(analyses)
        
        # Find most toxic tweet
        most_toxic_analysis = max(analyses, key=lambda x: x.toxicity_result.overall_score)
        
        report = {
            'summary': {
                'total_tweets_analyzed': total_tweets,
                'toxic_tweets_found': toxic_tweets,
                'toxicity_percentage': (toxic_tweets / total_tweets) * 100,
                'average_toxicity_score': avg_toxicity,
                'maximum_toxicity_score': max_toxicity,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'category_breakdown': category_breakdown,
            'most_toxic_tweet': {
                'tweet_id': most_toxic_analysis.tweet_data.id,
                'author': most_toxic_analysis.tweet_data.author_username,
                'text': most_toxic_analysis.tweet_data.text,
                'toxicity_score': most_toxic_analysis.toxicity_result.overall_score,
                'explanation': most_toxic_analysis.toxicity_result.explanation,
                'url': most_toxic_analysis.tweet_url
            },
            'recommendations': self._generate_recommendations(analyses)
        }
        
        if output_format == 'json':
            # Include detailed analysis data
            report['detailed_analyses'] = [
                {
                    'tweet_id': analysis.tweet_data.id,
                    'author': analysis.tweet_data.author_username,
                    'text': analysis.tweet_data.text,
                    'url': analysis.tweet_url,
                    'toxicity_score': analysis.toxicity_result.overall_score,
                    'categories': analysis.toxicity_result.categories,
                    'confidence': analysis.toxicity_result.confidence,
                    'explanation': analysis.toxicity_result.explanation,
                    'reformulation': analysis.toxicity_result.reformulation
                }
                for analysis in analyses
            ]
        
        return report
    
    def _generate_recommendations(self, analyses: List[TwitterToxicityAnalysis]) -> List[str]:
        """Generate actionable recommendations based on analysis results"""
        recommendations = []
        
        total_tweets = len(analyses)
        toxic_tweets = sum(1 for analysis in analyses 
                          if analysis.toxicity_result.overall_score >= self.toxicity_threshold)
        
        toxicity_percentage = (toxic_tweets / total_tweets) * 100
        
        if toxicity_percentage > 30:
            recommendations.append("High toxicity detected. Consider implementing content moderation.")
        elif toxicity_percentage > 10:
            recommendations.append("Moderate toxicity detected. Monitor discussions closely.")
        else:
            recommendations.append("Low toxicity levels. Current moderation appears effective.")
        
        # Category-specific recommendations
        category_breakdown = self._get_most_common_categories(analyses)
        top_category = max(category_breakdown.items(), key=lambda x: x[1])
        
        if top_category[1] > 0.3:
            if top_category[0] == 'threat':
                recommendations.append("Threats detected. Review for potential escalation.")
            elif top_category[0] == 'insult':
                recommendations.append("Personal attacks identified. Consider warnings or timeouts.")
            elif top_category[0] == 'identity_hate':
                recommendations.append("Identity-based hate detected. Immediate action recommended.")
        
        return recommendations
    
    def export_analysis(self, analysis: TwitterToxicityAnalysis, filepath: str):
        """Export single analysis to JSON file"""
        try:
            export_dict = {
                'tweet_data': {
                    'id': analysis.tweet_data.id,
                    'text': analysis.tweet_data.text,
                    'author_id': analysis.tweet_data.author_id,
                    'author_username': analysis.tweet_data.author_username,
                    'created_at': analysis.tweet_data.created_at.isoformat(),
                    'public_metrics': analysis.tweet_data.public_metrics
                },
                'toxicity_analysis': {
                    'overall_score': analysis.toxicity_result.overall_score,
                    'categories': analysis.toxicity_result.categories,
                    'confidence': analysis.toxicity_result.confidence,
                    'explanation': analysis.toxicity_result.explanation,
                    'reformulation': analysis.toxicity_result.reformulation,
                    'layer1_scores': analysis.toxicity_result.layer1_scores,
                    'layer2_scores': analysis.toxicity_result.layer2_scores
                },
                'metadata': {
                    'tweet_url': analysis.tweet_url,
                    'analysis_timestamp': analysis.analysis_timestamp.isoformat(),
                    'pipeline_version': '1.0'
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_dict, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Analysis exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    async def test_pipeline():
        logging.basicConfig(level=logging.INFO)
        
        # Initialize pipeline
        pipeline = TwitterToxicityPipeline('/teamspace/studios/this_studio/.env')
        
        print("🚀 TESTING TWITTER TOXICITY PIPELINE")
        print("=" * 60)
        
        # Test single tweet analysis (using dummy URL for demo)
        test_url = "https://twitter.com/user/status/1234567890123456789"
        
        print(f"📝 Testing single tweet analysis...")
        print(f"URL: {test_url}")
        print("Note: This will fail because the tweet doesn't exist")
        print("But it demonstrates the complete pipeline flow\n")
        
        # This will fail gracefully due to non-existent tweet
        result = await pipeline.analyze_tweet_from_url(test_url)
        
        if result:
            print(f"✅ Analysis successful!")
            print(f"Tweet: {result.tweet_data.text}")
            print(f"Toxicity Score: {result.toxicity_result.overall_score:.2f}")
        else:
            print("⚠️  Analysis failed (expected for demo URL)")
        
        print("\n✅ Pipeline test complete!")
        print("🚀 Ready for real Twitter URL analysis!")
    
    # Run test
    asyncio.run(test_pipeline())