import os
import logging
import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from dotenv import load_dotenv

@dataclass
class ToxicityResult:
    """Data class for toxicity detection results"""
    text: str
    overall_score: float
    categories: Dict[str, float]
    confidence: float
    layer1_scores: Dict[str, float]
    layer2_scores: Dict[str, float]
    explanation: str = ""
    reformulation: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ToxicityDetector:
    """
    Dual-layer toxicity detection system
    Layer 1: Pre-trained ML model for fast classification
    Layer 2: Gemini AI for contextual analysis and explanations
    """
    
    def __init__(self, env_file_path: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        self._load_credentials(env_file_path)
        
        # Initialize models
        self.ml_classifier = None
        self.tokenizer = None
        self.gemini_model = None
        
        # Toxicity categories
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Initialize both detection layers
        self._initialize_ml_layer()
        self._initialize_gemini_layer()
        
        # Fusion weights (can be tuned)
        self.fusion_weights = {
            'ml_weight': 0.6,
            'gemini_weight': 0.4
        }
    
    def _load_credentials(self, env_file_path: str = None):
        """Load API credentials from environment"""
        if env_file_path:
            load_dotenv(env_file_path)
        
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        
        if not self.gemini_api_key:
            self.logger.warning("Gemini API key not found")
    
    def _initialize_ml_layer(self):
        """Initialize pre-trained ML toxicity classifier"""
        try:
            self.logger.info("Loading ML toxicity classifier...")
            
            # Using a lightweight toxicity classification model
            # Alternative models: "unitary/toxic-bert", "martin-ha/toxic-comment-model"
            model_name = "unitary/toxic-bert"
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize text classification pipeline
            self.ml_classifier = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            self.logger.info("✅ ML layer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML layer: {str(e)}")
            # Fallback: simple keyword-based classification
            self._initialize_fallback_classifier()
    
    def _initialize_fallback_classifier(self):
        """Fallback classifier using keyword matching"""
        self.logger.info("Initializing fallback keyword-based classifier...")
        
        self.toxic_keywords = {
            'toxic': ['hate', 'stupid', 'idiot', 'moron', 'dumb', 'pathetic'],
            'severe_toxic': ['die', 'kill', 'murder', 'suicide'],
            'obscene': ['damn', 'hell', 'crap'],
            'threat': ['kill you', 'hurt you', 'destroy', 'attack'],
            'insult': ['loser', 'failure', 'worthless', 'garbage'],
            'identity_hate': ['racist', 'sexist', 'bigot', 'discrimination']
        }
        
        self.ml_classifier = None  # Mark as fallback mode
        self.logger.info("✅ Fallback classifier initialized")
    
    def _initialize_gemini_layer(self):
        """Initialize Gemini AI for contextual analysis"""
        try:
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                self.logger.info("✅ Gemini layer initialized successfully")
            else:
                self.logger.warning("Gemini API key missing - Layer 2 disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini layer: {str(e)}")
            self.gemini_model = None
    
    def _analyze_with_ml(self, text: str) -> Dict[str, float]:
        """
        Layer 1: Analyze text with pre-trained ML model
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Category scores from ML model
        """
        try:
            if self.ml_classifier:
                # Use transformer model
                results = self.ml_classifier(text)
                
                # Convert results to our category format
                ml_scores = {}
                
                for result in results[0]:  # results is a list of lists
                    label = result['label'].lower()
                    score = result['score']
                    
                    # Map model labels to our categories
                    if 'toxic' in label or 'negative' in label:
                        ml_scores['toxic'] = score
                    elif 'severe' in label:
                        ml_scores['severe_toxic'] = score
                    else:
                        # Default mapping
                        ml_scores['toxic'] = max(ml_scores.get('toxic', 0), score)
                
                # Ensure all categories are present
                for category in self.categories:
                    if category not in ml_scores:
                        ml_scores[category] = 0.0
                
                self.logger.debug(f"ML analysis complete for text: {text[:50]}...")
                return ml_scores
                
            else:
                # Use fallback keyword classifier
                return self._fallback_classify(text)
                
        except Exception as e:
            self.logger.error(f"ML analysis error: {str(e)}")
            return {category: 0.0 for category in self.categories}
    
    def _fallback_classify(self, text: str) -> Dict[str, float]:
        """Fallback keyword-based classification"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.toxic_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 0.3  # Increment score for each keyword match
            
            scores[category] = min(score, 1.0)  # Cap at 1.0
        
        return scores
    
    async def _analyze_with_gemini(self, text: str) -> Tuple[Dict[str, float], str, str]:
        """
        Layer 2: Analyze text with Gemini AI for context and explanations
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Tuple[Dict[str, float], str, str]: Scores, explanation, reformulation
        """
        if not self.gemini_model:
            return {category: 0.0 for category in self.categories}, "", ""
        
        try:
            prompt = f"""
            Analyze the following text for toxicity across these categories:
            1. toxic (general toxicity)
            2. severe_toxic (severe threats or violence)
            3. obscene (profanity or vulgar content)
            4. threat (direct threats)
            5. insult (personal attacks)
            6. identity_hate (hate based on identity)
            
            Text to analyze: "{text}"
            
            Provide your response in this JSON format:
            {{
                "scores": {{
                    "toxic": 0.0-1.0,
                    "severe_toxic": 0.0-1.0,
                    "obscene": 0.0-1.0,
                    "threat": 0.0-1.0,
                    "insult": 0.0-1.0,
                    "identity_hate": 0.0-1.0
                }},
                "explanation": "Brief explanation of why this text is/isn't toxic",
                "reformulation": "If toxic, suggest a polite reformulation. If not toxic, return empty string."
            }}
            
            Consider context, sarcasm, and intent. Be precise with scores (0.0 = not toxic, 1.0 = highly toxic).
            """
            
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                
                scores = result_json.get('scores', {})
                explanation = result_json.get('explanation', '')
                reformulation = result_json.get('reformulation', '')
                
                # Ensure all categories are present
                for category in self.categories:
                    if category not in scores:
                        scores[category] = 0.0
                
                self.logger.debug(f"Gemini analysis complete for text: {text[:50]}...")
                return scores, explanation, reformulation
            
            else:
                self.logger.error("Could not parse JSON from Gemini response")
                return {category: 0.0 for category in self.categories}, "", ""
                
        except Exception as e:
            self.logger.error(f"Gemini analysis error: {str(e)}")
            return {category: 0.0 for category in self.categories}, "", ""
    
    def _fuse_scores(self, ml_scores: Dict[str, float], gemini_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Fuse scores from both detection layers
        
        Args:
            ml_scores (Dict[str, float]): Scores from ML layer
            gemini_scores (Dict[str, float]): Scores from Gemini layer
            
        Returns:
            Dict[str, float]: Fused final scores
        """
        fused_scores = {}
        
        for category in self.categories:
            ml_score = ml_scores.get(category, 0.0)
            gemini_score = gemini_scores.get(category, 0.0)
            
            # Weighted average fusion
            fused_score = (
                ml_score * self.fusion_weights['ml_weight'] +
                gemini_score * self.fusion_weights['gemini_weight']
            )
            
            fused_scores[category] = fused_score
        
        return fused_scores
    
    async def analyze_text(self, text: str) -> ToxicityResult:
        """
        Main analysis function - combines both detection layers
        
        Args:
            text (str): Text to analyze for toxicity
            
        Returns:
            ToxicityResult: Complete analysis results
        """
        if not text or not text.strip():
            return ToxicityResult(
                text="",
                overall_score=0.0,
                categories={category: 0.0 for category in self.categories},
                confidence=1.0,
                layer1_scores={},
                layer2_scores={},
                explanation="Empty text provided",
                reformulation=""
            )
        
        self.logger.info(f"Analyzing text: {text[:100]}...")
        
        # Layer 1: ML Analysis
        ml_scores = self._analyze_with_ml(text)
        
        # Layer 2: Gemini Analysis
        gemini_scores, explanation, reformulation = await self._analyze_with_gemini(text)
        
        # Fuse scores
        final_scores = self._fuse_scores(ml_scores, gemini_scores)
        
        # Calculate overall toxicity score
        overall_score = max(final_scores.values())
        
        # Calculate confidence based on agreement between layers
        confidence = self._calculate_confidence(ml_scores, gemini_scores)
        
        result = ToxicityResult(
            text=text,
            overall_score=overall_score,
            categories=final_scores,
            confidence=confidence,
            layer1_scores=ml_scores,
            layer2_scores=gemini_scores,
            explanation=explanation,
            reformulation=reformulation
        )
        
        self.logger.info(f"Analysis complete - Overall score: {overall_score:.2f}")
        return result
    
    def _calculate_confidence(self, ml_scores: Dict[str, float], gemini_scores: Dict[str, float]) -> float:
        """
        Calculate confidence based on agreement between layers
        
        Args:
            ml_scores (Dict[str, float]): ML layer scores
            gemini_scores (Dict[str, float]): Gemini layer scores
            
        Returns:
            float: Confidence score (0.0-1.0)
        """
        if not ml_scores or not gemini_scores:
            return 0.5  # Medium confidence if only one layer available
        
        # Calculate correlation/agreement between scores
        differences = []
        for category in self.categories:
            ml_score = ml_scores.get(category, 0.0)
            gemini_score = gemini_scores.get(category, 0.0)
            differences.append(abs(ml_score - gemini_score))
        
        avg_difference = np.mean(differences)
        confidence = max(0.0, 1.0 - avg_difference)  # Higher agreement = higher confidence
        
        return confidence
    
    def batch_analyze(self, texts: List[str]) -> List[ToxicityResult]:
        """
        Analyze multiple texts
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[ToxicityResult]: Results for each text
        """
        async def analyze_batch():
            tasks = [self.analyze_text(text) for text in texts]
            return await asyncio.gather(*tasks)
        
        return asyncio.run(analyze_batch())
    
    def export_result(self, result: ToxicityResult, filepath: str):
        """
        Export toxicity analysis result to JSON
        
        Args:
            result (ToxicityResult): Analysis result to export
            filepath (str): File path for export
        """
        try:
            export_dict = {
                'text': result.text,
                'overall_score': result.overall_score,
                'categories': result.categories,
                'confidence': result.confidence,
                'layer1_scores': result.layer1_scores,
                'layer2_scores': result.layer2_scores,
                'explanation': result.explanation,
                'reformulation': result.reformulation,
                'timestamp': result.timestamp.isoformat(),
                'analysis_metadata': {
                    'fusion_weights': self.fusion_weights,
                    'model_info': {
                        'ml_layer': 'toxic-bert' if self.ml_classifier else 'fallback',
                        'gemini_layer': self.gemini_model_name if self.gemini_model else 'disabled'
                    }
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_dict, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Analysis result exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting result: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    async def test_toxicity_detector():
        # Initialize detector
        detector = ToxicityDetector('/teamspace/studios/this_studio/.env')
        
        # Test texts
        test_texts = [
            "Hello, how are you today?",  # Non-toxic
            "You are such an idiot and I hate you!",  # Toxic
            "This is a great post, thanks for sharing!",  # Non-toxic
            "I'm going to destroy your career!",  # Threat
            ""  # Empty text
        ]
        
        print("🔍 TESTING DUAL-LAYER TOXICITY DETECTION")
        print("=" * 60)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 Test {i}: {text if text else '(empty text)'}")
            
            result = await detector.analyze_text(text)
            
            print(f"Overall Score: {result.overall_score:.2f}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Top Category: {max(result.categories.items(), key=lambda x: x[1])}")
            
            if result.explanation:
                print(f"Explanation: {result.explanation}")
            
            if result.reformulation:
                print(f"Suggested Fix: {result.reformulation}")
            
            print("-" * 40)
        
        print("\n✅ Toxicity detection testing complete!")
    
    # Run test
    asyncio.run(test_toxicity_detector())