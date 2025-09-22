#!/usr/bin/env python3
import re
import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    is_helpful: bool
    is_accurate: bool
    is_complete: bool
    is_relevant: bool
    confidence_score: float
    issues: List[str]
    strengths: List[str]

class ImprovedEvaluator:    
    def __init__(self):
        self.price_patterns = [
            r'\$\d+(?:\.\d{2})?',  # $123.45
            r'\d+(?:\.\d{2})?\s*dollars?',  # 123.45 dollars
            r'price[:\s]*\$?\d+(?:\.\d{2})?',  # price: $123.45
        ]
        
        self.rating_patterns = [
            r'\d+(?:\.\d+)?\s*(?:stars?|out of 5)',  # 4.5 stars
            r'rating[:\s]*\d+(?:\.\d+)?',  # rating: 4.5
            r'\d+(?:\.\d+)?\/5',  # 4.5/5
        ]
        
        self.brand_patterns = [
            r'brand[:\s]*([A-Za-z][A-Za-z0-9\s&]+)',  # brand: Apple
            r'made by\s+([A-Za-z][A-Za-z0-9\s&]+)',  # made by Apple
            r'manufactured by\s+([A-Za-z][A-Za-z0-9\s&]+)',  # manufactured by Apple
        ]
    
    def evaluate_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> EvaluationResult:       
        # Basic checks
        is_empty = len(response.strip()) == 0
        contains_error = self._contains_error(response)
        
        if is_empty:
            return EvaluationResult(
                is_helpful=False, is_accurate=False, is_complete=False, is_relevant=False,
                confidence_score=0.0, issues=["Empty response"], strengths=[]
            )
        
        if contains_error:
            return EvaluationResult(
                is_helpful=False, is_accurate=False, is_complete=False, is_relevant=False,
                confidence_score=0.0, issues=["Contains error"], strengths=[]
            )
        
        # Analyze query intent
        query_intent = self._analyze_query_intent(query)
        
        # Evaluate based on intent
        if query_intent == 'price_inquiry':
            return self._evaluate_price_response(query, response, expected_product)
        elif query_intent == 'rating_inquiry':
            return self._evaluate_rating_response(query, response, expected_product)
        elif query_intent == 'brand_inquiry':
            return self._evaluate_brand_response(query, response, expected_product)
        elif query_intent == 'feature_inquiry':
            return self._evaluate_feature_response(query, response, expected_product)
        elif query_intent == 'feature_check':
            return self._evaluate_feature_check_response(query, response, expected_product)
        elif query_intent == 'shipping_inquiry':
            return self._evaluate_shipping_response(query, response, expected_product)
        elif query_intent == 'return_inquiry':
            return self._evaluate_return_response(query, response, expected_product)
        else:
            return self._evaluate_general_response(query, response, expected_product)
    
    def _analyze_query_intent(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['price', 'cost', 'much', 'expensive']):
            return 'price_inquiry'
        
        if any(word in query_lower for word in ['rating', 'rated', 'stars', 'review']):
            return 'rating_inquiry'
        
        if any(word in query_lower for word in ['brand', 'makes', 'company', 'manufacturer']):
            return 'brand_inquiry'
        
        if 'features' in query_lower:
            return 'feature_inquiry'
        
        if query_lower.startswith('is ') or query_lower.startswith('does ') or query_lower.startswith('has '):
            return 'feature_check'
        
        if any(word in query_lower for word in ['shipping', 'delivery', 'ship', 'deliver']):
            return 'shipping_inquiry'
        
        if any(word in query_lower for word in ['return', 'refund', 'policy']):
            return 'return_inquiry'
        
        return 'general_inquiry'
    
    def _evaluate_price_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> EvaluationResult:
        issues = []
        strengths = []
        
        price_found = False
        price_accurate = False
        
        for pattern in self.price_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                price_found = True
                break
        
        if price_found:
            # Check if the price is accurate
            expected_price = expected_product.get('price', 0)
            if expected_price > 0:
                price_str = str(expected_price)
                if price_str in response:
                    price_accurate = True
                    strengths.append("Accurate price information")
                else:
                    issues.append("Price information present but inaccurate")
            else:
                issues.append("Price information present but no expected price to compare")
        else:
            issues.append("No price information found")
        
        # Check if response is helpful
        is_helpful = price_found and (price_accurate or expected_product.get('price', 0) == 0)
        
        # Check completeness
        is_complete = price_found and len(response) > 20  # Not just a number
        
        # Check relevance
        is_relevant = 'price' in response.lower() or '$' in response
        
        confidence_score = self._calculate_confidence(is_helpful, price_accurate, is_complete, is_relevant)
        
        return EvaluationResult(
            is_helpful=is_helpful,
            is_accurate=price_accurate,
            is_complete=is_complete,
            is_relevant=is_relevant,
            confidence_score=confidence_score,
            issues=issues,
            strengths=strengths
        )
    
    def _evaluate_rating_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> EvaluationResult:
        issues = []
        strengths = []
        
        # Check if response contains rating information
        rating_found = False
        rating_accurate = False
        
        for pattern in self.rating_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                rating_found = True
                break
        
        if rating_found:
            expected_rating = expected_product.get('avg_rating', 0)
            if expected_rating > 0:
                rating_str = str(expected_rating)
                if rating_str in response:
                    rating_accurate = True
                    strengths.append("Accurate rating information")
                else:
                    issues.append("Rating information present but inaccurate")
            else:
                issues.append("Rating information present but no expected rating to compare")
        else:
            issues.append("No rating information found")
        
        is_helpful = rating_found and (rating_accurate or expected_product.get('avg_rating', 0) == 0)
        is_complete = rating_found and len(response) > 20
        is_relevant = 'rating' in response.lower() or 'star' in response.lower()
        
        confidence_score = self._calculate_confidence(is_helpful, rating_accurate, is_complete, is_relevant)
        
        return EvaluationResult(
            is_helpful=is_helpful,
            is_accurate=rating_accurate,
            is_complete=is_complete,
            is_relevant=is_relevant,
            confidence_score=confidence_score,
            issues=issues,
            strengths=strengths
        )
    
    def _evaluate_brand_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> EvaluationResult:
        issues = []
        strengths = []
        
        brand_found = False
        brand_accurate = False
        
        for pattern in self.brand_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                brand_found = True
                break
        
        if brand_found:
            expected_brand = expected_product.get('brand', '')
            if expected_brand:
                if expected_brand.lower() in response.lower():
                    brand_accurate = True
                    strengths.append("Accurate brand information")
                else:
                    issues.append("Brand information present but inaccurate")
            else:
                issues.append("Brand information present but no expected brand to compare")
        else:
            issues.append("No brand information found")
        
        is_helpful = brand_found and (brand_accurate or not expected_product.get('brand', ''))
        is_complete = brand_found and len(response) > 20
        is_relevant = 'brand' in response.lower()
        
        confidence_score = self._calculate_confidence(is_helpful, brand_accurate, is_complete, is_relevant)
        
        return EvaluationResult(
            is_helpful=is_helpful,
            is_accurate=brand_accurate,
            is_complete=is_complete,
            is_relevant=is_relevant,
            confidence_score=confidence_score,
            issues=issues,
            strengths=strengths
        )
    
    def _evaluate_feature_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> EvaluationResult:
        issues = []
        strengths = []
        
        feature_found = 'feature' in response.lower()
        feature_accurate = False
        
        if feature_found:
            expected_features = expected_product.get('features', [])
            if expected_features:
                found_features = []
                for feature in expected_features:
                    if feature.lower() in response.lower():
                        found_features.append(feature)
                
                if found_features:
                    feature_accurate = True
                    strengths.append(f"Found features: {', '.join(found_features)}")
                else:
                    issues.append("Feature information present but doesn't match expected features")
            else:
                issues.append("Feature information present but no expected features to compare")
        else:
            issues.append("No feature information found")
        
        is_helpful = feature_found and (feature_accurate or not expected_product.get('features', []))
        is_complete = feature_found and len(response) > 30
        is_relevant = 'feature' in response.lower()
        
        confidence_score = self._calculate_confidence(is_helpful, feature_accurate, is_complete, is_relevant)
        
        return EvaluationResult(
            is_helpful=is_helpful,
            is_accurate=feature_accurate,
            is_complete=is_complete,
            is_relevant=is_relevant,
            confidence_score=confidence_score,
            issues=issues,
            strengths=strengths
        )
    
    def _evaluate_feature_check_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> EvaluationResult:
        issues = []
        strengths = []
        feature_accurate = False  # Initialize variable
        issues = []
        strengths = []
        
        # Extract the feature being asked about
        query_lower = query.lower()
        asked_feature = None
        
        # Common features to check for
        common_features = [
            'organic', 'gluten free', 'dishwasher safe', 'heavy duty', 
            'professional grade', 'ergonomic', 'durable', 'gentle',
            'fragrance free', 'bpa free', 'non-stick', 'easy clean',
            'healthy', 'natural', 'non-gmo'
        ]
        
        for feature in common_features:
            if feature in query_lower:
                asked_feature = feature
                break
        
        if asked_feature:
            # Check if response addresses the specific feature
            response_lower = response.lower()
            
            has_yes = any(word in response_lower for word in ['yes', 'indeed', 'correct', 'true'])
            has_no = any(word in response_lower for word in ['no', 'not', 'false', 'incorrect'])
            
            if has_yes or has_no:
                # Check if the response is accurate based on expected features
                expected_features = [f.lower() for f in expected_product.get('features', [])]
                
                if asked_feature in expected_features:
                    # Product has the feature
                    if has_yes:
                        feature_accurate = True
                        strengths.append(f"Correctly identified {asked_feature} feature")
                    else:
                        issues.append(f"Incorrectly denied {asked_feature} feature")
                else:
                    # Product doesn't have the feature
                    if has_no:
                        feature_accurate = True
                        strengths.append(f"Correctly identified absence of {asked_feature} feature")
                    else:
                        issues.append(f"Incorrectly claimed {asked_feature} feature")
            else:
                issues.append("Response doesn't clearly answer yes/no")
        else:
            issues.append("Could not identify specific feature being asked about")
        
        is_helpful = asked_feature and (has_yes or has_no)
        is_complete = is_helpful and len(response) > 20
        is_relevant = asked_feature and asked_feature in response.lower()
        
        confidence_score = self._calculate_confidence(is_helpful, feature_accurate, is_complete, is_relevant)
        
        return EvaluationResult(
            is_helpful=is_helpful,
            is_accurate=feature_accurate,
            is_complete=is_complete,
            is_relevant=is_relevant,
            confidence_score=confidence_score,
            issues=issues,
            strengths=strengths
        )
    
    def _evaluate_shipping_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> EvaluationResult:
        issues = []
        strengths = []
        
        # Check if response contains shipping information
        shipping_keywords = ['shipping', 'delivery', 'days', 'express', 'overnight', 'standard']
        shipping_found = any(keyword in response.lower() for keyword in shipping_keywords)
        
        if shipping_found:
            strengths.append("Contains shipping information")
        else:
            issues.append("No shipping information found")
        
        is_helpful = shipping_found
        is_complete = shipping_found and len(response) > 30
        is_relevant = 'shipping' in response.lower() or 'delivery' in response.lower()
        
        confidence_score = self._calculate_confidence(is_helpful, True, is_complete, is_relevant)
        
        return EvaluationResult(
            is_helpful=is_helpful,
            is_accurate=True,  # Assume accurate if tool was called
            is_complete=is_complete,
            is_relevant=is_relevant,
            confidence_score=confidence_score,
            issues=issues,
            strengths=strengths
        )
    
    def _evaluate_return_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> EvaluationResult:
        issues = []
        strengths = []
        
        # Check if response contains return policy information
        return_keywords = ['return', 'policy', 'window', 'condition', 'restocking']
        return_found = any(keyword in response.lower() for keyword in return_keywords)
        
        if return_found:
            strengths.append("Contains return policy information")
        else:
            issues.append("No return policy information found")
        
        is_helpful = return_found
        is_complete = return_found and len(response) > 30
        is_relevant = 'return' in response.lower() or 'policy' in response.lower()
        
        confidence_score = self._calculate_confidence(is_helpful, True, is_complete, is_relevant)
        
        return EvaluationResult(
            is_helpful=is_helpful,
            is_accurate=True,  
            is_complete=is_complete,
            is_relevant=is_relevant,
            confidence_score=confidence_score,
            issues=issues,
            strengths=strengths
        )
    
    def _evaluate_general_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> EvaluationResult:
        issues = []
        strengths = []
        
        # Basic quality checks
        if len(response) < 10:
            issues.append("Response too short")
        else:
            strengths.append("Response has adequate length")
        
        if 'product' in response.lower() or expected_product.get('title', '').lower() in response.lower():
            strengths.append("Response mentions the product")
        else:
            issues.append("Response doesn't mention the product")
        
        is_helpful = len(response) > 10
        is_complete = len(response) > 30
        is_relevant = 'product' in response.lower() or expected_product.get('title', '').lower() in response.lower()
        
        confidence_score = self._calculate_confidence(is_helpful, True, is_complete, is_relevant)
        
        return EvaluationResult(
            is_helpful=is_helpful,
            is_accurate=True,
            is_complete=is_complete,
            is_relevant=is_relevant,
            confidence_score=confidence_score,
            issues=issues,
            strengths=strengths
        )
    
    def _contains_error(self, response: str) -> bool:
        error_indicators = [
            'error', 'not available', 'cannot find', 'unable to',
            'sorry', 'apologize', 'unfortunately', 'failed'
        ]
        return any(indicator in response.lower() for indicator in error_indicators)
    
    def _calculate_confidence(self, is_helpful: bool, is_accurate: bool, is_complete: bool, is_relevant: bool) -> float:
        score = 0.0
        if is_helpful:
            score += 0.3
        if is_accurate:
            score += 0.3
        if is_complete:
            score += 0.2
        if is_relevant:
            score += 0.2
        return score

if __name__ == "__main__":
    evaluator = ImprovedEvaluator()
    # Test
    query = "What is the price of Cetaphil Vitamins?"
    response = "The Cetaphil Vitamins - Personal Care costs $295.26."
    expected_product = {
        "title": "Cetaphil Vitamins - Personal Care",
        "price": 295.26,
        "brand": "Cetaphil",
        "features": ["Gentle", "Fragrance Free"],
        "avg_rating": 3.4
    }
    
    result = evaluator.evaluate_response(query, response, expected_product)
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Result: {result}")
