import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import json

class HealthDataCollector:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        self.cache = {}
        self.cache_duration = 3600  # 1 hour

    def get_health_recommendations(self) -> List[Dict]:
        """Collect health recommendations from reputable sources"""
        if self._is_cache_valid('health_recs'):
            return self.cache['health_recs']

        recommendations = []
        sources = {
            'mayo_clinic': 'https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/sleep/art-20048379',
            'sleep_foundation': 'https://www.sleepfoundation.org/sleep-hygiene',
            'cdc': 'https://www.cdc.gov/sleep/about_sleep/sleep_hygiene.html'
        }

        for source, url in sources.items():
            try:
                recs = self._scrape_recommendations(url, source)
                recommendations.extend(recs)
            except Exception as e:
                print(f"Error scraping {source}: {e}")

        self.cache['health_recs'] = recommendations
        self.cache['health_recs_timestamp'] = time.time()
        return recommendations

    def _scrape_recommendations(self, url: str, source: str) -> List[Dict]:
        """Scrape recommendations from a specific source"""
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        recommendations = []

        # Look for relevant content in different HTML elements
        for element in soup.find_all(['p', 'li', 'h2', 'h3']):
            text = element.get_text().strip()
            if len(text) > 20 and self._is_relevant(text):
                category = self._categorize_recommendation(text)
                recommendations.append({
                    'source': source,
                    'text': text,
                    'category': category,
                    'url': url
                })

        return recommendations

    def _is_relevant(self, text: str) -> bool:
        """Check if text contains relevant health information"""
        keywords = ['sleep', 'rest', 'activity', 'exercise', 'health', 
                   'routine', 'habit', 'lifestyle', 'wellness']
        return any(keyword in text.lower() for keyword in keywords)

    def _categorize_recommendation(self, text: str) -> str:
        """Categorize recommendation based on content"""
        categories = {
            'sleep': ['sleep', 'bed', 'rest', 'nap'],
            'exercise': ['exercise', 'activity', 'workout', 'movement'],
            'diet': ['food', 'drink', 'eat', 'nutrition'],
            'lifestyle': ['routine', 'habit', 'schedule'],
            'environment': ['room', 'temperature', 'light', 'noise'],
            'mental_health': ['stress', 'anxiety', 'relaxation', 'meditation']
        }

        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'general'

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        timestamp_key = f'{cache_key}_timestamp'
        if timestamp_key not in self.cache:
            return False
        return time.time() - self.cache[timestamp_key] < self.cache_duration