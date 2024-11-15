from typing import List, Dict, Any
from openai import OpenAI
import json

class HealthChatAssistant:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.context = []
        self.health_data = None

    def update_health_data(self, samsung_data: Dict, oura_data: Dict, 
                          recommendations: List[Dict]):
        """Update assistant's context with current health data"""
        self.health_data = {
            'samsung_data': samsung_data,
            'oura_data': oura_data,
            # Limit recommendations to 3 most relevant ones
            'recommendations': recommendations[:3] if recommendations else []
        }

    def get_response(self, user_input: str) -> str:
        """Get AI response to user query"""
        try:
            # Create simplified system message
            system_message = self._create_simplified_message()
            
            # Create conversation
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use GPT-3.5 for faster responses
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"I apologize, but I'm having trouble processing your request. Please try asking a more specific question about your sleep or health data."

    def _create_simplified_message(self) -> str:
        """Create a simplified system message with key data points"""
        sleep_score = self.health_data['oura_data'].get('sleep_score', 'N/A')
        steps = self.health_data['samsung_data'].get('steps', 'N/A')
        heart_rate = self.health_data['samsung_data'].get('heart_rate', 'N/A')

        return f"""You are a health assistant. Key data points:
        - Sleep Score: {sleep_score}
        - Daily Steps: {steps}
        - Heart Rate: {heart_rate}
        
        Provide brief, practical advice based on this data."""