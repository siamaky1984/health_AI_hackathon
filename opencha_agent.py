from typing import Dict, Any, List
import json

class SleepHealthAgent:
    def __init__(self):
        self.orchestrator = None
        self.data_pipe = None
        self.task_planner = None
        self.response_generator = None

    def process_health_data(self, samsung_data: Dict[str, Any], 
                          oura_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process multimodal health data through openCHA's Data Pipe"""
        structured_data = {
            'wearable_data': {
                'samsung': samsung_data,
                'oura': oura_data
            },
            'metadata': {
                'data_type': 'sleep_health',
                'timestamp': self.data_pipe.get_timestamp()
            }
        }
        return self.data_pipe.transform(structured_data)

    def generate_response(self, query: str, health_data: Dict) -> str:
        """Generate response using openCHA's Response Generator"""
        # Plan tasks based on query
        tasks = self.task_planner.plan(query, health_data)
        
        # Execute tasks through orchestrator
        results = self.orchestrator.execute_tasks(tasks)
        
        # Generate personalized response
        response = self.response_generator.generate(
            query=query,
            health_data=health_data,
            task_results=results
        )
        
        return response