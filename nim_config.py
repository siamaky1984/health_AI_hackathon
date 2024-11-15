from nimblephysics.blueprint import BlueprintConfig

config = BlueprintConfig(
    name="sleep_predictor",
    description="Sleep Quality Prediction and Analysis Service",
    version="1.0.0",
    endpoints=[
        {
            "path": "/predict",
            "method": "POST",
            "description": "Predict sleep quality",
            "request_schema": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["predict_sleep"]},
                    "samsung_data": {"type": "object"},
                    "oura_data": {"type": "object"}
                }
            }
        },
        {
            "path": "/visualize",
            "method": "POST",
            "description": "Generate sleep visualizations",
            "request_schema": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["get_visualizations"]},
                    "visualization_type": {"type": "string"}
                }
            }
        },
        {
            "path": "/weather",
            "method": "POST",
            "description": "Get weather data",
            "request_schema": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["get_weather"]},
                    "location": {"type": "string"}
                }
            }
        }
    ]
)