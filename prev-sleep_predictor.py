# Add these imports at the top
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
import threading
import queue
import fastapi
import time


# ... existing imports ...
from fastapi import FastAPI
from pydantic import BaseModel


# def build_sleep_predictor(samsung_data=None, oura_data=None):
#     """
#     Builds and returns the sleep predictor model
    
#     Args:
#         samsung_data: Historical Samsung health data
#         oura_data: Historical Oura ring data
    
#     Returns:
#         tuple: (predictor, results, combined_features)
#     """
#     # Initialize empty data if none provided
#     if samsung_data is None:
#         samsung_data = {}
#     if oura_data is None:
#         oura_data = {}
    
#     # Combine features from both data sources
#     combined_features = {
#         **samsung_data,
#         **oura_data
#     }
    

# Add this after imports but before any classes
def build_sleep_predictor(awake_times, hrv_data, imu_data, ppg_data, pedometer_data,
                         sleep_data, activity_data, readiness_data, heart_rate_data):
    """
    Builds and returns the sleep predictor model
    
    Args:
        awake_times: Samsung awake times data
        hrv_data: Samsung HRV data
        imu_data: Samsung IMU data
        ppg_data: Samsung PPG data
        pedometer_data: Samsung pedometer data
        sleep_data: Oura sleep data
        activity_data: Oura activity data
        readiness_data: Oura readiness data
        heart_rate_data: Oura heart rate data
    
    Returns:
        tuple: (predictor, results, combined_features)
    """
    # Combine features from both data sources
    samsung_data = {
        'awake_times': awake_times,
        'hrv_data': hrv_data,
        'imu_data': imu_data,
        'ppg_data': ppg_data,
        'pedometer_data': pedometer_data
    }
    
    oura_data = {
        'sleep_data': sleep_data,
        'activity_data': activity_data,
        'readiness_data': readiness_data,
        'heart_rate_data': heart_rate_data
    }
    
    combined_features = {
        **samsung_data,
        **oura_data
    }
    
    class SimplePredictor:
        def predict(self, samsung_data, oura_data):
            # Implement your actual prediction logic here
            return 75  # Placeholder sleep score
    
    predictor = SimplePredictor()
    results = {}  # Placeholder for model results
    
    return predictor, results, combined_features







class PredictionRequest(BaseModel):
    """Data model for prediction requests"""
    message: str
    samsung_data: dict
    oura_data: dict
    weather_data: dict
    health_news: list

class NIMConnector:
    """Handles communication with the Neural Interface Module"""
    def __init__(self, predictor):
        self.predictor = predictor
        self.app = FastAPI()
        
        # Register endpoints
        self.app.post("/predict")(self.predict_sleep)
    
    async def predict_sleep(self, request: PredictionRequest):
        """Process prediction request and return sleep recommendations"""
        try:
            # Use the predictor to generate sleep score and recommendations
            sleep_score = self.predictor.predict(request.samsung_data, request.oura_data)
            recommendations = self.generate_recommendations(
                sleep_score, 
                request.weather_data, 
                request.health_news
            )
            
            return {
                "sleep_score": sleep_score,
                "recommendations": recommendations
            }
        except Exception as e:
            return {"error": str(e)}
    
    def generate_recommendations(self, sleep_score, weather_data, health_news):
        """Generate sleep recommendations based on all available data"""
        recommendations = []
        
        # Add basic recommendations based on sleep score
        if sleep_score < 70:
            recommendations.append("Consider going to bed earlier tonight")
            recommendations.append("Limit screen time before bed")
        
        # Add weather-based recommendations
        if weather_data.get('humidity', 0) > 70:
            recommendations.append("High humidity - consider using a dehumidifier")
        
        # Add any relevant news-based recommendations
        if health_news:
            recommendations.append(f"Latest sleep tip: {health_news[0]}")
        
        return recommendations

# ... rest of your existing code ...


















class WebDataCollector:
    """Collects relevant health and weather data from websites"""
    def __init__(self):
        self.weather_data = {'temperature': 0, 'humidity': 0, 'pressure': 0, 'sunset_time': ''}
        self.health_news = ['No news available']  # Default value
        self.air_quality = {}
    
    def fetch_weather_data(self, location):
        """Fetch local weather data that might impact sleep"""
        try:
            # Example using a weather API (you'll need to replace with your preferred weather service)
            api_key = "YOUR_WEATHER_API_KEY"  # Replace with actual API key
            url = f"https://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}"
            response = requests.get(url)
            data = response.json()
            
            self.weather_data = {
                'temperature': data['current']['temp_c'],
                'humidity': data['current']['humidity'],
                'pressure': data['current']['pressure_mb'],
                'sunset_time': data['forecast']['forecastday'][0]['astro']['sunset']
            }
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            # Set default values if fetch fails
            self.weather_data = {
                'temperature': 0,
                'humidity': 0,
                'pressure': 0,
                'sunset_time': 'unknown'
            }
    

class SleepAppGUI:
    def __init__(self, predictor, nim_connector):
        self.root = tk.Tk()
        self.root.title("Sleep Quality Predictor")
        self.predictor = predictor
        self.nim_connector = nim_connector
        self.web_collector = WebDataCollector()
        self.message_queue = queue.Queue()
        
        self.setup_gui()
        self.start_data_collection()
    
    def setup_gui(self):
        # Create main containers
        self.notebook = ttk.Notebook(self.root)
        self.data_frame = ttk.Frame(self.notebook)
        self.chat_frame = ttk.Frame(self.notebook)
        self.visualization_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_frame, text="Sensor Data")
        self.notebook.add(self.chat_frame, text="Chat")
        self.notebook.add(self.visualization_frame, text="Visualizations")
        self.notebook.pack(expand=True, fill='both')
        
        # Sensor data display
        self.setup_sensor_display()
        
        # Chat interface
        self.setup_chat_interface()
        
        # Visualization controls
        self.setup_visualization_controls()
    
    def setup_sensor_display(self):
        # Create labels for sensor data
        self.samsung_label = ttk.Label(self.data_frame, text="Samsung Data:")
        self.samsung_label.pack()
        
        self.oura_label = ttk.Label(self.data_frame, text="Oura Data:")
        self.oura_label.pack()
        
        self.weather_label = ttk.Label(self.data_frame, text="Weather Data:")
        self.weather_label.pack()
        
        self.news_label = ttk.Label(self.data_frame, text="Health News:")
        self.news_label.pack()
    
    def setup_chat_interface(self):
        # Chat display
        self.chat_display = tk.Text(self.chat_frame, height=20, width=50)
        self.chat_display.pack(pady=10)
        
        # Input area
        self.input_frame = ttk.Frame(self.chat_frame)
        self.input_frame.pack(fill='x', pady=5)
        
        self.chat_input = ttk.Entry(self.input_frame)
        self.chat_input.pack(side='left', fill='x', expand=True)
        
        self.send_button = ttk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side='right')
    
    def setup_visualization_controls(self):
        # Add buttons for different visualizations
        ttk.Button(self.visualization_frame, text="Sleep Patterns", 
                  command=lambda: self.predictor.visualize_data(self.combined_features, self.results)
                 ).pack(pady=5)
        
        ttk.Button(self.visualization_frame, text="Activity Patterns",
                  command=lambda: SleepQualityVisualizer.plot_activity_patterns(self.combined_features)
                 ).pack(pady=5)
    
    def send_message(self):
        message = self.chat_input.get()
        if message:
            self.chat_input.delete(0, tk.END)
            self.process_message(message)
    
    async def process_message(self, message):
        # Prepare data for NIM
        current_data = {
            "message": message,
            "samsung_data": self.latest_samsung_data,
            "oura_data": self.latest_oura_data,
            "weather_data": self.web_collector.weather_data,
            "health_news": self.web_collector.health_news
        }
        
        # Send to NIM and get response
        try:
            response = await self.nim_connector.predict_sleep(PredictionRequest(**current_data))
            self.display_response(response)
        except Exception as e:
            self.chat_display.insert(tk.END, f"Error: {str(e)}\n")
    
    def display_response(self, response):
        # Display the prediction and recommendations
        self.chat_display.insert(tk.END, f"\nPredicted Sleep Score: {response.sleep_score}\n")
        self.chat_display.insert(tk.END, "Recommendations:\n")
        for rec in response.recommendations:
            self.chat_display.insert(tk.END, f"- {rec}\n")
        self.chat_display.see(tk.END)
    
    def start_data_collection(self):
        """Start background data collection"""
        def collect_data():
            while True:
                # Update sensor data
                self.update_samsung_data()
                self.update_oura_data()
                
                # Update web data
                self.web_collector.fetch_weather_data("your_location")
                self.web_collector.fetch_health_news()
                
                # Update GUI
                self.update_gui()
                
                time.sleep(300)  # Update every 5 minutes
        
        threading.Thread(target=collect_data, daemon=True).start()
    def update_samsung_data(self):
        try:
            # Initialize with empty data if no real data available
            self.latest_samsung_data = {
                'awake_times': [],
                'hrv_data': [],
                'imu_data': [],
                'ppg_data': [],
                'pedometer_data': []
            }
            # Add your actual Samsung data collection logic here
        except Exception as e:
            print(f"Error updating Samsung data: {e}")

    def update_oura_data(self):
        """Update Oura ring data"""
        try:
            # Initialize with empty data if no real data available
            self.latest_oura_data = {
                'sleep_data': [],
                'activity_data': [],
                'readiness_data': [],
                'heart_rate_data': []
            }
            # Add your actual Oura data collection logic here
        except Exception as e:
            print(f"Error updating Oura data: {e}")
    def update_gui(self):
        """Update GUI with latest data"""
        self.root.after(0, self._update_gui_labels)
    
    def _update_gui_labels(self):
        # Update sensor data displays
        self.samsung_label.config(text=f"Samsung Data: {self.latest_samsung_data}")
        self.oura_label.config(text=f"Oura Data: {self.latest_oura_data}")
        self.weather_label.config(text=f"Weather: {self.web_collector.weather_data}")
        self.news_label.config(text=f"Latest News: {self.web_collector.health_news[0]}")
    
    def run(self):
        self.root.mainloop()

# Modify your main block to include the GUI
if __name__ == "__main__":
    # ... your existing initialization code ...
    
    # Initialize the predictor and NIM connector
    predictor, results, combined_features = build_sleep_predictor(...)
    nim_connector = NIMConnector(predictor)
    
    # Create and run the GUI
    app = SleepAppGUI(predictor, nim_connector)
    
    # Run the FastAPI server in a separate thread
    def run_server():
        uvicorn.run(
            nim_connector.app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Run the GUI
    app.run()