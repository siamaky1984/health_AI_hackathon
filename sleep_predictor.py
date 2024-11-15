from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import uvicorn
from health_data_collector import HealthDataCollector
from chat_assistant import HealthChatAssistant
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = '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'

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("OpenAI API key not found! Please set OPENAI_API_KEY in your environment or .env file")
    api_key = "dummy_key"  # This will prevent the app from crashing, but chat won't work

# Initialize collectors and assistants
health_collector = HealthDataCollector()
chat_assistant = HealthChatAssistant(api_key=api_key)

from opencha import Interface, Orchestrator, DataPipe, TaskPlanner, ResponseGenerator
from opencha.sources import HealthcareDataSource, KnowledgeBase

class SleepPredictorInterface(Interface):
    def __init__(self):
        super().__init__()
        self.health_agent = SleepHealthAgent()
        self.knowledge_base = KnowledgeBase()
        
    def setup_components(self):
        """Initialize openCHA components"""
        self.health_agent.orchestrator = Orchestrator()
        self.health_agent.data_pipe = DataPipe()
        self.health_agent.task_planner = TaskPlanner()
        self.health_agent.response_generator = ResponseGenerator()
        
        # Connect to external sources
        self.health_agent.orchestrator.register_source(
            HealthcareDataSource(source_type="sleep_health")
        )
        self.health_agent.orchestrator.register_source(self.knowledge_base)
class SleepPredictor:
    def predict(self, samsung_data: Dict, oura_data: Dict) -> float:
        """Simple sleep quality prediction"""
        base_score = 75.0
        
        # Adjust based on samsung data
        if samsung_data.get('steps', 0) > 7500:
            base_score += 5
        if 60 <= samsung_data.get('heart_rate', 70) <= 80:
            base_score += 5
            
        # Adjust based on oura data
        if oura_data.get('readiness_score', 0) > 80:
            base_score += 5
        if oura_data.get('sleep_score', 0) > 80:
            base_score += 5
            
        return min(100, max(0, base_score))

    def get_recommendations(self, sleep_score: float) -> List[str]:
        """Generate sleep recommendations based on score"""
        if sleep_score < 70:
            return [
                "Consider going to bed earlier",
                "Limit screen time before bed",
                "Create a relaxing bedtime routine"
            ]
        elif sleep_score < 85:
            return [
                "Maintain your current sleep schedule",
                "Consider small improvements to sleep hygiene"
            ]
        return ["Great sleep habits! Keep it up!"]

class PredictionRequest(BaseModel):
    samsung_data: Dict[str, Any]
    oura_data: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: float
    recommendations: List[str]

@app.post("/predict", response_model=PredictionResponse)
async def predict_sleep(request: PredictionRequest):
    try:
        predictor = SleepPredictor()
        prediction = predictor.predict(request.samsung_data, request.oura_data)
        recommendations = predictor.get_recommendations(prediction)
        
        return PredictionResponse(
            prediction=prediction,
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_streamlit_interface():
    st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")
    st.title("Sleep Quality Predictor")

    tab1, tab2, tab3 = st.tabs(["Data Input", "Predictions", "Visualizations"])
    tab4, tab5 = st.tabs(["Health Recommendations", "Chat Assistant"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Samsung Health Data")
            steps = st.number_input("Steps", value=8500, step=100)
            heart_rate = st.number_input("Heart Rate", value=72, step=1)
            sleep_duration = st.number_input("Sleep Duration (hours)", value=7.5, step=0.1)
            
            samsung_data = {
                "steps": steps,
                "heart_rate": heart_rate,
                "sleep_duration": sleep_duration
            }
            st.json(samsung_data)

        with col2:
            st.subheader("Oura Ring Data")
            readiness = st.number_input("Readiness Score", value=85, step=1)
            sleep_score = st.number_input("Sleep Score", value=82, step=1)
            activity_score = st.number_input("Activity Score", value=88, step=1)
            
            oura_data = {
                "readiness_score": readiness,
                "sleep_score": sleep_score,
                "activity_score": activity_score
            }
            st.json(oura_data)

    with tab2:
        if st.button("Generate Prediction"):
            predictor = SleepPredictor()
            prediction = predictor.predict(samsung_data, oura_data)
            recommendations = predictor.get_recommendations(prediction)
            
            # Display prediction gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sleep Quality Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig)
            
            st.subheader("Recommendations")
            for rec in recommendations:
                st.info(rec)

    with tab3:
        # Sample visualization
        dates = pd.date_range(start='2024-01-01', periods=7)
        df = pd.DataFrame({
            'Date': dates,
            'Sleep Hours': [7.5, 6.8, 8.2, 7.0, 6.5, 8.0, 7.2],
            'Sleep Quality': [85, 70, 90, 75, 65, 88, 78]
        })
        
        fig = px.line(df, x='Date', y=['Sleep Hours', 'Sleep Quality'],
                     title='Sleep Patterns Over Time')
        st.plotly_chart(fig)
    
    with tab4:
        st.subheader("Health Recommendations")
        
        # Add filters
        categories = st.multiselect(
            "Filter by category",
            ["sleep", "exercise", "diet", "lifestyle", "environment", "mental_health", "general"],
            default=["sleep", "exercise"]
        )
        
        sources = st.multiselect(
            "Filter by source",
            ["mayo_clinic", "sleep_foundation", "cdc"],
            default=["mayo_clinic", "sleep_foundation", "cdc"]
        )

        # Get and display recommendations
        recommendations = health_collector.get_health_recommendations()
        filtered_recs = [
            rec for rec in recommendations
            if rec['category'] in categories and rec['source'] in sources
        ]
        for category in categories:
            category_recs = [rec for rec in filtered_recs if rec['category'] == category]
            if category_recs:
                st.subheader(f"📌 {category.title()}")
                for rec in category_recs:
                    with st.expander(f"{rec['source'].title()} Recommendation"):
                        st.write(rec['text'])
                        st.caption(f"Source: [{rec['source']}]({rec['url']})")

    with tab5:
        st.subheader("Chat with Health Assistant")
        
        if not os.getenv('OPENAI_API_KEY'):
            st.error("Please set up your OpenAI API key to use the chat feature.")
            st.markdown("""
            To set up your API key:
            1. Get your API key from [OpenAI](https://platform.openai.com/api-keys)
            2. Create a `.env` file in your project directory
            3. Add this line to your `.env` file:
            ```
            OPENAI_API_KEY=your-api-key-here
            ```
            """)
            return
        
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Chat input
        user_input = st.text_input("Ask about your health and sleep:", key="chat_input")
        
        if st.button("Send"):
            if user_input:
                # Update assistant's context
                chat_assistant.update_health_data(
                    samsung_data=samsung_data,
                    oura_data=oura_data,
                    recommendations=recommendations
                )
                
                # Get response
                response = chat_assistant.get_response(user_input)
                
                # Add to chat history
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Assistant", response))

        # Display chat history
        for role, message in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"🧑 **You:** {message}")
            else:
                st.markdown(f"🤖 **Assistant:** {message}")

        # Suggested questions
        st.sidebar.subheader("Suggested Questions")
        questions = [
            "How can I improve my sleep quality?",
            "What exercise routine would work best for me?",
            "Should I adjust my bedtime?",
            "How does my activity level affect my sleep?",
            "What dietary changes could help my sleep?"
        ]
        
        for question in questions:
            if st.sidebar.button(question):
                chat_assistant.update_health_data(
                    samsung_data=samsung_data,
                    oura_data=oura_data,
                    recommendations=recommendations
                )
                response = chat_assistant.get_response(question)
                st.session_state.chat_history.append(("You", question))
                st.session_state.chat_history.append(("Assistant", response))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        create_streamlit_interface()