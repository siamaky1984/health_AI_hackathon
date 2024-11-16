import streamlit as st
from sleep_predictor import NIMConnector, WebDataCollector
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def create_sleep_pattern_viz():
    dates = pd.date_range(start='2024-01-01', periods=7)
    sleep_data = {
        'Date': dates,
        'Sleep Hours': [7.5, 6.8, 8.2, 7.0, 6.5, 8.0, 7.2],
        'Sleep Quality': [85, 70, 90, 75, 65, 88, 78],
        'Deep Sleep (hrs)': [2.5, 2.0, 3.0, 2.2, 1.8, 2.8, 2.3],
        'REM Sleep (hrs)': [1.8, 1.5, 2.0, 1.7, 1.4, 1.9, 1.6],
        'Light Sleep (hrs)': [3.2, 3.3, 3.2, 3.1, 3.3, 3.3, 3.3]
    }
    return pd.DataFrame(sleep_data)

def create_heart_rate_viz():
    times = pd.date_range(start='2024-01-01', periods=24, freq='H')
    heart_rate_data = {
        'Time': times,
        'Heart Rate': [65 + 10*np.sin(x/2) + np.random.randint(-5, 5) for x in range(24)]
    }
    return pd.DataFrame(heart_rate_data)

def main():
    try:
        # Initialize data
        samsung_data = {
            'awake_times': [],
            'hrv_data': [],
            'imu_data': [],
            'ppg_data': [],
            'pedometer_data': []
        }
        
        oura_data = {
            'sleep_data': [],
            'activity_data': [],
            'readiness_data': [],
            'heart_rate_data': []
        }
        
        # Initialize predictor and connectors
        predictor, results, combined_features = build_sleep_predictor(
            samsung_data['awake_times'],
            samsung_data['hrv_data'],
            samsung_data['imu_data'],
            samsung_data['ppg_data'],
            samsung_data['pedometer_data'],
            oura_data['sleep_data'],
            oura_data['activity_data'],
            oura_data['readiness_data'],
            oura_data['heart_rate_data']
        )
        
        nim_connector = NIMConnector(predictor)
        web_collector = WebDataCollector()

        # Streamlit interface
        st.title("Sleep Quality Predictor")
        
        # Sidebar
        with st.sidebar:
            st.header("Controls")
            if st.button("Refresh Data"):
                web_collector.fetch_weather_data("your_location")
                web_collector.fetch_health_news()
                
            # Display weather and news in sidebar
            st.subheader("Weather Data")
            st.json(web_collector.weather_data)
            
            st.subheader("Health News")
            for news in web_collector.health_news:
                st.info(news)

        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["Data", "Predictions", "Visualizations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Samsung Health Data")
                st.json(samsung_data)
            with col2:
                st.subheader("Oura Ring Data")
                st.json(oura_data)
        
        with tab2:
            if st.button("Generate Prediction"):
                prediction = predictor.predict(samsung_data, oura_data)
                st.metric("Sleep Score", f"{prediction}%")
                
                # Gauge chart
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
        
        with tab3:
            viz_type = st.selectbox(
                "Select Visualization",
                ["Sleep Patterns", "Sleep Stages", "Heart Rate", "Weekly Overview"]
            )
            
            if viz_type == "Sleep Patterns":
                df = create_sleep_pattern_viz()
                fig = px.line(df, x='Date', y='Sleep Hours',
                             title='Sleep Duration Over Time')
                fig.add_scatter(x=df['Date'], y=df['Sleep Quality'],
                              name='Sleep Quality', yaxis='y2')
                fig.update_layout(
                    yaxis2=dict(
                        title='Sleep Quality',
                        overlaying='y',
                        side='right'
                    )
                )
                st.plotly_chart(fig)
                
            elif viz_type == "Sleep Stages":
                df = create_sleep_pattern_viz()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df['Date'], y=df['Deep Sleep (hrs)'],
                                   name='Deep Sleep'))
                fig.add_trace(go.Bar(x=df['Date'], y=df['REM Sleep (hrs)'],
                                   name='REM Sleep'))
                fig.add_trace(go.Bar(x=df['Date'], y=df['Light Sleep (hrs)'],
                                   name='Light Sleep'))
                fig.update_layout(
                    barmode='stack',
                    title='Sleep Stages Distribution',
                    yaxis_title='Hours'
                )
                st.plotly_chart(fig)
                
            elif viz_type == "Heart Rate":
                df = create_heart_rate_viz()
                fig = px.line(df, x='Time', y='Heart Rate',
                             title='24-Hour Heart Rate Pattern')
                fig.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Heart Rate (BPM)'
                )
                st.plotly_chart(fig)
                
            elif viz_type == "Weekly Overview":
                df = create_sleep_pattern_viz()
                fig = px.scatter(df, x='Date', y='Sleep Hours',
                               size='Sleep Quality', color='Deep Sleep (hrs)',
                               title='Weekly Sleep Overview')
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Total Sleep Hours',
                    coloraxis_colorbar_title='Deep Sleep (hrs)'
                )
                st.plotly_chart(fig)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Sleep", "7.3 hrs", "↑0.5 hrs")
            with col2:
                st.metric("Sleep Quality", "82%", "↑5%")
            with col3:
                st.metric("Deep Sleep", "2.4 hrs", "↓0.2 hrs")

    except Exception as e:
        st.error(f"Error starting application: {e}")

if __name__ == "__main__":
    main()