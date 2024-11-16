from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime,timedelta
import os
from dotenv import load_dotenv
import json
import argparse

# import cudf

from sleep_score_predict import SleepQualityPredictor

# Load environment variables
load_dotenv()

app = FastAPI()

#############################################################################
st.set_page_config(
      page_title="Health_AI_Hackathon_team57",
      page_icon=":)",
      layout="wide",
      initial_sidebar_state="expanded",
      menu_items={
          'Get Help': 'https://github.com/siamaky1984/health_AI_hackathon/',
          'About': "# Welcome to your personalized Health and Wellness Recommendation app!"
      }
  )

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
.medium-font {
    font-size:20px !important;
.small-font {
    font-size:10px !important;
}
</style>
""", unsafe_allow_html=True)

#############################################################################
#Configuring the title and sidebar for the app created by streamlit

#st.title("The Generative AI in Health Hackathon Project")
st.markdown('<p class="big-font">  </p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">  </p>', unsafe_allow_html=True)

st.sidebar.markdown('<p class="medium-font">The Generative AI in Health Hackathon Project</p> Siamak Yousefi, Hanieh Haeri, Smaneh Movassaghi </p>', unsafe_allow_html=True)


st.sidebar.markdown('<p class="small-font">This is a simple app created to analyze sleep and activity data collected\
                        from multiple participants and sensors such as Samsung watch, Oura, imu, ppg, and etc.\
                        The data source can be further studied at https://datadryad.org/stash/dataset/doi:10.7280/D1WH6T  </p> \
                    <p>This app analyses a participants sleep and activity and provides timeseries of health vitals as well as\
                    some basic statistics for these vitals.\
                    <p> The participant can further explore the app and receive mutimodal health recommendations along with\
                    references to visit as needed.</p> \
                    <p>For more information, visit our Git Hub at https://github.com/siamaky1984/health_AI_hackathon</p> \
                    Comments? <p>Email: </p>'
                    , unsafe_allow_html=True)
###########################################################################



# def load_oura_data():
#     """Load and process Oura Ring data from CSV files"""
#     all_data = []
    
#     if not os.path.exists(OURA_PATH):
#         st.error(f"Oura path not found: {OURA_PATH}")
#         return pd.DataFrame()
    
#     csv_files = []
#     for root, _, files in os.walk(OURA_PATH):
#         for file in files:
#             if file.endswith('.csv'):
#                 csv_files.append(os.path.join(root, file))
    
#     st.sidebar.write(f"Found {len(csv_files)} Oura CSV files")
    
#     # Process each CSV file
#     for file_path in csv_files:
#         try:
#             # df = pd.read_csv(file_path)
#             df = cudf.read_csv(file_path) ### use cudf 

#             st.sidebar.write(f"\nProcessing: {os.path.basename(file_path)}")
#             st.sidebar.write("Columns found:", list(df.columns))
            
#             # Map the columns to our expected format
#             column_mapping = {
#                 'Sleep Score': 'sleep_score',
#                 'Total Sleep Score': 'sleep_score',
#                 'Sleep score': 'sleep_score',
#                 'Readiness Score': 'readiness_score',
#                 'Activity Score': 'activity_score',
#                 'Deep Sleep': 'deep_sleep',
#                 'REM Sleep': 'rem_sleep',
#                 'Average HRV': 'hrv',
#                 'HRV': 'hrv',
#                 'Date': 'date',
#                 'Timestamp': 'date'
#             }


#             column_mapping={
#                 'inactive': 'inactive',
#                 'hr_average': 'hr_average',
#                 'hr_lowest':'hr_lowest',
#                 'awake': 


#             }


#             column_mapping ={
#                 'score_activity_balance': 'activity_score',

#             }

#             heart: 'heart_rate', 'heart_rmssd'
#             readiness: 'score', 'score_activity_balance', 'score_hrv_balance','score_previous_day',
#                         'score_previous_night', 'score_recovery_index','score_resting_hr', 'score_sleep_balance', 'score_temperature'
#             activity: 'date', 'score', 'score_stay_active', 'score_move_every_hour',
#                             'score_meet_daily_targets', 'score_training_frequency',
#                             'score_training_volume', 'score_recovery_time', 'cal_active',
#                             'cal_total', 'daily_movement', 'inactivity_alerts', 'steps', 'non_wear',
#                             'rest', 'inactive', 'low', 'medium', 'high', 'average_met',
#                             'met_min_inactive', 'met_min_low', 'met_min_medium', 'met_min_high',
#                             'day_start_timestamp', 'day_end_timestamp'
            
#             sleep: 'score', 'score_alignment', 'score_deep', 'score_disturbances',
#                         'score_efficiency', 'score_latency', 'score_rem', 'score_total',
#                         'duration', 'awake', 'light', 'rem', 'deep', 'total', 'onset_latency',
#                         'midpoint_time', 'efficiency', 'restless', 'hr_average', 'hr_lowest',
#                         'rmssd', 'breath_average', 'bedtime_start_midnight_delta',
#                         'bedtime_end_midnight_delta', 'temperature_delta',
#                         'bedtime_start_timestamp', 'bedtime_end_timestamp'
            
#             # Rename columns if they exist
#             df = df.rename(columns=column_mapping)

#             print('df head', file_path)
#             print(df.keys())
#             print( df.head())
            
#             all_data.append(df)
            
#         except Exception as e:
#             st.sidebar.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
    
#     if all_data:
#         combined_df = pd.concat(all_data, ignore_index=True)
#         st.sidebar.write("\nFinal columns after processing:", list(combined_df.columns))
#         return combined_df
#     else:
#         return pd.DataFrame()


# def load_samsung_data():
#     """Load and process Samsung Health data from CSV files"""
#     all_data = []
    
#     if not os.path.exists(SAMSUNG_PATH):
#         st.error(f"Samsung path not found: {SAMSUNG_PATH}")
#         return pd.DataFrame()
    
#     csv_files = []
#     for root, _, files in os.walk(SAMSUNG_PATH):
#         for file in files:
#             if file.endswith('.csv'):
#                 csv_files.append(os.path.join(root, file))
    
#     st.sidebar.write(f"Found {len(csv_files)} Samsung CSV files")
    
#     # Process each CSV file
#     for file_path in csv_files:
#         # if 'imu' in file_path:
#         #     continue
#         try:
#             # df = pd.read_csv(file_path)
#             df = cudf.read_csv(file_path)

#             st.sidebar.write(f"\nProcessing: {os.path.basename(file_path)}")
#             st.sidebar.write("Columns found:", list(df.columns))
            
#             # Map the columns to our expected format
#             column_mapping = {
#                 'Step count': 'num_total_steps',
#                 'Steps': 'num_total_steps',
#                 'Heart Rate': 'heart_rate',
#                 'Sleep Duration': 'sleep_duration',
#                 'Stress Level': 'stress_level',
#                 'Date': 'date',
#                 'Timestamp': 'date'
#             }

#             column_mapping = {
#                 'num_total_steps': 'steps',
#                 'hrv_meannn': 'heart_rate',



                 
#             }

#             awake_times: 'timestamp_start', 'timestamp_end', 'state'
#             hrv_1min:  'timestamp', 'hrv_meannn', 'hrv_sdnn', 'hrv_rmssd', 'hrv_sdsd',
#                     'hrv_cvnn', 'hrv_cvsd', 'hrv_mediannn', 'hrv_madnn', 'hrv_mcvnn',
#                     'hrv_iqrnn', 'hrv_prc80nn', 'hrv_pnn50', 'hrv_pnn20', 'hrv_minnn',
#                     'hrv_maxnn', 'hrv_tinn', 'hrv_hti', 'hrv_ulf', 'hrv_vlf', 'hrv_lf',
#                     'hrv_hf', 'hrv_vhf', 'hrv_lfhf', 'hrv_lfn', 'hrv_hfn', 'hrv_lnhf',
#                     'hrv_sd1', 'hrv_sd2', 'hrv_sd1sd2', 'hrv_s', 'hr', 'ulf', 'vlf', 'lf',
#                     'hf', 'vhf', 'lfhf', 'lfn', 'hfn', 'lnhf'
            
#             ppg: 'timestamp', 'ppg', 'hr'
#             pedometer: 'timestamp', 'num_total_steps', 'num_total_walking_steps',
#                     'num_total_running_steps', 'move_distance_meter', 'cal_burn_kcal',
#                     'last_speed_kmh', 'last_step_freq', 'last_state_level',
#                     'last_state_class' 
            
#             imu: 'timestamp', 'accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'
#             hrv_5min: 'timestamp', 'hrv_meannn', 'hrv_sdnn', 'hrv_rmssd', 'hrv_sdsd',
#                         'hrv_cvnn', 'hrv_cvsd', 'hrv_mediannn', 'hrv_madnn', 'hrv_mcvnn',
#                         'hrv_iqrnn', 'hrv_prc80nn', 'hrv_pnn50', 'hrv_pnn20', 'hrv_minnn',
#                         'hrv_maxnn', 'hrv_tinn', 'hrv_hti', 'hrv_ulf', 'hrv_vlf', 'hrv_lf',
#                         'hrv_hf', 'hrv_vhf', 'hrv_lfhf', 'hrv_lfn', 'hrv_hfn', 'hrv_lnhf',
#                         'hrv_sd1', 'hrv_sd2', 'hrv_sd1sd2', 'hrv_s', 'hr'
            
#             pressure: 'timestamp', 'pressure'

#             # Rename columns if they exist
#             df = df.rename(columns=column_mapping)

#             print('df head >>>>>>>>>', file_path)
#             print( df.keys())
#             print( df.head() )
            
#             all_data.append(df)
            
#         except Exception as e:
#             st.sidebar.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
    
#     if all_data:
#         combined_df = pd.concat(all_data, ignore_index=True)
#         st.sidebar.write("\nFinal columns after processing:", list(combined_df.columns))
#         return combined_df
#     else:
#         return pd.DataFrame()
    


def get_health_data(predictor):
    """Get and merge health data"""
    # Load data
    # oura_df = load_oura_data()
    # samsung_df = load_samsung_data()

    ## for samsung
    samsung_sensor_list = ['samsung_awake',
        'samsung_hrv',
        'samsung_imu',
        'samsung_ppg_data',
        'samsung_pedometer_data']
    
    awake_file = os.path.join(SAMSUNG_PATH, 'awake_times.csv')
    hrv_file = os.path.join(SAMSUNG_PATH, 'hrv_1min.csv')
    imu_file = os.path.join(SAMSUNG_PATH, 'imu.csv')
    ppg_file = os.path.join(SAMSUNG_PATH, 'ppg.csv')
    pressure_file = os.path.join(SAMSUNG_PATH, 'pressure.csv')
    pedometer_file = os.path.join(SAMSUNG_PATH, 'pedometer.csv')

    samsung_awake_data = pd.read_csv(awake_file)
    samsung_hrv_data = pd.read_csv(hrv_file)
    # samsung_imu_data = pd.read_csv(imu_file)
    # samsung_ppg_data = pd.read_csv(ppg_file)
    samsung_pressure_data = pd.read_csv(pressure_file)
    samsung_pedometer_data = pd.read_csv(pedometer_file)

    
    samsung_df = predictor.preprocess_samsung_data(
        samsung_awake_data, samsung_hrv_data, samsung_pedometer_data
    )
    
    print('samsung', samsung_df.keys())

    ## do it for oura
    oura_sensor_list = ['oura_sleep_data',
        'oura_activity_data',
        'oura_readiness_data',
        'oura_heart_rate_data']

    oura_sleep_file = os.path.join(OURA_PATH, 'sleep.csv')
    oura_activity_file = os.path.join(OURA_PATH, 'activity.csv')
    oura_readiness_file = os.path.join(OURA_PATH, 'readiness.csv')
    oura_heart_rate_file = os.path.join(OURA_PATH, 'heart_rate.csv')

    oura_activity_data = pd.read_csv(oura_activity_file)
    oura_readiness_data = pd.read_csv(oura_readiness_file)
    oura_heart_rate_data = pd.read_csv(oura_heart_rate_file)
    oura_sleep_data = pd.read_csv(oura_sleep_file)
    
    oura_df = predictor.preprocess_oura_data(
        oura_sleep_data, oura_activity_data, oura_readiness_data, oura_heart_rate_data
    )
    print('oura', oura_df.keys())


    combined_df = predictor.engineer_features(samsung_df, oura_df)

    print('combined', combined_df.keys())

    # Debug: Show data shapes
    st.sidebar.write("Oura data shape:", oura_df.shape if not oura_df.empty else "Empty")
    st.sidebar.write("Samsung data shape:", samsung_df.shape if not samsung_df.empty else "Empty")
    
    # Convert dates to datetime if needed
    if not oura_df.empty:
        date_column = [col for col in oura_df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
        oura_df['date'] = pd.to_datetime(oura_df[date_column])
    
    if not samsung_df.empty:
        date_column = [col for col in samsung_df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
        samsung_df['date'] = pd.to_datetime(samsung_df[date_column])
    
    ### TODO 
    # Merge data if both dataframes have data
    if not oura_df.empty and not samsung_df.empty:
        merged_df = pd.merge(oura_df, samsung_df, on='date', how='outer')
        merged_df = merged_df.sort_values('date')
        st.sidebar.write("Merged data shape:", merged_df.shape)
    else:
        merged_df = pd.DataFrame()
        if oura_df.empty:
            st.warning("No Oura data found")
        if samsung_df.empty:
            st.warning("No Samsung data found")


    return {
        'oura_data': {
            'sleep_score': np.mean(oura_df['score_sleep']), # if not oura_df.empty and 'sleep_score' in oura_df.columns else 75,
            'readiness_score': np.mean(oura_df['score_temperature']), # if not oura_df.empty and 'readiness_score' in oura_df.columns else 80,
            'activity_score': np.mean(oura_df['score_activity']), # if not oura_df.empty and 'activity_score' in oura_df.columns else 70,
            'deep_sleep': np.mean(oura_df['score_sleep_balance']), # if not oura_df.empty and 'deep_sleep' in oura_df.columns else 2,
            'rem_sleep': np.mean(oura_df['rem']), # if not oura_df.empty and 'rem_sleep' in oura_df.columns else 1.5,
            'hrv': np.mean(oura_df['hr_average']), # if not oura_df.empty and 'hrv' in oura_df.columns else 50
        },
        'samsung_data': {
            'steps': np.mean(samsung_df['num_total_steps']), # if not samsung_df.empty and 'steps' in samsung_df.columns else 8000,
            'heart_rate': np.mean(samsung_df['hr_mean']), # if not samsung_df.empty and 'heart_rate' in samsung_df.columns else 70,

            'sleep_duration': np.mean(samsung_df['total_awake_hours'] ), # if not samsung_df.empty and 'sleep_duration' in samsung_df.columns else 7,
            # 'stress_level': np.mean( samsung_df['stress_level'] ), # if not samsung_df.empty and 'stress_level' in samsung_df.columns else 30
            'cal_burn_kcal' : np.mean(samsung_df['cal_burn_kcal']),
  
        },
        'merged_df': merged_df
    }
    
    # # Create return dictionary with default values if needed
    # return {
    #     'oura_data': {
    #         'sleep_score': oura_df['sleep_score'].iloc[-1] if not oura_df.empty and 'sleep_score' in oura_df.columns else 75,
    #         'readiness_score': oura_df['readiness_score'].iloc[-1] if not oura_df.empty and 'readiness_score' in oura_df.columns else 80,
    #         'activity_score': oura_df['activity_score'].iloc[-1] if not oura_df.empty and 'activity_score' in oura_df.columns else 70,
    #         'deep_sleep': oura_df['deep_sleep'].iloc[-1] if not oura_df.empty and 'deep_sleep' in oura_df.columns else 2,
    #         'rem_sleep': oura_df['rem_sleep'].iloc[-1] if not oura_df.empty and 'rem_sleep' in oura_df.columns else 1.5,
    #         'hrv': oura_df['hrv'].iloc[-1] if not oura_df.empty and 'hrv' in oura_df.columns else 50
    #     },
    #     'samsung_data': {
    #         'steps': samsung_df['steps'].iloc[-1] if not samsung_df.empty and 'steps' in samsung_df.columns else 8000,
    #         'heart_rate': samsung_df['heart_rate'].iloc[-1] if not samsung_df.empty and 'heart_rate' in samsung_df.columns else 70,
    #         'sleep_duration': samsung_df['sleep_duration'].iloc[-1] if not samsung_df.empty and 'sleep_duration' in samsung_df.columns else 7,
    #         'stress_level': samsung_df['stress_level'].iloc[-1] if not samsung_df.empty and 'stress_level' in samsung_df.columns else 30
    #     },
    #     'merged_df': merged_df
    # }



def check_data_structure(file_path):
    """Debug function to check JSON structure"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            st.sidebar.write(f"Data structure for {os.path.basename(file_path)}:")
            st.sidebar.json(data)
    except Exception as e:
        st.sidebar.error(f"Error reading file: {str(e)}")



        
class HealthAgent:
    def __init__(self):
        self.external_sources = {
            "Mayo Clinic": "https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/sleep/art-20048379",
            "Sleep Foundation": "https://www.sleepfoundation.org/sleep-hygiene",
            "CDC": "https://www.cdc.gov/sleep/about_sleep/sleep_hygiene.html"
        }
        
        self.recommendations = {
            'Mayo Clinic': {
                'sleep': [
                    "Stick to a sleep schedule",
                    "Pay attention to what you eat and drink",
                    "Create a restful environment",
                    "Limit daytime naps",
                    "Include physical activity in your daily routine"
                ],
                # 'stress': [
                #     "Regular exercise",
                #     "Relaxation techniques",
                #     "Maintain social connections",
                #     "Set realistic goals",
                #     "Practice stress management"
                # ]
            },
            'Sleep Foundation': {
                'sleep': [
                    "Optimize your bedroom environment",
                    "Follow a consistent pre-bed routine",
                    "Use comfortable bedding",
                    "Avoid bright lights before bedtime",
                    "Practice relaxation techniques"
                ],
                'exercise': [
                    "Exercise regularly, but not too close to bedtime",
                    "Get exposure to natural daylight",
                    "Stay active throughout the day",
                    "Balance intensity of workouts",
                    "Include both cardio and strength training"
                ]
            },
            'CDC': {
                'sleep': [
                    "Be consistent with sleep schedule",
                    "Make sure bedroom is quiet, dark, and relaxing",
                    "Remove electronic devices from bedroom",
                    "Avoid large meals before bedtime",
                    "Get enough natural light during the day"
                ],
                'health': [
                    "Maintain a healthy diet",
                    "Stay physically active",
                    "Manage chronic conditions",
                    "Practice good sleep hygiene",
                    "Regular health check-ups"
                ]
            }
        }

        self.sleep_recommendations = {
            'poor': [
                "Establish a consistent sleep schedule",
                "Create a relaxing bedtime routine",
                "Limit screen time before bed",
                "Ensure your bedroom is dark and cool",
                "Consider meditation or deep breathing exercises"
            ],
            'moderate': [
                "Maintain your current sleep schedule",
                "Try to increase deep sleep phases",
                "Monitor caffeine intake after noon",
                "Get regular exercise, but not too close to bedtime"
            ],
            'good': [
                "Keep up your excellent sleep habits",
                "Fine-tune your sleep environment",
                "Continue monitoring your sleep patterns"
            ]
        }
        
        self.activity_recommendations = {
            'low': [
                "Start with short walks throughout the day",
                "Try gentle stretching exercises",
                "Set achievable daily step goals",
                "Consider low-impact activities like swimming",
                "Gradually increase activity duration"
            ],
            'moderate': [
                "Mix cardio and strength training",
                "Aim for 150 minutes of moderate activity per week",
                "Include flexibility exercises",
                "Try interval training",
                "Join group fitness classes"
            ],
            'high': [
                "Maintain your excellent activity level",
                "Ensure proper recovery between workouts",
                "Mix up your routine to prevent plateaus",
                "Consider training for an event",
                "Focus on form and technique"
            ]
        }
        
        self.suggested_questions = [
            "How can I improve my deep sleep?",
            "What's the ideal sleep schedule for my age?",
            # "How does stress affect my sleep quality?",
            "What's the relationship between exercise and sleep?",
            "How can I establish a better bedtime routine?",
            "What foods should I avoid before bedtime?",
            "How does screen time affect my sleep?",
            "What's the optimal bedroom temperature for sleep?",
            "How can I reduce sleep anxiety?",
            "Should I try meditation for better sleep?"
        ]

    def get_chat_response(self, prompt):
        """Generate response with recommendations from multiple sources"""
        response = "Here's what I found from multiple trusted sources:\n\n"
        
        # Add recommendations based on keywords
        keywords = {
            'sleep': ['sleep', 'bed', 'rest', 'nap', 'insomnia'],
            # 'stress': ['stress', 'anxiety', 'worried', 'tension'],
            'exercise': ['exercise', 'activity', 'workout', 'fitness'],
            'health': ['health', 'wellness', 'lifestyle', 'habits']
        }
        
        # Find matching categories based on prompt
        matching_categories = []
        for category, terms in keywords.items():
            if any(term in prompt.lower() for term in terms):
                matching_categories.append(category)
        
        # If no matches, default to sleep category
        if not matching_categories:
            matching_categories = ['sleep']
        
        # Add recommendations from each source
        for source, recommendations in self.recommendations.items():
            response += f"\n**{source} Recommendations:**\n"
            for category in matching_categories:
                if category in recommendations:
                    response += "\n".join([f"- {rec}" for rec in recommendations[category][:3]])
                    response += "\n"
        
        # Add references
        response += "\n**References:**\n"
        for source, url in self.external_sources.items():
            response += f"- [{source}]({url})\n"
        
        # Add suggested follow-up questions
        response += "\n**You might also want to ask:**\n"
        relevant_questions = [q for q in self.suggested_questions 
                            if any(cat in q.lower() for cat in matching_categories)]
        response += "\n".join([f"- {q}" for q in relevant_questions[:3]])
        
        return response

    def analyze_data(self, samsung_data, oura_data):
        """Analyze health data and generate insights"""
        analysis = {
            'sleep_quality': self._analyze_sleep(samsung_data, oura_data),
            'activity_level': self._analyze_activity(samsung_data, oura_data),
            'overall_health': self._analyze_overall_health(samsung_data, oura_data),
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis

    def _analyze_sleep(self, samsung_data, oura_data):
        """Analyze sleep metrics"""
        sleep_score = oura_data.get('sleep_score', 0)
        sleep_duration = samsung_data.get('sleep_duration', 0)
        
        quality_assessment = {
            'score': sleep_score,
            'duration': sleep_duration,
            'quality': 'poor' if sleep_score < 70 else 'moderate' if sleep_score < 85 else 'good',
            'issues': []
        }
        
        # Identify potential issues
        if sleep_duration < 7:
            quality_assessment['issues'].append("Insufficient sleep duration")
        if sleep_score < 70:
            quality_assessment['issues'].append("Low sleep quality")
        
        return quality_assessment

    def _analyze_activity(self, samsung_data, oura_data):
        """Analyze activity metrics"""
        steps = samsung_data.get('steps', 0)
        activity_score = oura_data.get('activity_score', 0)
        
        activity_level = {
            'steps': steps,
            'score': activity_score,
            'level': 'low' if steps < 5000 else 'moderate' if steps < 10000 else 'high',
            'recommendations': []
        }
        
        # Add specific recommendations
        if steps < 5000:
            activity_level['recommendations'].extend(self.activity_recommendations['low'])
        elif steps < 10000:
            activity_level['recommendations'].extend(self.activity_recommendations['moderate'])
        else:
            activity_level['recommendations'].extend(self.activity_recommendations['high'])
        
        return activity_level

    def _analyze_overall_health(self, samsung_data, oura_data):
        """Analyze overall health status"""
        readiness_score = oura_data.get('readiness_score', 0)
        heart_rate = samsung_data.get('heart_rate', 0)
        # stress_level = samsung_data.get('stress_level', 0)
        
        
        return {
            'readiness': readiness_score,
            'heart_rate': heart_rate,
            # 'stress_level': stress_level,
            'status': 'good' if readiness_score > 80 else 'moderate' if readiness_score > 60 else 'poor'
        }

    def _generate_recommendations(self, analysis):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Sleep recommendations
        sleep_quality = analysis['sleep_quality']['quality']
        recommendations.extend(self.sleep_recommendations[sleep_quality])
        
        # Activity recommendations
        activity_level = analysis['activity_level']['level']
        recommendations.extend(self.activity_recommendations[activity_level])
        
        return recommendations



# Update the create_streamlit_interface function to use HealthAgent
def create_streamlit_interface(sleep_quality_predictor):
    st.title("Sleep Quality Predictor")
    
    # Initialize HealthAgent
    health_agent = HealthAgent()
    
    # Load health data
    health_data = get_health_data(sleep_quality_predictor)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Input", "Predictions", "Visualizations", 
        "Health Recommendations", "Chat Assistant"
    ])
    
    # Data Input Tab (Tab 1)
    with tab1:
        st.header("Your Health Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Samsung Health Data")
            samsung_data = health_data['samsung_data']
            
            steps = st.number_input("Steps", 
                value=int(samsung_data['steps']), 
                min_value=0,
                max_value=50000,
                step=100,
                help="Daily step count")
            
            heart_rate = st.number_input("Heart Rate (bpm)", 
                value=int(samsung_data['heart_rate']),
                min_value=40,
                max_value=200, 
                step=1,
                help="Average heart rate")
            
            sleep_duration = st.number_input("Sleep Duration (hours)", 
                value=float(samsung_data['sleep_duration']),
                min_value=0.0,
                max_value=24.0, 
                step=0.1,
                help="Total sleep duration")
            
            # stress = st.number_input("Stress Level", 
            #     value=int(samsung_data['stress_level']),
            #     min_value=0,
            #     max_value=100, 
            #     step=1,
            #     help="Average stress level (0-100)")
            
            # # Display current values
            st.info("Current Samsung Health Metrics")
            st.metric("Daily Steps", f"{steps:,}")
            st.metric("Average Heart Rate", f"{heart_rate} bpm")
            st.metric("Sleep Duration", f"{sleep_duration:.1f} hours")
            # st.metric("Stress Level", f"{stress}/100")
        
        with col2:
            st.subheader("Oura Ring Data")
            oura_data = health_data['oura_data']
            
            sleep_score = st.number_input("Sleep Score", 
                value=int(oura_data['sleep_score']),
                min_value=0,
                max_value=100, 
                step=1,
                help="Overall sleep quality score")
            
            readiness = st.number_input("Readiness Score", 
                value=int(oura_data['readiness_score']),
                min_value=0,
                max_value=100, 
                step=1,
                help="Daily readiness score")
            
            activity_score = st.number_input("Activity Score", 
                value=int(oura_data['activity_score']),
                min_value=0,
                max_value=100, 
                step=1,
                help="Daily activity score")
            
            hrv = st.number_input("HRV", 
                value=int(oura_data['hrv']),
                min_value=0,
                max_value=200, 
                step=1,
                help="Average HRV")
            
            # Display current values
            st.info("Current Oura Ring Metrics")
            st.metric("Sleep Quality", f"{sleep_score}/100")
            st.metric("Readiness", f"{readiness}/100")
            st.metric("Activity", f"{activity_score}/100")
            st.metric("HRV", f"{hrv} ms")

    # Predictions Tab (Tab 2)
    with tab2:
        st.header("Sleep Quality Predictions")
        
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            st.subheader("Tonight's Sleep Prediction")
            
            # Calculate predicted sleep score
            predicted_score = (
                sleep_score * 0.4 +
                readiness * 0.3 +
                # (100 - stress) * 0.2 +
                (activity_score * 0.1)
            )
            
            # Display prediction gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Sleep Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig)
        
        with pred_col2:
            st.subheader("Sleep Factors Impact")
            
            # Impact analysis
            impact_factors = {
                # "Stress Level": -stress * 0.5,
                "Physical Activity": activity_score * 0.3,
                "Previous Sleep": sleep_score * 0.4,
                "Heart Rate": -abs(heart_rate - 70) * 0.2
            }
            
            # Impact chart
            impact_df = pd.DataFrame({
                'Factor': list(impact_factors.keys()),
                'Impact': list(impact_factors.values())
            })
            
            fig = px.bar(impact_df, x='Impact', y='Factor',
                        orientation='h',
                        title='Sleep Quality Impact Factors')
            st.plotly_chart(fig)

    # Visualizations Tab (Tab 3)
    with tab3:
        st.header("Health Data Visualizations")
        
        df = health_data['merged_df']
        
        if not df.empty:
            # Get available columns for visualization
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Let user select which metric to visualize
            metric_option = st.selectbox(
                "Select Metric to Visualize",
                numeric_columns
            )
            
            # Create time series plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[metric_option],
                mode='lines+markers',
                name=metric_option
            ))
            
            fig.update_layout(
                title=f'{metric_option} Over Time',
                xaxis_title='Date',
                yaxis_title=metric_option
            )
            st.plotly_chart(fig)
            
            # Show statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Average", f"{df[metric_option].mean():.2f}")
            col2.metric("Minimum", f"{df[metric_option].min():.2f}")
            col3.metric("Maximum", f"{df[metric_option].max():.2f}")
            
            # Correlation heatmap
            st.subheader("Correlations")
            corr_matrix = df[numeric_columns].corr()
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig)
        else:
            st.warning("No data available for visualization")
            st.write("Available columns in the data:")
            st.write(list(df.columns) if not df.empty else "No columns available")


    
    with tab4:
        st.subheader("Your Health Insights")
        
        # Get analysis from HealthAgent
        analysis = health_agent.analyze_data(
            health_data['samsung_data'],
            health_data['oura_data']
        )
        
        # Display insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sleep Analysis")
            st.write(f"Quality: {analysis['sleep_quality']['quality'].title()}")
            st.write(f"Score: {analysis['sleep_quality']['score']}")
            
            if analysis['sleep_quality']['issues']:
                st.warning("Areas for Improvement:")
                for issue in analysis['sleep_quality']['issues']:
                    st.write(f"• {issue}")
        
        with col2:
            st.markdown("### Activity Analysis")
            st.write(f"Level: {analysis['activity_level']['level'].title()}")
            st.write(f"Steps: {analysis['activity_level']['steps']:,}")
        
        st.markdown("### Personalized Recommendations")
        for i, rec in enumerate(analysis['recommendations'], 1):
            st.write(f"{i}. {rec}")
    
    # Chat Assistant Tab
    with tab5:
        st.subheader("Chat with Your Health Assistant")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about your sleep and activity..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            response = health_agent.get_chat_response(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sensor path')
    parser.add_argument('--data_path', type=str, default='ifh_affect', help='Mode of operation: train or generate')
    parser.add_argument('--par_ID', type=int, default=1)

    args = parser.parse_args()

    OURA_PATH = args.data_path + '/par_' + str(args.par_ID) + '/oura/'
    SAMSUNG_PATH = args.data_path+ '/par_' + str( args.par_ID ) + '/samsung/'

    print(OURA_PATH)
    print(SAMSUNG_PATH)
    

    # Load sleepQualityPredictor class
    sleep_quality_predictor = SleepQualityPredictor() 

    #st.sidebar.markdown("# Data Loading Debug")
    
    # Check basic path existence
    # st.sidebar.markdown("## Path Check")
    # st.sidebar.write(f"Oura path exists: {os.path.exists(OURA_PATH)}")
    # st.sidebar.write(f"Samsung path exists: {os.path.exists(SAMSUNG_PATH)}")
    
    # # Show directory contents
    # st.sidebar.markdown("## Directory Contents")
    # if os.path.exists(OURA_PATH):
    #     st.sidebar.write("Oura directory contents:")
    #     st.sidebar.write([f for f in os.listdir(OURA_PATH) if f.endswith('.csv')])
    
    # if os.path.exists(SAMSUNG_PATH):
    #     st.sidebar.write("Samsung directory contents:")
    #     st.sidebar.write([f for f in os.listdir(SAMSUNG_PATH) if f.endswith('.csv')])
    
    create_streamlit_interface(sleep_quality_predictor)
