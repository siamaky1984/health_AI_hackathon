import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

import os 
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import datetime





class NIMConnector:
    def __init__(self, predictor):
        self.predictor = predictor
    
    def predict_sleep(self, samsung_data, oura_data):
        try:
            return self.predictor.predict(samsung_data, oura_data)
        except Exception as e:
            return {"error": str(e)}

class WebDataCollector:
    def __init__(self):
        self.weather_data = {
            'temperature': 20,
            'humidity': 50,
            'pressure': 1013,
            'sunset_time': '19:00'
        }
        self.health_news = ['Regular sleep schedule improves overall health']
    
    def fetch_weather_data(self, location):
        try:
            # Placeholder for actual API call
            pass
        except Exception as e:
            st.error(f"Error fetching weather data: {e}")
    
    def fetch_health_news(self):
        try:
            self.health_news = [
                "Regular sleep schedule improves overall health",
                "Exercise during the day promotes better sleep",
                "Avoid screens before bedtime"
            ]
        except Exception as e:
            st.error(f"Error fetching health news: {e}")

# def build_sleep_predictor(df_sleep):

#     awake_times, hrv_data, imu_data, ppg_data, pedometer_data,\
#     sleep_data, activity_data, readiness_data, heart_rate_data = df_sleep
    
#     class SimplePredictor:
#         def predict(self, samsung_data, oura_data):
#             return 75  # Placeholder prediction
    
#     predictor = SimplePredictor()
#     results = {}
#     combined_features = {
#         'samsung_data': {
#             'awake_times': awake_times,
#             'hrv_data': hrv_data,
#             'imu_data': imu_data,
#             'ppg_data': ppg_data,
#             'pedometer_data': pedometer_data
#         },
#         'oura_data': {
#             'sleep_data': sleep_data,
#             'activity_data': activity_data,
#             'readiness_data': readiness_data,
#             'heart_rate_data': heart_rate_data
#         }
#     }
    
#     return predictor, results, combined_features



class SleepQualityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_samsung_data(self, awake_times, hrv_data, pedometer_data):
        """Preprocess Samsung sensor data"""
        # Process awake times
        awake_df = pd.DataFrame()
        for _, row in awake_times.iterrows():
            date = pd.to_datetime(row['timestamp_start'], unit='ms').date()
            duration = (pd.to_datetime(row['timestamp_end'], unit='ms') - 
                       pd.to_datetime(row['timestamp_start'], unit='ms')).total_seconds() / 3600
            if date in awake_df.index:
                awake_df.loc[date, 'total_awake_hours'] += duration
            else:
                awake_df.loc[date, 'total_awake_hours'] = duration

        
        # Convert index to datetime
        awake_df = awake_df.reset_index()
        awake_df['date'] = pd.to_datetime(awake_df['index'])
        awake_df = awake_df.drop('index', axis=1)
                
        # Process HRV data
        hrv_data['date'] = pd.to_datetime(hrv_data['timestamp'], unit='ms').dt.date
        hrv_data['date'] = pd.to_datetime(hrv_data['date'])  # Convert to datetime

        print(hrv_data.keys())
        daily_hrv = hrv_data.groupby('date').agg({
            'hr': ['mean', 'std', 'max', 'min'],
            'hrv_sdnn': 'mean',
            'hrv_rmssd': 'mean',
            'hrv_lfhf': 'mean'
        })

        print(daily_hrv.head())
        # Flatten the multi-level columns
        daily_hrv.columns = [
            'hr_mean', 'hr_std', 'hr_max', 'hr_min',
            'hrv_sdnn_mean', 'hrv_rmssd_mean', 'hrv_lfhf_mean'
        ]
        daily_hrv = daily_hrv.reset_index()
        
        print(daily_hrv.head())

        # Process pedometer data
        pedometer_data['date'] = pd.to_datetime(pedometer_data['timestamp'], unit='ms').dt.date
        pedometer_data['date'] = pd.to_datetime(pedometer_data['date'])  # Convert to datetime

        daily_activity = pedometer_data.groupby('date').agg({
            'num_total_steps': 'max',
            'move_distance_meter': 'max',
            'cal_burn_kcal': 'max',
            'last_speed_kmh': 'mean'
        }).reset_index()

        print('awake')
        print('awake',awake_df.head())
        print( 'hrv', daily_hrv.head())
        print('activity', daily_activity.head())
        
        # Combine all Samsung features
        samsung_features = pd.merge(awake_df, daily_hrv, on='date', how='outer')
        samsung_features = pd.merge(samsung_features, daily_activity, on='date', how='outer')
        
        # Ensure date is datetime
        samsung_features['date'] = pd.to_datetime(samsung_features['date'])

        print('samsung features,')
        print(samsung_features.head())

        return samsung_features
    
    def preprocess_oura_data(self, sleep_data, activity_data, readiness_data, heart_rate_data):
        """Preprocess Oura ring data"""
        # Process sleep data
        sleep_data['date'] = pd.to_datetime(sleep_data['date'])
        sleep_features = sleep_data[['date', 'score', 'duration', 'efficiency', 'deep', 'rem', 
                                   'hr_average', 'hr_lowest', 'rmssd', 'temperature_delta']]
        
        # Process activity data
        activity_data['date'] = pd.to_datetime(activity_data['date'])
        activity_features = activity_data[['date', 'score', 'cal_active', 'cal_total', 'steps',
                                         'inactive', 'low', 'medium', 'high', 'average_met']]
        
        # Process readiness data
        readiness_data['date'] = pd.to_datetime(readiness_data['date'])
        readiness_features = readiness_data[['date', 'score', 'score_activity_balance',
                                           'score_sleep_balance', 'score_temperature']]
        
        # Combine all Oura features
        oura_features = pd.merge(sleep_features, activity_features, on='date', 
                               how='outer', suffixes=('_sleep', '_activity'))
        oura_features = pd.merge(oura_features, readiness_features, on='date', 
                               how='outer', suffixes=('', '_readiness'))
        
        # Ensure date is datetime
        oura_features['date'] = pd.to_datetime(oura_features['date'])

        print('oura features, ')
        print(oura_features.head())

        return oura_features
    
    def engineer_features(self, samsung_features, oura_features):
        """Create combined feature set with engineered features"""

        print(samsung_features.head())
        print(oura_features.head())

        # Merge Samsung and Oura data
        combined_features = pd.merge(samsung_features, oura_features, 
                                   left_on='date', right_on='date', how='outer')
        
        # Create time-based features
        combined_features['day_of_week'] = combined_features['date'].dt.dayofweek
        combined_features['is_weekend'] = combined_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Create activity intensity features
        combined_features['activity_ratio'] = (combined_features['medium'] + combined_features['high']) / \
                                            (combined_features['inactive'] + combined_features['low'] + 1e-6)
        
        # Create sleep pressure features (time since last good sleep)
        combined_features['sleep_pressure'] = combined_features['score_sleep'].shift(1)
        
        # Create cumulative fatigue features
        combined_features['rolling_activity_load'] = combined_features['cal_active'].rolling(3).mean()
        
        return combined_features
    
    def prepare_training_data(self, combined_features):
        """Prepare features and target for model training"""
        # Define target variable (next night's sleep score)
        target = combined_features['score_sleep'].shift(-1)

        print('target', target)
        
        # Select features for prediction
        feature_columns = [
            'total_awake_hours', 'hr_mean', 'hr_std', 'hr_max', 'hr_min',
            'hrv_sdnn_mean', 'hrv_rmssd_mean', 'hrv_lfhf_mean',
            'num_total_steps',
            'cal_burn_kcal', 'score_activity', 'cal_active', 'inactive',
            'medium', 'high', 'average_met', 
            'score', 'score_activity_balance', 'score_sleep_balance','score_temperature', 
            'day_of_week', 'is_weekend', 'activity_ratio',
            'sleep_pressure', 'rolling_activity_load'
        ]
        
        print('combined features', combined_features.keys())
        # Remove rows with NaN values
        features = combined_features[feature_columns].copy()
        valid_idx = ~target.isna() & ~features.isna().any(axis=1)
        
        X = features[valid_idx]
        y = target[valid_idx]
        
        return X, y
    
    def train_model(self, X, y):
        """Train the sleep quality prediction model with enhanced result tracking"""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Generate predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Get feature importance
        # feature_importance = pd.DataFrame({
        #     'feature': X.columns,
        #     'importance': self.model.feature_importances_
        # }).sort_values('importance', ascending=False)

        feature_importance = {
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }
        
        # Store test and prediction data for visualization
        self.y_test = y_test
        self.y_pred = y_pred
        
        return {
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict_sleep_quality(self, current_day_features):
        """Predict sleep quality for the current day"""
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        
        # Scale features
        scaled_features = self.scaler.transform(current_day_features)
        
        # Make prediction
        predicted_sleep_score = self.model.predict(scaled_features)
        
        return predicted_sleep_score[0]
    


class SleepQualityVisualizer:
    @staticmethod
    def plot_sleep_patterns(combined_features):
        """Visualize sleep patterns and quality over time"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Sleep Score Timeline
        plt.subplot(2, 1, 1)
        plt.plot(combined_features['date'], combined_features['score_sleep'], 
                marker='o', linestyle='-', label='Sleep Score')
        plt.fill_between(combined_features['date'], 
                        combined_features['score_sleep'] - combined_features['score_sleep'].std(),
                        combined_features['score_sleep'] + combined_features['score_sleep'].std(),
                        alpha=0.2)
        plt.title('Sleep Score Over Time')
        plt.ylabel('Sleep Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Sleep Components
        plt.subplot(2, 1, 2)
        plt.stackplot(combined_features['date'],
                     combined_features['deep'] / 60,  # Convert to hours
                     combined_features['rem'] / 60,
                    #  combined_features['light'] / 60,
                     labels=['Deep Sleep', 'REM Sleep', 'Light Sleep'])
        plt.title('Sleep Stage Distribution')
        plt.ylabel('Hours')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        # plt.show()
    
    @staticmethod
    def plot_activity_patterns(combined_features):
        """Visualize daily activity patterns"""
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Daily Steps and Activity Score
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(combined_features['date'], combined_features['steps'], 
                color='blue', label='Steps')
        ax1.set_ylabel('Steps', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(combined_features['date'], combined_features['score_activity'],
                color='red', label='Activity Score')
        ax2.set_ylabel('Activity Score', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Daily Steps and Activity Score')
        plt.xticks(rotation=45)
        
        # Plot 2: Activity Distribution
        plt.subplot(3, 1, 2)
        activity_data = pd.melt(combined_features[['date', 'inactive', 'low', 'medium', 'high']], 
                               id_vars=['date'])
        sns.boxplot(x='variable', y='value', data=activity_data)
        plt.title('Distribution of Activity Levels')
        plt.ylabel('Minutes')
        
        # Plot 3: Movement Intensity Timeline
        plt.subplot(3, 1, 3)
        plt.stackplot(combined_features['date'],
                     combined_features['inactive'],
                     combined_features['low'],
                     combined_features['medium'],
                     combined_features['high'],
                     labels=['Inactive', 'Low', 'Medium', 'High'])
        plt.title('Daily Activity Composition')
        plt.ylabel('Minutes')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        # plt.show()
    
    @staticmethod
    def plot_physiological_metrics(combined_features):
        """Visualize physiological metrics"""
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Heart Rate Metrics
        plt.subplot(3, 1, 1)
        plt.plot(combined_features['date'], combined_features['hr_mean'], 
                label='Average HR', color='red')
        plt.fill_between(combined_features['date'],
                        combined_features['hr_min'],
                        combined_features['hr_max'],
                        alpha=0.2, color='red')
        plt.title('Heart Rate Range')
        plt.ylabel('BPM')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Plot 2: HRV Metrics
        plt.subplot(3, 1, 2)
        plt.plot(combined_features['date'], combined_features['hrv_sdnn_mean'],
                label='SDNN', color='blue')
        plt.plot(combined_features['date'], combined_features['hrv_rmssd_mean'],
                label='RMSSD', color='green')
        plt.title('Heart Rate Variability Metrics')
        plt.ylabel('ms')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Plot 3: Temperature and Movement
        plt.subplot(3, 1, 3)
        plt.plot(combined_features['date'], combined_features['temperature_delta'],
                label='Temperature Deviation', color='purple')
        plt.title('Temperature Deviation from Baseline')
        plt.ylabel('°C')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        # plt.show()
    
    @staticmethod
    def plot_prediction_analysis(y_test, y_pred, feature_importance):
        """Visualize model predictions and feature importance"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Predicted vs Actual
        plt.subplot(2, 1, 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', label='Perfect Prediction')
        plt.title('Predicted vs Actual Sleep Scores')
        plt.xlabel('Actual Sleep Score')
        plt.ylabel('Predicted Sleep Score')
        plt.legend()
        
        # Plot 2: Feature Importance
        plt.subplot(2, 1, 2)
        # top_features = feature_importance.head(10)
        # sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Feature Importance')
        
        plt.tight_layout()
        # plt.show()
    
    @staticmethod
    def plot_correlation_matrix(combined_features):
        """Visualize feature correlations"""
        # Select relevant numerical columns
        numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
        correlation_matrix = combined_features[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

# Add visualization methods to the original SleepQualityPredictor class
class EnhancedSleepQualityPredictor(SleepQualityPredictor):
    def visualize_data(self, combined_features, results):
        """Generate comprehensive visualizations"""
        visualizer = SleepQualityVisualizer()
        
        print("Plotting sleep patterns...")
        visualizer.plot_sleep_patterns(combined_features)
        
        print("Plotting activity patterns...")
        visualizer.plot_activity_patterns(combined_features)
        
        print("Plotting physiological metrics...")
        visualizer.plot_physiological_metrics(combined_features)
        
        print("Plotting prediction analysis...")
        # visualizer.plot_prediction_analysis(results)
        visualizer.plot_prediction_analysis(
            results['y_test'],
            results['y_pred'],
            results['feature_importance']
        )
        
        print("Plotting correlation matrix...")
        visualizer.plot_correlation_matrix(combined_features)



def build_sleep_predictor(df_list):
    """Main function to build and train the sleep predictor"""
    # predictor = SleepQualityPredictor()

    samsung_awake, samsung_hrv, samsung_pedometer, \
        oura_sleep, oura_activity, oura_readiness, oura_heart_rate = df_list
    

    predictor = EnhancedSleepQualityPredictor()
    
    # Preprocess data
    samsung_features = predictor.preprocess_samsung_data(
        samsung_awake, samsung_hrv, samsung_pedometer
    )
    
    oura_features = predictor.preprocess_oura_data(
        oura_sleep, oura_activity, oura_readiness, oura_heart_rate
    )
    
    # Engineer features
    combined_features = predictor.engineer_features(samsung_features, oura_features)
    
    # Prepare training data
    X, y = predictor.prepare_training_data(combined_features)
    
    # Train and evaluate model
    results = predictor.train_model(X, y)

    # print('>>>>>>>>', type(results))
    # print(results.keys())


    # # Generate visualizations
    print("Generating visualizations...")
    predictor.visualize_data(combined_features, results)
    
    
    return predictor, results, combined_features
    


if __name__ == "__main__":

    dataset_source = "./ifh_affect"
    par_ID ='par_1'
    file_path_samsung = os.path.join(dataset_source, 'par_1/samsung')
    file_path_oura = os.path.join(dataset_source, 'par_1/oura')

    awake_file = os.path.join(file_path_samsung, 'awake_times.csv')
    hrv_file = os.path.join(file_path_samsung, 'hrv_1min.csv')

    # imu_file = os.path.join(file_path_samsung, 'imu.csv')
    # ppg_file = os.path.join(file_path_samsung, 'ppg.csv')

    pressure_file = os.path.join(file_path_samsung, 'pressure.csv')
    pedometer_file = os.path.join(file_path_samsung, 'pedometer.csv')


    oura_sleep_file = os.path.join(file_path_oura, 'sleep.csv')
    oura_activity_file = os.path.join(file_path_oura, 'activity.csv')
    oura_readiness_file = os.path.join(file_path_oura, 'readiness.csv')
    oura_heart_rate_file = os.path.join(file_path_oura, 'heart_rate.csv')



    # awake_file = os.path.join(file_path_samsung, 'samsung_awake_times_short.csv')
    # hrv_file = os.path.join(file_path_samsung, 'samsung_hrv_1min_short.csv')
    # imu_file = os.path.join(file_path_samsung, 'samsung_imu_short.csv')
    # pressure_file = os.path.join(file_path_samsung, 'samsung_pressure_short.csv')
    # ppg_file = os.path.join(file_path_samsung, 'samsung_ppg_short.csv')
    # pedometer_file = os.path.join(file_path_samsung, 'samsung_pedometer_short.csv')

    # oura_sleep_file = os.path.join(file_path_oura, 'oura_sleep_short.csv')
    # oura_activity_file = os.path.join(file_path_oura, 'oura_activity_short.csv')
    # oura_readiness_file = os.path.join(file_path_oura, 'oura_readiness_short.csv')
    # oura_heart_rate_file = os.path.join(file_path_oura, 'oura_heart_rate_short.csv')



    samsung_awake_data = pd.read_csv(awake_file)
    samsung_hrv_data = pd.read_csv(hrv_file)
    # samsung_imu_data = pd.read_csv(imu_file)
    # samsung_ppg_data = pd.read_csv(ppg_file)

    samsung_imu_data = []
    samsung_ppg_data = []
    samsung_pedometer_data = pd.read_csv(pedometer_file)

    oura_activity_data = pd.read_csv(oura_activity_file)
    oura_readiness_data = pd.read_csv(oura_readiness_file)
    oura_heart_rate_data = pd.read_csv(oura_heart_rate_file)
    oura_sleep_data = pd.read_csv(oura_sleep_file)


    df_list = [samsung_awake_data,
        samsung_hrv_data,
        # samsung_imu_data,
        # samsung_ppg_data,
        samsung_pedometer_data,
        oura_sleep_data,
        oura_activity_data,
        oura_readiness_data,
        oura_heart_rate_data]

    # Load your data
    predictor, results, combined_features = build_sleep_predictor(
        df_list
    )

    # Check model performance
    print(f"Model R² Score: {results['r2']}")
    print("\nMost important features:")
    print(results['feature_importance'])

    # Make predictions for new data
    ### select a test data for example another participant
    test_par = 'par_25'

    file_path_samsung_test = os.path.join(dataset_source, 'samsung')
    file_path_oura_test = os.path.join(dataset_source, 'par_1/oura')

    # tomorrow_sleep_score = predictor.predict_sleep_quality(current_day_features)


