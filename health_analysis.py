import os 
import sys
import numpy as np
import pandas as pd

from ifh_affect import DataLoader, DataTransform


import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Replace with the actual path to your dataset folder or zip file
dataset_source = "./ifh_affect"

# Create a DataLoader instance
data_loader = DataLoader(dataset_source)

participant_id = "par_1"

# ema_daily_df = data_loader.load_single_df(modality="ema", file="daily", participant=participant_id)
# print("EMA Daily Data for par_1:")
# print(ema_daily_df.head())

# ema_head = ema_daily_df.head()

# csv_file_name= "ema_daily_short.csv"
# path = os.path.join( dataset_source,  participant_id + '/ema' ) 
# ema_head.to_csv( os.path.join(path,csv_file_name), index=False)




# file_list = [ 'imu', 'ppg', 'pedometer']
# for file_i in file_list:
#     ## load samsung 
#     samsung_daily_df = data_loader.load_single_df(modality="samsung", file=file_i, participant=participant_id)
#     print("Samsung data for par_1:", file_i)
#     print(samsung_daily_df.head())

#     samsung_df_head = samsung_daily_df.head()

#     ## save to csv
#     csv_file_name = f"samsung_{file_i}_short.csv"
#     path = os.path.join( dataset_source, participant_id+ '/samsung' ) 
#     samsung_df_head.to_csv( os.path.join(path,csv_file_name), index=False)




# # Example 2: Load 'heart_rate' data for all participants
# heart_rate_df = data_loader.load_df(modality="oura", file="heart_rate")
# print("\nHeart Rate Data for All Participants:")
# print(heart_rate_df.head())

# file_list = ['activity', 'activity_level', 'heart_rate', 'readiness', 'sleep_hypnogram', 'sleep']
# for file_i in file_list:
#     print(file_i)
#     oura_df = data_loader.load_single_df(modality="oura", file=file_i, participant= participant_id)
#     print("Oura data for par_1:", file_i)
#     print(oura_df.head())

#     oura_df_head = oura_df.head()   
    
#     csv_file_name = f"oura_{file_i}_short.csv"
#     path = os.path.join(dataset_source, participant_id + '/oura')
#     oura_df_head.to_csv(os.path.join(path, csv_file_name), index=False)




# # Example 3: Load 'sleep' data for specific participants with participant IDs added
# sleep_df = data_loader.load_df(
#     modality="samsung", file="pedometer", participants=["par_3", "par_7", "par_12"], add_id=True
# )
# print("\nSamsung Pedometer Data for Selected Participants:")
# print(sleep_df.head())




class ComprehensiveHealthAnalyzer:
    def __init__(self):
        # Previous sensor data
        self.awake_data = None
        self.hrv_data = None
        self.imu_data = None
        self.pressure_data = None
        self.ppg_data = None
        self.pedometer_data = None
        
        # Oura ring data
        self.oura_sleep = None
        self.oura_activity = None
        self.oura_readiness = None
        self.oura_heart_rate = None
        self.oura_hypnogram = None
        
    def load_oura_data(self, sleep_file, activity_file, readiness_file, 
                      heart_rate_file, hypnogram_file):
        """Load Oura ring data"""
        self.oura_sleep = pd.read_csv(sleep_file)
        self.oura_sleep['date'] = pd.to_datetime(self.oura_sleep['date'])
        self.oura_sleep['bedtime_start'] = pd.to_datetime(self.oura_sleep['bedtime_start_timestamp'], unit='ms')
        self.oura_sleep['bedtime_end'] = pd.to_datetime(self.oura_sleep['bedtime_end_timestamp'], unit='ms')
        
        self.oura_activity = pd.read_csv(activity_file)
        print(self.oura_activity.head())
        self.oura_activity['date'] = pd.to_datetime(self.oura_activity['date'])
        
        self.oura_readiness = pd.read_csv(readiness_file)
        self.oura_readiness['date'] = pd.to_datetime(self.oura_readiness['date'])
        
        self.oura_heart_rate = pd.read_csv(heart_rate_file)
        self.oura_heart_rate['timestamp'] = pd.to_datetime(self.oura_heart_rate['timestamp'], unit='ms')
        
        self.oura_hypnogram = pd.read_csv(hypnogram_file)
        self.oura_hypnogram['timestamp'] = pd.to_datetime(self.oura_hypnogram['timestamp'], unit='ms')
    
    def analyze_sleep_architecture(self):
        """Analyze detailed sleep patterns and quality"""
        sleep_analysis = {
            'averages': {
                'total_sleep_duration': self.oura_sleep['duration'].mean() / 3600,  # Convert to hours
                'sleep_efficiency': self.oura_sleep['efficiency'].mean(),
                'deep_sleep_percentage': (self.oura_sleep['deep'].mean() / self.oura_sleep['duration'].mean()) * 100,
                'rem_sleep_percentage': (self.oura_sleep['rem'].mean() / self.oura_sleep['duration'].mean()) * 100,
                'sleep_score': self.oura_sleep['score'].mean(),
                'hr_during_sleep': self.oura_sleep['hr_average'].mean(),
                'hrv_during_sleep': self.oura_sleep['rmssd'].mean()
            },
            'sleep_timing': {
                'average_bedtime': self.oura_sleep['bedtime_start_midnight_delta'].mean() / 3600,  # Convert to hours
                'average_wake_time': self.oura_sleep['bedtime_end_midnight_delta'].mean() / 3600,
                'sleep_midpoint': self.oura_sleep['midpoint_time'].mean() / 3600
            }
        }
        return sleep_analysis
    
    def analyze_daily_activity(self):
        """Analyze activity patterns and intensity levels"""
        activity_analysis = {
            'averages': {
                'daily_steps': self.oura_activity['steps'].mean(),
                'active_calories': self.oura_activity['cal_active'].mean(),
                'total_calories': self.oura_activity['cal_total'].mean(),
                'activity_score': self.oura_activity['score'].mean(),
                'movement_consistency': self.oura_activity['score_move_every_hour'].mean()
            },
            'activity_distribution': {
                'inactive_time': self.oura_activity['inactive'].mean(),
                'low_activity_time': self.oura_activity['low'].mean(),
                'medium_activity_time': self.oura_activity['medium'].mean(),
                'high_activity_time': self.oura_activity['high'].mean()
            },
            'metabolic_metrics': {
                'average_met': self.oura_activity['average_met'].mean(),
                'total_daily_movement': self.oura_activity['daily_movement'].mean()
            }
        }
        return activity_analysis
    
    def analyze_readiness_trends(self):
        """Analyze overall readiness and recovery patterns"""
        readiness_analysis = {
            'overall_readiness': self.oura_readiness['score'].mean(),
            'component_scores': {
                'activity_balance': self.oura_readiness['score_activity_balance'].mean(),
                'sleep_balance': self.oura_readiness['score_sleep_balance'].mean(),
                'resting_hr': self.oura_readiness['score_resting_hr'].mean(),
                'temperature': self.oura_readiness['score_temperature'].mean()
            }
        }
        return readiness_analysis
    
    def plot_comprehensive_patterns(self):
        """Create comprehensive visualizations of all health metrics"""
        plt.figure(figsize=(10, 12))
        
        # Plot 1: Sleep Architecture
        plt.subplot(4, 1, 1)
        sleep_data = pd.melt(self.oura_sleep[['date', 'deep', 'rem', 'light', 'awake']], 
                            id_vars=['date'])
        sns.barplot(data=sleep_data, x='date', y='value', hue='variable')
        plt.title('Sleep Architecture Distribution')
        plt.xticks(rotation=45)
        
        # Plot 2: Activity Distribution
        plt.subplot(4, 1, 2)
        activity_data = pd.melt(self.oura_activity[['date', 'inactive', 'low', 'medium', 'high']], 
                               id_vars=['date'])
        sns.barplot(data=activity_data, x='date', y='value', hue='variable')
        plt.title('Daily Activity Distribution')
        plt.xticks(rotation=45)
        
        # Plot 3: Heart Rate Trends
        plt.subplot(4, 1, 3)
        plt.plot(self.oura_heart_rate['timestamp'], self.oura_heart_rate['heart_rate'], 
                label='Heart Rate')
        plt.plot(self.oura_heart_rate['timestamp'], self.oura_heart_rate['heart_rmssd'], 
                label='HRV (RMSSD)')
        plt.title('Heart Rate and HRV Trends')
        plt.legend()
        
        # Plot 4: Readiness Scores
        plt.subplot(4, 1, 4)
        plt.plot(self.oura_readiness['date'], self.oura_readiness['score'], 
                marker='o', label='Overall Readiness')
        plt.title('Daily Readiness Score')
        plt.xticks(rotation=45)
        



        plt.tight_layout()
        plt.show()
    
    def generate_health_summary(self):
        """Generate comprehensive health and wellness summary"""
        sleep = self.analyze_sleep_architecture()
        activity = self.analyze_daily_activity()
        readiness = self.analyze_readiness_trends()
        
        return {
            'sleep_quality': sleep,
            'activity_patterns': activity,
            'readiness_metrics': readiness,
            'key_insights': {
                'average_sleep_score': sleep['averages']['sleep_score'],
                'average_activity_score': activity['averages']['activity_score'],
                'average_readiness': readiness['overall_readiness'],
                'daily_steps': activity['averages']['daily_steps'],
                'sleep_efficiency': sleep['averages']['sleep_efficiency']
            }
        }

def analyze_health_data(oura_file_path):
    
    oura_sleep_file= os.path.join( oura_file_path, 'oura_sleep_short.csv')
    oura_activity_file= os.path.join( oura_file_path, 'oura_activity_short.csv')
    oura_readiness_file= os.path.join( oura_file_path, 'oura_readiness_short.csv')
    oura_heart_rate_file= os.path.join( oura_file_path, 'oura_heart_rate_short.csv')
    oura_hypnogram_file= os.path.join( oura_file_path, 'oura_sleep_hypnogram_short.csv')
    oura_activity_level_file= os.path.join( oura_file_path, 'oura_activity_level_short.csv')


    # oura_sleep_file= os.path.join( oura_file_path, 'sleep.csv')
    # oura_activity_file= os.path.join( oura_file_path, 'activity.csv')
    # oura_readiness_file= os.path.join( oura_file_path, 'readiness.csv')
    # oura_heart_rate_file= os.path.join( oura_file_path, 'heart_rate.csv')
    # oura_hypnogram_file = os.path.join( oura_file_path, 'sleep_hypnogram.csv')
    # oura_activity_level_file= os.path.join( oura_file_path, 'activity_level.csv')

    
    """Main function to analyze all health data"""
    analyzer = ComprehensiveHealthAnalyzer()
    analyzer.load_oura_data(oura_sleep_file, oura_activity_file, oura_readiness_file,
                           oura_heart_rate_file, oura_hypnogram_file)
    
    summary = analyzer.generate_health_summary()
    analyzer.plot_comprehensive_patterns()
    
    return summary




class EnhancedSensorAnalyzer:
    def __init__(self):
        self.awake_data = None
        self.hrv_data = None
        self.imu_data = None
        self.pressure_data = None
        self.ppg_data = None
        self.pedometer_data = None
        
    def load_data(self, awake_file, hrv_file, imu_file, pressure_file, ppg_file, pedometer_file):
        """Load and preprocess all sensor data"""
        # Load previous sensor data
        self.awake_data = pd.read_csv(awake_file)
        self.awake_data['timestamp_start'] = pd.to_datetime(self.awake_data['timestamp_start'], unit='ms')
        self.awake_data['timestamp_end'] = pd.to_datetime(self.awake_data['timestamp_end'], unit='ms')
        
        self.hrv_data = pd.read_csv(hrv_file)
        self.hrv_data['timestamp'] = pd.to_datetime(self.hrv_data['timestamp'], unit='ms')
        
        self.imu_data = pd.read_csv(imu_file)
        self.imu_data['timestamp'] = pd.to_datetime(self.imu_data['timestamp'], unit='ms')
        
        self.pressure_data = pd.read_csv(pressure_file)
        self.pressure_data['timestamp'] = pd.to_datetime(self.pressure_data['timestamp'], unit='ms')
        
        # Load new sensor data
        self.ppg_data = pd.read_csv(ppg_file)
        self.ppg_data['timestamp'] = pd.to_datetime(self.ppg_data['timestamp'], unit='ms')
        
        self.pedometer_data = pd.read_csv(pedometer_file)
        self.pedometer_data['timestamp'] = pd.to_datetime(self.pedometer_data['timestamp'], unit='ms')
        
    def analyze_physical_activity(self):
        """Analyze physical activity patterns from pedometer data"""
        activity_analysis = {
            'total_steps': self.pedometer_data['num_total_steps'].max(),
            'walking_steps': self.pedometer_data['num_total_walking_steps'].max(),
            'running_steps': self.pedometer_data['num_total_running_steps'].max(),
            'total_distance': self.pedometer_data['move_distance_meter'].max(),
            'calories_burned': self.pedometer_data['cal_burn_kcal'].max(),
            'activity_states': self.pedometer_data['last_state_class'].value_counts().to_dict()
        }
        
        # Calculate average speeds during active periods
        active_periods = self.pedometer_data[self.pedometer_data['last_speed_kmh'] > 0]
        if not active_periods.empty:
            activity_analysis['average_speed'] = active_periods['last_speed_kmh'].mean()
            activity_analysis['average_step_freq'] = active_periods['last_step_freq'].mean()
        
        return activity_analysis
    
    def analyze_cardiovascular_metrics(self):
        """Analyze cardiovascular health metrics from PPG and HRV data"""
        # Combine PPG and HRV data for comprehensive heart analysis
        ppg_analysis = {
            'mean_ppg': self.ppg_data['ppg'].mean(),
            'ppg_variability': self.ppg_data['ppg'].std(),
            'hr_from_ppg': self.ppg_data[self.ppg_data['hr'] != 0]['hr'].mean()
        }
        
        hrv_analysis = {
            'mean_hr': self.hrv_data['hr'].mean(),
            'mean_hrv_sdnn': self.hrv_data['hrv_sdnn'].mean(),
            'mean_hrv_rmssd': self.hrv_data['hrv_rmssd'].mean(),
            'stress_indicator': self.hrv_data['hrv_lfhf'].mean()  # LF/HF ratio as stress indicator
        }
        
        return {'ppg_metrics': ppg_analysis, 'hrv_metrics': hrv_analysis}
    
    def analyze_environmental_context(self):
        """Analyze environmental conditions from pressure data"""
        return {
            'mean_pressure': self.pressure_data['pressure'].mean(),
            'pressure_variability': self.pressure_data['pressure'].std(),
            'pressure_range': self.pressure_data['pressure'].max() - self.pressure_data['pressure'].min()
        }
    
    def plot_comprehensive_patterns(self):
        """Create comprehensive visualizations of all sensor data"""
        plt.figure(figsize=(10, 12))
        
        # Plot 1: Physical Activity
        plt.subplot(5, 1, 1)
        plt.plot(self.pedometer_data['timestamp'], self.pedometer_data['last_speed_kmh'], 
                label='Speed (km/h)', color='blue')
        plt.title('Physical Activity Pattern')
        plt.legend()
        
        # Plot 2: Steps Accumulation
        plt.subplot(5, 1, 2)
        plt.plot(self.pedometer_data['timestamp'], self.pedometer_data['num_total_steps'], 
                label='Total Steps', color='green')
        plt.title('Cumulative Steps')
        plt.legend()
        
        # Plot 3: Heart Metrics
        plt.subplot(5, 1, 3)
        plt.plot(self.ppg_data['timestamp'], self.ppg_data['ppg'], 
                label='PPG Signal', alpha=0.5, color='red')
        plt.title('PPG Signal Pattern')
        plt.legend()
        
        # Plot 4: HRV Metrics
        plt.subplot(5, 1, 4)
        plt.plot(self.hrv_data['timestamp'], self.hrv_data['hrv_sdnn'], 
                label='HRV (SDNN)', color='purple')
        plt.title('Heart Rate Variability')
        plt.legend()
        
        # Plot 5: Environmental Pressure
        plt.subplot(5, 1, 5)
        plt.plot(self.pressure_data['timestamp'], self.pressure_data['pressure'], 
                label='Atmospheric Pressure', color='orange')
        plt.title('Environmental Pressure')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_summary(self):
        """Generate a comprehensive summary of all sensor data"""
        physical_activity = self.analyze_physical_activity()
        cardiovascular = self.analyze_cardiovascular_metrics()
        environmental = self.analyze_environmental_context()
        
        # Calculate activity intensity periods
        activity_states = self.pedometer_data['last_state_class'].value_counts(normalize=True) * 100
        
        summary = {
            'physical_activity': {
                'total_steps': physical_activity['total_steps'],
                'total_distance_meters': physical_activity['total_distance'],
                'calories_burned': physical_activity['calories_burned'],
                'activity_distribution': activity_states.to_dict()
            },
            'cardiovascular_health': {
                'average_heart_rate': cardiovascular['hrv_metrics']['mean_hr'],
                'hrv_quality': cardiovascular['hrv_metrics']['mean_hrv_sdnn'],
                'stress_level': cardiovascular['hrv_metrics']['stress_indicator']
            },
            'environmental_context': environmental
        }
        
        return summary

def analyze_all_sensor_data( file_path):
    awake_file = os.path.join(file_path, 'samsung_awake_times_short.csv')
    hrv_file = os.path.join(file_path, 'samsung_hrv_1min_short.csv')
    imu_file = os.path.join(file_path, 'samsung_imu_short.csv')
    pressure_file = os.path.join(file_path, 'samsung_pressure_short.csv')
    ppg_file = os.path.join(file_path, 'samsung_ppg_short.csv')
    pedometer_file = os.path.join(file_path, 'samsung_pedometer_short.csv')
    
    """Main function to analyze all sensor data"""
    analyzer = EnhancedSensorAnalyzer()
    analyzer.load_data(awake_file, hrv_file, imu_file, pressure_file, ppg_file, pedometer_file)
    
    # Generate comprehensive analysis
    summary = analyzer.generate_comprehensive_summary()
    
    # Create visualizations
    analyzer.plot_comprehensive_patterns()
    
    return summary


if __name__ == "__main__":
    dataset_source = "./ifh_affect"
    samsung_path = os.path.join(dataset_source, 'par_1/samsung')
    summary = analyze_all_sensor_data(samsung_path)

    oura_path = os.path.join(dataset_source, 'par_1/oura')
    # oura_path = dataset_source
    summary = analyze_health_data(oura_path)


