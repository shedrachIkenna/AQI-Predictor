import requests
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
import hopsworks 
import os 
from dotenv import load_dotenv 

load_dotenv() 

class AQIFeaturePipeline:
    def __init__(self, city="barcelona"):
        self.city = city
        self.base_url = f"https://api.waqi.info/feed/{city}/"
        self.api_key = os.getenv("WAQI_API_KEY")

    def fetch_aqi_data(self, date=None):
        """Fetch current or historical AQI data"""
        url = f"{self.base_url}?token={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data['status'] != "ok":
                raise Exception(f"API Error: {data}")
            return data['data']
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        
    def extract_features(self, raw_data):
        """Extract and engineer features from raw API data"""
        if not raw_data:
            return None 
        iaqi = raw_data.get("iaqi", {})
        features = {
            'timestamp': datetime.now(),
            'city': raw_data.get('city', {}).get('name', self.city),
            'aqi': raw_data.get('aqi', np.nan),
            
            # Pollutant measurements
            'pm25': iaqi.get('pm25', {}).get('v', np.nan),
            'pm10': iaqi.get('pm10', {}).get('v', np.nan),
            'o3': iaqi.get('o3', {}).get('v', np.nan),
            'no2': iaqi.get('no2', {}).get('v', np.nan),
            'so2': iaqi.get('so2', {}).get('v', np.nan),
            'co': iaqi.get('co', {}).get('v', np.nan),
            
            # Weather data
            'temperature': iaqi.get('t', {}).get('v', np.nan),
            'humidity': iaqi.get('h', {}).get('v', np.nan),
            'pressure': iaqi.get('p', {}).get('v', np.nan),
            'wind_speed': iaqi.get('w', {}).get('v', np.nan),
        }

        # Time-based features 
        dt = features['timestamp']
        features.update({
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'is_weekend': dt.weekday() >= 5
        })

        return features 
    
    def create_targets(self, df):
        """Create target variable (AQI for the next 1, 2, 3, days)"""
        df = df.sort_values('timestamp')
        df['aqi_next_1d'] = df['aqi'].shift(-24) 
        df['aqi_next_2d'] = df['aqi'].shift(-48)
        df['aqi_next_3d'] = df['aqi'].shift(-72)    
        return df 

    def save_to_feature_store(self, features_df):
        """Saves Features to Hopworks Feature Store"""
        try: 
            project = hopsworks.login()   
            fs = project.get_feature_store()

            aqi_fg = fs.get_or_create_feature_group(
                name = "aqi_features",
                version = 1, 
                primary_key = ['timestamp', 'city'],
                description = "Air Quality Index features and targets"
            ) 
            aqi_fg.insert(features_df)
            print(f"Saved {len(features_df)} records to feature store")
        except Exception as e: 
            print(f"Error saving to feature store: {e}")
            # Fallback: Save data locally as backup if feature store fails 
            output_dir = 'data'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir) 
            features_df.to_csv('data/aqi_features.csv', index=False) 
            print("Saved data locally as fallback")

    def run_pipeline(self):
        """Execute the complete feature pipeline"""
        print("Starting Feature Pipeline")
        raw_data = self.fetch_aqi_data()
        if raw_data:
            features = self.extract_features(raw_data)
            features_df = pd.DataFrame([features])
            self.save_to_feature_store(features_df)
            print("Feature pipeline completed successfully")
            return features_df
        else:
            print("Failed to fetch data")
            return None 

if __name__ == "__main__":
    pipeline = AQIFeaturePipeline()
    pipeline.run_pipeline()