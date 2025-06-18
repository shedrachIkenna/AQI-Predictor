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
        self.api_key = os.getenv("WAQI_API_KEY")  # Get from aqicn.org
        
    def fetch_aqi_data(self, date=None):
        """Fetch current or historical AQI data"""
        url = f"{self.base_url}?token={self.api_key}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'ok':
                raise Exception(f"API Error: {data}")
                
            return data['data']
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def extract_features(self, raw_data):
        """Extract and engineer features from raw AQI data"""
        if not raw_data:
            return None
            
        # Extract basic measurements
        iaqi = raw_data.get('iaqi', {})
        
        features = {
            'timestamp': datetime.now(),
            'city': raw_data.get('city', {}).get('name', self.city),
            'aqi': float(raw_data.get('aqi', 0)) if raw_data.get('aqi') is not None else 0.0,
            
            # Pollutant measurements - ensure consistent types
            'pm25': float(iaqi.get('pm25', {}).get('v', 0)) if iaqi.get('pm25', {}).get('v') is not None else 0.0,
            'pm10': float(iaqi.get('pm10', {}).get('v', 0)) if iaqi.get('pm10', {}).get('v') is not None else 0.0,
            'o3': float(iaqi.get('o3', {}).get('v', 0)) if iaqi.get('o3', {}).get('v') is not None else 0.0,
            'no2': float(iaqi.get('no2', {}).get('v', 0)) if iaqi.get('no2', {}).get('v') is not None else 0.0,
            'so2': float(iaqi.get('so2', {}).get('v', 0)) if iaqi.get('so2', {}).get('v') is not None else 0.0,
            'co': float(iaqi.get('co', {}).get('v', 0)) if iaqi.get('co', {}).get('v') is not None else 0.0,
            
            # Weather data - ensure consistent types
            'temperature': int(iaqi.get('t', {}).get('v', 20)) if iaqi.get('t', {}).get('v') is not None else 20,
            'humidity': float(iaqi.get('h', {}).get('v', 50)) if iaqi.get('h', {}).get('v') is not None else 50.0,
            'pressure': float(iaqi.get('p', {}).get('v', 1013)) if iaqi.get('p', {}).get('v') is not None else 1013.0,
            'wind_speed': float(iaqi.get('w', {}).get('v', 0)) if iaqi.get('w', {}).get('v') is not None else 0.0,
        }
        
        # Time-based features
        dt = features['timestamp']
        features.update({
            'hour': int(dt.hour),
            'day_of_week': int(dt.weekday()),
            'month': int(dt.month),
            'is_weekend': bool(dt.weekday() >= 5)
        })
        
        return features
    
    def create_targets(self, df):
        """Create target variables (AQI for next 1, 2, 3 days)"""
        df = df.sort_values('timestamp')
        
        # Shift AQI values to create future targets
        df['aqi_next_1d'] = df['aqi'].shift(-24).astype(float)  # Assuming hourly data
        df['aqi_next_2d'] = df['aqi'].shift(-48).astype(float)
        df['aqi_next_3d'] = df['aqi'].shift(-72).astype(float)
        
        return df
    
    def get_feature_group_schema(self):
        """Define the schema for the feature group"""
        return [
            {"name": "timestamp", "type": "timestamp"},
            {"name": "city", "type": "string"},
            {"name": "aqi", "type": "double"},
            {"name": "pm25", "type": "double"},
            {"name": "pm10", "type": "double"},
            {"name": "o3", "type": "double"},
            {"name": "no2", "type": "double"},
            {"name": "so2", "type": "double"},
            {"name": "co", "type": "double"},
            {"name": "temperature", "type": "bigint"},
            {"name": "humidity", "type": "double"},
            {"name": "pressure", "type": "double"},
            {"name": "wind_speed", "type": "double"},
            {"name": "hour", "type": "int"},
            {"name": "day_of_week", "type": "int"},
            {"name": "month", "type": "int"},
            {"name": "is_weekend", "type": "boolean"},
            {"name": "aqi_next_1d", "type": "double"},
            {"name": "aqi_next_2d", "type": "double"},
            {"name": "aqi_next_3d", "type": "double"}
        ]
    
    def save_to_feature_store(self, features_df):
        """Save features to Hopsworks Feature Store"""
        try:
            project = hopsworks.login(
                api_key_value=os.getenv("HOPSWORKS_API_KEY")
            )
            fs = project.get_feature_store()
            
            # Try to get existing feature group, if it doesn't exist or has schema issues, create new one
            try:
                # Try to get the existing feature group
                aqi_fg = fs.get_feature_group("aqi_features", version=1)
                print("Found existing feature group")
                
                # Check if we need to add the target columns
                existing_features = [f.name for f in aqi_fg.features]
                target_cols = ['aqi_next_1d', 'aqi_next_2d', 'aqi_next_3d']
                
                if not all(col in existing_features for col in target_cols):
                    print("Target columns missing, creating new version...")
                    # Create new version with target columns
                    aqi_fg = fs.create_feature_group(
                        name="aqi_features",
                        version=2,  # New version
                        primary_key=["timestamp", "city"],
                        description="Air Quality Index features and targets with prediction columns"
                    )
                
            except Exception as e:
                print(f"Creating new feature group: {e}")
                # Create new feature group
                aqi_fg = fs.create_feature_group(
                    name="aqi_features",
                    version=1,
                    primary_key=["timestamp", "city"],
                    description="Air Quality Index features and targets"
                )
            
            # Ensure data types match schema expectations
            print("Preparing data for insertion...")
            
            # Make sure timestamp is properly formatted
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            
            # Ensure numeric columns are the right type
            numeric_columns = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 
                             'humidity', 'pressure', 'wind_speed', 
                             'aqi_next_1d', 'aqi_next_2d', 'aqi_next_3d']
            
            for col in numeric_columns:
                if col in features_df.columns:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce').astype(float)
            
            # Ensure integer columns
            int_columns = ['temperature', 'hour', 'day_of_week', 'month']
            for col in int_columns:
                if col in features_df.columns:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce').astype(int)
            
            # Ensure boolean columns
            if 'is_weekend' in features_df.columns:
                features_df['is_weekend'] = features_df['is_weekend'].astype(bool)
            
            print(f"Data shape: {features_df.shape}")
            print(f"Data types:\n{features_df.dtypes}")
            
            # Insert data
            aqi_fg.insert(features_df, write_options={"wait_for_job": False})
            print(f"Successfully saved {len(features_df)} records to feature store")
            
        except Exception as e:
            print(f"Error saving to feature store: {e}")
            # Fallback: save locally
            os.makedirs('data', exist_ok=True)
            features_df.to_csv('data/aqi_features.csv', index=False)
            print("Saved data locally as fallback")
    
    def run_pipeline(self):
        """Execute the complete feature pipeline"""
        print("Starting feature pipeline...")
        
        # Fetch raw data
        raw_data = self.fetch_aqi_data()
        
        if raw_data:
            # Extract features
            features = self.extract_features(raw_data)
            features_df = pd.DataFrame([features])
            
            # Save to feature store
            self.save_to_feature_store(features_df)
            
            print("Feature pipeline completed successfully!")
            return features_df
        else:
            print("Failed to fetch data")
            return None

if __name__ == "__main__":
    pipeline = AQIFeaturePipeline()
    pipeline.run_pipeline()