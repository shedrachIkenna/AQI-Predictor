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
        self.base_url = f"https://api.wapi.info/feed/{city}/"
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
        