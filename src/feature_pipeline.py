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
        