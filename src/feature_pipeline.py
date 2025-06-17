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
