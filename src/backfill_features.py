import pandas as pd 
from datetime import datetime, timedelta
from feature_pipeline import AQIFeaturePipeline
import time 
from tqdm import tqdm
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

def reset_feature_group():
    """Reset the feature group to handle schema changes"""
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        
        # Try to delete existing feature group versions
        try:
            for version in [1, 2, 3]:
                try:
                    fg = fs.get_feature_group("aqi_features", version=version)
                    fg.delete()
                    print(f"Deleted feature group version {version}")
                except:
                    pass
        except Exception as e:
            print(f"Note: {e}")
        
        print("Feature group reset complete")
        
    except Exception as e:
        print(f"Error resetting feature group: {e}")

def backfill_historical_data(days=7, city="barcelona", reset_fg=False):
    """Backfill historical AQI data"""
    
    if reset_fg:
        print("Resetting feature group...")
        reset_feature_group()
        time.sleep(5)  # Wait a bit for the deletion to process
    
    pipeline = AQIFeaturePipeline(city)
    all_features = []
    
    print(f"Backfilling {days} days of data for {city}...")
    
    # Simulate historical data with some variation
    base_time = datetime.now()
    
    for i in tqdm(range(days * 24), desc="Fetching data"):
        try: 
            # Simulate fetching historical data
            raw_data = pipeline.fetch_aqi_data()

            if raw_data:
                features = pipeline.extract_features(raw_data)
                
                # Adjust timestamp for historical simulation 
                features['timestamp'] = base_time - timedelta(hours=i)
                
                # Add some variation to simulate historical changes
                if i > 0:  # Don't modify the first (most recent) data point
                    variation_factor = 1 + (np.random.random() * 0.3 - 0.15)  # ±15% variation
                    
                    # Vary pollutant levels
                    for pollutant in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
                        if features[pollutant] > 0:
                            features[pollutant] = max(0, features[pollutant] * variation_factor)
                    
                    # Vary AQI accordingly
                    if features['aqi'] > 0:
                        features['aqi'] = max(0, features['aqi'] * variation_factor)
                
                all_features.append(features)

            # Rate Limiting - reduced for faster processing
            time.sleep(0.5)
            
        except Exception as e: 
            print(f"Error at iteration {i}: {e}")
            continue 
    
    if not all_features:
        print("No data collected!")
        return None
    
    # Create dataframe and add targets
    print("Processing data and creating targets...")
    df = pd.DataFrame(all_features)
    
    # Sort by timestamp (oldest first) for proper target creation
    df = df.sort_values('timestamp')
    df = df.reset_index(drop=True)
    
    # Create targets
    df = pipeline.create_targets(df)

    # Remove rows without targets (last few days)
    original_len = len(df)
    df = df.dropna(subset=['aqi_next_1d', 'aqi_next_2d', 'aqi_next_3d'])
    print(f"Removed {original_len - len(df)} rows without targets")

    if len(df) == 0:
        print("No valid data after target creation!")
        return None

    # Save to feature store 
    print("Saving to feature store...")
    pipeline.save_to_feature_store(df)

    print(f"Backfilled {len(df)} records successfully")
    return df 

if __name__ == "__main__":
    import numpy as np  # Add this import for the variation calculation
    
    # Ask user if they want to reset the feature group
    reset = input("Do you want to reset the feature group? (y/N): ").lower().strip()
    reset_fg = reset in ['y', 'yes']
    
    result = backfill_historical_data(reset_fg=reset_fg)
    
    if result is not None:
        print("\nData summary:")
        print(f"Shape: {result.shape}")
        print(f"Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
        print(f"Average AQI: {result['aqi'].mean():.1f}")
        print("\nSample of data:")
        print(result[['timestamp', 'city', 'aqi', 'pm25', 'aqi_next_1d']].head())
    else:
        print("Backfill failed!")