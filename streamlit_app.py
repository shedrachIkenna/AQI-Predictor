import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

@st.cache_data
def load_models():
    """Load trained models and metadata"""
    try:
        models = {}
        targets = ['aqi_next_1d', 'aqi_next_2d', 'aqi_next_3d']
        
        # Check if models directory exists (try both locations)
        models_dir = 'src/models'  # Direct path since we know it exists
        
        if not os.path.exists(models_dir):
            return None, None, None, f"Models directory not found at: {models_dir}. Current working directory: {os.getcwd()}"
        
        # Debug: print the directory being used
        print(f"Using models directory: {models_dir}")
        print(f"Directory contents: {os.listdir(models_dir) if os.path.exists(models_dir) else 'Directory not found'}")
        
        # Load models
        missing_models = []
        for target in targets:
            model_path = f"{models_dir}/{target}_model.joblib"
            print(f"Looking for model at: {model_path}")
            if os.path.exists(model_path):
                models[target] = joblib.load(model_path)
                print(f"Successfully loaded: {model_path}")
            else:
                missing_models.append(f"{target}_model.joblib")
                print(f"Missing model file: {model_path}")
        
        if missing_models:
            return None, None, None, f"Missing model files: {missing_models}. Using directory: {models_dir}"
        
        # Load feature columns
        feature_path = f'{models_dir}/feature_columns.joblib'
        if not os.path.exists(feature_path):
            return None, None, None, f"Feature columns file not found at: {feature_path}"
        feature_cols = joblib.load(feature_path)
        
        # Load metadata
        metadata_path = f'{models_dir}/model_metadata.joblib'
        metadata = None
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
        
        return models, feature_cols, metadata, None
    except Exception as e:
        return None, None, None, f"Error loading models: {str(e)}"

def fetch_current_data(city="barcelona"):
    """Fetch current AQI data"""
    api_key = os.getenv("WAQI_API_KEY")
    
    if not api_key:
        return None, "WAQI API key not found. Please set WAQI_API_KEY in your .env file"
    
    url = f"https://api.waqi.info/feed/{city}/?token={api_key}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'ok':
            return None, f"API Error: {data.get('data', 'Unknown error')}"
            
        return data['data'], None
    except requests.exceptions.Timeout:
        return None, "Request timeout. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def extract_features_for_prediction(raw_data):
    """Extract features for model prediction"""
    if not raw_data:
        return None
        
    iaqi = raw_data.get('iaqi', {})
    
    features = {
        'pm25': iaqi.get('pm25', {}).get('v', np.nan),
        'pm10': iaqi.get('pm10', {}).get('v', np.nan),
        'o3': iaqi.get('o3', {}).get('v', np.nan),
        'no2': iaqi.get('no2', {}).get('v', np.nan),
        'so2': iaqi.get('so2', {}).get('v', np.nan),
        'co': iaqi.get('co', {}).get('v', np.nan),
        'temperature': iaqi.get('t', {}).get('v', np.nan),
        'humidity': iaqi.get('h', {}).get('v', np.nan),
        'pressure': iaqi.get('p', {}).get('v', np.nan),
        'wind_speed': iaqi.get('w', {}).get('v', np.nan),
    }
    
    # Time-based features
    dt = datetime.now()
    features.update({
        'hour': dt.hour,
        'day_of_week': dt.weekday(),
        'month': dt.month,
        'is_weekend': dt.weekday() >= 5
    })
    
    return features

def get_aqi_category(aqi_value):
    """Get AQI category and color"""
    if pd.isna(aqi_value):
        return "Unknown", "gray"
    
    if aqi_value <= 50:
        return "Good", "green"
    elif aqi_value <= 100:
        return "Moderate", "yellow"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi_value <= 200:
        return "Unhealthy", "red"
    elif aqi_value <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"

def main():
    st.title("🌍 Air Quality Index Predictor")
    st.markdown("Predict AQI for the next 3 days using real-time data and machine learning")
    
    # Sidebar for city selection
    st.sidebar.header("Configuration")
    city = st.sidebar.selectbox(
        "Select City",
        ["barcelona", "london", "paris", "madrid", "rome", "beijing", "delhi", "tokyo"],
        index=0
    )
    
    # Load models
    models, feature_cols, metadata, error_msg = load_models()
    
    if models is None:
        st.error(f"Could not load models: {error_msg}")
        
        # Show instructions for training models
        st.markdown("""
        ### 🔧 Setup Instructions
        
        To use this app, you need to train the models first:
        
        1. **Run the training pipeline:**
           ```bash
           python training_pipeline.py
           ```
        
        2. **Make sure you have the required data:**
           - Hopsworks API key in your `.env` file
           - Or local training data in `data/aqi_features.csv`
        
        3. **Ensure these files are created in the `models/` directory:**
           - `aqi_next_1d_model.joblib`
           - `aqi_next_2d_model.joblib`
           - `aqi_next_3d_model.joblib`
           - `feature_columns.joblib`
           - `model_metadata.joblib`
        """)
        return
    
    # Fetch current data
    with st.spinner(f"Fetching current air quality data for {city.title()}..."):
        raw_data, api_error = fetch_current_data(city)
    
    if raw_data is None:
        st.error(f"Could not fetch current data: {api_error}")
        
        # Show demo mode
        st.info("Running in demo mode with sample data")
        
        # Create sample data for demonstration
        features = {
            'pm25': 25.0, 'pm10': 35.0, 'o3': 80.0, 'no2': 40.0, 'so2': 15.0, 'co': 1.2,
            'temperature': 22.0, 'humidity': 65.0, 'pressure': 1013.0, 'wind_speed': 3.5,
            'hour': datetime.now().hour, 'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month, 'is_weekend': datetime.now().weekday() >= 5
        }
        current_aqi = 75
    else:
        # Extract features
        features = extract_features_for_prediction(raw_data)
        current_aqi = raw_data.get('aqi', 'N/A')
    
    if features is None:
        st.error("Could not extract features from current data.")
        return
    
    # Current AQI display
    st.header(f"Current AQI in {city.title()}: {current_aqi}")
    
    if current_aqi != 'N/A':
        category, color = get_aqi_category(current_aqi)
        st.markdown(f"**Status:** :{color}[{category}]")
    
    # Display current measurements
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PM2.5", f"{features.get('pm25', 'N/A')} μg/m³")
        st.metric("PM10", f"{features.get('pm10', 'N/A')} μg/m³")
    
    with col2:
        st.metric("Ozone (O₃)", f"{features.get('o3', 'N/A')} μg/m³")
        st.metric("NO₂", f"{features.get('no2', 'N/A')} μg/m³")
    
    with col3:
        st.metric("Temperature", f"{features.get('temperature', 'N/A')}°C")
        st.metric("Humidity", f"{features.get('humidity', 'N/A')}%")
    
    # Make predictions
    st.header("📈 AQI Predictions")
    
    try:
        # Prepare features for prediction
        feature_df = pd.DataFrame([features])
        
        # Handle missing values (simple imputation with median values)
        for col in feature_df.columns:
            if feature_df[col].isna().any():
                # Use reasonable defaults for missing values
                defaults = {
                    'pm25': 25, 'pm10': 35, 'o3': 50, 'no2': 30, 'so2': 10, 'co': 1.0,
                    'temperature': 20, 'humidity': 60, 'pressure': 1013, 'wind_speed': 2.0
                }
                feature_df[col] = feature_df[col].fillna(defaults.get(col, 0))
        
        # Ensure all required features are present
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        feature_df = feature_df[feature_cols]
        
        # Make predictions
        predictions = {}
        for target, model in models.items():
            pred = model.predict(feature_df)[0]
            predictions[target] = max(0, pred)  # Ensure non-negative
        
        # Display predictions
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        days = ['Tomorrow', 'Day After Tomorrow', 'In 3 Days']
        targets = ['aqi_next_1d', 'aqi_next_2d', 'aqi_next_3d']
        
        for i, (col, day, target) in enumerate(zip([pred_col1, pred_col2, pred_col3], days, targets)):
            with col:
                if target in predictions:
                    pred_value = round(predictions[target])
                    category, color = get_aqi_category(pred_value)
                    
                    st.metric(day, pred_value)
                    st.markdown(f":{color}[{category}]")
                else:
                    st.metric(day, "N/A")
                    st.markdown("Model not available")
        
        # Visualization
        st.header("📊 Prediction Visualization")
        
        # Create prediction chart
        dates = [datetime.now() + timedelta(days=i+1) for i in range(3)]
        pred_values = [predictions.get(target, np.nan) for target in targets]
        
        fig = go.Figure()
        
        # Add prediction line
        valid_dates = [date for date, val in zip(dates, pred_values) if not pd.isna(val)]
        valid_values = [val for val in pred_values if not pd.isna(val)]
        
        if valid_dates:
            fig.add_trace(go.Scatter(
                x=valid_dates,
                y=valid_values,
                mode='lines+markers',
                name='Predicted AQI',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ))
        
        # Add current AQI point
        if current_aqi != 'N/A' and not pd.isna(current_aqi):
            fig.add_trace(go.Scatter(
                x=[datetime.now()],
                y=[current_aqi],
                mode='markers',
                name='Current AQI',
                marker=dict(size=15, color='red')
            ))
        
        fig.update_layout(
            title="AQI Prediction Timeline",
            xaxis_title="Date",
            yaxis_title="AQI Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.info("Please check that your models are properly trained and saved.")
    
    # Model performance
    if metadata:
        st.header("🤖 Model Performance")
        
        for target, meta in metadata.items():
            with st.expander(f"Model for {target}"):
                st.write(f"**Model Type:** {meta['model_name']}")
                st.write(f"**R² Score:** {meta['metrics']['r2']:.3f}")
                st.write(f"**Mean Absolute Error:** {meta['metrics']['mae']:.2f}")
                st.write(f"**Mean Squared Error:** {meta['metrics']['mse']:.2f}")

if __name__ == "__main__":
    main()