import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import hopsworks
import joblib
import os 
from dotenv import load_dotenv

load_dotenv()

class AQITrainingPipeline:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            "xgboost": XGBRegressor(random_state=24)
        }

        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None], 
                'min_samples_split': [2, 5], 

            }, 

            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'learning_rate': [0.01, 0.1, 0.2],
            },
        }

    ################

    def load_training_data(self):
        """Load training data from feature store"""
        try:
            project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
            fs = project.get_feature_store()
            
            aqi_fg = fs.get_feature_group("aqi_features", version=1)
            df = aqi_fg.read()
            
            return df
        except Exception as e:
            print(f"Error loading from feature store: {e}")
            # Fallback to local data
            return pd.read_csv('data/aqi_features.csv')
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Select feature columns
        feature_cols = [
            'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
            'temperature', 'humidity', 'pressure', 'wind_speed',
            'hour', 'day_of_week', 'month', 'is_weekend'
        ]
        
        target_cols = ['aqi_next_1d', 'aqi_next_2d', 'aqi_next_3d']
        
        # Clean data
        df = df.dropna(subset=feature_cols + target_cols)
        
        X = df[feature_cols]
        y = df[target_cols]
        
        return X, y, feature_cols, target_cols
    
    def train_models(self, X, y):
        """Train and evaluate multiple models"""
        results = {}
        trained_models = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        for target_idx, target_name in enumerate(y.columns):
            print(f"Training models for {target_name}...")
            
            y_target_train = y_train.iloc[:, target_idx]
            y_target_test = y_test.iloc[:, target_idx]
            
            target_results = {}
            target_models = {}
            
            for model_name, model in self.models.items():
                print(f"  Training {model_name}...")
                
                # Grid search for best parameters
                grid_search = GridSearchCV(
                    model, self.param_grids[model_name],
                    cv=3, scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_target_train)
                best_model = grid_search.best_estimator_
                
                # Evaluate
                y_pred = best_model.predict(X_test)
                
                metrics = {
                    'mse': mean_squared_error(y_target_test, y_pred),
                    'mae': mean_absolute_error(y_target_test, y_pred),
                    'r2': r2_score(y_target_test, y_pred),
                    'best_params': grid_search.best_params_
                }
                
                target_results[model_name] = metrics
                target_models[model_name] = best_model
                
                print(f"    MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
            
            results[target_name] = target_results
            trained_models[target_name] = target_models
        
        return results, trained_models
    
    def select_best_models(self, results, trained_models):
        """Select best model for each target"""
        best_models = {}
        
        for target_name in results.keys():
            target_results = results[target_name]
            best_model_name = min(target_results.keys(), 
                                key=lambda x: target_results[x]['mse'])
            
            best_models[target_name] = {
                'model': trained_models[target_name][best_model_name],
                'model_name': best_model_name,
                'metrics': target_results[best_model_name]
            }
            
            print(f"Best model for {target_name}: {best_model_name}")
        
        return best_models
    
    def save_models(self, best_models, feature_cols):
        """Save trained models"""
        os.makedirs('models', exist_ok=True)
        
        for target_name, model_info in best_models.items():
            model_path = f"models/{target_name}_model.joblib"
            joblib.dump(model_info['model'], model_path)
            print(f"Saved model for {target_name}")
        
        # Save feature columns
        joblib.dump(feature_cols, 'models/feature_columns.joblib')
        
        # Save model metadata
        metadata = {
            target: {
                'model_name': info['model_name'],
                'metrics': info['metrics']
            }
            for target, info in best_models.items()
        }
        
        joblib.dump(metadata, 'models/model_metadata.joblib')
    
    def run_training_pipeline(self):
        """Execute complete training pipeline"""
        print("Starting training pipeline...")
        
        # Load data
        df = self.load_training_data()
        print(f"Loaded {len(df)} training samples")
        
        # Prepare features
        X, y, feature_cols, target_cols = self.prepare_features(df)
        print(f"Features: {len(feature_cols)}, Targets: {len(target_cols)}")
        
        # Train models
        results, trained_models = self.train_models(X, y)
        
        # Select best models
        best_models = self.select_best_models(results, trained_models)
        
        # Save models
        self.save_models(best_models, feature_cols)
        
        print("Training pipeline completed!")
        return best_models

if __name__ == "__main__":
    pipeline = AQITrainingPipeline()
    pipeline.run_training_pipeline()
