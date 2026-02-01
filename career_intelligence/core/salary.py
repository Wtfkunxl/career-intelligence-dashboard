import joblib
import numpy as np
import os

def predict_salary(model, user_embedding, experience_years):
    """
    Predicts salary range based on user embedding + experience.
    Model is expected to be a RandomForestRegressor.
    Input Feature Vector: [Emb_0, ..., Emb_383, Experience]
    """
    if model is None:
        return (0, 0)
        
    # Create feature vector
    # embedding is (384,), need to append experience
    features = np.append(user_embedding, experience_years)
    
    # Reshape for prediction (1, 385)
    features = features.reshape(1, -1)
    
    predicted_avg = model.predict(features)[0]
    
    # Create a nice range +/- 15%
    low = predicted_avg * 0.85
    high = predicted_avg * 1.15
    
    return round(low, 1), round(high, 1)
