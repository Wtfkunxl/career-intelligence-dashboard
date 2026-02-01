import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
import os

# To match roles, we need Pre-computed Role Embeddings.
# For this implementation, we will compare User Skill Vector vs (Job Role Vectors)

def match_roles(user_embedding, role_df):
    """
    Compare user_embedding (1D array) against all roles in role_df.
    role_df should have 'role' and 'embedding' columns.
    
    Returns top 3 roles with match % and salary data.
    """
    if user_embedding is None or role_df is None or role_df.empty:
        return []
    
    # Stack embeddings
    role_matrix = np.stack(role_df['embedding'].values)
    
    # Reshape user_embedding for sklearn (1, n_features)
    user_vec_reshaped = user_embedding.reshape(1, -1)
    
    # Compute Cosine Similarity
    sim_scores = cosine_similarity(user_vec_reshaped, role_matrix)[0]
    
    # Add to temporary DF to sort
    results = role_df.copy()
    results['match_score'] = sim_scores
    
    # Get top 3
    top_matches = results.sort_values('match_score', ascending=False).head(3)
    
    output = []
    for _, row in top_matches.iterrows():
        output.append({
            "role": row['Role'],
            "match_pct": int(row['match_score'] * 100),
            "avg_salary": row['Avg_Salary'],
            "demand": row['Demand_Level']
        })
        
    return output
