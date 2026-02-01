import pandas as pd
import numpy as np
import joblib
import os
import random
from sklearn.ensemble import RandomForestRegressor
from nlp.embedder import get_mean_embedding

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Mock Data Config
ROLES = [
    ("Frontend Engineer", ["react", "javascript", "html", "css", "typescript", "git", "redux"], 5.0, 18.0),
    ("Backend Engineer", ["python", "django", "sql", "docker", "aws", "redis", "fastapi"], 6.0, 22.0),
    ("Machine Learning Engineer", ["python", "machine learning", "tensorflow", "pytorch", "scikit-learn", "docker", "aws"], 8.0, 30.0),
    ("Data Scientist", ["python", "pandas", "numpy", "statistics", "sql", "visualization", "jupyter"], 7.0, 25.0),
    ("DevOps Engineer", ["linux", "docker", "kubernetes", "aws", "terraform", "ci/cd", "bash"], 7.0, 28.0),
    ("Product Manager", ["agile", "jira", "communication", "roadmap", "analytics", "user research"], 10.0, 35.0),
    ("Full Stack Developer", ["react", "python", "node.js", "sql", "mongo", "aws", "git"], 6.0, 24.0)
]

from core.ingestion import fetch_remoteok_jobs

def generate_mock_data(n=300):
    # Try Live Data First
    try:
        print("Attempting to fetch Live Data...")
        df_live = fetch_remoteok_jobs()
        if not df_live.empty:
            # We need to map live data columns to match what train_models expects
            # Live Data cols: Role, Company, Skills, Salary, Min_Salary, Max_Salary, Experience, Source
            # Expected cols: Role, Skills, Salary, Experience
            
            # The ingestion script already normalizes it well.
            # Let's just filter/select needed columns for CSV
            df_final = df_live[['Role', 'Skills', 'Salary', 'Experience']]
            
            # Save Raw Data
            df_final.to_csv(os.path.join(DATA_DIR, 'jobs.csv'), index=False)
            return df_final
    except Exception as e:
        print(f"Live data fetch failed: {e}. Falling back to mock.")

    data = []
    print(f"Generating {n} mock job entries (Fallback)...")
    
    for _ in range(n):
        role_name, skills, min_sal, max_sal = random.choice(ROLES)
        
        # Vary skills: pick 70-100% of core skills + 0-2 random extras
        num_skills = int(len(skills) * random.uniform(0.7, 1.0))
        sample_skills = random.sample(skills, max(1, num_skills))
        
        # Add random extra
        if random.random() > 0.7:
             extra = random.choice(ROLES)[1]
             sample_skills.append(random.choice(extra))
             
        sample_skills = list(set(sample_skills)) # unique
        
        # Salary logic: Base + noise
        salary = random.uniform(min_sal, max_sal)
        
        # Experience needed (simulated)
        exp_needed = random.randint(1, 8)
        
        # Higher salary for higher exp
        salary += (exp_needed * 1.5)
        
        data.append({
            "Role": role_name,
            "Skills": sample_skills, # List
            "Salary": round(salary, 2),
            "Experience": exp_needed
        })
        
    df = pd.DataFrame(data)
    
    # Save Raw Data
    df.to_csv(os.path.join(DATA_DIR, 'jobs.csv'), index=False)
    return df

def train_models(df):
    print("Computing embeddings... (This involves downloading/loading model)")
    
    # 1. Compute Embeddings for each job row (Feature for Training)
    # This might be slow for 300 rows if using CPU. It's okay for Setup.
    # We map 'Skills' list -> Mean Vector
    df['embedding'] = df['Skills'].apply(lambda x: get_mean_embedding(x))
    
    # 2. Train Salary Model
    # X = [Embedding (384) + Experience (1)] -> 385 dims
    print("Training Salary Model...")
    
    X_list = []
    for _, row in df.iterrows():
        # Concat embedding + exp
        vec = np.append(row['embedding'], row['Experience'])
        X_list.append(vec)
        
    X = np.array(X_list)
    y = df['Salary'].values
    
    salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
    salary_model.fit(X, y)
    
    joblib.dump(salary_model, os.path.join(MODEL_DIR, 'salary.pkl'))
    
    # 3. Create Role Prototypes for Matching
    print("Creating Role Prototypes...")
    distinct_roles = []
    
    # Map raw live titles to broader categories for cleaner aggregation
    def map_role_category(title):
        t = title.lower()
        if 'front' in t or 'react' in t or 'vue' in t or 'web' in t: return 'Frontend Engineer'
        if 'back' in t or 'api' in t or 'node' in t or 'django' in t: return 'Backend Engineer'
        if 'data' in t and 'scien' in t: return 'Data Scientist'
        if 'machine' in t or 'ai' in t or 'learning' in t: return 'Machine Learning Engineer'
        if 'devops' in t or 'cloud' in t or 'sre' in t: return 'DevOps Engineer'
        if 'product' in t: return 'Product Manager'
        if 'full' in t: return 'Full Stack Developer'
        return 'Software Engineer' # Default

    df['Category'] = df['Role'].apply(map_role_category)
    
    grouped = df.groupby('Category')
    from core.demand import calculate_demand_map
    demand_map = calculate_demand_map(df)
    
    for role_name, group in grouped:
        # Mean of means
        role_embedding = np.mean(np.stack(group['embedding'].values), axis=0)
        avg_sal = group['Salary'].mean()
        
        # Demand Level
        count = len(group)
        # Adjust thresholds for live data size (usually smaller than 300)
        demand_lvl = "High" if count > 15 else "Medium"
        
        distinct_roles.append({
            "Role": role_name,
            "embedding": role_embedding,
            "Avg_Salary": round(avg_sal, 1),
            "Demand_Level": demand_lvl,
            "Core_Skills": group.iloc[0]['Skills'] 
        })
        
    role_df = pd.DataFrame(distinct_roles)
    joblib.dump(role_df, os.path.join(MODEL_DIR, 'roles.pkl'))
    
    # Save Demand Map
    joblib.dump(demand_map, os.path.join(MODEL_DIR, 'demand_map.pkl'))
    
    print("All Models and Data Saved successfully.")

if __name__ == "__main__":
    df = generate_mock_data(300)
    train_models(df)
