import joblib
import os
import pandas as pd
from collections import Counter

# access local files
base_dir = os.path.dirname(os.path.abspath(__file__))
demand_map_path = os.path.join(base_dir, 'models', 'demand_map.pkl')
jobs_csv_path = os.path.join(base_dir, 'data', 'jobs.csv')

print(f"Checking demand map at: {demand_map_path}")

if os.path.exists(demand_map_path):
    demand_map = joblib.load(demand_map_path)
    print("\n--- Demand Map (Sample) ---")
    # Sort by value
    sorted_demand = sorted(demand_map.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_demand[:20]:
        print(f"{k}: {v}")
    
    print(f"\nScore for 'machine learning': {demand_map.get('machine learning', 'NOT FOUND')}")
    print(f"Score for 'python': {demand_map.get('python', 'NOT FOUND')}")

if os.path.exists(jobs_csv_path):
    print(f"\nChecking jobs data at: {jobs_csv_path}")
    df = pd.read_csv(jobs_csv_path)
    # Check simple counts
    # Skills are stored as string representation of list in CSV, need to parse
    # But wait, generate_data.py imports fetch_remoteok_jobs which saves it. 
    # Or generate_mock_data saves it.
    
    # In generate_data.py: df.to_csv(..., index=False)
    # The 'Skills' column might be a string literal of a list "['a', 'b']"
    
    import ast
    try:
        all_skills = []
        for raw in df['Skills']:
            # it might be a list if not saved to csv yet, but reading from csv makes it str
            if isinstance(raw, str):
                try:
                    s_list = ast.literal_eval(raw)
                    all_skills.extend([s.lower() for s in s_list])
                except:
                    pass
        
        counts = Counter(all_skills)
        print("\n--- Raw Counts from CSV ---")
        print(f"Total jobs: {len(df)}")
        print(f"Python count: {counts['python']}")
        print(f"Machine Learning count: {counts['machine learning']}")
        if counts['python'] > 0:
            print(f"Ratio ML/Python: {counts['machine learning'] / counts['python']:.2f}")
    except Exception as e:
        print(f"Error parsing CSV: {e}")
