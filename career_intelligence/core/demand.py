from collections import Counter
import pandas as pd

def calculate_demand_map(df):
    """
    Calculates demand score (0-100) for each skill based on frequency in dataset.
    Returns Dictionary {skill_lower: score}
    """
    # Flatten all skills
    # Assuming df['Skills'] is list of strings
    all_skills = [s for sublist in df['Skills'] for s in sublist]
    counts = Counter(all_skills)
    
    if not counts:
        return {}
        
    max_count = counts.most_common(1)[0][1]
    
    import math
    demand_map = {}
    if not counts:
        return {}
        
    max_count = counts.most_common(1)[0][1]
    log_max = math.log(max_count + 1)
    
    for skill, count in counts.items():
        # Logarithmic normalization to boost specialist skills
        # score = log(count) / log(max) * 100
        log_val = math.log(count + 1)
        score = int((log_val / log_max) * 100)
        demand_map[skill] = score
        
    return demand_map

def get_gap_skills(user_skills, target_role_skills):
    """
    Identify missing skills.
    """
    user_set = set(k.lower() for k in user_skills)
    target_set = set(k.lower() for k in target_role_skills)
    
    missing = list(target_set - user_set)
    return missing
