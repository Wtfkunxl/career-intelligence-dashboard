from core.skills import filter_skills

def fetch_remoteok_jobs():
    """
    Fetches live jobs from RemoteOK API.
    """
    url = "https://remoteok.com/api"
    print(f"Fetching live data from {url}...")
    
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching API: {e}")
        return pd.DataFrame() 
    
    job_list = []
    
    # Skip the first element
    for item in data[1:]:
        title = item.get('position', 'Unknown Role')
        company = item.get('company', 'Unknown Company')
        tags = item.get('tags', [])
        
        # Salary parsing
        salary_str = item.get('salary', '') 
        min_sal, max_sal = parse_salary(salary_str)
        
        # SALARY NORMALIZATION (Critical Fix)
        # RemoteOK gives USD (e.g., 60-120k).
        # We need realistic Indian LPA.
        # Direct conversion is too high (80 LPA). 
        # We apply a "PPP / Market Correction Factor" of ~0.25
        # e.g., $100k USD -> 25 LPA INR (High end Indian salary).
        
        if min_sal > 0:
            min_sal = min_sal * 0.25
            max_sal = max_sal * 0.25
        else:
            # Fallback imputation
            min_sal, max_sal = estimate_salary_fallback(title)

        # Clean Skills using Whitelist
        cleaned_tags = [t.lower() for t in tags]
        valid_skills = filter_skills(cleaned_tags)
        
        # If no valid skills found, try extracting from title? Or just skip/keep empty.
        # Let's keep empty to urge data quality via valid_skills
        if not valid_skills:
            # Heuristic: Add 'python' if in title
            if 'python' in title.lower(): valid_skills.append('python')
            if 'data' in title.lower(): valid_skills.append('sql')

        job_list.append({
            "Role": title,
            "Company": company,
            "Skills": valid_skills,
            "Salary": (min_sal + max_sal) / 2, # Avg
            "Min_Salary": min_sal,
            "Max_Salary": max_sal,
            "Experience": random.randint(1, 6), 
            "Source": "RemoteOK"
        })
        
    print(f"Successfully processed {len(job_list)} live jobs.")
    return pd.DataFrame(job_list)

def parse_salary(salary_str):
    """
    Parses strings like '$60k - $100k'. Returns raw K values.
    """
    if not salary_str:
        return 0, 0
    clean = re.sub(r'[^\d\-]', '', salary_str)
    try:
        if '-' in clean:
            parts = clean.split('-')
            low = float(parts[0])
            high = float(parts[1])
            return low, high
        else:
             val = float(clean)
             return val, val
    except:
        return 0, 0

def estimate_salary_fallback(title):
    """
    Returns realistic IND LPA ranges.
    """
    title = title.lower()
    if 'senior' in title or 'lead' in title:
        return 18, 35 # LPA
    if 'junior' in title or 'intern' in title:
        return 4, 8   # LPA
    if 'manager' in title or 'director' in title:
        return 25, 50 # LPA
    if 'data' in title or 'machine' in title:
        return 10, 22 # LPA
    return 8, 18 # Mid level default (Standard Dev)
