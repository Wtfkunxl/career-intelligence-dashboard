def generate_roadmap(missing_skills):
    """
    Distributes missing skills across 3 months.
    """
    if not missing_skills:
        return {}

    # Basic clean up
    skills = sorted(list(set(missing_skills)))
    n = len(skills)
    
    roadmap = {
        "Month 1": [],
        "Month 2": [],
        "Month 3": []
    }
    
    # Dynamic distribution
    for i, skill in enumerate(skills):
        if i % 3 == 0:
            roadmap["Month 1"].append(skill)
        elif i % 3 == 1:
            roadmap["Month 2"].append(skill)
        else:
            roadmap["Month 3"].append(skill)
            
    return roadmap
