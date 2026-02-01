# Comprehensive Tech Skill Whitelist to filter out noise
VALID_SKILLS = {
    # Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "golang", "rust", "swift", "kotlin", "php", "ruby", "scala", "r", "c",
    
    # Frameworks & Libraries
    "react", "angular", "vue", "next.js", "node.js", "django", "flask", "fastapi", "spring boot", "ruby on rails", "laravel", 
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy", "matplotlib", "seaborn", "nltk", "spacy", "opencv",
    "express.js", "graphql", "redux", "jquery", "bootstrap", "tailwind",
    
    # Data & Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb", "oracle", "sql server", 
    "firebase", "snowflake", "bigquery", "redshift", "spark", "hadoop", "kafka", "airflow", "tableau", "power bi", "looker",
    
    # DevOps & Cloud
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab ci", "github actions", "circleci", "terraform", "ansible",
    "linux", "bash", "git", "nginx", "apache", "prometheus", "grafana", "elk stack", "splunk", "datadog",
    
    # Algorithms & Others
    "machine learning", "deep learning", "nlp", "computer vision", "data structures", "algorithms", "system design",
    "microservices", "rest api", "soap", "agile", "scrum", "jira"
}

def filter_skills(skill_list):
    """Returns only valid technical skills from a list."""
    return [s for s in skill_list if s.lower() in VALID_SKILLS]
