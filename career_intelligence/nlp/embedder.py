from sentence_transformers import SentenceTransformer
import numpy as np

# Load a lightweight model for speed but good quality
MODEL_NAME = 'all-MiniLM-L6-v2'
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def get_embedding(text):
    """
    Returns a numpy array embedding for the given text.
    Text can be a single string or a list of strings.
    """
    model = get_model()
    embeddings = model.encode(text)
    return embeddings

def get_mean_embedding(skills_list):
    """
    Returns the mean embedding vector for a list of skills.
    Use this to represent a User Profile or a Job Role based on its skills.
    """
    if not skills_list:
        return np.zeros(384) # 384 dim for MiniLM
    
    # We treat each skill as a phrase, get individual embeddings, then average them.
    # Alternatively, join them into one string "python, sql, aws".
    # Averaging individual tags usually captures multi-modal skills better than one sentence.
    embeddings = get_embedding(skills_list)
    return np.mean(embeddings, axis=0)
