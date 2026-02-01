# 🚀 Career Intelligence Platform

An ML-powered career analytics dashboard that helps users understand job market demand, predict salary trajectories, identify skill gaps, match suitable roles, and generate personalized learning roadmaps.

Built as an end-to-end product using Python, NLP, Machine Learning, and Streamlit.

---

## 🔍 Overview

The Career Intelligence Platform is designed for students, freshers, and early-career professionals to make data-driven career decisions.

Users can enter their skills and target role to get:

- Market demand analysis  
- Salary projections based on experience  
- Role matching using similarity scoring  
- Skill gap identification  
- Personalized learning roadmap  

The project demonstrates full-stack ML engineering: data processing, NLP, model logic, visualization, and deployment.

---

## ✨ Features

- 📊 Executive dashboard with KPI metrics  
- 📈 Skill demand analytics (market-based)  
- 💰 Salary trajectory visualization by experience  
- 🎯 Role matching using cosine similarity  
- 🧩 Skill gap analysis  
- 🗺 Personalized 3-month learning roadmap  
- 📉 Interactive charts using Plotly  

---

## 🧠 System Architecture

High-level flow:

Job Dataset  
→ Skill Processing (NLP + embeddings)  
→ Demand Scoring  
→ Salary Estimation  
→ Role Matching  
→ Streamlit Dashboard  

Key techniques used:

- NLP skill extraction  
- Embedding-based similarity  
- Regression-style salary estimation  
- Market frequency analysis  
- Interactive data visualization  

---

## 🛠 Tech Stack

**Backend / ML**
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Sentence Transformers  

**Visualization / UI**
- Streamlit  
- Plotly  

**Other**
- Joblib (model persistence)  
- Custom CSS for UI styling  

---

## 📁 Project Structure
career_intelligence/
│
├── app.py
├── core/
│ ├── salary.py
│ ├── demand.py
│ └── matcher.py
│
├── nlp/
│ └── embedder.py
│
├── roadmap/
│ └── generator.py
│
├── data/
│ └── jobs.csv
│
├── assets/
│ └── style.css
│
├── models/
│
├── requirements.txt
└── README.md
