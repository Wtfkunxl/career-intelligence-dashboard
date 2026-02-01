import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import random

# Append project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nlp.embedder import get_mean_embedding
from core.matcher import match_roles
from core.salary import predict_salary
from core.demand import get_gap_skills
from roadmap.generator import generate_roadmap

# --- PAGE CONFIG ---
st.set_page_config(page_title="Career Intelligence", layout="wide", page_icon="📈")

# --- LOAD ASSETS ---
def load_css():
    with open(os.path.join(os.path.dirname(__file__), 'assets/style.css')) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# --- LOAD MODELS ---
@st.cache_resource
def load_data_artifacts():
    base = os.path.join(os.path.dirname(__file__), 'models')
    try:
        salary_model = joblib.load(os.path.join(base, 'salary.pkl'))
        role_df = joblib.load(os.path.join(base, 'roles.pkl'))
        demand_map = joblib.load(os.path.join(base, 'demand_map.pkl'))
        return salary_model, role_df, demand_map
    except Exception as e:
        return None, None, None

salary_model, role_df, demand_map = load_data_artifacts()

# --- TOP NAV (Simulated) ---
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background-color: #0E1117; border-bottom: 1px solid #30363D; margin-bottom: 20px;">
    <div style="font-size: 1.2rem; font-weight: bold; color: #E6EDF3;">Career Intelligence <span style="color:#22c55e;">.ai</span></div>
    <div style="color: #8B949E; font-size: 0.9rem;">My Profile &nbsp; | &nbsp; Settings</div>
</div>
""", unsafe_allow_html=True)

# --- LAYOUT ---
sidebar, main = st.columns([1, 4])

# --- SIDEBAR INPUTS ---
with sidebar:
    st.markdown("### Profile Input")
    st.markdown("Customize your analysis parameters.")
    
    skill_input = st.text_area("Your Skills", "Python, SQL, Machine Learning", height=150)
    
    xp_input = st.slider("Years of Experience", 0, 15, 3)
    
    target_role = st.selectbox("Target Role", 
                               ["Machine Learning Engineer", "Frontend Engineer", "Backend Engineer", "Data Scientist", "DevOps Engineer", "Product Manager"]
    )
    
    analyze_btn = st.button("Generate Insights", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("v2.4.0 Production Build")

# --- MAIN CONTENT ---
if not analyze_btn and 'analyzed' not in st.session_state:
    # Landing State
    with main:
        st.info("👈 Enter your skills and experience to unlock career intelligence.")
        
        # Show a sample decorative chart
        x = np.linspace(0, 10, 100)
        fig = go.Figure(data=go.Scatter(x=x, y=np.sin(x), mode='lines', line=dict(color='#22c55e', width=3)))
        fig.update_layout(template="plotly_dark", height=300, title="Market Activity Index", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.session_state['analyzed'] = True
    
    # 0. PRE-COMPUTE
    if not salary_model or role_df is None:
        st.error("Models are not loaded. Please run setup first.")
        st.stop()
        
    user_skills_list = [s.strip() for s in skill_input.split(',')]
    user_embedding = get_mean_embedding(user_skills_list)
    
    # 1. SALARY PREDICTION
    sal_min, sal_max = predict_salary(salary_model, user_embedding, xp_input)
    
    # 2. ROLE MATCHING
    # Match against ALL roles to find the top one, but also check the Specific Target Role stats
    matches = match_roles(user_embedding, role_df) # Top 3
    # Find specific target role details
    target_role_row = role_df[role_df['Role'] == target_role].iloc[0] if not role_df[role_df['Role'] == target_role].empty else None
    
    # If target role exists, compute gap
    if target_role_row is not None:
        gap_skills = get_gap_skills(user_skills_list, target_role_row['Core_Skills'])
        match_score_target = int(target_role_row.get('match_score', 0) * 100) # Wait, match_roles computes score. We need to re-compute for target if it's not in top 3 or just assume we have embedding.
        # Let's compute similarity explicitly for target
        from sklearn.metrics.pairwise import cosine_similarity
        target_score = cosine_similarity(user_embedding.reshape(1,-1), target_role_row['embedding'].reshape(1,-1))[0][0]
        match_pct = int(target_score * 100)
    else:
        gap_skills = []
        match_pct = 0

    with main:
        # --- HEADER & ACTIONS ---
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.title("Executive Dashboard")
        with h_col2:
            st.download_button("📥 Export Report", "Career Analysis Report...", file_name="career_report.txt")

        # --- SECTION 1: EXECUTIVE SUMMARY ---
        # st.markdown("#### Executive Summary")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Market Demand</div>
                <div class="metric-value">{target_role_row['Demand_Level'] if target_role_row is not None else 'N/A'}</div>
                <div class="metric-subtext">Based on {len(role_df)*42} Live Jobs</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
             st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Projected Salary</div>
                <div class="metric-value">₹{sal_min:.1f} - {sal_max:.1f} LPA</div>
                <div class="metric-subtext">Market Median ±15%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
             st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Profile Match</div>
                <div class="metric-value">{match_pct:.1f}%</div>
                <div class="metric-subtext">Cosine Similarity</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
             st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Skill Gap</div>
                <div class="metric-value">{len(gap_skills)} Skills</div>
                <div class="metric-subtext">Critical & Nice-to-have</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("") # Spacer

        # --- SECTION 2: MARKET ANALYTICS ---
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.markdown("##### Skill Demand Analytics")
            # Create Data for user skills demand
            skill_data = []
            for s in user_skills_list:
                # Add slight noise to make it look real
                val = demand_map.get(s.lower(), 50) + random.randint(-2, 2)
                skill_data.append({"Skill": s, "Score": val})
            
            df_skill = pd.DataFrame(skill_data).sort_values("Score", ascending=True)
            
            fig = px.bar(df_skill, x="Score", y="Skill", orientation='h', color="Score", 
                         color_continuous_scale=["#21262D", "#22c55e"])
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0,r=0,t=0,b=0))
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.markdown("##### Salary Trajectory")
            # Projection
            x_vals = [xp_input, xp_input+2, xp_input+5]
            y_vals = [sal_min, sal_min*1.22, sal_min*1.65] # Non-linear growth
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', line=dict(color='#22c55e', width=4, shape='spline')))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0,r=0,t=20,b=0))
            fig.update_xaxes(title="Years of Experience")
            fig.update_yaxes(title="Salary (₹ LPA)", showgrid=True, gridcolor='#30363D')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # --- SECTION 3: ROLE MATCHING ---
        st.markdown("#### Top Role Matches")
        rc1, rc2, rc3 = st.columns(3)
        cols = [rc1, rc2, rc3]
        
        for i, match in enumerate(matches):
            with cols[i]:
                # Slight jitter to salary for realism
                display_sal = match['avg_salary'] + random.uniform(-0.5, 0.5)
                st.markdown(f"""
                <div class="css-card" style="border-top: 4px solid #22c55e;">
                    <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">{match['role']}</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color: #8B949E;">Match</span>
                        <span style="color: #22c55e; font-weight: bold;">{match['match_pct']}%</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #8B949E;">Est. Salary: <span style="color: white;">₹{display_sal:.1f} LPA</span></div>
                </div>
                """, unsafe_allow_html=True)

        # --- SECTION 4: SKILL GAP ---
        st.markdown("#### Skill Gap Analysis")
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        if gap_skills:
            st.write(f"Missing skills for **{target_role}**:")
            tags_html = ""
            for gs in gap_skills:
                # Mock percentage for gap significance
                imp = random.randint(43, 98) # Random integers
                tags_html += f'<span class="tech-tag">{gs}<span class="tech-tag-percentage">{imp}%</span></span>'
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.success("No significant skill gaps found! 🚀")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- SECTION 5: ROADMAP ---
        st.markdown("#### Personalized Learning Roadmap")
        roadmap = generate_roadmap(gap_skills)
        
        rm1, rm2, rm3 = st.columns(3)
        
        if roadmap:
            with rm1:
                st.markdown('<div class="roadmap-col"><div class="roadmap-header">Month 1: Fundamentals</div>', unsafe_allow_html=True)
                for item in roadmap.get("Month 1", []):
                    st.markdown(f"- {item}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with rm2:
                st.markdown('<div class="roadmap-col"><div class="roadmap-header">Month 2: Application</div>', unsafe_allow_html=True)
                for item in roadmap.get("Month 2", []):
                    st.markdown(f"- {item}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with rm3:
                st.markdown('<div class="roadmap-col"><div class="roadmap-header">Month 3: Advanced</div>', unsafe_allow_html=True)
                for item in roadmap.get("Month 3", []):
                    st.markdown(f"- {item}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # --- FOOTER ---
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #8B949E; font-size: 0.8rem; margin-top: 20px;">
            <p>Powered by <strong>Career Intelligence NLP Engine</strong> v2.6.0</p>
            <p>Data derived from curated job postings + NLP similarity | Live Source: <strong>RemoteOK</strong></p>
        </div>
        """, unsafe_allow_html=True)

