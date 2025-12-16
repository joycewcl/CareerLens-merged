"""
Visualization components for job matching analysis.

Contains enhanced visualization functions for displaying job match analytics,
skill distributions, and comparative charts.
"""

import streamlit as st
from collections import Counter
import datetime
from typing import Dict, List

# Lazy imports for heavy visualization libraries
_pd = None
_plt = None
_go = None
_np = None


def _get_pandas():
    """Lazy load pandas"""
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def _get_matplotlib():
    """Lazy load matplotlib"""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_plotly():
    """Lazy load plotly"""
    global _go
    if _go is None:
        import plotly.graph_objects as go
        _go = go
    return _go


def _get_numpy():
    """Lazy load numpy"""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np



import streamlit as st
import plotly.graph_objs as go
from collections import Counter

def create_enhanced_visualizations(matched_jobs, job_seeker_data=None):
    if not matched_jobs or len(matched_jobs) == 0:
        st.info("No matched jobs available for visualization.")
        return

    job_titles = []
    sim_scores = []
    skill_scores = []
    exp_scores = []
    match_percentages = []
    avg_salaries = []
    salary_labels = []
    industries = []
    employment_types = []
    posting_dates = []
    skill_match_counts = []
    missing_skill_counts = []

    for job in matched_jobs:
        label = f"{job.get('job_title', 'N/A')} @ {job.get('company_name', '')}"
        job_titles.append(label)
        sim_scores.append(job.get("cosine_similarity_score", 0))
        skill_scores.append(job.get("skill_match_score", 0))
        exp_scores.append(job.get("experience_match_score", 0))
        match_percentages.append(job.get("match_percentage", 0))

        # Salary
        sal_min = job.get("salary_min")
        sal_max = job.get("salary_max")
        if sal_min is not None and sal_max is not None:
            try:
                salary = (float(sal_min) + float(sal_max)) / 2
                labelstr = f"{sal_min:.0f}-{sal_max:.0f}"
            except Exception:
                salary = None
                labelstr = "N/A"
        elif sal_min is not None:
            salary = float(sal_min)
            labelstr = f"{sal_min:.0f}"
        elif sal_max is not None:
            salary = float(sal_max)
            labelstr = f"{sal_max:.0f}"
        else:
            salary = None
            labelstr = "N/A"
        avg_salaries.append(salary)
        salary_labels.append(labelstr)

        # Industry & employment type
        industries.append(job.get("industry", "N/A"))
        employment_types.append(job.get("employment_type", "N/A"))

        # Post date
        dtxt = job.get("posted_date")
        if dtxt:
            posting_dates.append(str(dtxt))

        # Skills
        skill_val = job.get("matched_skills")
        if isinstance(skill_val, list):
            skill_match_counts.append(len(skill_val))
        elif isinstance(skill_val, str):
            skill_match_counts.append(len([x.strip() for x in skill_val.split(",") if x.strip()]))
        else:
            skill_match_counts.append(0)

        miss_val = job.get("missing_skills")
        if isinstance(miss_val, list):
            missing_skill_counts.append(len(miss_val))
        elif isinstance(miss_val, str):
            missing_skill_counts.append(len([x.strip() for x in miss_val.split(",") if x.strip()]))
        else:
            missing_skill_counts.append(0)

    # 1. Match Score Comparison
    st.subheader("Match Scores for Each Job")
    match_fig = go.Figure()
    match_fig.add_trace(go.Bar(x=job_titles, y=sim_scores, name="Cosine Similarity"))
    match_fig.add_trace(go.Bar(x=job_titles, y=skill_scores, name="Skill Match Score"))
    match_fig.add_trace(go.Bar(x=job_titles, y=exp_scores, name="Experience Match Score"))
    match_fig.add_trace(go.Bar(x=job_titles, y=match_percentages, name="Match Percentage"))
    match_fig.update_layout(barmode='group', xaxis_tickangle=-45, yaxis=dict(title="Score / Percent"))
    st.plotly_chart(match_fig, use_container_width=True)

    # 2. Salary Distribution
    st.subheader("Average Salary per Job")
    if any(s is not None for s in avg_salaries):
        base_salary = [s if s is not None else 0 for s in avg_salaries]
        salary_fig = go.Figure([go.Bar(x=job_titles, y=base_salary, text=salary_labels, textposition='auto')])
        salary_fig.update_layout(xaxis_tickangle=-45, yaxis_title="Average Salary")
        st.plotly_chart(salary_fig, use_container_width=True)

    # 3. Industry Distribution
    st.subheader("Industry Distribution")
    inds = [i for i in industries if i and i != "N/A"]
    if inds:
        inds_ct = Counter(inds)
        fig = go.Figure([go.Bar(x=list(inds_ct.keys()), y=list(inds_ct.values()))])
        fig.update_layout(yaxis_title="Number of Jobs")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No industry data available.")

    # 4. Employment Type Frequencies
    st.subheader("Employment Type Frequencies")
    etypes = [e for e in employment_types if e and e != "N/A"]
    if etypes:
        emp_ct = Counter(etypes)
        fig = go.Figure([go.Bar(x=list(emp_ct.keys()), y=list(emp_ct.values()))])
        fig.update_layout(yaxis_title="Number of Jobs")
        st.plotly_chart(fig, use_container_width=True)

    # 5. Posting Date Histogram
    if posting_dates:
        st.subheader("Job Posting Trend")
        date_ct = Counter(posting_dates)
        xs = sorted(date_ct.keys())
        ys = [date_ct[x] for x in xs]
        fig = go.Figure([go.Bar(x=xs, y=ys)])
        fig.update_layout(xaxis_title="Posting Date", yaxis_title="Jobs Posted")
        st.plotly_chart(fig, use_container_width=True)

    # 6. Skill Match/Gap Comparison
    st.subheader("Matched and Missing Skill Counts per Job")
    skills_fig = go.Figure()
    skills_fig.add_trace(go.Bar(x=job_titles, y=skill_match_counts, name='Matched Skills Count'))
    skills_fig.add_trace(go.Bar(x=job_titles, y=missing_skill_counts, name='Missing Skills Count'))
    skills_fig.update_layout(barmode='group', xaxis_tickangle=-45, yaxis_title='Skill Count')
    st.plotly_chart(skills_fig, use_container_width=True)


def create_job_comparison_radar(matched_jobs: list[dict]):
    """Create radar chart for top 3 job comparisons"""
    
    # Lazy load plotly only when radar chart is created
    go = _get_plotly()
        
    try:
        st.markdown("### 📊 Job Comparison Radar")
        
        # Define comparison categories
        categories = ['Skill Match', 'Role Relevance', 'Total Fit', 'Location Match', 'Salary Alignment']
        
        # Calculate scores for each category (simplified for demo)
        job_scores = []
        for job in matched_jobs[:1]:
            scores = [
                job.get('skill_match_percentage', 0),
                job.get('semantic_score', 0),
                job.get('combined_score', 0),  # Simulated experience fit
                75,  # Simulated location match
                70   # Simulated salary alignment
            ]
            job_scores.append(scores)
        
        fig = go.Figure()

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, scores in enumerate(job_scores):
            job_title = matched_jobs[i].get('title', f'Job {i+1}')[:25]
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],  # Close the radar
                theta=categories + [categories[0]],
                fill='toself',
                name=f"{job_title}",
                line=dict(color=colors[i], width=2),
                opacity=0.7
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            showlegend=True,
            title=dict(
                text="Multi-dimensional Job Comparison",
                x=0.5,
                font=dict(size=16)
            ),
            height=500,
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
