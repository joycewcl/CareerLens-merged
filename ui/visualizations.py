"""
Visualization components for job matching analysis.

Contains enhanced visualization functions for displaying job match analytics,
skill distributions, and comparative charts.
"""

import streamlit as st
from collections import Counter
import datetime

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



def create_enhanced_visualizations(matched_jobs):
    if not matched_jobs or len(matched_jobs) == 0:
        st.info("No matched jobs available for visualization.")
        return
    go = _get_plotly()
    
    job_titles, similarity_scores, skill_scores, exp_scores = [], [], [], []
    avg_salaries, years_experience = [], []
    num_applicants, salary_labels = [], []
    company_ratings, rating_counts = [], []
    company_sizes, remote_flags = [], []
    urgent_flags, high_volume_flags = [], []
    job_types, benefit_counts, publishing_dates = [], [], []

    for job in matched_jobs:
        # Title/Company
        label = f"{job.get('title', 'N/A')} @ {job.get('companyName', '')}"
        job_titles.append(label)

        # Match scores (from your system)
        similarity_scores.append(job.get("similarity_score", 0))
        skill_scores.append(job.get("skill_match_score", 0))
        exp_scores.append(job.get("experience_match_score", 0))

        # Salary (same as before)
        sal_block = job.get("salary", {})
        sal_min, sal_max = sal_block.get("salaryMin"), sal_block.get("salaryMax")
        salary = (float(sal_min) + float(sal_max)) / 2 if sal_min is not None and sal_max is not None \
                 else float(sal_min) if sal_min is not None else float(sal_max) if sal_max is not None else None
        avg_salaries.append(salary)
        salary_labels.append(sal_block.get("salaryText", "N/A") if salary else "N/A")
        years_experience.append(None)  # Not present in API

        # Applicants
        num_applicants.append(job.get("numOfCandidates"))

        # Company rating
        rating = job.get("rating", {}).get("rating")
        company_ratings.append(rating)
        rating_counts.append(job.get("rating", {}).get("count"))

        # Company size/employee group
        company_sizes.append(job.get('companyNumEmployees', 'N/A'))

        # REMOTE flag
        remote_flags.append(bool(job.get('isRemote', False)))

        # Hiring urgency
        urgent_flags.append(bool(job.get('hiringDemand', {}).get('isUrgentHire')))
        high_volume_flags.append(bool(job.get('hiringDemand', {}).get('isHighVolumeHiring')))

        # Job types & benefit count
        job_types.extend(job.get("jobType", []))
        benefit_counts.append(len(job.get("benefits", [])))

        # Date published
        date_str = job.get("datePublished", None)
        try:
            if date_str:
                publishing_dates.append(datetime.date.fromisoformat(date_str))
        except Exception:
            pass

    # 1. Match Scores
    st.subheader("Match Scores for Each Job")
    match_fig = go.Figure()
    match_fig.add_trace(go.Bar(x=job_titles, y=similarity_scores, name="Similarity Score"))
    match_fig.add_trace(go.Bar(x=job_titles, y=skill_scores, name="Skill Match Score"))
    match_fig.add_trace(go.Bar(x=job_titles, y=exp_scores, name="Experience Match Score"))
    match_fig.update_layout(barmode='group', xaxis_tickangle=-45, yaxis=dict(title="Score"))
    st.plotly_chart(match_fig, use_container_width=True)

    # 2. Salary (Y) vs. Years (X: always None/0 now)
    st.subheader("Salary Distribution")
    if any(s is not None for s in avg_salaries):
        base_salary = [s if s is not None else 0 for s in avg_salaries]
        salary_fig = go.Figure([go.Bar(x=job_titles, y=base_salary, text=salary_labels, textposition='auto')])
        salary_fig.update_layout(xaxis_tickangle=-45, yaxis_title="Avg. Salary")
        st.plotly_chart(salary_fig, use_container_width=True)

    # 3. Applicants
    st.subheader("Number of Applicants per Job")
    if any(n is not None for n in num_applicants):
        appl_fig = go.Figure()
        appl_fig.add_trace(go.Bar(
            x=job_titles,
            y=[n if n is not None else 0 for n in num_applicants],
            name="Applicants",
            text=[str(n) if n is not None else "N/A" for n in num_applicants],
            textposition='auto'
        ))
        appl_fig.update_layout(
            xaxis_tickangle=-45,
            yaxis=dict(title="Number of Applicants"),
        )
        st.plotly_chart(appl_fig, use_container_width=True)
    else:
        st.caption("No applicant data available.")

    # 4. Company Ratings
    st.subheader("Company Ratings")
    if any(r is not None for r in company_ratings):
        fig = go.Figure([go.Bar(
            x=job_titles,
            y=[r if r is not None else 0 for r in company_ratings],
            text=[f"Based on {c} reviews" if c else "" for c in rating_counts],
            textposition="auto",
            marker_color='orange'
        )])
        fig.update_layout(xaxis_tickangle=-45, yaxis_title="Company Rating (out of 5)")
        st.plotly_chart(fig, use_container_width=True)

    # 5. Remote Ratio
    st.subheader("Remote vs. Non-Remote Jobs")
    remote_count = sum(remote_flags)
    nonremote_count = len(remote_flags) - remote_count
    if len(remote_flags) > 0:
        fig = go.Figure(data=[go.Pie(
            labels=["Remote", "Non-Remote"],
            values=[remote_count, nonremote_count],
            hole=0.3
        )])
        st.plotly_chart(fig, use_container_width=True)

    # 6. Urgent/High Volume Hiring Pie
    st.subheader("Urgent/High Volume Hiring")
    st.write("Urgent Hiring")
    urgent_n = sum(urgent_flags)
    not_urgent_n = len(urgent_flags) - urgent_n
    fig = go.Figure(data=[go.Pie(labels=["Urgent", "Not Urgent"], values=[urgent_n, not_urgent_n], hole=0.3)])
    st.plotly_chart(fig, use_container_width=True)
    st.write("High Volume Hiring")
    highvol_n = sum(high_volume_flags)
    nothighvol_n = len(high_volume_flags) - highvol_n
    fig2 = go.Figure(data=[go.Pie(labels=["High Volume", "Other"], values=[highvol_n, nothighvol_n], hole=0.3)])
    st.plotly_chart(fig2, use_container_width=True)

    # 7. Company size breakdown
    st.subheader("Company Size Distribution (by employees)")
    empgroups = [size for size in company_sizes if size and size != "N/A"]
    if empgroups:
        emp_counter = Counter(empgroups)
        fig = go.Figure(data=[go.Bar(x=list(emp_counter.keys()), y=list(emp_counter.values()))])
        fig.update_layout(yaxis_title="Num Jobs")
        st.plotly_chart(fig, use_container_width=True)

    # 8. Job Posting Trends (Dates Published)
    if publishing_dates:
        st.subheader("Job Posting Trend")
        date_counter = Counter(publishing_dates)
        xs = sorted(date_counter.keys())
        ys = [date_counter[x] for x in xs]
        fig = go.Figure(data=[go.Bar(x=[str(x) for x in xs], y=ys)])
        fig.update_layout(xaxis_title="Date Published", yaxis_title="Number of Jobs")
        st.plotly_chart(fig, use_container_width=True)

    # 9. Job Type Frequency
    if job_types:
        st.subheader("Job Type Frequencies")
        jt_counter = Counter(job_types)
        fig = go.Figure(data=[go.Bar(x=list(jt_counter.keys()), y=list(jt_counter.values()))])
        fig.update_layout(yaxis_title="Number of Jobs")
        st.plotly_chart(fig, use_container_width=True)

    # 10. Number of Benefits per job
    st.subheader("Number of Benefits per Job (Bar)")
    fig = go.Figure([go.Bar(x=job_titles, y=benefit_counts, marker_color='lightgreen')])
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Benefit Count")
    st.plotly_chart(fig, use_container_width=True)


def create_job_comparison_radar(matched_jobs: List[Dict]):
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
