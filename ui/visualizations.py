"""
Visualization components for job matching analysis.

Contains enhanced visualization functions for displaying job match analytics,
skill distributions, and comparative charts.
"""

import streamlit as st
from typing import List, Dict, TYPE_CHECKING

# Type checking imports for Pylance
if TYPE_CHECKING:
    import matplotlib.pyplot

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


def create_enhanced_visualizations(matched_jobs: List[Dict], job_seeker_data: Dict = None):
    """Create enhanced visualizations for job matching analysis"""
    if not matched_jobs:
        st.warning("No visualization data available - no jobs matched")
        return

    # Lazy load heavy libraries only when visualizations are created
    np = _get_numpy()
    plt = _get_matplotlib()
    pd = _get_pandas()

    st.markdown("---")
    st.subheader("📊 Advanced Match Analysis")
    
    try:
        # 1. Score comparison chart
        st.markdown("### 🎯 Match Score Breakdown")
        
        # Prepare data for top 5 jobs
        top_jobs = matched_jobs[:5]
        jobs_display = [f"Job {i+1}" for i in range(len(top_jobs))]
        combined_scores = [j.get('combined_score', 0) for j in top_jobs]
        semantic_scores = [j.get('semantic_score', 0) for j in top_jobs]
        skill_scores = [j.get('skill_match_percentage', 0) for j in top_jobs]

        # Create detailed score comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Scores by job - improved styling
        x = np.arange(len(jobs_display))
        width = 0.25

        bars1 = ax1.bar(x - width, combined_scores, width, label='Combined', color='#7e22ce', alpha=0.8)
        bars2 = ax1.bar(x, semantic_scores, width, label='Semantic', color='#3b82f6', alpha=0.8)
        bars3 = ax1.bar(x + width, skill_scores, width, label='Skills', color='#10b981', alpha=0.8)

        ax1.set_xlabel('Jobs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Match Scores by Job Position', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(jobs_display, fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 2. Enhanced skill distribution
        all_skills = []
        skill_categories = {}
        
        for job in matched_jobs:
            skills = job.get('matched_skills', [])
            all_skills.extend(skills)
            
            # Categorize skills
            for skill in skills:
                skill_lower = skill.lower()
                if any(keyword in skill_lower for keyword in ['python', 'java', 'c++', 'javascript', 'sql', 'r']):
                    category = 'Programming'
                elif any(keyword in skill_lower for keyword in ['machine learning', 'ai', 'deep learning', 'nlp', 'computer vision']):
                    category = 'AI/ML'
                elif any(keyword in skill_lower for keyword in ['tableau', 'power bi', 'excel', 'analysis', 'analytics']):
                    category = 'Analytics'
                elif any(keyword in skill_lower for keyword in ['project', 'management', 'leadership', 'team']):
                    category = 'Management'
                elif any(keyword in skill_lower for keyword in ['communication', 'presentation', 'writing', 'english']):
                    category = 'Communication'
                else:
                    category = 'Other'
                
                skill_categories[category] = skill_categories.get(category, 0) + 1

        # Skill frequency chart
        skill_counts = pd.Series(all_skills).value_counts().head(10)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(skill_counts)))
        bars = ax2.barh(range(len(skill_counts)), skill_counts.values, color=colors, alpha=0.8)
        
        ax2.set_yticks(range(len(skill_counts)))
        ax2.set_yticklabels(skill_counts.index, fontsize=10)
        ax2.set_xlabel('Frequency Across All Matched Jobs', fontsize=12, fontweight='bold')
        ax2.set_title('Most Common Matched Skills', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
                    str(int(width)), ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 3. Job Match Quality Distribution
        st.markdown("### 📈 Job Match Quality Distribution")
        
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Score distribution histogram
        all_scores = [job.get('combined_score', 0) for job in matched_jobs]
        ax3.hist(all_scores, bins=10, color='#8b5cf6', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Combined Match Score (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Jobs', fontsize=12, fontweight='bold')
        ax3.set_title('Distribution of Match Scores', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        avg_score = np.mean(all_scores)
        ax3.axvline(avg_score, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_score:.1f}%')
        ax3.legend()

        # Skill match vs semantic match scatter plot
        semantic_scores_all = [job.get('semantic_score', 0) for job in matched_jobs]
        skill_scores_all = [job.get('skill_match_percentage', 0) for job in matched_jobs]
        
        scatter = ax4.scatter(semantic_scores_all, skill_scores_all, 
                             c=all_scores, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Semantic Match Score (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Skill Match Score (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Semantic vs Skill Match Correlation', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Combined Score (%)', fontsize=10, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # 4. Detailed Skill Analysis
        st.markdown("### 🔧 Detailed Skill Analysis")
        
        if job_seeker_data:
            candidate_skills = set()
            hard_skills = job_seeker_data.get('hard_skills', '')
            if hard_skills:
                candidate_skills.update([skill.strip().lower() for skill in hard_skills.split(',')])
            
            # Analyze skill coverage
            total_required_skills = set()
            matched_skills_per_job = []
            
            for job in matched_jobs[:5]:  # Top 5 jobs
                required_skills = set(job.get('matched_skills', []))
                total_required_skills.update(required_skills)
                matched_skills_per_job.append(len(required_skills))
            
            skill_coverage = len(total_required_skills) / len(candidate_skills) * 100 if candidate_skills else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Unique Skills", len(candidate_skills))
            with col2:
                st.metric("Skills Required by Top Jobs", len(total_required_skills))
            with col3:
                st.metric("Skill Coverage", f"{skill_coverage:.1f}%")

        # 5. Job Quality Indicators
        st.markdown("### 🎯 Job Quality Indicators")
        
        # Calculate various metrics
        high_match_jobs = len([j for j in matched_jobs if j.get('combined_score', 0) >= 80])
        avg_semantic = np.mean([j.get('semantic_score', 0) for j in matched_jobs])
        avg_skill = np.mean([j.get('skill_match_percentage', 0) for j in matched_jobs])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", len(matched_jobs))
        with col2:
            st.metric("High Quality Matches", high_match_jobs)
        with col3:
            st.metric("Avg Semantic Match", f"{avg_semantic:.1f}%")
        with col4:
            st.metric("Avg Skill Match", f"{avg_skill:.1f}%")

        # 6. Recommendations based on analysis
        st.markdown("### 💡 Personalized Recommendations")
        
        if avg_skill < 50:
            st.warning("**Skill Development Opportunity**: Your skill match is relatively low. Consider:")
            st.write("- Focus on learning the most frequently required skills shown above")
            st.write("- Take online courses for high-demand technologies")
            st.write("- Work on projects that demonstrate these skills")
        
        if avg_semantic > avg_skill:
            st.info("**Strength in Role Fit**: Your experience and background are well-aligned with these roles, even if specific skills need development.")
        
        if high_match_jobs >= 3:
            st.success("**Strong Market Position**: You have multiple high-quality matches! Focus on applying to these top positions.")

    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        st.info("Please try again with different search parameters")


def create_job_comparison_radar(matched_jobs: List[Dict]):
    """Create radar chart for top 3 job comparisons"""
    if len(matched_jobs) < 2:
        return
    
    # Lazy load plotly only when radar chart is created
    go = _get_plotly()
        
    try:
        st.markdown("### 📊 Job Comparison Radar")
        
        # Define comparison categories
        categories = ['Skill Match', 'Role Relevance', 'Experience Fit', 'Location Match', 'Salary Alignment']
        
        # Calculate scores for each category (simplified for demo)
        job_scores = []
        for job in matched_jobs[:3]:
            scores = [
                job.get('skill_match_percentage', 0),
                job.get('semantic_score', 0),
                min(job.get('semantic_score', 0) * 0.8, 100),  # Simulated experience fit
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
