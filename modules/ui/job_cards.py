"""Job card display components"""
import streamlit as st


def display_job_card(result, index):
    """Display a job card with match score and details"""
    job = result['job']
    score = result.get('similarity_score', 0.0)
    
    remote_badge = "🏠 Remote" if job['is_remote'] else ""
    rating = job['company_rating']
    stars = "⭐" * int(rating) if rating > 0 else ""
    
    st.markdown(f"""
    <div class="job-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
            <div style="flex-grow: 1;">
                <h3 style="margin: 0; color: var(--primary-accent);">#{index} {job['title']}</h3>
                <p style="margin: 0.5rem 0; color: var(--text-muted); font-size: 0.95rem;">
                    🏢 <strong>{job['company']}</strong> {stars} • 📍 {job['location']} {remote_badge}
                </p>
            </div>
            <div class="match-score">
                {score:.1%} Match
            </div>
        </div>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem; color: var(--text-muted);">
            <span>⏰ {job['job_type']}</span>
            <span>💰 {job['salary']}</span>
            <span>📅 {job['posted_date']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if job['benefits']:
            st.write("**Benefits:**")
            for benefit in job['benefits']:
                st.markdown(f'<span class="tag">✓ {benefit}</span>', unsafe_allow_html=True)
    
    with col2:
        if job['skills']:
            st.write("**Skills:**")
            skills_text = " ".join([f'<span class="tag">{skill}</span>' for skill in job['skills'][:8]])
            st.markdown(skills_text, unsafe_allow_html=True)
    
    st.write("")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        with st.expander("📝 View Full Description", expanded=False):
            description_text = job['description']
            if len(description_text) > 10000:
                st.info(f"📄 Full description ({len(description_text):,} characters)")
                st.text_area(
                    "Job Description",
                    value=description_text,
                    height=400,
                    key=f"desc_{index}",
                    label_visibility="collapsed"
                )
            else:
                st.write(description_text)
    
    with col2:
        col2a, col2b = st.columns(2)
        with col2a:
            if job['url'] != '#':
                st.link_button("Apply →", job['url'], use_container_width=True)
        with col2b:
            if st.button("📄 Resume", key=f"resume_{index}", use_container_width=True, type="primary"):
                st.session_state.selected_job = job
                st.session_state.show_resume_generator = True
                st.rerun()
