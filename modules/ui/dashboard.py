"""Dashboard display components"""
import streamlit as st
import pandas as pd
import gc
from backend import calculate_salary_band
from modules.analysis import filter_jobs_by_domains, filter_jobs_by_salary
from modules.semantic_search import SemanticJobSearch, fetch_jobs_with_cache, generate_and_store_resume_embedding
from modules.utils import get_embedding_generator, get_job_scraper, get_text_generator
from modules.utils.config import _determine_index_limit


def calculate_match_scores(jobs, user_skills_str):
    """Calculate detailed match scores using main app formula.
    
    Formula: 60% semantic similarity + 40% skill match (all on 0-100 scale)
    Based on _calculate_match_scores from streamlit_app.py
    
    Edge cases handled:
    - Empty user skills: skill_match_pct = 0
    - Out-of-bounds semantic scores: clamped to 0-100
    - Missing job data: graceful fallbacks
    """
    if not user_skills_str:
        user_skills_str = ''
    
    candidate_skills = set([s.lower().strip() for s in str(user_skills_str).split(',') if s.strip()])
    total_skills = len(candidate_skills)
    
    for job_result in jobs:
        # Handle both nested 'job' structure and direct job properties
        job = job_result.get('job', job_result)
        description = job.get('description', '').lower()
        title = job.get('title', '').lower()
        
        # Count skill matches
        matched_skills = []
        for skill in candidate_skills:
            if skill in description or skill in title:
                matched_skills.append(skill)
        
        # Handle edge cases for skill match calculation
        if total_skills == 0:
            skill_match_pct = 0  # No skills to match against
        else:
            skill_match_pct = (len(matched_skills) / total_skills) * 100
        
        # Semantic similarity - handle both 0-1 scale and 0-100 scale
        raw_semantic = job_result.get('similarity_score', 0)
        if raw_semantic <= 1:
            raw_semantic = raw_semantic * 100  # Normalize to 0-100
        
        # Ensure semantic_score is bounded (0-100)
        semantic_score = max(0, min(100, raw_semantic))
        
        # Combined score: 60% semantic + 40% skill match (0-100 scale)
        combined_score = (0.6 * semantic_score) + (0.4 * skill_match_pct)
        
        # Ensure combined score is also bounded
        combined_score = max(0, min(100, combined_score))
        
        # Get missing skills
        job_skills = job.get('skills', [])
        job_skills_lower = [s.lower().strip() for s in job_skills if isinstance(s, str)]
        missing_skills = [js for js in job_skills_lower if not any(cs in js or js in cs for cs in candidate_skills)]
        
        # Add scores to result (0-100 scale)
        job_result['skill_match_percentage'] = round(skill_match_pct, 1)
        job_result['skill_match_score'] = round(skill_match_pct, 1)  # Alias for compatibility
        job_result['matched_skills'] = list(matched_skills)[:10]
        job_result['matched_skills_count'] = len(matched_skills)
        job_result['missing_skills'] = missing_skills[:5]
        job_result['combined_score'] = round(combined_score, 1)
        job_result['combined_match_score'] = round(combined_score, 1)  # Alias for compatibility
        job_result['semantic_score'] = round(semantic_score, 1)
    
    return jobs


def display_skill_matching_matrix(user_profile):
    """Display skill matching calculation matrix to help users understand ranking
    
    Uses layout from dashboard with flexible field access from main app.
    """
    st.markdown("---")
    st.markdown("### 📊 How Job Ranking Works")
    
    # Support both 'hard_skills' and 'skills' fields (from streamlit_app.py)
    user_skills = user_profile.get('hard_skills', '') if user_profile else ''
    if not user_skills:
        user_skills = user_profile.get('skills', '') if user_profile else ''
    
    if not user_skills:
        st.info("💡 **Skill-Based Ranking**: Jobs are ranked by how many required skills you match. Upload your CV to see your skills analyzed.")
        return
    
    user_skills_list = [s.strip() for s in str(user_skills).split(',') if s.strip()]
    
    if not user_skills_list:
        st.info("💡 **Skill-Based Ranking**: Jobs are ranked by how many required skills you match.")
        return
    
    st.markdown("#### Your Skills")
    skills_display = ", ".join(user_skills_list[:10])
    if len(user_skills_list) > 10:
        skills_display += f" (+{len(user_skills_list) - 10} more)"
    st.markdown(f"**{len(user_skills_list)} skills identified:** {skills_display}")
    
    st.markdown("---")
    
    st.markdown("#### Ranking Formula")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Skill Match Score =**
        
        ```
        Matched Skills
        ──────────────
        Required Skills
        ```
        
        **Example:**
        - Job requires: Python, SQL, React, Docker
        - You have: Python, SQL, React
        - **Score: 3/4 = 75%**
        """)
    
    with col2:
        st.markdown("""
        **Ranking Logic:**
        
        1. ✅ Jobs are fetched from job boards
        2. 🔍 Your skills are matched against each job
        3. 📊 Jobs sorted by match score (highest first)
        4. 🎯 Top matches appear at the top
        """)
    
    st.markdown("---")
    
    st.markdown("#### Matching Method")
    
    method_col1, method_col2 = st.columns(2)
    
    with method_col1:
        st.markdown("""
        **Semantic Matching** (Primary)
        - Uses AI embeddings to understand skill similarity
        - Recognizes related skills (e.g., "JavaScript" ≈ "JS")
        - Handles variations and synonyms
        - Threshold: 70% similarity required
        """)
    
    with method_col2:
        st.markdown("""
        **String Matching** (Fallback)
        - Used when semantic matching unavailable
        - Direct text comparison
        - Case-insensitive matching
        - Handles partial matches
        """)
    
    if 'matched_jobs' in st.session_state and st.session_state.matched_jobs:
        st.markdown("---")
        st.markdown("#### Example: Top Match Breakdown")
        
        top_match = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
        if top_match:
            # Handle both nested 'job' structure and direct job properties
            job = top_match.get('job', top_match)
            job_skills = job.get('skills', []) or job.get('matched_skills', [])
            # Handle both 0-1 scale and 0-100 scale scores
            skill_score = top_match.get('skill_match_score', top_match.get('skill_match_percentage', 0))
            if skill_score > 1:
                skill_score = skill_score / 100  # Normalize to 0-1
            matched_count = int(skill_score * len(job_skills)) if job_skills else 0
            
            if job_skills:
                st.markdown(f"**{job.get('title', 'Job')} at {job.get('company', 'Company')}**")
                st.markdown(f"**Match Score: {int(skill_score * 100)}%** ({matched_count}/{len(job_skills)} skills matched)")
                
                job_skills_lower = [s.lower().strip() for s in job_skills if isinstance(s, str)]
                user_skills_lower = [s.lower().strip() for s in user_skills_list]
                
                matched_skills_list = []
                missing_skills_list = []
                
                for js in job_skills_lower:
                    matched = False
                    for us in user_skills_lower:
                        if js in us or us in js:
                            matched_skills_list.append(js)
                            matched = True
                            break
                    if not matched:
                        missing_skills_list.append(js)
                
                if matched_skills_list:
                    st.success(f"✅ **Matched Skills:** {', '.join(matched_skills_list[:5])}")
                if missing_skills_list:
                    st.warning(f"⚠️ **Missing Skills:** {', '.join(missing_skills_list[:5])}")


def display_market_positioning_profile(matched_jobs, user_profile):
    """Display Dashboard with 3 key metric cards: Match Score, Est. Salary, Skill Gaps
    
    Uses dashboard layout with flexible data access from main app to handle both
    job seeker page results and market dashboard results.
    """
    if not matched_jobs:
        return
    
    # Calculate average match score - handle both 'combined_score' and 'combined_match_score'
    # Also handle both 0-1 scale and 0-100 scale (from streamlit_app.py)
    avg_match_score = 0
    for result in matched_jobs:
        if isinstance(result, dict):
            score = result.get('combined_score', result.get('combined_match_score', 0))
            avg_match_score += score
    avg_match_score = avg_match_score / len(matched_jobs) if matched_jobs else 0
    
    # Normalize score if it's already a percentage (0-100 scale)
    if avg_match_score > 1:
        match_score_pct = int(avg_match_score)
    else:
        match_score_pct = int(avg_match_score * 100)
    
    if match_score_pct >= 80:
        match_delta = "Excellent fit"
    elif match_score_pct >= 60:
        match_delta = "Good fit"
    else:
        match_delta = "Room to improve"
    
    # Calculate salary band
    salary_min, salary_max = calculate_salary_band(matched_jobs)
    avg_salary = (salary_min + salary_max) // 2
    
    user_salary_expectation = st.session_state.get('salary_expectation', 0)
    if user_salary_expectation > 0:
        salary_delta_pct = ((avg_salary - user_salary_expectation) / user_salary_expectation * 100) if user_salary_expectation > 0 else 0
        if salary_delta_pct > 0:
            salary_delta = f"+{salary_delta_pct:.0f}% vs target"
        elif salary_delta_pct < 0:
            salary_delta = f"{salary_delta_pct:.0f}% vs target"
        else:
            salary_delta = "Matches target"
    else:
        salary_delta = "Market rate"
    
    # Calculate skill gaps - support both 'hard_skills' and 'skills' fields
    user_skills = user_profile.get('hard_skills', '') or user_profile.get('skills', '')
    all_job_skills = []
    for result in matched_jobs:
        # Handle both nested 'job' structure and direct job properties
        job = result.get('job', result)
        job_skills = job.get('skills', []) or job.get('matched_skills', [])
        all_job_skills.extend(job_skills)
    
    user_skills_list = [s.lower().strip() for s in str(user_skills).split(',') if s.strip()]
    skill_gaps = set()
    for job_skill in all_job_skills:
        if isinstance(job_skill, str):
            job_skill_lower = job_skill.lower().strip()
            if job_skill_lower and not any(us in job_skill_lower or job_skill_lower in us for us in user_skills_list):
                skill_gaps.add(job_skill_lower)
    
    num_skill_gaps = min(len(skill_gaps), 20)  # Cap at 20
    
    if num_skill_gaps <= 3:
        gap_delta = "Well positioned"
    elif num_skill_gaps <= 7:
        gap_delta = "Some upskilling needed"
    else:
        gap_delta = "Focus on learning"
    
    st.markdown("### 📊 Your Market Position Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="dashboard-metric-card">
            <div class="dashboard-metric-label">🎯 Match Score</div>
            <div class="dashboard-metric-value">{match_score_pct}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"📈 {match_delta}")
    
    with col2:
        st.markdown(f"""
        <div class="dashboard-metric-card">
            <div class="dashboard-metric-label">💰 Est. Salary</div>
            <div class="dashboard-metric-value">HKD {avg_salary // 1000}k</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"💰 {salary_delta}")
    
    with col3:
        st.markdown(f"""
        <div class="dashboard-metric-card">
            <div class="dashboard-metric-label">📈 Skill Gaps</div>
            <div class="dashboard-metric-value">{num_skill_gaps}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"🎯 {gap_delta}")
    
    # Show top skill gaps in an expander (from streamlit_app.py)
    if skill_gaps:
        with st.expander("🔧 Top Skills to Develop"):
            gap_list = list(skill_gaps)[:10]
            cols = st.columns(2)
            for i, skill in enumerate(gap_list):
                with cols[i % 2]:
                    st.write(f"• {skill}")


def display_refine_results_section(matched_jobs, user_profile):
    """Display Refine Results section with filters"""
    st.markdown("---")
    with st.expander("🔧 Refine Results", expanded=False):
        st.markdown("### Adjust Search Criteria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_domains = st.session_state.get('target_domains', [])
            target_domains = st.multiselect(
                "Target Domains (HK Focus)",
                options=["FinTech", "ESG & Sustainability", "Data Analytics", "Digital Transformation", 
                        "Investment Banking", "Consulting", "Technology", "Healthcare", "Education"],
                default=current_domains,
                key="refine_domains"
            )
        
        with col2:
            current_salary = st.session_state.get('salary_expectation', 0)
            salary_expectation = st.slider(
                "Min. Monthly Salary (HKD)",
                min_value=0,
                max_value=150000,
                value=current_salary,
                step=5000,
                help="Set to 0 to disable salary filtering",
                key="refine_salary"
            )
        
        force_refresh = st.checkbox(
            "Force new API fetch",
            value=False,
            help="Bypass cached results (only if results seem stale).",
            key="force_refresh_jobs_toggle"
        )
        
        # Check if user has uploaded a profile
        has_profile = (
            st.session_state.get('resume_text') or 
            (st.session_state.get('user_profile') and st.session_state.user_profile.get('skills'))
        )
        
        button_text = "🔄 Apply Filters & Refresh" if st.session_state.get('matched_jobs') else "🔍 Search Jobs"
        
        if st.button(button_text, type="primary", use_container_width=True):
            if not has_profile:
                st.warning("⚠️ Please upload your CV in the sidebar first to get personalized job matches.")
                return
            
            st.session_state.target_domains = target_domains
            st.session_state.salary_expectation = salary_expectation
            
            search_query = " ".join(target_domains) if target_domains else "Hong Kong jobs"
            scraper = get_job_scraper()
            
            if scraper is None:
                st.error("⚠️ Job scraper not configured.")
                return
            
            with st.spinner("🔄 Refreshing results from Indeed..."):
                jobs = fetch_jobs_with_cache(
                    scraper,
                    search_query,
                    location="Hong Kong",
                    max_rows=25,
                    job_type="fulltime",
                    country="hk",
                    force_refresh=force_refresh
                )
                
                if not jobs:
                    st.error("❌ No jobs found from Indeed.")
                    return
                
                total_fetched = len(jobs)
                
                if target_domains:
                    jobs = filter_jobs_by_domains(jobs, target_domains)
                
                if salary_expectation > 0:
                    jobs = filter_jobs_by_salary(jobs, salary_expectation)
                
                if not jobs:
                    st.warning(f"⚠️ No jobs match your filters. Found {total_fetched} jobs but none passed your criteria.")
                    return
                
                embedding_gen = get_embedding_generator()
                desired_matches = min(15, len(jobs))
                jobs_to_index_limit = _determine_index_limit(len(jobs), desired_matches)
                top_match_count = min(desired_matches, jobs_to_index_limit)
                search_engine = SemanticJobSearch(embedding_gen)
                search_engine.index_jobs(jobs, max_jobs_to_index=jobs_to_index_limit)
                
                resume_embedding = st.session_state.get('resume_embedding')
                if not resume_embedding and st.session_state.resume_text:
                    resume_embedding = generate_and_store_resume_embedding(
                        st.session_state.resume_text,
                        st.session_state.user_profile if st.session_state.user_profile else None
                    )
                
                resume_query = None
                if not resume_embedding:
                    if st.session_state.resume_text:
                        resume_query = st.session_state.resume_text
                        if st.session_state.user_profile.get('summary'):
                            profile_data = f"{st.session_state.user_profile.get('summary', '')} {st.session_state.user_profile.get('experience', '')} {st.session_state.user_profile.get('skills', '')}"
                            resume_query = f"{resume_query} {profile_data}"
                    else:
                        resume_query = f"{st.session_state.user_profile.get('summary', '')} {st.session_state.user_profile.get('experience', '')} {st.session_state.user_profile.get('skills', '')} {st.session_state.user_profile.get('education', '')}"
                
                results = search_engine.search(query=resume_query, top_k=top_match_count, resume_embedding=resume_embedding)
                
                # Calculate match scores using main app formula (0-100 scale)
                user_skills = st.session_state.user_profile.get('hard_skills', '') or st.session_state.user_profile.get('skills', '')
                results = calculate_match_scores(results, user_skills)
                
                results.sort(key=lambda x: x.get('combined_score', x.get('combined_match_score', 0)), reverse=True)
                
                st.session_state.matched_jobs = results
                st.session_state.dashboard_ready = True
                
                gc.collect()
                
                st.rerun()


def display_ranked_matches_table(matched_jobs, user_profile):
    """Display Smart Ranked Matches Table with interactive dataframe"""
    if not matched_jobs:
        return
    
    st.markdown("---")
    st.markdown("### Top AI-Ranked Opportunities")
    st.caption("💡 **Tip:** Click any row to expand and see full job description, match analysis, and application copilot")
    
    # Support both 'hard_skills' and 'skills' fields
    user_skills = user_profile.get('hard_skills', '') or user_profile.get('skills', '')
    
    # Ensure all results have scores calculated (if not already done)
    for result in matched_jobs:
        if 'combined_score' not in result and 'combined_match_score' not in result:
            # Calculate scores using centralized function
            calculate_match_scores([result], user_skills)
    
    # Sort by combined score (0-100 scale)
    matched_jobs.sort(key=lambda x: x.get('combined_score', x.get('combined_match_score', 0)), reverse=True)
    
    table_data = []
    for i, result in enumerate(matched_jobs):
        # Handle both nested 'job' structure and direct job properties
        job = result.get('job', result)
        
        # Get scores (0-100 scale)
        match_score = result.get('combined_score', result.get('combined_match_score', 0))
        
        job_skills = job.get('skills', [])
        matching_skills = []
        user_skills_list = [s.lower().strip() for s in str(user_skills).split(',') if s.strip()]
        for js in job_skills[:6]:
            if isinstance(js, str):
                js_lower = js.lower().strip()
                if any(us in js_lower or js_lower in us for us in user_skills_list):
                    matching_skills.append(js)
                    if len(matching_skills) >= 4:
                        break
        
        missing_critical = result.get('missing_skills', [])
        missing_critical_skill = missing_critical[0] if missing_critical else "None"
        
        # Score is already on 0-100 scale from calculate_match_scores
        # Handle edge case where old 0-1 scale data might still exist
        display_score = int(match_score) if match_score > 1 else int(match_score * 100)
        
        table_data.append({
            'Rank': i + 1,
            'Match Score': display_score,
            'Job Title': job.get('title', 'Unknown'),
            'Company': job.get('company', 'Unknown'),
            'Location': job.get('location', 'Unknown'),
            'Key Matching Skills': matching_skills[:4] if matching_skills else [],
            'Missing Critical Skill': missing_critical_skill,
            '_index': i
        })
    
    df = pd.DataFrame(table_data)
    
    column_config = {
        'Rank': st.column_config.NumberColumn(
            'Rank',
            help='Job ranking position (fixed, does not change when sorting)',
            width='small',
            format='%d'
        ),
        'Match Score': st.column_config.ProgressColumn(
            'Match Score',
            help='Combined match score: 60% semantic similarity + 40% skill overlap (jobs ranked by this)',
            min_value=0,
            max_value=100,
            format='%d%%'
        ),
        'Job Title': st.column_config.TextColumn(
            'Job Title',
            width='medium',
            help='Click to select and view full details'
        ),
        'Company': st.column_config.TextColumn(
            'Company',
            width='medium'
        ),
        'Location': st.column_config.TextColumn(
            'Location',
            width='small'
        ),
        'Key Matching Skills': st.column_config.ListColumn(
            'Key Matching Skills',
            help='Top skills you have that match this role'
        ),
        'Missing Critical Skill': st.column_config.TextColumn(
            'Missing Critical Skill',
            help='Most important skill gap for this role',
            width='medium'
        ),
        '_index': st.column_config.NumberColumn(
            '_index',
            width='small',
            help=None
        )
    }
    
    column_order = ['Rank', 'Match Score', 'Job Title', 'Company', 'Location', 'Key Matching Skills', 'Missing Critical Skill']
    
    df_display = df[column_order].copy()
    
    selected_rows = st.dataframe(
        df_display,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    if selected_rows.selection.rows:
        selected_idx = df.iloc[selected_rows.selection.rows[0]]['_index']
        st.session_state.selected_job_index = int(selected_idx)
    else:
        st.session_state.selected_job_index = None


def display_match_breakdown(matched_jobs, user_profile):
    """Display Match Breakdown & Application Copilot in expander"""
    if st.session_state.selected_job_index is None:
        return
    
    selected_result = matched_jobs[st.session_state.selected_job_index]
    # Handle both nested 'job' structure and direct job properties
    job = selected_result.get('job', selected_result)
    
    # Get scores - handle both 0-1 and 0-100 scales
    semantic_score = selected_result.get('semantic_score', selected_result.get('similarity_score', 0))
    skill_score = selected_result.get('skill_match_percentage', selected_result.get('skill_match_score', 0))
    missing_skills = selected_result.get('missing_skills', [])
    
    # Normalize to 0-100 scale if needed
    if semantic_score <= 1:
        semantic_score = semantic_score * 100
    if skill_score <= 1:
        skill_score = skill_score * 100
    
    combined_score = selected_result.get('combined_score', selected_result.get('combined_match_score', 0))
    if combined_score <= 1:
        combined_score = combined_score * 100
    
    # Support both 'hard_skills' and 'skills' fields
    user_skills = user_profile.get('hard_skills', '') or user_profile.get('skills', '')
    job_skills = job.get('skills', []) or job.get('matched_skills', [])
    user_skills_list = [s.lower().strip() for s in str(user_skills).split(',') if s.strip()]
    job_skills_list = [s.lower().strip() for s in job_skills if isinstance(s, str) and s.strip()]
    
    matched_skills_count = selected_result.get('matched_skills_count', 0)
    if matched_skills_count == 0:
        for js in job_skills_list:
            if any(us in js or js in us for us in user_skills_list):
                matched_skills_count += 1
    
    total_required = len(job_skills_list) if job_skills_list else 1
    
    text_gen = get_text_generator()
    if text_gen is None:
        recruiter_note = "AI analysis unavailable. Please configure Azure OpenAI credentials."
    else:
        recruiter_note = text_gen.generate_recruiter_note(job, user_profile, semantic_score / 100, skill_score / 100)
    
    rank_position = st.session_state.selected_job_index + 1 if st.session_state.selected_job_index is not None else 0
    
    expander_title = f"📋 Rank #{rank_position}: {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}"
    
    with st.expander(expander_title, expanded=True):
        st.markdown("#### 📝 Full Job Description")
        description_text = job.get('description', 'No description available.')
        if len(description_text) > 10000:
            st.info(f"📄 Full description ({len(description_text):,} characters)")
            st.text_area(
                "Job Description",
                value=description_text,
                height=400,
                key=f"job_desc_{st.session_state.selected_job_index}",
                label_visibility="collapsed"
            )
        else:
            st.markdown(description_text)
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎯 Match Analysis & Why This is a Fit")
            
            st.markdown(f"""
            **Match Score Breakdown:**
            - **🎯 Skill Match Score:** {skill_score:.1f}% ({matched_skills_count}/{total_required} skills matched)
            - **📊 Semantic Similarity Score:** {semantic_score:.1f}%
              - Measures how well your experience contextually aligns with role requirements.
            - **⚖️ Combined Match Score:** {combined_score:.1f}%
              - Weighted combination: 60% semantic + 40% skill overlap
            """)
            
            if matched_skills_count > 0:
                matched_skills_display = selected_result.get('matched_skills', [])
                if not matched_skills_display:
                    for js in job_skills_list:
                        if any(us in js or js in us for us in user_skills_list):
                            matched_skills_display.append(js)
                            if len(matched_skills_display) >= 10:
                                break
                if matched_skills_display:
                    st.success(f"✅ **Matched Skills:** {', '.join(matched_skills_display[:10])}")
            
            if missing_skills:
                st.warning(f"⚠️ **Missing Skills:** {', '.join(missing_skills[:5])}")
            
            st.markdown("---")
            st.info(f"**🤖 AI Recruiter Analysis:**\n\n{recruiter_note}")
        
        with col2:
            st.markdown("#### Application Copilot")
            
            if missing_skills:
                top_missing = missing_skills[0]
                cert_keywords = ['certification', 'certified', 'accreditation', 'license', 'pmp', 'scrum', 'hkicpa', 'cpa', 'cfa', 'cpa', 'aws', 'azure', 'gcp']
                is_cert = any(kw in top_missing.lower() for kw in cert_keywords)
                
                if is_cert:
                    st.warning(f"⚠️ **Crucial Gap:** This job highly values {top_missing}. Consider starting this certification.")
                else:
                    st.warning(f"⚠️ **Skill Gap:** Consider developing expertise in {top_missing}.")
            
            if st.button("✨ Tailor Resume for this Job", use_container_width=True, type="primary", key="tailor_resume_button"):
                st.session_state.selected_job = job
                st.session_state.show_resume_generator = True
                st.rerun()
            
            st.caption("Generates a citation-locked, AI-optimized CV emphasizing your matching skills.")
            
            job_url = job.get('url', '#')
            if job_url and job_url != '#':
                st.markdown("---")
                st.link_button("🚀 Apply to Job", job_url, use_container_width=True, type="secondary")
