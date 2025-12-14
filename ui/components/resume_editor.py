"""Resume editor and generator UI components"""
import json
import streamlit as st
import time
import requests
from utils import get_text_generator, get_embedding_generator, api_call_with_retry
from ui.components.match_feedback import display_match_score_feedback

# Lazy imports for heavy resume generation modules (docx, reportlab)
_resume_formatters = None


def _get_resume_formatters():
    """Lazy load resume formatters (docx, pdf generation)"""
    global _resume_formatters
    if _resume_formatters is None:
        from core.resume_generator import generate_docx_from_json, generate_pdf_from_json, format_resume_as_text
        _resume_formatters = {
            'docx': generate_docx_from_json,
            'pdf': generate_pdf_from_json,
            'text': format_resume_as_text
        }
    return _resume_formatters


def render_structured_resume_editor(resume_data):
    """Render structured resume JSON in editable Streamlit form"""
    if not resume_data:
        return None
    
    edited_data = {}
    
    st.subheader("📋 Your Tailored Resume")
    st.caption("Edit the sections below to customize your resume")
    
    # Header Section
    with st.expander("👤 Header Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            edited_data['header'] = {
                'name': st.text_input("Full Name", value=resume_data.get('header', {}).get('name', ''), key='resume_name'),
                'title': st.text_input("Professional Title", value=resume_data.get('header', {}).get('title', ''), key='resume_title'),
                'email': st.text_input("Email", value=resume_data.get('header', {}).get('email', ''), key='resume_email'),
                'phone': st.text_input("Phone", value=resume_data.get('header', {}).get('phone', ''), key='resume_phone'),
            }
        with col2:
            edited_data['header']['location'] = st.text_input("Location", value=resume_data.get('header', {}).get('location', ''), key='resume_location')
            edited_data['header']['linkedin'] = st.text_input("LinkedIn URL", value=resume_data.get('header', {}).get('linkedin', ''), key='resume_linkedin')
            edited_data['header']['portfolio'] = st.text_input("Portfolio URL", value=resume_data.get('header', {}).get('portfolio', ''), key='resume_portfolio')
    
    # Summary
    # Check if there's a pending refined summary to display
    if '_pending_refined_summary' in st.session_state:
        # Apply the pending refinement by updating the widget key directly
        st.session_state['resume_summary'] = st.session_state['_pending_refined_summary']
        del st.session_state['_pending_refined_summary']
    
    # Get the summary value - prefer session state (for edited/refined), then fall back to resume data
    summary_value = resume_data.get('summary', '')
    
    col_summary1, col_summary2 = st.columns([4, 1])
    with col_summary1:
        edited_data['summary'] = st.text_area(
            "Professional Summary",
            value=summary_value,
            height=100,
            key='resume_summary'
        )
    with col_summary2:
        if st.button("✨ Refine with AI", key='refine_summary', use_container_width=True, help="Use AI to improve this section"):
            with st.spinner("🤖 Refining summary..."):
                try:
                    text_gen = get_text_generator()
                    if text_gen is None:
                        st.error("⚠️ Azure OpenAI is not configured.")
                    else:
                        # Get the current value from the widget's session state
                        current_summary = st.session_state.get('resume_summary', summary_value)
                        if not current_summary or not current_summary.strip():
                            st.warning("⚠️ Please enter a summary first before refining.")
                        else:
                            refinement_prompt = f"""Improve this professional summary. Make it more impactful, quantified, and tailored. Keep it concise (2-3 sentences).

Current Summary:
{current_summary}

Return ONLY the improved summary text, no additional explanation."""
                            
                            payload = {
                                "messages": [
                                    {"role": "system", "content": "You are a resume writing expert. Improve professional summaries to be more impactful and quantified."},
                                    {"role": "user", "content": refinement_prompt}
                                ],
                                "max_tokens": 200,
                                "temperature": 0.7
                            }
                            
                            def make_request():
                                return requests.post(text_gen.url, headers=text_gen.headers, json=payload, timeout=30)
                            
                            response = api_call_with_retry(make_request, max_retries=2)
                            if response and response.status_code == 200:
                                result = response.json()
                                refined_text = result['choices'][0]['message']['content'].strip()
                                # Store in pending key - will be applied on next rerun before widget renders
                                st.session_state['_pending_refined_summary'] = refined_text
                                st.rerun()
                            elif response:
                                st.error(f"⚠️ API Error: {response.status_code}. Please try again.")
                            else:
                                st.error("⚠️ Failed to connect to AI service. Please try again.")
                except Exception as e:
                    st.error(f"⚠️ Error refining summary: {str(e)}")
    
    # Skills
    skills_list = resume_data.get('skills_highlighted', [])
    skills_text = ', '.join(skills_list) if skills_list else ''
    skills_input = st.text_area(
        "Highlighted Skills (comma-separated)",
        value=skills_text,
        height=60,
        key='resume_skills',
        help="List skills separated by commas"
    )
    edited_data['skills_highlighted'] = [s.strip() for s in skills_input.split(',') if s.strip()]
    
    # Experience
    st.subheader("💼 Work Experience")
    edited_data['experience'] = []
    
    experience_list = resume_data.get('experience', [])
    for i, exp in enumerate(experience_list):
        with st.expander(f"📌 {exp.get('company', 'Company')} - {exp.get('title', 'Position')}", expanded=(i == 0)):
            col1, col2 = st.columns([2, 1])
            with col1:
                company = st.text_input("Company", value=exp.get('company', ''), key=f'exp_company_{i}')
                title = st.text_input("Job Title", value=exp.get('title', ''), key=f'exp_title_{i}')
            with col2:
                dates = st.text_input("Date Range", value=exp.get('dates', ''), key=f'exp_dates_{i}')
            
            st.write("**Key Achievements:**")
            bullets = exp.get('bullets', [])
            edited_bullets = []
            for j, bullet in enumerate(bullets):
                bullet_key = f'exp_bullet_{i}_{j}'
                pending_bullet_key = f'_pending_refined_bullet_{i}_{j}'
                
                # Check for pending refined bullet and apply it BEFORE widget renders
                if pending_bullet_key in st.session_state:
                    st.session_state[bullet_key] = st.session_state[pending_bullet_key]
                    del st.session_state[pending_bullet_key]
                
                col_bullet1, col_bullet2 = st.columns([4, 1])
                with col_bullet1:
                    bullet_text = st.text_area(
                        f"Bullet {j+1}",
                        value=bullet,
                        height=60,
                        key=bullet_key
                    )
                with col_bullet2:
                    if st.button("✨", key=f'refine_bullet_{i}_{j}', help="Refine this bullet with AI", use_container_width=True):
                        with st.spinner("🤖 Refining..."):
                            try:
                                text_gen = get_text_generator()
                                if text_gen is None:
                                    st.error("⚠️ Azure OpenAI is not configured.")
                                else:
                                    # Get the current value from the widget's session state
                                    current_bullet = st.session_state.get(bullet_key, bullet)
                                    if not current_bullet or not current_bullet.strip():
                                        st.warning("⚠️ Please enter content first before refining.")
                                    else:
                                        refinement_prompt = f"""Improve this resume bullet point. Make it more quantified, impactful, and achievement-focused. Use numbers, percentages, or metrics when possible.

Current Bullet:
{current_bullet}

Return ONLY the improved bullet point, no additional text."""
                                        
                                        payload = {
                                            "messages": [
                                                {"role": "system", "content": "You are a resume writing expert. Improve bullet points to be quantified and achievement-focused."},
                                                {"role": "user", "content": refinement_prompt}
                                            ],
                                            "max_tokens": 150,
                                            "temperature": 0.7
                                        }
                                        
                                        def make_request():
                                            return requests.post(text_gen.url, headers=text_gen.headers, json=payload, timeout=30)
                                        
                                        response = api_call_with_retry(make_request, max_retries=2)
                                        if response and response.status_code == 200:
                                            result = response.json()
                                            refined_text = result['choices'][0]['message']['content'].strip()
                                            # Store in pending key - will be applied on next rerun before widget renders
                                            st.session_state[pending_bullet_key] = refined_text
                                            st.rerun()
                                        elif response:
                                            st.error(f"⚠️ API Error: {response.status_code}")
                                        else:
                                            st.error("⚠️ Failed to connect to AI service.")
                            except Exception as e:
                                st.error(f"⚠️ Error: {str(e)}")
                
                if bullet_text.strip():
                    edited_bullets.append(bullet_text.strip())
            
            if st.button(f"➕ Add Bullet Point", key=f'add_bullet_{i}'):
                edited_bullets.append("")
                st.rerun()
            
            edited_data['experience'].append({
                'company': company,
                'title': title,
                'dates': dates,
                'bullets': edited_bullets
            })
    
    # Education
    edited_data['education'] = st.text_area(
        "Education",
        value=resume_data.get('education', ''),
        height=100,
        key='resume_education'
    )
    
    # Certifications
    edited_data['certifications'] = st.text_area(
        "Certifications & Awards",
        value=resume_data.get('certifications', ''),
        height=100,
        key='resume_certifications'
    )
    
    return edited_data


def display_resume_generator():
    """Display the resume generator interface with structured resume editing"""
    if st.session_state.selected_job is None:
        st.warning("No job selected. Please select a job first.")
        if st.button("← Back to Jobs"):
            st.session_state.show_resume_generator = False
            st.rerun()
        return
    
    job = st.session_state.selected_job
    
    st.markdown('<h1 class="main-header">📄 Resume Generator</h1>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="job-card">
        <h3 style="color: var(--primary-accent); margin: 0;">{job['title']}</h3>
        <p style="margin: 0.5rem 0; color: var(--text-muted);">🏢 {job['company']} • 📍 {job['location']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.user_profile.get('name') or not st.session_state.user_profile.get('experience'):
        st.error("⚠️ Please complete your profile first!")
        if st.button("← Go to Profile"):
            st.session_state.show_resume_generator = False
            st.rerun()
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Your Profile:**", st.session_state.user_profile.get('name', 'N/A'))
    
    with col2:
        if st.button("← Back to Jobs"):
            st.session_state.show_resume_generator = False
            st.session_state.generated_resume = None
            st.session_state.match_score = None
            st.session_state.missing_keywords = None
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🚀 Generate Tailored Resume", type="primary", use_container_width=True):
        with st.spinner("🤖 Creating your personalized resume using AI..."):
            text_gen = get_text_generator()
            if text_gen is None:
                st.error("⚠️ Azure OpenAI is not configured.")
                return
            raw_resume_text = st.session_state.get('resume_text')
            resume_data = text_gen.generate_resume(
                st.session_state.user_profile, 
                job,
                raw_resume_text=raw_resume_text
            )
            
            if resume_data:
                st.session_state.generated_resume = resume_data
                
                with st.spinner("📊 Analyzing resume match..."):
                    embedding_gen = get_embedding_generator()
                    resume_text = json.dumps(resume_data, indent=2)
                    match_score, missing_keywords = text_gen.calculate_match_score(
                        resume_text,
                        job.get('description', ''),
                        embedding_gen
                    )
                    st.session_state.match_score = match_score
                    st.session_state.missing_keywords = missing_keywords
                
                st.success("✅ Resume generated successfully!")
                st.balloons()
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Failed to generate resume. Please try again.")
    
    if st.session_state.generated_resume and st.session_state.get('match_score') is not None:
        display_match_score_feedback(
            st.session_state.match_score,
            st.session_state.missing_keywords,
            job['title']
        )
    
    if st.session_state.generated_resume:
        st.markdown("---")
        
        edited_resume_data = render_structured_resume_editor(st.session_state.generated_resume)
        
        if edited_resume_data:
            st.session_state.generated_resume = edited_resume_data
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Get formatters (lazy loaded)
        formatters = _get_resume_formatters()
        
        with col1:
            pdf_file = formatters['pdf'](
                st.session_state.generated_resume,
                filename=f"resume_{job['company']}_{job['title']}.pdf"
            )
            if pdf_file:
                st.download_button(
                    label="📥 Download as PDF",
                    data=pdf_file,
                    file_name=f"resume_{job['company']}_{job['title']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        with col2:
            docx_file = formatters['docx'](
                st.session_state.generated_resume,
                filename=f"resume_{job['company']}_{job['title']}.docx"
            )
            if docx_file:
                st.download_button(
                    label="📥 Download as DOCX",
                    data=docx_file,
                    file_name=f"resume_{job['company']}_{job['title']}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
        
        with col3:
            json_data = json.dumps(st.session_state.generated_resume, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"resume_{job['company']}_{job['title']}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col4:
            txt_content = formatters['text'](st.session_state.generated_resume)
            st.download_button(
                label="📥 Download as TXT",
                data=txt_content,
                file_name=f"resume_{job['company']}_{job['title']}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col5:
            if job['url'] != '#':
                st.link_button(
                    "🚀 Apply to Job",
                    job['url'],
                    use_container_width=True,
                    type="primary"
                )
        
        if st.button("🔄 Recalculate Match Score", use_container_width=True):
            with st.spinner("📊 Recalculating match score..."):
                text_gen = get_text_generator()
                if text_gen is None:
                    st.error("⚠️ Azure OpenAI is not configured.")
                    return
                embedding_gen = get_embedding_generator()
                resume_text = json.dumps(st.session_state.generated_resume, indent=2)
                match_score, missing_keywords = text_gen.calculate_match_score(
                    resume_text,
                    job.get('description', ''),
                    embedding_gen
                )
                st.session_state.match_score = match_score
                st.session_state.missing_keywords = missing_keywords
                st.rerun()
