"""
Job Seeker Dashboard Page - CV Upload and Profile Management.

This module contains the main analyzer page where job seekers can:
- Upload their CV (PDF/DOCX)
- Have GPT-4 analyze their skills and experience
- Complete and save their profile information
"""

import streamlit as st
from typing import Dict, Any


def main_analyzer_page():
    """Main Page - CareerLens Job Seeker Dashboard"""
    # Import dependencies inside function to avoid circular imports
    from core.job_processor import JobSeekerBackend
    from database import save_job_seeker_info
    from core.resume_parser import extract_structured_profile
    
    # Import WebSocket utilities with fallback
    try:
        from utils.helpers import ProgressTracker, _websocket_keepalive
    except ImportError:
        # Fallback implementations
        def _websocket_keepalive(message=None, force=False):
            pass
        
        class ProgressTracker:
            def __init__(self, description="Processing", total_steps=100, show_progress=True):
                self.description = description
                self.total_steps = total_steps
                self.show_progress = show_progress
                self.current_step = 0
                self.progress_bar = None
            
            def __enter__(self):
                if self.show_progress:
                    self.progress_bar = st.progress(0, text=f"⏳ {self.description}...")
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.progress_bar:
                    self.progress_bar.empty()
                return False
            
            def update(self, step=None, message=None):
                if step is not None:
                    self.current_step = step
                else:
                    self.current_step += 1
                progress = min(self.current_step / self.total_steps, 1.0)
                if self.show_progress and self.progress_bar:
                    display_message = message or f"⏳ {self.description}... ({int(progress * 100)}%)"
                    self.progress_bar.progress(progress, text=display_message)
    
    # Initialize backend - cached to prevent re-initialization
    @st.cache_resource(show_spinner="Loading CareerLens...")
    def load_backend():
        return JobSeekerBackend()
    
    backend = load_backend()
    
    st.markdown("""
    <style>
        .main-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: -1px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .main-title .brand-span {
            color: var(--brand-core);
        }
        .main-title .lens-span {
            color: var(--brand-glow);
        }
        .main-tagline {
            text-align: center;
            color: var(--text-secondary-light);
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
    </style>
    <h1 class="main-title"><span class="brand-span">Career</span><span class="lens-span">Lens</span></h1>
    <p class="main-tagline">AI Career Copilot • Hong Kong</p>
    """, unsafe_allow_html=True)
    st.markdown("Upload your CV and let **AI** find matching jobs globally, ranked by match quality!")

    # Define helper functions
    def smart_select_match(value, options):
        """Smart match select box options"""
        if not value:
            return 0
        
        value_str = str(value).lower()
        for i, option in enumerate(options):
            if option.lower() in value_str or value_str in option.lower():
                return i
        return 0

    def format_ai_data(data, default=""):
        """Format AI returned data"""
        if isinstance(data, list):
            return ", ".join(data)
        elif isinstance(data, str):
            return data
        else:
            return default

    # Main Page - CV Upload Section
    st.header("📁 Upload Your CV")
    cv_file = st.file_uploader("Choose your CV", type=['pdf', 'docx'], key="cv_uploader")

    # Initialize variables
    autofill_data = {}
    analysis_complete = False
    ai_analysis = {}  # Initialize ai_analysis

    if cv_file:
        st.success(f"✅ Uploaded: **{cv_file.name}**")

        if st.button("🔍 Analyze with AI", type="primary", use_container_width=True, key="analyze_button"):

            # STEP 1: Analyze Resume with WebSocket keepalive
            with ProgressTracker("Analyzing your resume with AI", total_steps=3) as tracker:
                try:
                    tracker.update(1, "Extracting text from your CV...")
                    _websocket_keepalive("Extracting text")
                    
                    tracker.update(2, "AI is analyzing your skills and experience...")
                    _websocket_keepalive("AI analysis in progress")
                    
                    resume_data, ai_analysis = backend.process_resume(cv_file, cv_file.name)
                    
                    # Also extract detailed structured profile for resume tailoring
                    tracker.update(2, "Extracting detailed experience history...")
                    _websocket_keepalive("Detailed extraction")
                    structured_profile = extract_structured_profile(resume_data['raw_text'])
                    
                    tracker.update(3, "Analysis complete!")
                    _websocket_keepalive("Complete")
                    
                    # Check if analysis failed or was incomplete
                    analysis_failed = ai_analysis.get('_analysis_failed', False)
                    analysis_incomplete = ai_analysis.get('_analysis_incomplete', False)
                    
                    if analysis_failed:
                        error_msg = ai_analysis.get('_error', 'Unknown error')
                        st.error(
                            "❌ **AI Analysis Failed**\n\n"
                            f"**Error:** {error_msg}\n\n"
                        )
                        
                        # Show helpful instructions if it's a configuration error
                        config_error_keywords = ["Missing required secrets", "Connection error", "placeholder API key", "Authentication error", "not configured"]
                        if any(keyword in error_msg for keyword in config_error_keywords):
                            st.info(
                                "💡 **How to fix this:**\n\n"
                                "1. **In Streamlit Cloud:** \n"
                                "   - Go to your app dashboard\n"
                                "   - Click on **⚙️ Settings** → **Secrets**\n"
                                "   - Add the missing API keys\n\n"
                                "2. **For local development:**\n"
                                "   - Create a file `.streamlit/secrets.toml` in your project\n"
                                "   - Add your API keys in TOML format\n\n"
                                "3. **Get API keys:**\n"
                                "   - Azure OpenAI: https://azure.microsoft.com/en-us/products/ai-services/openai-service\n"
                                "   - RapidAPI: https://rapidapi.com/\n\n"
                                "After adding secrets, refresh the page."
                            )
                        
                        st.warning(
                            "⚠️ Please fill in your career details manually in the form below."
                        )
                    elif analysis_incomplete:
                        st.warning(
                            "⚠️ **AI Analysis Incomplete**\n\n"
                            "Could not fully analyze your resume. Please review and complete the form below."
                        )
                    else:
                        st.balloons()

                    # Display analysis results
                    st.markdown("---")
                    st.subheader("🤖 AI Career Analysis")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        primary_role = ai_analysis.get('primary_role', '') or 'Not detected'
                        st.metric("🎯 Primary Role", primary_role)

                    with col2:
                        confidence = ai_analysis.get('confidence', 0) * 100
                        st.metric("💯 Confidence", f"{confidence:.0f}%")

                    with col3:
                        st.metric("📊 Seniority", ai_analysis.get('seniority_level', '') or 'Not detected')

                    # Skills detected by GPT-4
                    st.markdown("### 💡 Skills Detected by AI")
                    skills = ai_analysis.get('skills', [])
                    if skills:
                        # Create skill tags
                        skills_html = ""
                        for skill in skills[:10]:
                            skills_html += f'<span style="background-color: #E8F4FD; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
                        st.markdown(skills_html, unsafe_allow_html=True)

                        if len(skills) > 10:
                            with st.expander(f"➕ Show all {len(skills)} skills"):
                                more_skills_html = ""
                                for skill in skills[10:]:
                                    more_skills_html += f'<span style="background-color: #F0F0F0; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
                                st.markdown(more_skills_html, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No skills detected")

                    # Core strengths
                    st.markdown("### 💪 Core Strengths")
                    strengths = ai_analysis.get('core_strengths', [])
                    if strengths:
                        cols = st.columns(min(3, len(strengths)))
                        for i, strength in enumerate(strengths):
                            with cols[i % len(cols)]:
                                st.info(f"✓ {strength}")

                    # Extract and format data
                    detailed_exp = ""
                    if structured_profile and structured_profile.get('experience'):
                        detailed_exp = structured_profile.get('experience')
                        
                    autofill_data = {
                        # Educational background
                        "education_level": format_ai_data(ai_analysis.get('education_level', '')),
                        "major": format_ai_data(ai_analysis.get('major', '')),
                        "graduation_status": format_ai_data(ai_analysis.get('graduation_status', '')),
                        "university_background": format_ai_data(ai_analysis.get('university_background', '')),
                        
                        # Languages and certificates
                        "languages": format_ai_data(ai_analysis.get('languages', '')),
                        "certificates": format_ai_data(ai_analysis.get('certificates', '')),
                        
                        # Skills - directly use detected skills
                        "hard_skills": format_ai_data(skills),  # Use detected skills
                        "soft_skills": format_ai_data(ai_analysis.get('core_strengths', [])),  # Use core strengths
                        
                        # Work experience
                        "work_experience": format_ai_data(ai_analysis.get('work_experience', '')),
                        "project_experience": format_ai_data(ai_analysis.get('project_experience', '')),
                        "detailed_experience": detailed_exp,
                        
                        # Preferences
                        "location_preference": format_ai_data(ai_analysis.get('location_preference', '')),
                        "industry_preference": format_ai_data(ai_analysis.get('industry_preference', '')),
                        
                        # Salary
                        "salary_expectation": format_ai_data(ai_analysis.get('salary_expectation', '')),
                        "benefits_expectation": format_ai_data(ai_analysis.get('benefits_expectation', '')),
                        
                        # New fields
                        "primary_role": format_ai_data(ai_analysis.get('primary_role', '')),
                        "simple_search_terms": format_ai_data(ai_analysis.get('simple_search_terms', ''))
                    }
                    
                    analysis_complete = True
                    
                    # Store in session state
                    st.session_state.autofill_data = autofill_data
                    st.session_state.analysis_complete = True
                    st.session_state.ai_analysis = ai_analysis  # Save ai_analysis for later use

                    st.success("🎉 Resume analysis complete! Form has been auto-filled with your information.")

                except Exception as e:
                    st.error(f"❌ Error analyzing resume: {str(e)}")
                    st.stop()

    else:
        # Welcome screen
        st.info("📄 **Upload your CV above to get started!**")

        # Instructions
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📋 How it works:

            1. **📄 Upload** your CV (PDF or DOCX)
            2. **🤖 AI** analyzes your skills, experience, and ideal roles
            3. **🔍 Search** LinkedIn jobs via RapidAPI (global search)
            4. **🎯 Rank** all jobs by match quality using AI
            5. **📊 See** your best matches with detailed scores!
            """)

        st.markdown("---")
        st.success("💡 **Pro Tip:** Jobs are searched globally (not filtered by Hong Kong) and ranked by how well they match your profile!")

    # ========== Form Area ==========
    if st.session_state.get('analysis_complete', False) or not cv_file:
        with st.form("job_seeker_form"):
            st.subheader("📝 Complete Your Profile")
            
            if st.session_state.get('analysis_complete', False):
                st.success("✅ Form auto-filled with your resume analysis!")
            
            st.markdown("Review and edit the auto-filled information from your CV analysis:")

            # Use data from session_state
            current_data = st.session_state.get('autofill_data', {})

            # Career Preferences - new fields at top of form
            st.subheader("🎯 Career Preferences")
            col_career1, col_career2 = st.columns(2)
            
            with col_career1:
                primary_role = st.text_input("Primary Role*", 
                                           value=current_data.get("primary_role", ""),
                                           placeholder="e.g., Project Manager, Software Engineer, Data Analyst")
            
            with col_career2:
                simple_search_terms = st.text_input("Search Keywords*", 
                                                  value=current_data.get("simple_search_terms", ""),
                                                  placeholder="e.g., python developer, project management, data science")

            # Educational background
            st.subheader("🎓 Educational background")
            col1, col2 = st.columns(2)

            with col1:
                education_options = ["Please select", "PhD", "Master", "Bachelor", "Diploma", "High School"]
                ed_level = current_data.get("education_level", "")
                education_index = smart_select_match(ed_level, education_options)
                
                education_level = st.selectbox(
                    "Educational level*",
                    education_options,
                    index=education_index
                )
                
                major = st.text_input("Major", 
                                    value=current_data.get("major", ""),
                                    placeholder="e.g., Computer Science, Business Administration")
                
                grad_options = ["Please select", "Graduated", "Fresh graduates", "Currently studying"]
                grad_status = current_data.get("graduation_status", "")
                grad_index = smart_select_match(grad_status, grad_options)
                
                graduation_status = st.selectbox(
                    "Graduation status*",
                    grad_options,
                    index=grad_index
                )

            with col2:
                uni_options = ["Please select", "985 Universities", "211 Universities", "Overseas Universities", "Regular Undergraduate Universities", "Other"]
                uni_bg = current_data.get("university_background", "")
                uni_index = smart_select_match(uni_bg, uni_options)
                
                university_background = st.selectbox(
                    "University background*",
                    uni_options,
                    index=uni_index
                )
                
                languages = st.text_input("Languages", 
                                        value=current_data.get("languages", ""),
                                        placeholder="e.g., English, Mandarin, Cantonese")
                
                certificates = st.text_input("Certificates", 
                                           value=current_data.get("certificates", ""),
                                           placeholder="e.g., PMP, CFA, AWS Certified")

            # Skills
            st.subheader("💼 Skills")
            hard_skills = st.text_area("Technical Skills", 
                                     value=current_data.get("hard_skills", ""),
                                     placeholder="e.g., Python, JavaScript, SQL, Machine Learning",
                                     height=100)
            
            soft_skills = st.text_area("Core Strengths", 
                                     value=current_data.get("soft_skills", ""),
                                     placeholder="e.g., Leadership, Communication, Problem Solving",
                                     height=100)

            # Work Experience
            st.subheader("📈 Work Experience")
            
            detailed_experience = st.text_area("Detailed Work Experience (for AI Resume Tailoring)", 
                                             value=current_data.get("detailed_experience", ""),
                                             placeholder="Paste your full work history here...",
                                             height=150,
                                             help="This detailed experience is used by the AI to generate tailored resumes. It should include company names, titles, dates, and bullet points.")
            
            col3, col4 = st.columns(2)

            with col3:
                work_exp_options = ["Please select", "Recent Graduate", "1-3 years", "3-5 years", "5-10 years", "10+ years"]
                work_exp = current_data.get("work_experience", "")
                work_index = smart_select_match(work_exp, work_exp_options)
                
                work_experience = st.selectbox(
                    "Work experience years*",
                    work_exp_options,
                    index=work_index
                )

            with col4:
                project_experience = st.text_area("Project experience", 
                                                value=current_data.get("project_experience", ""),
                                                placeholder="Describe your key projects and achievements",
                                                height=100)

            # Work preferences
            st.subheader("📍 Work preferences")
            col5, col6 = st.columns(2)

            with col5:
                loc_options = ["Please select", "Hong Kong", "Mainland China", "Overseas", "No Preference"]
                loc_pref = current_data.get("location_preference", "")
                loc_index = smart_select_match(loc_pref, loc_options)
                
                location_preference = st.selectbox(
                    "Location Preference*",
                    loc_options,
                    index=loc_index
                )
             
            with col6:
                industry_preference = st.text_input("Industry Preference", 
                                                  value=current_data.get("industry_preference", ""),
                                                  placeholder="e.g., Technology, Finance, Healthcare")
       
            # Salary and benefits expectations
            st.subheader("💰 Salary and Benefits Expectations")
            salary_expectation = st.text_input("Expected Salary Range", 
                                             value=current_data.get("salary_expectation", ""),
                                             placeholder="e.g., HKD 30,000 - 40,000")
            
            benefits_expectation = st.text_area("Benefits Requirements", 
                                              value=current_data.get("benefits_expectation", ""),
                                              placeholder="e.g., Medical insurance, Flexible working hours",
                                              height=80)
            

            # Submit button
            submitted = st.form_submit_button("💾 Save Information", use_container_width=True)

            if submitted:
                if (education_level == "Please select" or graduation_status == "Please select" or
                    university_background == "Please select" or work_experience == "Please select" or
                    location_preference == "Please select" or not primary_role.strip() or not simple_search_terms.strip()):
                    st.error("Please complete all required fields (marked with *)!")
                else:
                    # Save to database - pass as dictionary
                    profile_data = {
                        "education_level": education_level,
                        "major": major,
                        "graduation_status": graduation_status,
                        "university_background": university_background,
                        "languages": languages,
                        "certificates": certificates,
                        "hard_skills": hard_skills,
                        "soft_skills": soft_skills,
                        "work_experience": work_experience,
                        "project_experience": project_experience,
                        "detailed_experience": detailed_experience,
                        "location_preference": location_preference,
                        "industry_preference": industry_preference,
                        "salary_expectation": salary_expectation,
                        "benefits_expectation": benefits_expectation,
                        "primary_role": primary_role,
                        "simple_search_terms": simple_search_terms
                    }
                    job_seeker_id = save_job_seeker_info(profile_data)
                    
                    if job_seeker_id:
                        # Save to session state
                        st.session_state.job_seeker_id = job_seeker_id
                        st.success(f"✅ Information saved successfully! Your ID: {job_seeker_id}")
                        st.balloons()
                        
                        # Display success message
                        st.info(f"🔑 Your job seeker ID has been saved: **{job_seeker_id}**")
                        st.info("💡 You can use this ID on the Job Match page to view personalized job recommendations")
                    else:
                        st.error("❌ Failed to save information, please try again")
