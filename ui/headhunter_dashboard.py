"""
Headhunter Dashboard - Job Publishing and Management.

This module contains pages for headhunters/recruiters to:
- Publish new job positions
- View and manage published positions
- View position statistics
"""

import streamlit as st
from datetime import datetime, timedelta


def enhanced_head_hunter_page():
    """Enhanced Head Hunter Page - Job Publishing and Management"""
    st.title("🎯 Head Hunter Portal")

    # Page selection
    page_option = st.sidebar.radio(
        "Select Function",
        ["Publish New Position", "View Published Positions", "Position Statistics"]
    )

    if page_option == "Publish New Position":
        publish_new_job()
    elif page_option == "View Published Positions":
        view_published_jobs()
    elif page_option == "Position Statistics":
        show_job_statistics()


def publish_new_job():
    """Publish New Position Form"""
    from database import save_head_hunter_job, HeadhunterDB
    from core.resume_parser import ResumeParser, extract_job_posting_from_text
    
    st.header("📝 Publish New Position")

    # Initialize session state for form fields if not present
    if 'hh_job_form_data' not in st.session_state:
        st.session_state.hh_job_form_data = {}

    # File Uploader for Auto-fill
    with st.expander("📂 Import from File (PDF/DOCX)", expanded=True):
        st.info("Upload a job description file to auto-fill the form below.")
        uploaded_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx', 'txt'], key="job_uploader")
        
        if uploaded_file is not None:
            if st.button("✨ Parse & Auto-fill Form", type="primary"):
                with st.spinner("Parsing job description..."):
                    try:
                        parser = ResumeParser()
                        # Extract text
                        text = parser.extract_text(uploaded_file, uploaded_file.name)
                        
                        if len(text) < 50:
                            st.warning("Not enough text extracted from file.")
                        else:
                            job_data = extract_job_posting_from_text(text)
                            
                            if job_data:
                                st.session_state.hh_job_form_data = job_data
                                st.success("✅ Form auto-filled! Please review details below.")
                                st.rerun()
                            else:
                                st.error("❌ Could not extract information from the file.")
                    except Exception as e:
                        st.error(f"Error parsing file: {e}")

    # Helper to get value safely
    def get_val(key, default=''):
        return st.session_state.hh_job_form_data.get(key, default)

    # Helper for selectbox index
    def get_idx(options, key):
        val = st.session_state.hh_job_form_data.get(key)
        if val and val in options:
            return options.index(val)
        return 0

    with st.form("head_hunter_job_form"):
        # Basic Position Information
        st.subheader("🎯 Basic Position Information")

        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Position Title*", 
                                    value=get_val('job_title'), 
                                    placeholder="e.g.: Senior Frontend Engineer")
        with col2:
            opts_emp = ["Please select", "Full-time", "Part-time", "Contract", "Internship"]
            employment_type = st.selectbox("Employment Type*", opts_emp, index=get_idx(opts_emp, 'employment_type'))

        job_description = st.text_area("Job Description*", height=100,
                                      value=get_val('job_description'),
                                      placeholder="Detailed introduction of position main content and team situation...")

        main_responsibilities = st.text_area("Main Responsibilities*", height=100,
                                           value=get_val('main_responsibilities'),
                                           placeholder="List main responsibilities with bullet points, one per line...")

        required_skills = st.text_area("Required Skills & Qualifications*", height=100,
                                     value=get_val('required_skills'),
                                     placeholder="e.g.: 5+ years experience, proficient in React.js, Computer Science degree...")

        # Company and Client Information
        st.subheader("🏢 Company and Client Information")

        col3, col4 = st.columns(2)
        with col3:
            client_company = st.text_input("Client Company Name*", value=get_val('client_company'), placeholder="Company official name")
            opts_ind = ["Please select", "Technology", "Finance", "Consulting", "Healthcare", "Education", "Manufacturing", "Retail", "Other"]
            industry = st.selectbox("Industry*", opts_ind, index=get_idx(opts_ind, 'industry'))
        with col4:
            opts_loc = ["Please select", "Hong Kong", "Mainland China", "Overseas", "Remote"]
            work_location = st.selectbox("Work Location*", opts_loc, index=get_idx(opts_loc, 'work_location'))
            opts_size = ["Please select", "Startup (1-50)", "SME (51-200)", "Large Enterprise (201-1000)", "Multinational (1000+)"]
            company_size = st.selectbox("Company Size*", opts_size, index=get_idx(opts_size, 'company_size'))

        opts_type = ["Please select", "Remote", "Hybrid", "Office"]
        work_type = st.selectbox("Work Type*", opts_type, index=get_idx(opts_type, 'work_type'))

        # Employment Details
        st.subheader("💼 Employment Details")

        col5, col6 = st.columns(2)
        with col5:
            opts_exp = ["Please select", "Fresh Graduate", "1-3 years", "3-5 years", "5-10 years", "10+ years"]
            experience_level = st.selectbox("Experience Level*", opts_exp, index=get_idx(opts_exp, 'experience_level'))
        with col6:
            opts_visa = ["Not provided", "Work Visa", "Assistance provided", "Must have own visa"]
            visa_support = st.selectbox("Visa Support", opts_visa, index=get_idx(opts_visa, 'visa_support'))

        # Salary and Application Method
        st.subheader("💰 Salary and Application Method")

        col7, col8, col9 = st.columns([2, 2, 1])
        with col7:
            min_sal_val = get_val('min_salary', 30000)
            if isinstance(min_sal_val, str):
                try: min_sal_val = int(min_sal_val)
                except: min_sal_val = 30000
            min_salary = st.number_input("Minimum Salary*", min_value=0, value=min_sal_val, step=5000)
        with col8:
            max_sal_val = get_val('max_salary', 50000)
            if isinstance(max_sal_val, str):
                try: max_sal_val = int(max_sal_val)
                except: max_sal_val = 50000
            max_salary = st.number_input("Maximum Salary*", min_value=0, value=max_sal_val, step=5000)
        with col9:
            opts_curr = ["HKD", "USD", "CNY", "EUR", "GBP"]
            currency = st.selectbox("Currency", opts_curr, index=get_idx(opts_curr, 'currency'))

        benefits = st.text_area("Benefits", height=80,
                              value=get_val('benefits'),
                              placeholder="e.g.: Medical insurance, 15 days annual leave, performance bonus, stock options...")

        app_method_default = "Please send resume to recruit@headhunter.com, include position title in email subject"
        application_method = st.text_area("Application Method*", height=80,
                                        value=get_val('application_method', app_method_default),
                                        placeholder="Application process and contact information...")

        job_valid_until = st.date_input("Position Posting Validity Period*",
                                      value=datetime.now().date() + timedelta(days=30))

        # Submit button
        submitted = st.form_submit_button("💾 Publish Position", type="primary", use_container_width=True)

        if submitted:
            # Validate required fields
            required_fields = [
                job_title, job_description, main_responsibilities, required_skills,
                client_company, industry, work_location, work_type, company_size,
                employment_type, experience_level, min_salary, max_salary, application_method
            ]

            if "Please select" in [employment_type, industry, work_location, work_type, company_size, experience_level]:
                st.error("Please complete all required fields (marked with *)!")
            elif not all(required_fields):
                st.error("Please complete all required fields (marked with *)!")
            elif min_salary >= max_salary:
                st.error("Maximum salary must be greater than minimum salary!")
            
            else:
                # Create dictionary object
                job_data = {
                    'job_title': job_title,
                    'job_description': job_description,
                    'main_responsibilities': main_responsibilities,
                    'required_skills': required_skills,
                    'client_company': client_company,
                    'industry': industry,
                    'work_location': work_location,
                    'work_type': work_type,
                    'company_size': company_size,
                    'employment_type': employment_type,
                    'experience_level': experience_level,
                    'visa_support': visa_support,
                    'min_salary': min_salary,
                    'max_salary': max_salary,
                    'currency': currency,
                    'benefits': benefits,
                    'application_method': application_method,
                    'job_valid_until': job_valid_until.strftime("%Y-%m-%d")
                }
                
                # Save to database
                success = save_head_hunter_job(job_data)

                if success:
                    st.success("✅ Position published successfully!")
                    st.balloons()
                    # Clear session state
                    if 'hh_job_form_data' in st.session_state:
                        del st.session_state.hh_job_form_data
                else:
                    st.error("❌ Position publishing failed, please try again")


def view_published_jobs():
    """View Published Positions"""
    from database import HeadhunterDB
    
    @st.cache_resource
    def get_headhunter_db():
        return HeadhunterDB()
    
    db2 = get_headhunter_db()
    
    st.header("📋 Published Positions")

    jobs = db2.get_all_head_hunter_jobs()

    if not jobs:
        st.info("No positions published yet")
        return

    st.success(f"Published {len(jobs)} positions")

    # Search and filter
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search position title or company")
    with col2:
        filter_industry = st.selectbox("Filter by industry", ["All industries"] + ["Technology", "Finance", "Consulting", "Healthcare", "Education", "Manufacturing", "Retail", "Other"])

    # Filter positions
    filtered_jobs = jobs
    if search_term:
        filtered_jobs = [job for job in jobs if search_term.lower() in job[2].lower() or search_term.lower() in job[6].lower()]
    if filter_industry != "All industries":
        filtered_jobs = [job for job in filtered_jobs if job[7] == filter_industry]

    if not filtered_jobs:
        st.warning("No matching positions found")
        return

    # Display position list
    for job in filtered_jobs:
        with st.expander(f"#{job[0]} {job[2]} - {job[6]}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Published Time:** {job[1]}")
                st.write(f"**Company:** {job[6]}")
                st.write(f"**Industry:** {job[7]}")
                st.write(f"**Location:** {job[8]} ({job[9]})")
                st.write(f"**Size:** {job[10]}")

            with col2:
                st.write(f"**Type:** {job[11]}")
                st.write(f"**Experience:** {job[12]}")
                st.write(f"**Salary:** {job[14]:,} - {job[15]:,} {job[16]}")
                st.write(f"**Valid Until:** {job[19]}")
                if job[13] != "Not provided":
                    st.write(f"**Visa:** {job[13]}")

            st.write("**Description:**")
            st.write(job[3][:200] + "..." if len(job[3]) > 200 else job[3])


def show_job_statistics():
    """Display Position Statistics"""
    from database import HeadhunterDB
    
    @st.cache_resource
    def get_headhunter_db():
        return HeadhunterDB()
    
    db2 = get_headhunter_db()
    
    st.header("📊 Position Statistics")

    jobs = db2.get_all_head_hunter_jobs()

    if not jobs:
        st.info("No statistics available yet")
        return

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Positions", len(jobs))
    with col2:
        active_jobs = len([job for job in jobs if datetime.strptime(job[19], "%Y-%m-%d").date() >= datetime.now().date()])
        st.metric("Active Positions", active_jobs)
    with col3:
        expired_jobs = len(jobs) - active_jobs
        st.metric("Expired Positions", expired_jobs)
    with col4:
        avg_salary = sum((job[14] + job[15]) / 2 for job in jobs) / len(jobs)
        st.metric("Average Salary", f"{avg_salary:,.0f}")

    # Industry distribution
    st.subheader("🏭 Industry Distribution")
    industry_counts = {}
    for job in jobs:
        industry = job[7]
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

    for industry, count in industry_counts.items():
        st.write(f"• **{industry}:** {count} positions ({count/len(jobs)*100:.1f}%)")

    # Location distribution
    st.subheader("📍 Work Location Distribution")
    location_counts = {}
    for job in jobs:
        location = job[8]
        location_counts[location] = location_counts.get(location, 0) + 1

    for location, count in location_counts.items():
        st.write(f"• **{location}:** {count} positions")

    # Experience requirement distribution
    st.subheader("🎯 Experience Requirement Distribution")
    experience_counts = {}
    for job in jobs:
        experience = job[12]
        experience_counts[experience] = experience_counts.get(experience, 0) + 1

    for experience, count in experience_counts.items():
        st.write(f"• **{experience}:** {count} positions")
