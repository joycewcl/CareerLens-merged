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
    
    st.header("📝 Publish New Position")

    with st.form("head_hunter_job_form"):
        # Basic Position Information
        st.subheader("🎯 Basic Position Information")

        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Position Title*", placeholder="e.g.: Senior Frontend Engineer")
        with col2:
            employment_type = st.selectbox("Employment Type*", ["Please select", "Full-time", "Part-time", "Contract", "Internship"])

        job_description = st.text_area("Job Description*", height=100,
                                      placeholder="Detailed introduction of position main content and team situation...")

        main_responsibilities = st.text_area("Main Responsibilities*", height=100,
                                           placeholder="List main responsibilities with bullet points, one per line...")

        required_skills = st.text_area("Required Skills & Qualifications*", height=100,
                                     placeholder="e.g.: 5+ years experience, proficient in React.js, Computer Science degree...")

        # Company and Client Information
        st.subheader("🏢 Company and Client Information")

        col3, col4 = st.columns(2)
        with col3:
            client_company = st.text_input("Client Company Name*", placeholder="Company official name")
            industry = st.selectbox("Industry*", ["Please select", "Technology", "Finance", "Consulting", "Healthcare", "Education", "Manufacturing", "Retail", "Other"])
        with col4:
            work_location = st.selectbox("Work Location*", ["Please select", "Hong Kong", "Mainland China", "Overseas", "Remote"])
            company_size = st.selectbox("Company Size*", ["Please select", "Startup (1-50)", "SME (51-200)", "Large Enterprise (201-1000)", "Multinational (1000+)"])

        work_type = st.selectbox("Work Type*", ["Please select", "Remote", "Hybrid", "Office"])

        # Employment Details
        st.subheader("💼 Employment Details")

        col5, col6 = st.columns(2)
        with col5:
            experience_level = st.selectbox("Experience Level*", ["Please select", "Fresh Graduate", "1-3 years", "3-5 years", "5-10 years", "10+ years"])
        with col6:
            visa_support = st.selectbox("Visa Support", ["Not provided", "Work Visa", "Assistance provided", "Must have own visa"])

        # Salary and Application Method
        st.subheader("💰 Salary and Application Method")

        col7, col8, col9 = st.columns([2, 2, 1])
        with col7:
            min_salary = st.number_input("Minimum Salary*", min_value=0, value=30000, step=5000)
        with col8:
            max_salary = st.number_input("Maximum Salary*", min_value=0, value=50000, step=5000)
        with col9:
            currency = st.selectbox("Currency", ["HKD", "USD", "CNY", "EUR", "GBP"])

        benefits = st.text_area("Benefits", height=80,
                              placeholder="e.g.: Medical insurance, 15 days annual leave, performance bonus, stock options...")

        application_method = st.text_area("Application Method*", height=80,
                                        value="Please send resume to recruit@headhunter.com, include position title in email subject",
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
