"""
AI Mock Interview Dashboard.

This module contains pages for AI-powered mock interviews:
- Start Mock Interview
- Interview Preparation Guide
- Usage Instructions
"""

import streamlit as st


def ai_interview_dashboard():
    """AI Interview Dashboard"""
    from backend import get_jobs_for_interview, get_job_seeker_profile, ai_interview_page
    
    st.title("🤖 AI Mock Interview System")

    # Quick statistics
    jobs = get_jobs_for_interview()
    seeker_profile = get_job_seeker_profile()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Positions", len(jobs) if jobs else 0)
    with col2:
        st.metric("Personal Profile", "✅" if seeker_profile else "❌")
    with col3:
        if 'interview' in st.session_state:
            progress = st.session_state.interview['current_question']
            total = st.session_state.interview['total_questions']
            st.metric("Interview Progress", f"{progress}/{total}")
        else:
            st.metric("Interview Status", "Not Started")

    # Page selection
    page_option = st.sidebar.radio(
        "Select Function",
        ["Start Mock Interview", "Interview Preparation Guide", "Instructions"]
    )

    if page_option == "Start Mock Interview":
        ai_interview_page()
    elif page_option == "Interview Preparation Guide":
        show_interview_guidance()
    else:
        show_interview_instructions()


def show_interview_guidance():
    """Display Interview Preparation Guide"""
    st.header("🎯 Interview Preparation Guide")

    st.info("""
    **Interview Preparation Suggestions:**

    ### 📚 Technical Interview Preparation
    1. **Review Core Skills**: Ensure mastery of key technologies required for the position
    2. **Prepare Project Cases**: Prepare 2-3 projects that demonstrate your capabilities
    3. **Practice Coding Problems**: Prepare algorithms and data structures for technical positions

    ### 💼 Behavioral Interview Preparation
    1. **STAR Method**: Situation-Task-Action-Result
    2. **Prepare Success Stories**: Show how you solve problems and create value
    3. **Understand Company Culture**: Research company values and work style

    ### 🎯 Communication Skills
    1. **Clear Expression**: Structure your answers
    2. **Active Listening**: Ensure understanding of question core
    3. **Show Enthusiasm**: Express interest in position and company
    """)


def show_interview_instructions():
    """Display Usage Instructions"""
    st.header("📖 AI Mock Interview Usage Instructions")

    st.info("""
    **AI Mock Interview Function Guide:**

    ### 🚀 Start Interview
    1. **Select Position**: Choose a position from headhunter published positions for mock interview
    2. **Start Interview**: AI will generate relevant questions based on position requirements
    3. **Answer Questions**: Provide detailed answers for each question

    ### 📊 Interview Process
    - **10 Questions**: Includes various types like technical, behavioral, situational
    - **Real-time Evaluation**: AI evaluates quality of each answer
    - **Personalized Questions**: Follow-up questions based on your previous answers

    ### 🎯 Get Feedback
    - **Detailed Scoring**: Specific scoring and feedback for each question
    - **Overall Evaluation**: Complete interview performance summary
    - **Improvement Suggestions**: Targeted career development advice

    **Tip**: Please ensure use in stable network environment for AI to generate questions and evaluate answers normally.
    """)
