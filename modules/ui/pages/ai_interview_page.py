"""
AI Mock Interview Dashboard.

This module contains the UI pages for AI-powered mock interviews:
- AI Interview Page (main interview interface)
- Interview Preparation Guide
- Usage Instructions
"""

import json
import streamlit as st
from typing import Dict, Optional

from database.queries import get_jobs_for_interview, get_job_seeker_profile, get_job_seeker_profile_tuple
from core.interview import (
    initialize_interview_session,
    generate_interview_question,
    evaluate_answer,
    generate_final_summary
)


def ai_interview_page():
    """AI Interview Page - Streamlit UI.
    
    This function renders the complete AI mock interview interface.
    """
    st.title("🤖 AI Mock Interview")

    # Get position information
    jobs = get_jobs_for_interview()
    seeker_profile = get_job_seeker_profile_tuple()

    if not jobs:
        st.warning("❌ No available position information, please first publish positions in the headhunter module")
        return

    if not seeker_profile:
        st.warning("❌ Please first fill in your information on the Job Seeker page")
        return

    st.success("🎯 Select the position you want to interview for to start the mock interview")

    # Select position
    job_options = {f"#{job[0]} {job[1]} - {job[5]}": job for job in jobs}
    selected_job_key = st.selectbox("Select Interview Position", list(job_options.keys()))
    selected_job = job_options[selected_job_key]

    # Display position information
    with st.expander("📋 Position Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Position:** {selected_job[1]}")
            st.write(f"**Company:** {selected_job[5]}")
            st.write(f"**Industry:** {selected_job[6]}")
        with col2:
            st.write(f"**Experience Requirement:** {selected_job[7]}")
            st.write(f"**Skill Requirements:** {selected_job[4][:100]}...")

    # Initialize interview session
    if 'interview' not in st.session_state:
        st.session_state.interview = initialize_interview_session(selected_job)
    interview = st.session_state.interview

    # Start/continue interview
    if not interview['completed']:
        if interview['current_question'] == 0:
            if st.button("🚀 Start Mock Interview", type="primary", use_container_width=True):
                # Generate first question
                with st.spinner("AI is preparing interview questions..."):
                    first_question = generate_interview_question(selected_job, seeker_profile)
                    if not first_question.startswith("AI question generation failed"):
                        interview['questions'].append(first_question)
                        interview['current_question'] = 1
                        st.rerun()
                    else:
                        st.error(first_question)

        # Display current question
        if interview['current_question'] > 0 and interview['current_question'] <= interview['total_questions']:
            st.subheader(f"❓ Question {interview['current_question']}/{interview['total_questions']}")
            st.info(interview['questions'][-1])

            # Answer input
            answer = st.text_area("Your Answer:", height=150,
                                placeholder="Please describe your answer in detail...",
                                key=f"answer_{interview['current_question']}")

            if st.button("📤 Submit Answer", type="primary", use_container_width=True):
                if answer.strip():
                    with st.spinner("AI is evaluating your answer..."):
                        # Evaluate current answer
                        evaluation = evaluate_answer(
                            interview['questions'][-1],
                            answer,
                            selected_job
                        )

                        try:
                            eval_data = json.loads(evaluation)
                            if 'error' not in eval_data:
                                # Save answer and evaluation
                                interview['answers'].append(answer)
                                interview['scores'].append(eval_data)

                                # Check if all questions are completed
                                if interview['current_question'] == interview['total_questions']:
                                    # Generate final summary
                                    with st.spinner("AI is generating interview summary..."):
                                        summary = generate_final_summary(interview, selected_job)
                                        try:
                                            summary_data = json.loads(summary)
                                            interview['summary'] = summary_data
                                            interview['completed'] = True
                                        except (json.JSONDecodeError, KeyError, TypeError):
                                            interview['summary'] = {"error": "Summary parsing failed"}
                                            interview['completed'] = True
                                else:
                                    # Generate next question
                                    previous_qa = {
                                        'question': interview['questions'][-1],
                                        'answer': answer
                                    }
                                    next_question = generate_interview_question(
                                        selected_job, seeker_profile, previous_qa
                                    )
                                    if not next_question.startswith("AI question generation failed"):
                                        interview['questions'].append(next_question)
                                        interview['current_question'] += 1
                                    else:
                                        st.error(next_question)

                                st.rerun()
                            else:
                                st.error(eval_data['error'])
                        except json.JSONDecodeError:
                            st.error("Evaluation result parsing failed")
                else:
                    st.warning("Please enter your answer")

            # Display progress
            progress = interview['current_question'] / interview['total_questions']
            st.progress(progress)
            st.write(f"Progress: {interview['current_question']}/{interview['total_questions']} questions")

    # Display interview results
    if interview['completed'] and interview['summary']:
        st.subheader("🎯 Interview Summary Report")

        summary = interview['summary']

        if 'error' in summary:
            st.error(summary['error'])
        else:
            # Overall score
            col1, col2, col3 = st.columns(3)
            with col1:
                score = summary.get('overall_score', 0)
                st.metric("Overall Score", f"{score}/100")
            with col2:
                st.metric("Job Fit", summary.get('job_fit', 'N/A'))
            with col3:
                st.metric("Questions Answered", f"{len(interview['answers'])}/{interview['total_questions']}")

            # Overall evaluation
            st.write("### 📊 Overall Evaluation")
            st.info(summary.get('summary', ''))

            # Core strengths
            st.write("### ✅ Core Strengths")
            strengths = summary.get('key_strengths', [])
            for strength in strengths:
                st.write(f"🎯 {strength}")

            # Improvement areas
            st.write("### 📈 Improvement Suggestions")
            improvements = summary.get('improvement_areas', [])
            for improvement in improvements:
                st.write(f"💡 {improvement}")

            # Detailed recommendations
            st.write("### 🎯 Career Development Recommendations")
            recommendations = summary.get('recommendations', [])
            for rec in recommendations:
                st.write(f"🌟 {rec}")

            # Detailed Q&A records
            with st.expander("📝 View Detailed Q&A Records"):
                for i, (question, answer, score_data) in enumerate(zip(
                    interview['questions'],
                    interview['answers'],
                    interview['scores']
                )):
                    st.write(f"#### Question {i+1}")
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {answer}")
                    if isinstance(score_data, dict):
                        st.write(f"**Score:** {score_data.get('score', 'N/A')}/10")
                        st.write(f"**Feedback:** {score_data.get('feedback', '')}")
                    st.markdown("---")

            # Restart interview
            if st.button("🔄 Restart Interview", use_container_width=True):
                del st.session_state.interview
                st.rerun()


def ai_interview_dashboard():
    """AI Interview Dashboard - Main entry point for the interview system."""
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
