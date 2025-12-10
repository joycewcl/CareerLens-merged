"""Profile extraction from resume text using AI"""
import json
import re
import streamlit as st
import requests
from modules.utils import get_text_generator, api_call_with_retry, _websocket_keepalive
from modules.utils.config import ENABLE_PROFILE_PASS2


def extract_relevant_resume_sections(resume_text):
    """Extract only Experience and Education sections from resume text to reduce token usage in Pass 2 verification"""
    if not resume_text:
        return ""
    
    experience_keywords = [
        r'experience', r'work experience', r'employment', r'employment history',
        r'professional experience', r'work history', r'career history', r'positions held'
    ]
    education_keywords = [
        r'education', r'academic background', r'academic qualifications',
        r'educational background', r'qualifications', r'degrees'
    ]
    
    lines = resume_text.split('\n')
    relevant_sections = []
    current_section = None
    in_experience = False
    in_education = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        line_lower = line_stripped.lower()
        
        if any(re.search(rf'\b{kw}\b', line_lower) for kw in experience_keywords):
            if not in_experience:
                in_experience = True
                in_education = False
                if current_section:
                    relevant_sections.append(current_section)
                current_section = line + '\n'
            continue
        
        if any(re.search(rf'\b{kw}\b', line_lower) for kw in education_keywords):
            if not in_education:
                in_education = True
                if current_section:
                    relevant_sections.append(current_section)
                current_section = line + '\n'
            continue
        
        major_sections = [r'summary', r'objective', r'skills', r'certifications', 
                         r'awards', r'publications', r'projects', r'contact', r'personal']
        if any(re.search(rf'\b{section}\b', line_lower) for section in major_sections):
            if in_experience or in_education:
                if current_section:
                    relevant_sections.append(current_section)
                current_section = None
                in_experience = False
                in_education = False
            continue
        
        if in_experience or in_education:
            if current_section:
                current_section += line + '\n'
    
    if current_section and (in_experience or in_education):
        relevant_sections.append(current_section)
    
    result = '\n'.join(relevant_sections)
    
    if not result or len(result) < 100:
        date_pattern = r'\b(19|20)\d{2}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'
        result_lines = []
        for line in lines:
            if re.search(date_pattern, line, re.IGNORECASE):
                result_lines.append(line)
            elif result_lines:
                if len([l for l in result_lines[-3:] if l.strip()]) < 3:
                    result_lines.append(line)
                else:
                    break
        if result_lines:
            result = '\n'.join(result_lines[:50])
    
    if result:
        return result[:2000] if len(result) > 2000 else result
    
    return ""


def extract_profile_from_resume(resume_text):
    """Use Azure OpenAI to extract structured profile information from resume text with two-pass self-correction"""
    try:
        text_gen = get_text_generator()
        
        if text_gen is None:
            st.error("⚠️ Azure OpenAI is not configured. Please configure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in your Streamlit secrets.")
            return None
        
        # FIRST PASS: Initial extraction
        prompt_pass1 = f"""You are an expert at parsing resumes. Extract structured information from the following resume text.

RESUME TEXT:
{resume_text}

Please extract and return the following information in JSON format:
{{
    "name": "Full name",
    "email": "Email address",
    "phone": "Phone number",
    "location": "City, State/Country",
    "linkedin": "LinkedIn URL if mentioned",
    "portfolio": "Portfolio/website URL if mentioned",
    "summary": "Professional summary or objective (2-3 sentences)",
    "experience": "Work experience in chronological order with job titles, companies, dates, and key achievements (formatted as bullet points)",
    "education": "Education details including degrees, institutions, and graduation dates",
    "skills": "Comma-separated list of technical and soft skills",
    "certifications": "Professional certifications, awards, publications, or other achievements"
}}

Important:
- If information is not found, use "N/A" or empty string
- Format experience with clear job titles, companies, dates, and bullet points for achievements
- Extract all relevant skills mentioned
- Keep the summary concise but informative
- Return ONLY valid JSON, no additional text or markdown"""
        
        payload_pass1 = {
            "messages": [
                {"role": "system", "content": "You are a resume parser. Extract structured information and return only valid JSON."},
                {"role": "user", "content": prompt_pass1}
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        _websocket_keepalive("Extracting profile information...")
        
        def make_request_pass1():
            return requests.post(
                text_gen.url,
                headers=text_gen.headers,
                json=payload_pass1,
                timeout=45
            )
        
        response_pass1 = api_call_with_retry(make_request_pass1, max_retries=3)
        
        if not response_pass1 or response_pass1.status_code != 200:
            if response_pass1 and response_pass1.status_code == 429:
                st.error("🚫 Rate limit reached for profile extraction after retries. Please wait a few minutes and try again.")
            else:
                error_detail = response_pass1.text[:200] if response_pass1 and response_pass1.text else "No error details"
                endpoint_info = f"Endpoint: {text_gen.url.split('/deployments')[0]}" if text_gen else "Endpoint: Not configured"
                st.error(f"API Error: {response_pass1.status_code if response_pass1 else 'Unknown'} - {error_detail}\n\n{endpoint_info}")
            return None
        
        result_pass1 = response_pass1.json()
        content_pass1 = result_pass1['choices'][0]['message']['content']
        
        if text_gen.token_tracker and 'usage' in result_pass1:
            usage = result_pass1['usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            text_gen.token_tracker.add_completion_tokens(prompt_tokens, completion_tokens)
        
        try:
            profile_data_pass1 = json.loads(content_pass1)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', content_pass1, re.DOTALL)
            if json_match:
                profile_data_pass1 = json.loads(json_match.group())
            else:
                st.error("Could not parse extracted profile data from first pass. Please try again.")
                return None
        
        # SECOND PASS: Self-correction (optional)
        if not ENABLE_PROFILE_PASS2:
            return profile_data_pass1
        
        relevant_resume_sections = extract_relevant_resume_sections(resume_text)
        
        if relevant_resume_sections:
            resume_context = f"""RELEVANT RESUME SECTIONS (Experience and Education only):
{relevant_resume_sections}"""
        else:
            resume_context = f"""RELEVANT RESUME SECTIONS (limited):
{resume_text[:1500]}"""
        
        prompt_pass2 = f"""You are a resume quality checker. Review the extracted profile data against the relevant resume sections and verify accuracy, especially for dates and company names.

{resume_context}

EXTRACTED PROFILE DATA (from first pass):
{json.dumps(profile_data_pass1, indent=2)}

Please review and correct the extracted data, paying special attention to:
1. **Dates** - Verify all employment dates, education dates, and certification dates are accurate
2. **Company Names** - Verify all company/organization names are spelled correctly
3. **Job Titles** - Verify job titles are accurate
4. **Education Institutions** - Verify institution names are correct

Return the corrected profile data in the same JSON format. If everything is correct, return the data as-is. If corrections are needed, return the corrected version.

Return ONLY valid JSON with this structure:
{{
    "name": "Full name",
    "email": "Email address",
    "phone": "Phone number",
    "location": "City, State/Country",
    "linkedin": "LinkedIn URL if mentioned",
    "portfolio": "Portfolio/website URL if mentioned",
    "summary": "Professional summary or objective (2-3 sentences)",
    "experience": "Work experience in chronological order with job titles, companies, dates, and key achievements (formatted as bullet points)",
    "education": "Education details including degrees, institutions, and graduation dates",
    "skills": "Comma-separated list of technical and soft skills",
    "certifications": "Professional certifications, awards, publications, or other achievements"
}}

Return ONLY valid JSON, no additional text or markdown."""
        
        payload_pass2 = {
            "messages": [
                {"role": "system", "content": "You are a resume quality checker. Verify and correct extracted data, especially dates and company names. Return only valid JSON."},
                {"role": "user", "content": prompt_pass2}
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        _websocket_keepalive("Verifying profile data...")
        
        def make_request_pass2():
            return requests.post(
                text_gen.url,
                headers=text_gen.headers,
                json=payload_pass2,
                timeout=45
            )
        
        response_pass2 = api_call_with_retry(make_request_pass2, max_retries=3)
        
        if response_pass2 and response_pass2.status_code == 200:
            result_pass2 = response_pass2.json()
            content_pass2 = result_pass2['choices'][0]['message']['content']
            
            if text_gen.token_tracker and 'usage' in result_pass2:
                usage = result_pass2['usage']
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                text_gen.token_tracker.add_completion_tokens(prompt_tokens, completion_tokens)
            
            try:
                profile_data_corrected = json.loads(content_pass2)
                return profile_data_corrected
            except json.JSONDecodeError:
                st.warning("⚠️ Self-correction pass failed, using initial extraction. Some details may need manual verification.")
                return profile_data_pass1
        else:
            st.warning("⚠️ Self-correction pass failed, using initial extraction. Some details may need manual verification.")
            return profile_data_pass1
            
    except Exception as e:
        st.error(f"Error extracting profile: {e}")
        return None


def verify_profile_data_pass2(profile_data, resume_text):
    """
    Run Pass 2 verification on existing profile data (Lazy Pass 2).
    
    This function is called on-demand before resume generation to verify
    the accuracy of dates, company names, job titles, and education details.
    
    Args:
        profile_data: Dict containing extracted profile data from Pass 1
        resume_text: Original resume text for cross-reference
        
    Returns:
        Dict with verified/corrected profile data, or original data if verification fails
    """
    if not profile_data or not resume_text:
        return profile_data
    
    # Check if already verified in this session
    if st.session_state.get('profile_verified', False):
        return profile_data
    
    try:
        text_gen = get_text_generator()
        
        if text_gen is None:
            st.warning("⚠️ AI service unavailable. Using unverified profile data.")
            return profile_data
        
        # Extract relevant sections for verification
        relevant_resume_sections = extract_relevant_resume_sections(resume_text)
        
        if relevant_resume_sections:
            resume_context = f"""RELEVANT RESUME SECTIONS (Experience and Education only):
{relevant_resume_sections}"""
        else:
            resume_context = f"""RELEVANT RESUME SECTIONS (limited):
{resume_text[:1500]}"""
        
        prompt_pass2 = f"""You are a resume quality checker. Review the extracted profile data against the relevant resume sections and verify accuracy, especially for dates and company names.

{resume_context}

EXTRACTED PROFILE DATA (to verify):
{json.dumps(profile_data, indent=2)}

Please review and correct the extracted data, paying special attention to:
1. **Dates** - Verify all employment dates, education dates, and certification dates are accurate
2. **Company Names** - Verify all company/organization names are spelled correctly
3. **Job Titles** - Verify job titles are accurate
4. **Education Institutions** - Verify institution names are correct

Return the corrected profile data in the same JSON format. If everything is correct, return the data as-is. If corrections are needed, return the corrected version.

Return ONLY valid JSON with the same structure as the input. No additional text or markdown."""
        
        payload_pass2 = {
            "messages": [
                {"role": "system", "content": "You are a resume quality checker. Verify and correct extracted data, especially dates and company names. Return only valid JSON."},
                {"role": "user", "content": prompt_pass2}
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        _websocket_keepalive("Verifying profile data for resume generation...")
        
        def make_request_pass2():
            return requests.post(
                text_gen.url,
                headers=text_gen.headers,
                json=payload_pass2,
                timeout=45
            )
        
        response_pass2 = api_call_with_retry(make_request_pass2, max_retries=2)
        
        if response_pass2 and response_pass2.status_code == 200:
            result_pass2 = response_pass2.json()
            content_pass2 = result_pass2['choices'][0]['message']['content']
            
            if text_gen.token_tracker and 'usage' in result_pass2:
                usage = result_pass2['usage']
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                text_gen.token_tracker.add_completion_tokens(prompt_tokens, completion_tokens)
            
            try:
                profile_data_verified = json.loads(content_pass2)
                # Mark as verified to avoid re-verification in same session
                st.session_state.profile_verified = True
                st.success("✅ Profile data verified for accuracy")
                return profile_data_verified
            except json.JSONDecodeError:
                st.warning("⚠️ Verification parsing failed. Using original profile data.")
                return profile_data
        else:
            st.warning("⚠️ Verification service unavailable. Using original profile data.")
            return profile_data
            
    except Exception as e:
        st.warning(f"⚠️ Verification error: {e}. Using original profile data.")
        return profile_data
