"""
API Client Classes for CareerLens Application

This module contains all API client classes including:
- APIMEmbeddingGenerator: Azure OpenAI embedding generation
- AzureOpenAITextGenerator: Azure OpenAI text generation  
- IndeedScraperAPI: Job scraping via RapidAPI

Rate limiting and token tracking are imported from core.rate_limiting:
- RateLimiter: Rate limiting for API calls (canonical source: core/rate_limiting.py)
- TokenUsageTracker: Token usage tracking (canonical source: core/rate_limiting.py)
"""

import os
import time
import json
import re
import hashlib
import streamlit as st
import requests

# Import canonical implementations from core
from core.rate_limiting import TokenUsageTracker, RateLimiter

# Lazy imports for heavy modules - only load when needed
_tiktoken = None
_tiktoken_encoding = None
_np = None
_cosine_similarity = None


def _get_tiktoken_encoding():
    """Lazy load tiktoken encoding"""
    global _tiktoken, _tiktoken_encoding
    if _tiktoken_encoding is None:
        import tiktoken
        _tiktoken = tiktoken
        _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoding


def _get_numpy():
    """Lazy load numpy"""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_cosine_similarity():
    """Lazy load sklearn cosine_similarity"""
    global _cosine_similarity
    if _cosine_similarity is None:
        from sklearn.metrics.pairwise import cosine_similarity
        _cosine_similarity = cosine_similarity
    return _cosine_similarity


from utils.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_BATCH_DELAY,
    RAPIDAPI_MAX_REQUESTS_PER_MINUTE,
    USE_FAST_SKILL_MATCHING
)
from utils.helpers import (
    api_call_with_retry,
    _websocket_keepalive,
    _chunked_sleep,
    _is_streamlit_cloud,
    _ensure_websocket_alive
)


class APIMEmbeddingGenerator:
    """Azure OpenAI Embedding Generator"""
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        endpoint = endpoint.rstrip('/')
        if endpoint.endswith('/openai'):
            endpoint = endpoint[:-7]
        self.endpoint = endpoint
        self.deployment = "text-embedding-3-small"
        self.api_version = "2024-02-01"
        self.url = f"{self.endpoint}/openai/deployments/{self.deployment}/embeddings?api-version={self.api_version}"
        self.headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        self._encoding = None  # Lazy load
    
    @property
    def encoding(self):
        """Lazy load tiktoken encoding"""
        if self._encoding is None:
            self._encoding = _get_tiktoken_encoding()
        return self._encoding
    
    def get_embedding(self, text):
        """Generate embedding for a single text."""
        try:
            payload = {"input": text, "model": self.deployment}
            estimated_tokens = len(self.encoding.encode(text))
            
            def make_request():
                return requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            
            response = api_call_with_retry(make_request, max_retries=3)
            
            if response and response.status_code == 200:
                result = response.json()
                embedding = result['data'][0]['embedding']
                tokens_used = result['usage'].get('total_tokens', 0) if 'usage' in result else estimated_tokens
                return embedding, tokens_used
            elif response:
                # Provide specific error messages based on status code
                if response.status_code == 404:
                    st.error(f"❌ API Error 404: Endpoint not found. Please check your Azure OpenAI endpoint URL: {self.endpoint}")
                elif response.status_code == 401:
                    st.error(f"❌ API Error 401: Unauthorized. Please check your Azure OpenAI API key.")
                elif response.status_code == 403:
                    st.error(f"❌ API Error 403: Forbidden. Your API key may not have access to this resource.")
                elif response.status_code == 429:
                    st.error(f"❌ API Error 429: Rate limit exceeded. Please wait and try again.")
                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text[:200]}")
                return None, 0
            else:
                st.error("❌ No response from Azure OpenAI API. Please check your network connection.")
                return None, 0
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            return None, 0
    
    def get_embeddings_batch(self, texts, batch_size=None):
        """Generate embeddings for a batch of texts.
        
        This method includes WebSocket keepalive calls to prevent connection
        timeouts during long-running embedding operations.
        """
        if not texts:
            return [], 0
        
        effective_batch_size = batch_size or DEFAULT_EMBEDDING_BATCH_SIZE
        if effective_batch_size <= 0:
            effective_batch_size = DEFAULT_EMBEDDING_BATCH_SIZE
        
        embeddings = []
        total_tokens_used = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_batches = (len(texts) + effective_batch_size - 1) // effective_batch_size
        
        # Initial keepalive before starting batch processing
        _websocket_keepalive("Starting embedding generation...", force=True)
        
        for i in range(0, len(texts), effective_batch_size):
            batch = texts[i:i + effective_batch_size]
            batch_num = i // effective_batch_size + 1
            progress = (i + len(batch)) / len(texts)
            progress_bar.progress(progress)
            status_text.text(f"🔄 Generating embeddings: {i + len(batch)}/{len(texts)} (batch {batch_num}/{total_batches})")
            
            # Keepalive before each batch
            _ensure_websocket_alive()
            
            if i > 0 and EMBEDDING_BATCH_DELAY > 0:
                _chunked_sleep(EMBEDDING_BATCH_DELAY, f"Batch {batch_num}/{total_batches}")
            
            try:
                payload = {"input": batch, "model": self.deployment}
                estimated_batch_tokens = sum(len(self.encoding.encode(text)) for text in batch)
                _websocket_keepalive(f"Processing batch {batch_num}/{total_batches}...")
                
                def make_request():
                    return requests.post(self.url, headers=self.headers, json=payload, timeout=30)
                
                response = api_call_with_retry(make_request, max_retries=3)
                
                # Keepalive after API call completes
                _ensure_websocket_alive()
                
                if response and response.status_code == 200:
                    data = response.json()
                    sorted_data = sorted(data['data'], key=lambda x: x['index'])
                    embeddings.extend([item['embedding'] for item in sorted_data])
                    tokens_used = data['usage'].get('total_tokens', 0) if 'usage' in data else estimated_batch_tokens
                    total_tokens_used += tokens_used
                elif response and response.status_code == 429:
                    st.warning(f"⚠️ Rate limit reached after retries. Skipping batch {batch_num}/{total_batches}.")
                    _websocket_keepalive()
                elif response and response.status_code == 404:
                    st.error(f"❌ API Error 404: Endpoint not found. Please check your Azure OpenAI endpoint URL: {self.endpoint}")
                    break  # Stop processing as the endpoint is invalid
                elif response and response.status_code == 401:
                    st.error(f"❌ API Error 401: Unauthorized. Please check your Azure OpenAI API key.")
                    break  # Stop processing as authentication failed
                elif response and response.status_code == 403:
                    st.error(f"❌ API Error 403: Forbidden. Your API key may not have access to this resource.")
                    break  # Stop processing as access is denied
                else:
                    error_msg = f"Batch {batch_num} API Error"
                    if response:
                        error_msg += f" {response.status_code}: {response.text[:100]}"
                    st.warning(f"⚠️ {error_msg}. Trying individual calls...")
                    _websocket_keepalive("Retrying with individual calls...")
                    for idx, text in enumerate(batch):
                        if idx % 2 == 0:
                            _ensure_websocket_alive()
                        emb, tokens = self.get_embedding(text)
                        if emb:
                            embeddings.append(emb)
                            total_tokens_used += tokens
            except Exception as e:
                st.warning(f"⚠️ Error processing batch {batch_num}, trying individual calls: {e}")
                _websocket_keepalive("Recovering from error...")
                for idx, text in enumerate(batch):
                    if idx % 2 == 0:
                        _ensure_websocket_alive()
                    emb, tokens = self.get_embedding(text)
                    if emb:
                        embeddings.append(emb)
                        total_tokens_used += tokens
        
        progress_bar.empty()
        status_text.empty()
        _websocket_keepalive("Embedding generation complete", force=True)
        return embeddings, total_tokens_used


class AzureOpenAITextGenerator:
    """Azure OpenAI Text Generator for resume generation and analysis"""
    def __init__(self, api_key, endpoint, token_tracker=None):
        self.api_key = api_key
        endpoint = endpoint.rstrip('/')
        if endpoint.endswith('/openai'):
            endpoint = endpoint[:-7]
        self.endpoint = endpoint
        self.deployment = "gpt-4o-mini"
        self.api_version = "2024-02-01"
        self.url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        self.headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        self.token_tracker = token_tracker
        self._encoding = None  # Lazy load
    
    @property
    def encoding(self):
        """Lazy load tiktoken encoding"""
        if self._encoding is None:
            self._encoding = _get_tiktoken_encoding()
        return self._encoding
    
    def generate_resume(self, user_profile, job_posting, raw_resume_text=None):
        """Generate a tailored resume based on user profile and job posting using Context Sandwich approach.
        Returns structured JSON data instead of formatted text."""
        from utils.helpers import _websocket_keepalive, api_call_with_retry
        
        system_instructions = """You are an expert resume writer with expertise in ATS optimization and career coaching.
Your task is to create a tailored resume by analyzing the job description and adapting the user's profile.
Return ONLY valid JSON - no markdown, no additional text, no code blocks."""

        job_description = f"""JOB POSTING TO MATCH:
Title: {job_posting.get('title', 'N/A')}
Company: {job_posting.get('company', 'N/A')}
Description: {job_posting.get('description', 'N/A')}
Required Skills: {', '.join(job_posting.get('skills', []))}"""

        structured_profile = f"""STRUCTURED PROFILE:
Name: {user_profile.get('name', 'N/A')}
Email: {user_profile.get('email', 'N/A')}
Phone: {user_profile.get('phone', 'N/A')}
Location: {user_profile.get('location', 'N/A')}
LinkedIn: {user_profile.get('linkedin', 'N/A')}
Portfolio: {user_profile.get('portfolio', 'N/A')}
Summary: {user_profile.get('summary', 'N/A')}
Experience: {user_profile.get('experience', 'N/A')}
Education: {user_profile.get('education', 'N/A')}
Skills: {user_profile.get('skills', 'N/A')}
Certifications: {user_profile.get('certifications', 'N/A')}"""

        raw_resume_section = ""
        if raw_resume_text:
            raw_resume_section = f"\n\nORIGINAL RESUME TEXT (for reference and context):\n{raw_resume_text[:3000]}"

        prompt = f"""{system_instructions}

{job_description}

{structured_profile}{raw_resume_section}

INSTRUCTIONS:
1. Analyze the job posting requirements and identify key skills, technologies, and qualifications needed
2. Tailor the user's profile to match the job description by:
   - Rewriting the summary to emphasize relevant experience
   - Highlighting skills that match the job requirements
   - Rewriting experience bullet points to emphasize relevant achievements
   - Using keywords from the job description for ATS optimization
3. Focus on achievements and measurable results
4. Maintain accuracy - only use information from the provided profile

Return your response as a JSON object with this exact structure:
{{
  "header": {{
    "name": "Full Name",
    "title": "Professional Title (tailored to job)",
    "email": "email@example.com",
    "phone": "phone number",
    "location": "City, State/Country",
    "linkedin": "LinkedIn URL or empty string",
    "portfolio": "Portfolio URL or empty string"
  }},
  "summary": "2-3 sentence professional summary tailored to the job description, emphasizing relevant experience and skills",
  "skills_highlighted": ["Skill 1", "Skill 2", "Skill 3", ...],
  "experience": [
    {{
      "company": "Company Name",
      "title": "Job Title",
      "dates": "Date Range",
      "bullets": [
        "Rewritten bullet point emphasizing relevant achievement...",
        "Another tailored bullet point..."
      ]
    }}
  ],
  "education": "Education details formatted as text",
  "certifications": "Certifications, awards, or other achievements formatted as text"
}}

IMPORTANT: Return ONLY the JSON object, no markdown code blocks, no additional text."""
        
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 3000,
                "temperature": 0.7,
                "response_format": {"type": "json_object"}
            }
            
            _websocket_keepalive("Generating resume...")
            
            def make_request():
                return requests.post(self.url, headers=self.headers, json=payload, timeout=45)
            
            response = api_call_with_retry(make_request, max_retries=3)
            
            if response and response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                if self.token_tracker and 'usage' in result:
                    usage = result['usage']
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    self.token_tracker.add_completion_tokens(prompt_tokens, completion_tokens)
                
                try:
                    content = content.strip()
                    if content.startswith("```"):
                        lines = content.split('\n')
                        content = '\n'.join(lines[1:-1]) if lines[-1].startswith('```') else '\n'.join(lines[1:])
                    
                    resume_data = json.loads(content)
                    return resume_data
                except json.JSONDecodeError as e:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        resume_data = json.loads(json_match.group())
                        return resume_data
                    else:
                        st.error(f"Could not parse JSON response: {e}")
                        return None
            else:
                if response:
                    # Provide specific error messages based on status code
                    if response.status_code == 404:
                        st.error(f"❌ API Error 404: Endpoint not found. Please check your Azure OpenAI endpoint URL: {self.endpoint}")
                        st.info("💡 Tip: Ensure your endpoint URL is correct and includes the proper deployment name.")
                    elif response.status_code == 401:
                        st.error(f"❌ API Error 401: Unauthorized. Please check your Azure OpenAI API key.")
                        st.info("💡 Tip: Update your API key in .streamlit/secrets.toml")
                    elif response.status_code == 403:
                        st.error(f"❌ API Error 403: Forbidden. Your API key may not have access to this resource.")
                    elif response.status_code == 429:
                        st.error(f"❌ API Error 429: Rate limit exceeded. Please wait and try again.")
                    else:
                        error_detail = response.text[:200] if response.text else "No error details"
                        st.error(f"❌ API Error {response.status_code}: {error_detail}")
                else:
                    st.error("❌ No response from Azure OpenAI API. Please check your network connection and API credentials.")
                return None
        except Exception as e:
            st.error(f"Error generating resume: {e}")
            return None
    
    def calculate_match_score(self, resume_content, job_description, embedding_generator):
        """Calculate match score between resume and job description, and identify missing keywords.
        Returns (None, None) if embeddings cannot be generated."""
        from utils.helpers import api_call_with_retry
        
        try:
            resume_embedding, resume_tokens = embedding_generator.get_embedding(resume_content)
            job_embedding, job_tokens = embedding_generator.get_embedding(job_description)
            
            # Token tracker is accessed via session state to avoid circular import
            if 'token_tracker' in st.session_state:
                st.session_state.token_tracker.add_embedding_tokens(resume_tokens + job_tokens)
            
            if not resume_embedding or not job_embedding:
                return None, None
            
            np = _get_numpy()
            cosine_sim = _get_cosine_similarity()
            
            resume_emb = np.array(resume_embedding).reshape(1, -1)
            job_emb = np.array(job_embedding).reshape(1, -1)
            similarity = cosine_sim(resume_emb, job_emb)[0][0]
            match_score = float(similarity)
            
            job_desc_for_keywords = job_description[:8000] if len(job_description) > 8000 else job_description
            if len(job_description) > 8000:
                job_desc_for_keywords += "\n\n[Description truncated for keyword extraction - full description available for matching]"
            
            keyword_prompt = f"""Extract the most important technical skills, tools, technologies, and qualifications mentioned in this job description. 
Return ONLY a JSON object with a "keywords" array, no additional text.

Job Description:
{job_desc_for_keywords}

Return format: {{"keywords": ["keyword1", "keyword2", "keyword3", ...]}}"""
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a keyword extraction expert. Extract only the most important technical and professional keywords. Return JSON with a 'keywords' array."},
                    {"role": "user", "content": keyword_prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            
            def make_request():
                return requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            
            response = api_call_with_retry(make_request, max_retries=2)
            
            missing_keywords = []
            if response and response.status_code == 200:
                try:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    if self.token_tracker and 'usage' in result:
                        usage = result['usage']
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        completion_tokens = usage.get('completion_tokens', 0)
                        self.token_tracker.add_completion_tokens(prompt_tokens, completion_tokens)
                    
                    keyword_data = json.loads(content)
                    job_keywords = keyword_data.get('keywords', [])
                    
                    resume_lower = resume_content.lower()
                    for keyword in job_keywords:
                        if isinstance(keyword, str) and keyword.lower() not in resume_lower:
                            missing_keywords.append(keyword)
                except Exception as e:
                    pass
            
            return match_score, missing_keywords[:10]
            
        except Exception as e:
            st.warning(f"Could not calculate match score: {e}")
            return None, None
    
    def analyze_seniority_level(self, job_titles):
        """Analyze job titles to determine seniority level"""
        from utils.helpers import api_call_with_retry
        
        if not job_titles:
            return "Mid-Senior Level"
        
        titles_text = "\n".join([f"- {title}" for title in job_titles[:10]])
        prompt = f"""Analyze these job titles and determine the most common seniority level.
        
Job Titles:
{titles_text}

Return ONLY a JSON object with this structure:
{{
    "seniority": "Entry Level" | "Mid Level" | "Mid-Senior Level" | "Senior Level" | "Executive Level",
    "confidence": "high" | "medium" | "low"
}}

Choose the most appropriate seniority level based on the job titles."""
        
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a career analyst. Analyze job titles and return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            
            def make_request():
                return requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            
            response = api_call_with_retry(make_request, max_retries=2)
            if response and response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                if self.token_tracker and 'usage' in result:
                    usage = result['usage']
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    self.token_tracker.add_completion_tokens(prompt_tokens, completion_tokens)
                
                data = json.loads(content)
                return data.get('seniority', 'Mid-Senior Level')
        except:
            pass
        
        all_titles = " ".join(job_titles).lower()
        if any(word in all_titles for word in ['executive', 'director', 'vp', 'vice president', 'head of']):
            return "Executive Level"
        elif any(word in all_titles for word in ['senior', 'sr.', 'lead', 'principal']):
            return "Senior Level"
        elif any(word in all_titles for word in ['junior', 'jr.', 'entry', 'associate', 'graduate']):
            return "Entry Level"
        else:
            return "Mid-Senior Level"
    
    def recommend_accreditations(self, job_descriptions, user_skills):
        """Recommend accreditations based on job requirements"""
        from utils.helpers import api_call_with_retry
        
        if not job_descriptions:
            return "PMP or Scrum Master"
        
        combined_desc = "\n\n".join([desc[:1000] for desc in job_descriptions[:5]])
        user_skills_str = user_skills if user_skills else "Not specified"
        
        prompt = f"""Analyze these job descriptions and recommend the most valuable professional accreditation or certification for Hong Kong market.

Job Descriptions:
{combined_desc}

User's Current Skills: {user_skills_str}

Return ONLY a JSON object:
{{
    "accreditation": "Name of certification (e.g., PMP, HKICPA, AWS Certified)",
    "reason": "Brief reason why this certification is valuable"
}}

Focus on certifications that are:
1. Highly valued in Hong Kong market
2. Frequently mentioned in these job descriptions
3. Would unlock more opportunities for the user"""
        
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a career advisor specializing in Hong Kong market. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.5,
                "response_format": {"type": "json_object"}
            }
            
            def make_request():
                return requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            
            response = api_call_with_retry(make_request, max_retries=2)
            if response and response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                if self.token_tracker and 'usage' in result:
                    usage = result['usage']
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    self.token_tracker.add_completion_tokens(prompt_tokens, completion_tokens)
                
                data = json.loads(content)
                return data.get('accreditation', 'PMP or Scrum Master')
        except:
            pass
        
        return "PMP or Scrum Master"
    
    def generate_recruiter_note(self, job, user_profile, semantic_score, skill_score):
        """Generate a personalized recruiter note"""
        from utils.helpers import api_call_with_retry
        
        job_title = job.get('title', '')
        job_desc = job.get('description', '')[:2000]
        user_summary = user_profile.get('summary', '')[:500]
        user_experience = user_profile.get('experience', '')[:500]
        
        prompt = f"""You are a professional recruiter in Hong Kong. Write a brief, actionable note about why this candidate is a good fit for this role.

Job Title: {job_title}
Job Description (excerpt): {job_desc}

Candidate Summary: {user_summary}
Candidate Experience (excerpt): {user_experience}

Match Scores:
- Semantic Match: {semantic_score:.0%}
- Skill Match: {skill_score:.0%}

Write a 2-3 sentence recruiter note that:
1. Highlights the strongest match points
2. Mentions any specific experience or skills that align well
3. Provides actionable feedback

Return ONLY the recruiter note text, no labels or formatting."""
        
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional recruiter. Write concise, actionable notes."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }
            
            def make_request():
                return requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            
            response = api_call_with_retry(make_request, max_retries=2)
            if response and response.status_code == 200:
                result = response.json()
                
                if self.token_tracker and 'usage' in result:
                    usage = result['usage']
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    self.token_tracker.add_completion_tokens(prompt_tokens, completion_tokens)
                
                return result['choices'][0]['message']['content'].strip()
            elif response:
                # Log but don't show error to user for non-critical feature
                if response.status_code == 404:
                    st.warning(f"⚠️ API Error 404: Could not generate recruiter note - endpoint not found.")
                elif response.status_code in [401, 403]:
                    st.warning(f"⚠️ API authentication issue - using fallback recruiter note.")
        except Exception as e:
            # Log exception for debugging but don't disrupt user experience
            import logging
            logging.warning(f"Recruiter note generation failed: {e}")
        
        if semantic_score >= 0.7:
            return f"This role heavily emphasizes recent experience in {job.get('skills', ['relevant skills'])[0] if job.get('skills') else 'relevant skills'}, which is a strong point in your profile."
        else:
            return "Consider highlighting more relevant experience from your background to strengthen your application."


# Note: RateLimiter is now imported from core.rate_limiting
# The canonical version supports a custom sleep_func parameter for WebSocket keepalive


class IndeedScraperAPI:
    """Job scraper using Indeed Scraper API via RapidAPI."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://indeed-scraper-api.p.rapidapi.com/api/job"
        self.headers = {
            'Content-Type': 'application/json',
            'x-rapidapi-host': 'indeed-scraper-api.p.rapidapi.com',
            'x-rapidapi-key': api_key
        }
        # Use RateLimiter with _chunked_sleep for WebSocket keepalive on Streamlit Cloud
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=RAPIDAPI_MAX_REQUESTS_PER_MINUTE,
            sleep_func=_chunked_sleep
        )
    
    def search_jobs(self, query, location="Hong Kong", max_rows=15, job_type="fulltime", country="hk"):
        """Search for jobs using Indeed Scraper API.
        
        Includes WebSocket keepalive calls to prevent connection timeouts
        during the job search API call.
        """
        from utils.helpers import _websocket_keepalive, api_call_with_retry, _ensure_websocket_alive
        
        payload = {
            "scraper": {
                "maxRows": max_rows,
                "query": query,
                "location": location,
                "jobType": job_type,
                "radius": "50",
                "sort": "relevance",
                "fromDays": "7",
                "country": country
            }
        }
        
        try:
            _websocket_keepalive("Preparing job search...", force=True)
            self.rate_limiter.wait_if_needed()
            _websocket_keepalive("Searching jobs...")
            
            def make_request():
                return requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            
            response = api_call_with_retry(make_request, max_retries=3, initial_delay=3)
            
            # Keepalive after API response
            _ensure_websocket_alive()
            
            if response and response.status_code == 201:
                data = response.json()
                jobs = []
                
                _websocket_keepalive("Processing job results...")
                
                if 'returnvalue' in data and 'data' in data['returnvalue']:
                    job_list = data['returnvalue']['data']
                    
                    for idx, job_data in enumerate(job_list):
                        # Keepalive every 5 jobs during parsing
                        if idx % 5 == 0:
                            _ensure_websocket_alive()
                        parsed_job = self._parse_job(job_data)
                        if parsed_job:
                            jobs.append(parsed_job)
                
                _websocket_keepalive("Job search complete", force=True)
                return jobs
            else:
                if response:
                    if response.status_code == 429:
                        st.error("🚫 Rate limit reached for Indeed API. Please wait a few minutes and try again.")
                    else:
                        error_detail = response.text[:200] if response.text else "No error details"
                        st.error(f"API Error: {response.status_code} - {error_detail}")
                return []
                
        except Exception as e:
            st.error(f"Error: {e}")
            return []
    
    def _parse_job(self, job_data):
        """Parse job data from API response."""
        try:
            location_data = job_data.get('location', {})
            location = location_data.get('formattedAddressShort') or location_data.get('city', 'Hong Kong')
            
            job_types = job_data.get('jobType', [])
            job_type = ', '.join(job_types) if job_types else 'Full-time'
            
            benefits = job_data.get('benefits', [])
            attributes = job_data.get('attributes', [])
            
            full_description = job_data.get('descriptionText', 'No description')
            description = full_description[:50000] if len(full_description) > 50000 else full_description
            
            return {
                'title': job_data.get('title', 'N/A'),
                'company': job_data.get('companyName', 'N/A'),
                'location': location,
                'description': description,
                'salary': 'Not specified',
                'job_type': job_type,
                'url': job_data.get('jobUrl', '#'),
                'posted_date': job_data.get('age', 'Recently'),
                'benefits': benefits[:5],
                'skills': attributes[:10],
                'company_rating': job_data.get('rating', {}).get('rating', 0),
                'is_remote': job_data.get('isRemote', False)
            }
        except:
            return None


# Note: TokenUsageTracker is now imported from core.rate_limiting


# Factory functions for getting API clients
def get_token_tracker():
    """Get or create token usage tracker."""
    if 'token_tracker' not in st.session_state:
        st.session_state.token_tracker = TokenUsageTracker()
    return st.session_state.token_tracker


@st.cache_resource(show_spinner=False)
def _create_embedding_generator_resource(api_key, endpoint):
    return APIMEmbeddingGenerator(api_key, endpoint)


@st.cache_resource(show_spinner=False)
def _create_text_generator_resource(api_key, endpoint):
    return AzureOpenAITextGenerator(api_key, endpoint)


def get_embedding_generator():
    """Get cached embedding generator instance."""
    try:
        AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
        
        if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
            st.error("⚠️ Azure OpenAI credentials are missing.")
            return None
        
        generator = _create_embedding_generator_resource(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
        return generator
    except KeyError as e:
        st.error(f"⚠️ Missing required secret: {e}")
        return None
    except Exception as e:
        st.error(f"⚠️ Error initializing embedding generator: {e}")
        return None


def get_text_generator():
    """Get cached text generator instance."""
    try:
        AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
        
        if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
            st.error("⚠️ Azure OpenAI credentials are missing.")
            return None
        
        generator = _create_text_generator_resource(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
        generator.token_tracker = get_token_tracker()
        return generator
    except KeyError as e:
        st.error(f"⚠️ Missing required secret: {e}")
        return None
    except Exception as e:
        st.error(f"⚠️ Error initializing text generator: {e}")
        return None


def get_job_scraper():
    """Get Indeed job scraper."""
    if 'job_scraper' not in st.session_state:
        RAPIDAPI_KEY = st.secrets.get("RAPIDAPI_KEY", "")
        if not RAPIDAPI_KEY:
            st.error("⚠️ RAPIDAPI_KEY is required in secrets.")
            return None
        st.session_state.job_scraper = IndeedScraperAPI(RAPIDAPI_KEY)
    return st.session_state.job_scraper
