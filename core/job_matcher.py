"""
Job matching and scoring functionality.

This module provides the JobMatcher class for semantic job matching using
Pinecone vector database and SentenceTransformer embeddings.
"""

import time
import streamlit as st
from typing import Dict, List, Optional, TYPE_CHECKING

# Type checking imports for Pylance
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone, ServerlessSpec

# Lazy imports for heavy ML/embedding libraries
_SentenceTransformer = None
_Pinecone = None
_ServerlessSpec = None


def _get_sentence_transformer_class():
    """Lazy load SentenceTransformer class."""
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


def _get_pinecone_classes():
    """Lazy load Pinecone classes."""
    global _Pinecone, _ServerlessSpec
    if _Pinecone is None:
        from pinecone import Pinecone, ServerlessSpec
        _Pinecone = Pinecone
        _ServerlessSpec = ServerlessSpec
    return _Pinecone, _ServerlessSpec


# ============================================================================
# CACHED MODEL LOADING - Prevents re-downloading on every page load
# ============================================================================

@st.cache_resource(show_spinner=False)
def _get_sentence_transformer_model_cached(model_name: str):
    """Load and cache SentenceTransformer model - only loaded once."""
    print("📦 Loading sentence transformer model (first time only)...")
    SentenceTransformer = _get_sentence_transformer_class()
    model = SentenceTransformer(model_name)
    print("✅ Model loaded and cached!")
    return model


@st.cache_resource(show_spinner=False)
def _get_pinecone_client_cached(api_key: str):
    """Get cached Pinecone client - only initialized once."""
    print("🔗 Initializing Pinecone client (first time only)...")
    Pinecone, _ = _get_pinecone_classes()
    pc = Pinecone(api_key=api_key)
    print("✅ Pinecone client cached!")
    return pc


@st.cache_resource(show_spinner=False)
def _get_pinecone_index_cached(_pc, index_name: str, embedding_dimension: int, environment: str):
    """Get cached Pinecone index - only initialized once."""
    _, ServerlessSpec = _get_pinecone_classes()
    existing_indexes = _pc.list_indexes()
    index_names = [idx['name'] for idx in existing_indexes]
    
    if index_name not in index_names:
        print(f"🔨 Creating new Pinecone index: {index_name}")
        _pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=environment
            )
        )
        time.sleep(2)
    else:
        print(f"✅ Using existing Pinecone index: {index_name}")
    
    return _pc.Index(index_name)


# ============================================================================
# JOB MATCHER CLASS
# ============================================================================

class JobMatcher:
    """Match resume to jobs using Pinecone semantic search and skill matching.
    
    This class provides semantic job matching capabilities using:
    - SentenceTransformer for embedding generation
    - Pinecone vector database for similarity search
    - Skill-based matching for additional scoring
    """
    
    def __init__(self, config=None):
        """Initialize JobMatcher with lazy-loading for heavy resources.
        
        Args:
            config: Optional config object. If None, will import from config module.
        """
        if config is None:
            from config import Config
            config = Config
        
        self._config = config
        self._pc = None
        self._model = None
        self._index = None
    
    @property
    def pc(self):
        """Lazy-load Pinecone client."""
        if self._pc is None:
            self._pc = _get_pinecone_client_cached(self._config.PINECONE_API_KEY)
        return self._pc
    
    @property
    def model(self):
        """Lazy-load SentenceTransformer model."""
        if self._model is None:
            self._model = _get_sentence_transformer_model_cached(self._config.MODEL_NAME)
        return self._model
    
    @property
    def index(self):
        """Lazy-load Pinecone index."""
        if self._index is None:
            self._index = _get_pinecone_index_cached(
                self.pc,
                self._config.INDEX_NAME,
                self._config.EMBEDDING_DIMENSION,
                self._config.PINECONE_ENVIRONMENT
            )
        return self._index
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        text = str(text).strip()
        if not text:
            text = "empty"
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def index_jobs(self, jobs: List[Dict]) -> int:
        """Index jobs in Pinecone vector database.
        
        Args:
            jobs: List of job dictionaries to index
            
        Returns:
            Number of jobs successfully indexed
        """
        if not jobs:
            return 0
        
        vectors_to_upsert = []
        
        for job in jobs:
            try:
                job_text = f"{job['title']} {job['company']} {job['description']}"
                embedding = self.generate_embedding(job_text)
                
                vectors_to_upsert.append({
                    'id': job['id'],
                    'values': embedding,
                    'metadata': {
                        'title': job['title'][:512],
                        'company': job['company'][:512],
                        'location': job['location'][:512],
                        'description': job['description'][:1000],
                        'url': job.get('url', '')[:512],
                        'posted_date': str(job.get('posted_date', ''))[:100]
                    }
                })
                
            except Exception as e:
                print(f"⚠️ Error indexing job {job.get('id', 'unknown')}: {e}")
                continue
        
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            return len(vectors_to_upsert)
        
        return 0
    
    def search_similar_jobs(self, resume_data: Dict, ai_analysis: Dict, top_k: int = 20) -> List[Dict]:
        """Search for similar jobs using semantic similarity.
        
        Args:
            resume_data: Parsed resume data with 'raw_text' key
            ai_analysis: AI-extracted analysis with 'primary_role' and 'skills' keys
            top_k: Number of top matches to return
            
        Returns:
            List of matched job dictionaries with similarity scores
        """
        try:
            # Create rich query from resume + AI analysis
            primary_role = ai_analysis.get('primary_role', '')
            skills = ' '.join(ai_analysis.get('skills', [])[:20])
            resume_snippet = resume_data.get('raw_text', '')[:1000]
            
            query_text = f"{primary_role} {skills} {resume_snippet}"
            
            print(f"🎯 Creating semantic embedding for resume...")
            query_embedding = self.generate_embedding(query_text)
            
            print(f"🔍 Searching Pinecone for top {top_k} matches...")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            matched_jobs = []
            for match in results['matches']:
                job = {
                    'id': match['id'],
                    'similarity_score': float(match['score']) * 100,
                    **match['metadata']
                }
                matched_jobs.append(job)
            
            print(f"✅ Found {len(matched_jobs)} semantic matches")
            return matched_jobs
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def analyze_simple_match(self, job_data: tuple, seeker_data: tuple) -> Dict:
        """Simple match analysis between job and seeker.
        
        This is a method wrapper around the module-level analyze_match_simple function.
        
        Args:
            job_data: Tuple of job fields from database query
            seeker_data: Tuple of seeker fields from database query
            
        Returns:
            Dictionary with match analysis results
        """
        return analyze_match_simple(job_data, seeker_data)


# ============================================================================
# MATCH SCORING FUNCTIONS
# ============================================================================

def calculate_match_scores(jobs: List[Dict], ai_analysis: Dict) -> List[Dict]:
    """Calculate detailed match scores - 60% semantic + 40% skill match.
    
    Args:
        jobs: List of job dictionaries with 'similarity_score' from semantic search
        ai_analysis: AI analysis with 'skills' key
        
    Returns:
        Jobs with added score fields
    """
    candidate_skills = set([s.lower() for s in ai_analysis.get('skills', [])])
    
    print(f"📊 Calculating match scores using {len(candidate_skills)} candidate skills...")
    
    for job in jobs:
        description = job.get('description', '').lower()
        title = job.get('title', '').lower()
        
        # Count skill matches
        matched_skills = []
        for skill in candidate_skills:
            if skill in description or skill in title:
                matched_skills.append(skill)
        
        # Calculate skill match percentage
        skill_match_pct = (len(matched_skills) / len(candidate_skills) * 100) if candidate_skills else 0
        
        # Semantic similarity (from Pinecone)
        semantic_score = job.get('similarity_score', 0)
        
        # Combined score: 60% semantic + 40% skill match
        combined_score = (0.6 * semantic_score) + (0.4 * skill_match_pct)
        
        # Add to job
        job['skill_match_percentage'] = round(skill_match_pct, 1)
        job['matched_skills'] = list(matched_skills)[:10]
        job['matched_skills_count'] = len(matched_skills)
        job['combined_score'] = round(combined_score, 1)
        job['semantic_score'] = round(semantic_score, 1)
    
    return jobs


def analyze_match_simple(job_data: tuple, seeker_data: tuple) -> Dict:
    """Simple match analysis between job and seeker.
    
    Args:
        job_data: Tuple of job fields from database query
        seeker_data: Tuple of seeker fields from database query
        
    Returns:
        Dictionary with match analysis results
    """
    match_score = 50  # Basic Score

    # Skills matching
    job_skills = str(job_data[4]).lower()
    seeker_skills = str(seeker_data[2]).lower()
    skill_match = len(set(job_skills.split()) & set(seeker_skills.split())) / max(len(job_skills.split()), 1)
    match_score += skill_match * 20

    # Experience matching
    experience_map = {"fresh graduate": 0, "1-3 years": 1, "3-5 years": 2, "5-10 years": 3, "10+ years": 4}
    job_exp = job_data[11]
    seeker_exp = seeker_data[3]

    if job_exp in experience_map and seeker_exp in experience_map:
        exp_diff = abs(experience_map[job_exp] - experience_map[seeker_exp])
        match_score -= exp_diff * 5

    # Industry matching
    job_industry = str(job_data[6]).lower()
    seeker_industry = str(seeker_data[6]).lower()
    if job_industry in seeker_industry or seeker_industry in job_industry:
        match_score += 10

    # Location matching
    job_location = str(job_data[8]).lower()
    seeker_location = str(seeker_data[7]).lower()
    if job_location in seeker_location or seeker_location in job_location:
        match_score += 5

    match_score = max(0, min(100, match_score))

    # Analyze based on score
    if match_score >= 80:
        strengths = ["High skill match", "Experience meets requirements", "Strong industry relevance"]
        gaps = []
        recommendation = "Highly recommend for interview"
    elif match_score >= 60:
        strengths = ["Core skills match", "Basic experience aligns"]
        gaps = ["Some skills need improvement", "Slight experience gap"]
        recommendation = "Recommend further communication"
    else:
        strengths = ["Has relevant background"]
        gaps = ["Low skill match", "Experience does not meet requirements"]
        recommendation = "Further evaluation needed"

    return {
        "match_score": int(match_score),
        "key_strengths": strengths,
        "potential_gaps": gaps,
        "recommendation": recommendation,
        "salary_match": "Good" if match_score > 70 else "Average",
        "culture_fit": "High" if match_score > 75 else "Medium"
    }


def calculate_job_match_score(job_seeker_data: Dict, job_data: Dict) -> Dict:
    """Calculate job match score between job seeker and job data.
    
    Args:
        job_seeker_data: Job seeker profile dictionary
        job_data: Job posting dictionary
        
    Returns:
        Match result dictionary with score and details
    """
    try:
        score = 0
        matched_skills = []
        
        # 1. Skill match (40%)
        job_seeker_skills = job_seeker_data.get('hard_skills', '').lower()
        job_description = job_data.get('job_description', '').lower()
        
        if job_seeker_skills:
            skills_list = [skill.strip().lower() for skill in job_seeker_skills.split(',')]
            for skill in skills_list:
                if skill and skill in job_description:
                    score += 5  # Each match adds 5 points
                    matched_skills.append(skill)
                    if score >= 40:  # Max skill points at 40
                        score = 40
                        break
        
        # 2. Experience match (20%)
        job_seeker_experience = job_seeker_data.get('work_experience', '').lower()
        job_title = job_data.get('job_title', '').lower()
        if 'senior' in job_title and 'senior' in job_seeker_experience:
            score += 20
        elif 'junior' in job_title and 'junior' in job_seeker_experience:
            score += 20
        elif 'entry' in job_title and 'fresh' in job_seeker_experience:
            score += 20
        else:
            score += 10  # 10 points for general experience match
        
        # 3. Location match (20%)
        job_seeker_location = job_seeker_data.get('location_preference', '').lower()
        job_location = job_data.get('job_city', '').lower()
        
        if job_seeker_location and job_location:
            if job_seeker_location in job_location or job_location in job_seeker_location:
                score += 20
            else:
                score += 5  # Unmatched location but give base score of 5
        
        # 4. Job Title Match (20%)
        job_seeker_role = job_seeker_data.get('primary_role', '').lower()
        
        if job_seeker_role and job_title:
            if job_seeker_role in job_title:
                score += 20
            else:
                # Search for keywords in job title
                search_terms = job_seeker_data.get('simple_search_terms', '').lower()
                if search_terms:
                    terms = [term.strip() for term in search_terms.split(',')]
                    for term in terms:
                        if term in job_title:
                            score += 15
                            break
        
        # Make sure the score is between 0 and 100
        score = min(max(score, 0), 100)
        
        return {
            'overall_score': score,
            'matched_skills': matched_skills,
            'skill_match': len(matched_skills),
            'experience_match': 'senior' in job_seeker_experience and 'senior' in job_title,
            'location_match': job_seeker_location in job_location if job_seeker_location and job_location else False
        }
        
    except Exception as e:
        print(f"❌ Error when calculating matching score: {e}")
        return {
            'overall_score': 0,
            'matched_skills': [],
            'skill_match': 0,
            'experience_match': False,
            'location_match': False
        }
