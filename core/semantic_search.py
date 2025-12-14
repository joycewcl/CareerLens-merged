# core/semantic_search.py
"""
Semantic job search using embeddings and caching.

This module provides:
- SemanticJobSearch class for embedding-based job matching
- Job caching with TTL support
- Resume embedding generation and storage

Consolidated from:
- modules/semantic_search/job_search.py
- modules/semantic_search/cache.py
- modules/semantic_search/embeddings.py
"""

import os
import hashlib
import time
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Lazy imports for heavy modules - only load when needed
_np = None
_cosine_similarity = None
_chromadb = None


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


def _get_chromadb():
    """Lazy load chromadb"""
    global _chromadb
    if _chromadb is None:
        import chromadb
        _chromadb = chromadb
    return _chromadb


# ============================================================================
# HELPER IMPORTS (lazy to avoid circular imports)
# ============================================================================

def _get_utils():
    """Lazy load utility functions to avoid circular imports"""
    from utils import get_token_tracker, get_embedding_generator
    from utils.helpers import _websocket_keepalive, _ensure_websocket_alive, _chunked_sleep
    from utils.config_utils import DEFAULT_MAX_JOBS_TO_INDEX, USE_FAST_SKILL_MATCHING
    return {
        'get_token_tracker': get_token_tracker,
        'get_embedding_generator': get_embedding_generator,
        '_websocket_keepalive': _websocket_keepalive,
        '_ensure_websocket_alive': _ensure_websocket_alive,
        '_chunked_sleep': _chunked_sleep,
        'DEFAULT_MAX_JOBS_TO_INDEX': DEFAULT_MAX_JOBS_TO_INDEX,
        'USE_FAST_SKILL_MATCHING': USE_FAST_SKILL_MATCHING,
    }


def _is_streamlit_cloud():
    """Detect if running on Streamlit Cloud (ephemeral filesystem)."""
    return (
        os.environ.get('STREAMLIT_SHARING_MODE') is not None or
        os.environ.get('STREAMLIT_SERVER_PORT') is not None or
        os.path.exists('/mount/src') or
        'streamlit.app' in os.environ.get('HOSTNAME', '')
    )


# ============================================================================
# CACHE FUNCTIONS (from modules/semantic_search/cache.py)
# ============================================================================

def is_cache_valid(cache_entry: Dict) -> bool:
    """Check if cache entry is still valid (not expired)."""
    if not cache_entry or not isinstance(cache_entry, dict):
        return False
    
    expires_at = cache_entry.get('expires_at')
    if expires_at is None:
        return False
    
    if isinstance(expires_at, str):
        try:
            expires_at = datetime.fromisoformat(expires_at)
        except ValueError:
            try:
                expires_at = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return False
    
    return datetime.now() < expires_at


def _build_jobs_cache_key(query: str, location: str, max_rows: int, job_type: str, country: str) -> str:
    """Create a unique cache key for job searches."""
    normalized_query = (query or "").strip().lower()
    return "|".join([
        normalized_query,
        (location or "").strip().lower(),
        str(max_rows),
        (job_type or "").strip().lower(),
        (country or "").strip().lower()
    ])


def _ensure_jobs_cache_structure():
    """Ensure jobs_cache is always a dict keyed by cache keys (handles legacy formats)."""
    if 'jobs_cache' not in st.session_state or not isinstance(st.session_state.jobs_cache, dict):
        st.session_state.jobs_cache = {}
        return
    cache = st.session_state.jobs_cache
    if cache and 'jobs' in cache and isinstance(cache['jobs'], list):
        cache_key = cache.get('cache_key') or _build_jobs_cache_key(
            cache.get('query', ''),
            cache.get('location', 'Hong Kong'),
            cache.get('count', len(cache.get('jobs', []))),
            cache.get('job_type', 'fulltime'),
            cache.get('country', 'hk')
        )
        st.session_state.jobs_cache = {cache_key: {**cache, 'cache_key': cache_key}}


def _get_cached_jobs(query: str, location: str, max_rows: int, job_type: str, country: str) -> Optional[Dict]:
    """Return cached jobs for a given search signature if valid."""
    _ensure_jobs_cache_structure()
    cache_key = _build_jobs_cache_key(query, location, max_rows, job_type, country)
    cache_entry = st.session_state.jobs_cache.get(cache_key)
    if not cache_entry:
        return None
    if not is_cache_valid(cache_entry):
        st.session_state.jobs_cache.pop(cache_key, None)
        return None
    return cache_entry


def _store_jobs_in_cache(query: str, location: str, max_rows: int, job_type: str, 
                         country: str, jobs: List[Dict], cache_ttl_hours: int = 168) -> Dict:
    """Persist job results in cache with TTL metadata."""
    _ensure_jobs_cache_structure()
    cache_key = _build_jobs_cache_key(query, location, max_rows, job_type, country)
    now = datetime.now()
    expires_at = now + timedelta(hours=cache_ttl_hours)
    st.session_state.jobs_cache[cache_key] = {
        'jobs': jobs,
        'count': len(jobs),
        'timestamp': now.isoformat(),
        'query': query,
        'location': location,
        'job_type': job_type,
        'country': country,
        'cache_key': cache_key,
        'expires_at': expires_at.isoformat()
    }
    return st.session_state.jobs_cache[cache_key]


def fetch_jobs_with_cache(scraper, query: str, location: str = "Hong Kong", max_rows: int = 25, 
                          job_type: str = "fulltime", country: str = "hk", 
                          cache_ttl_hours: int = 168, force_refresh: bool = False) -> List[Dict]:
    """
    Fetch jobs with session-level caching to avoid RapidAPI rate limits.
    Set force_refresh=True to bypass cache for a particular query.
    
    Includes WebSocket keepalive to prevent connection timeouts during API calls.
    """
    if scraper is None:
        return []
    
    utils = _get_utils()
    _websocket_keepalive = utils['_websocket_keepalive']
    
    _websocket_keepalive("Checking job cache...", force=True)
    _ensure_jobs_cache_structure()
    cache_key = _build_jobs_cache_key(query, location, max_rows, job_type, country)
    
    if force_refresh:
        if cache_key in st.session_state.jobs_cache:
            st.caption("🔁 Forcing a fresh job search (cache bypassed)")
        st.session_state.jobs_cache.pop(cache_key, None)
    else:
        cache_entry = _get_cached_jobs(query, location, max_rows, job_type, country)
        if cache_entry:
            timestamp = cache_entry.get('timestamp')
            expires_at = cache_entry.get('expires_at')
            expires_in_minutes = None
            if isinstance(expires_at, str):
                try:
                    expires_dt = datetime.fromisoformat(expires_at)
                    expires_in_minutes = max(0, int((expires_dt - datetime.now()).total_seconds() // 60))
                except ValueError:
                    pass
            if timestamp and isinstance(timestamp, str):
                try:
                    ts_dt = datetime.fromisoformat(timestamp)
                    human_ts = ts_dt.strftime("%b %d %H:%M")
                except ValueError:
                    human_ts = timestamp
            else:
                human_ts = "earlier"
            remaining_text = f" (~{expires_in_minutes} min left)" if expires_in_minutes is not None else ""
            st.caption(f"♻️ Using cached job results from {human_ts}{remaining_text}")
            _websocket_keepalive()
            return cache_entry.get('jobs', [])
    
    _websocket_keepalive("Fetching jobs from API...")
    jobs = scraper.search_jobs(query, location, max_rows, job_type, country)
    
    if jobs:
        _websocket_keepalive("Caching job results...")
        _store_jobs_in_cache(query, location, max_rows, job_type, country, jobs, cache_ttl_hours)
    
    _websocket_keepalive("Job fetch complete", force=True)
    return jobs


# ============================================================================
# EMBEDDING FUNCTIONS (from modules/semantic_search/embeddings.py)
# ============================================================================

def generate_and_store_resume_embedding(resume_text: str, user_profile: Optional[Dict] = None) -> Optional[List[float]]:
    """Generate embedding for resume and store in session state.
    
    This is called once when resume is uploaded/updated, so we can reuse
    the embedding for all subsequent searches without regenerating it.
    """
    if not resume_text:
        st.session_state.resume_embedding = None
        return None
    
    utils = _get_utils()
    get_embedding_generator = utils['get_embedding_generator']
    get_token_tracker = utils['get_token_tracker']
    
    # Build resume query text
    if user_profile:
        profile_data = f"{user_profile.get('summary', '')} {user_profile.get('experience', '')} {user_profile.get('skills', '')}"
        resume_query = f"{resume_text} {profile_data}"
    else:
        resume_query = resume_text
    
    # Generate embedding
    embedding_gen = get_embedding_generator()
    if not embedding_gen:
        return None
    
    embedding, tokens_used = embedding_gen.get_embedding(resume_query)
    
    # Update token tracker
    token_tracker = get_token_tracker()
    if token_tracker:
        token_tracker.add_embedding_tokens(tokens_used)
    
    if embedding:
        st.session_state.resume_embedding = embedding
        return embedding
    
    return None


# ============================================================================
# SEMANTIC JOB SEARCH CLASS (from modules/semantic_search/job_search.py)
# ============================================================================

class SemanticJobSearch:
    """Semantic job search using embeddings"""
    
    def __init__(self, embedding_generator, use_persistent_store: bool = True):
        self.embedding_gen = embedding_generator
        self.job_embeddings = []
        self.jobs = []
        self.chroma_client = None
        self.collection = None
        self._chroma_initialized = False
        
        if _is_streamlit_cloud():
            use_persistent_store = False
        
        self.use_persistent_store = use_persistent_store
    
    def _init_chroma_lazy(self):
        """Lazy initialize ChromaDB only when needed"""
        if self._chroma_initialized:
            return
        
        self._chroma_initialized = True
        chromadb = _get_chromadb()
        
        if self.use_persistent_store:
            try:
                chroma_db_path = os.path.join(os.getcwd(), ".chroma_db")
                os.makedirs(chroma_db_path, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
                self.collection = self.chroma_client.get_or_create_collection(
                    name="job_embeddings",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                st.warning(f"⚠️ Could not initialize persistent vector store: {e}. Using in-memory storage.")
                self.use_persistent_store = False
        
        if not self.use_persistent_store and self.chroma_client is None:
            try:
                self.chroma_client = chromadb.EphemeralClient()
                self.collection = self.chroma_client.get_or_create_collection(
                    name="job_embeddings",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                pass
    
    def _get_job_hash(self, job: Dict) -> str:
        """Generate a hash for a job to use as ID."""
        job_str = f"{job.get('title', '')}_{job.get('company', '')}_{job.get('url', '')}"
        return hashlib.md5(job_str.encode()).hexdigest()
    
    def index_jobs(self, jobs: List[Dict], max_jobs_to_index: Optional[int] = None):
        """Simplified job indexing: Check if job exists, if not, embed and store.
        
        Includes WebSocket keepalive calls to prevent connection timeouts.
        """
        if not jobs:
            st.warning("⚠️ No jobs available to index.")
            self.jobs = []
            self.job_embeddings = []
            return
        
        utils = _get_utils()
        _websocket_keepalive = utils['_websocket_keepalive']
        _ensure_websocket_alive = utils['_ensure_websocket_alive']
        get_token_tracker = utils['get_token_tracker']
        DEFAULT_MAX_JOBS_TO_INDEX = utils['DEFAULT_MAX_JOBS_TO_INDEX']
        
        _websocket_keepalive("Starting job indexing...", force=True)
        
        effective_limit = max_jobs_to_index or min(len(jobs), DEFAULT_MAX_JOBS_TO_INDEX)
        effective_limit = max(1, min(effective_limit, len(jobs)))
        if effective_limit < len(jobs):
            st.info(f"⚙️ Indexing first {effective_limit} of {len(jobs)} jobs to reduce embedding API calls.")
        jobs_to_index = jobs[:effective_limit]
        self.jobs = jobs_to_index
        
        _ensure_websocket_alive()
        
        job_texts = [
            f"{job['title']} at {job['company']}. {job['description']} Skills: {', '.join(job['skills'][:5])}"
            for job in jobs_to_index
        ]
        
        st.info(f"📊 Indexing {len(jobs_to_index)} jobs...")
        _websocket_keepalive("Preparing embeddings...")
        
        # Lazy init ChromaDB only when actually indexing
        self._init_chroma_lazy()
        
        if self.use_persistent_store and self.collection:
            try:
                job_hashes = [self._get_job_hash(job) for job in jobs_to_index]
                existing_data = self.collection.get(ids=job_hashes, include=['embeddings'])
                existing_ids = set(existing_data.get('ids', [])) if existing_data else set()
                
                jobs_to_embed = []
                indices_to_embed = []
                for idx, job_hash in enumerate(job_hashes):
                    if job_hash not in existing_ids:
                        jobs_to_embed.append(job_texts[idx])
                        indices_to_embed.append(idx)
                
                if jobs_to_embed:
                    st.info(f"🔄 Generating embeddings for {len(jobs_to_embed)} new jobs...")
                    new_embeddings, tokens_used = self.embedding_gen.get_embeddings_batch(jobs_to_embed)
                    
                    token_tracker = get_token_tracker()
                    if token_tracker:
                        token_tracker.add_embedding_tokens(tokens_used)
                    
                    for idx, emb in zip(indices_to_embed, new_embeddings):
                        if emb:
                            job_hash = job_hashes[idx]
                            self.collection.upsert(
                                ids=[job_hash],
                                embeddings=[emb],
                                documents=[job_texts[idx]],
                                metadatas=[{"job_index": idx}]
                            )
                
                retrieved = self.collection.get(ids=job_hashes, include=['embeddings'])
                if retrieved and 'embeddings' in retrieved and retrieved['embeddings'] is not None and len(retrieved['embeddings']) > 0:
                    hash_to_emb = {h: e for h, e in zip(retrieved['ids'], retrieved['embeddings'])}
                    self.job_embeddings = [hash_to_emb.get(h, None) for h in job_hashes]
                    self.job_embeddings = [e for e in self.job_embeddings if e is not None]
                    st.success(f"✅ Indexed {len(self.job_embeddings)} jobs (using persistent store)")
                else:
                    self.job_embeddings, tokens_used = self.embedding_gen.get_embeddings_batch(job_texts)
                    token_tracker = get_token_tracker()
                    if token_tracker:
                        token_tracker.add_embedding_tokens(tokens_used)
                    st.success(f"✅ Indexed {len(self.job_embeddings)} jobs")
            except Exception as e:
                st.warning(f"⚠️ Error using persistent store: {e}. Generating new embeddings...")
                self.job_embeddings, tokens_used = self.embedding_gen.get_embeddings_batch(job_texts)
                token_tracker = get_token_tracker()
                if token_tracker:
                    token_tracker.add_embedding_tokens(tokens_used)
                self.use_persistent_store = False
                st.success(f"✅ Indexed {len(self.job_embeddings)} jobs")
        else:
            self.job_embeddings, tokens_used = self.embedding_gen.get_embeddings_batch(job_texts)
            token_tracker = get_token_tracker()
            if token_tracker:
                token_tracker.add_embedding_tokens(tokens_used)
            st.success(f"✅ Indexed {len(self.job_embeddings)} jobs")
    
    def search(self, query: Optional[str] = None, top_k: int = 10, 
               resume_embedding: Optional[List[float]] = None) -> List[Dict]:
        """Simplified search: Use pre-computed resume embedding if available, otherwise generate from query.
        
        Includes WebSocket keepalive during search operations.
        """
        if not self.job_embeddings:
            return []
        
        utils = _get_utils()
        _websocket_keepalive = utils['_websocket_keepalive']
        _ensure_websocket_alive = utils['_ensure_websocket_alive']
        get_token_tracker = utils['get_token_tracker']
        
        _websocket_keepalive("Searching jobs...", force=True)
        
        if resume_embedding is not None:
            query_embedding = resume_embedding
        elif query:
            _websocket_keepalive("Generating query embedding...")
            query_embedding, tokens_used = self.embedding_gen.get_embedding(query)
            token_tracker = get_token_tracker()
            if token_tracker:
                token_tracker.add_embedding_tokens(tokens_used)
            if not query_embedding:
                return []
        else:
            return []
        
        _ensure_websocket_alive()
        
        np = _get_numpy()
        cosine_similarity = _get_cosine_similarity()
        
        query_emb = np.array(query_embedding).reshape(1, -1)
        job_embs = np.array(self.job_embeddings)
        
        similarities = cosine_similarity(query_emb, job_embs)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        _websocket_keepalive("Ranking results...")
        
        results = []
        for idx in top_indices:
            results.append({
                'job': self.jobs[idx],
                'similarity_score': float(similarities[idx]),
                'rank': len(results) + 1
            })
        
        return results
    
    def calculate_skill_match(self, user_skills: str, job_skills: List[str]) -> Tuple[float, List[str]]:
        """Calculate skill-based match score.
        
        Uses semantic matching with embeddings when available, with automatic
        fallback to string matching. Includes WebSocket keepalive.
        """
        if not user_skills or not job_skills:
            return 0.0, []
        
        user_skills_list = [s.strip() for s in str(user_skills).split(',') if s.strip()]
        job_skills_list = [s.strip() for s in job_skills if isinstance(s, str) and s.strip()]
        
        if not user_skills_list or not job_skills_list:
            return 0.0, []
        
        utils = _get_utils()
        USE_FAST_SKILL_MATCHING = utils['USE_FAST_SKILL_MATCHING']
        _ensure_websocket_alive = utils['_ensure_websocket_alive']
        get_token_tracker = utils['get_token_tracker']
        
        if USE_FAST_SKILL_MATCHING:
            return self._calculate_skill_match_string_based(user_skills_list, job_skills_list)
        
        try:
            _ensure_websocket_alive()
            
            user_skills_key = ",".join(sorted(user_skills_list))
            if user_skills_key in st.session_state.user_skills_embeddings_cache:
                user_skill_embeddings = st.session_state.user_skills_embeddings_cache[user_skills_key]
                user_tokens = 0
            else:
                user_skill_embeddings, user_tokens = self.embedding_gen.get_embeddings_batch(user_skills_list, batch_size=10)
                if user_skill_embeddings:
                    st.session_state.user_skills_embeddings_cache[user_skills_key] = user_skill_embeddings
            
            _ensure_websocket_alive()
            
            job_skills_key = ",".join(sorted(job_skills_list))
            if job_skills_key in st.session_state.skill_embeddings_cache:
                job_skill_embeddings = st.session_state.skill_embeddings_cache[job_skills_key]
                job_tokens = 0
            else:
                job_skill_embeddings, job_tokens = self.embedding_gen.get_embeddings_batch(job_skills_list, batch_size=10)
                if job_skill_embeddings:
                    st.session_state.skill_embeddings_cache[job_skills_key] = job_skill_embeddings
            
            if user_tokens > 0 or job_tokens > 0:
                token_tracker = get_token_tracker()
                if token_tracker:
                    token_tracker.add_embedding_tokens(user_tokens + job_tokens)
            
            if not user_skill_embeddings or not job_skill_embeddings:
                return self._calculate_skill_match_string_based(user_skills_list, job_skills_list)
            
            np = _get_numpy()
            cosine_similarity = _get_cosine_similarity()
            
            user_embs = np.array(user_skill_embeddings)
            job_embs = np.array(job_skill_embeddings)
            
            similarity_matrix = cosine_similarity(job_embs, user_embs)
            
            similarity_threshold = 0.7
            matched_skills = []
            matched_indices = set()
            
            for job_idx, job_skill in enumerate(job_skills_list):
                best_match_idx = np.argmax(similarity_matrix[job_idx])
                best_similarity = similarity_matrix[job_idx][best_match_idx]
                
                if best_similarity >= similarity_threshold and best_match_idx not in matched_indices:
                    matched_skills.append(job_skill)
                    matched_indices.add(best_match_idx)
            
            match_score = len(matched_skills) / len(job_skills_list) if job_skills_list else 0.0
            missing_skills = [js for js in job_skills_list if js not in matched_skills]
            
            return min(match_score, 1.0), missing_skills[:5]
            
        except Exception as e:
            return self._calculate_skill_match_string_based(user_skills_list, job_skills_list)
    
    def _calculate_skill_match_string_based(self, user_skills_list: List[str], 
                                            job_skills_list: List[str]) -> Tuple[float, List[str]]:
        """Fallback string-based skill matching"""
        user_skills_lower = [s.lower() for s in user_skills_list]
        job_skills_lower = [s.lower() for s in job_skills_list]
        
        matched_skills = []
        for job_skill in job_skills_lower:
            for user_skill in user_skills_lower:
                if job_skill in user_skill or user_skill in job_skill:
                    matched_skills.append(job_skill)
                    break
        
        match_score = len(matched_skills) / len(job_skills_lower) if job_skills_lower else 0.0
        missing_skills = [job_skills_list[i] for i, js in enumerate(job_skills_lower) if js not in matched_skills]
        
        return min(match_score, 1.0), missing_skills[:5]
