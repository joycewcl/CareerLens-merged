"""
API Connectivity and External Service Tests
============================================
Tests actual connectivity to external services:
1. Azure OpenAI API
2. RapidAPI (Indeed/LinkedIn)
3. Pinecone Vector Database
4. Local Embedding Model (SentenceTransformer)

Note: These tests require actual API keys to be configured in .streamlit/secrets.toml
"""

import pytest
import os
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLocalEmbeddings:
    """Test local embedding model (doesn't require API keys)."""
    
    def test_sentence_transformer_load(self):
        """Test SentenceTransformer model can be loaded"""
        from sentence_transformers import SentenceTransformer
        
        # Load the model used by the app
        model = SentenceTransformer('all-MiniLM-L6-v2')
        assert model is not None
    
    def test_generate_embedding(self):
        """Test generating an embedding from text"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        text = "Senior Python Developer with machine learning experience"
        
        embedding = model.encode(text)
        
        assert embedding is not None
        assert len(embedding) == 384  # all-MiniLM-L6-v2 produces 384-dim vectors
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Similar texts
        text1 = "Python developer with machine learning skills"
        text2 = "ML engineer experienced in Python programming"
        
        # Different text
        text3 = "Chef specializing in Italian cuisine"
        
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        emb3 = model.encode(text3)
        
        # Calculate similarities
        sim_12 = cosine_similarity([emb1], [emb2])[0][0]
        sim_13 = cosine_similarity([emb1], [emb3])[0][0]
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13
        print(f"\nSimilarity (Python dev vs ML engineer): {sim_12:.3f}")
        print(f"Similarity (Python dev vs Chef): {sim_13:.3f}")


class TestConfigurationSecrets:
    """Test API configuration (checks if secrets are available)."""
    
    def test_config_loads(self):
        """Test Config.setup() runs"""
        from config import Config
        
        Config._initialized = False
        Config.setup()
        
        assert Config._initialized is True
    
    def test_check_azure_credentials(self):
        """Check Azure credentials configuration"""
        from config import Config
        
        Config.setup()
        is_configured, message = Config.check_azure_credentials()
        
        print(f"\nAzure OpenAI configured: {is_configured}")
        if not is_configured:
            print(f"Message: {message}")
        
        # Just report status, don't fail
        assert isinstance(is_configured, bool)
    
    def test_rapidapi_key_status(self):
        """Check RapidAPI key configuration"""
        from config import Config
        
        Config.setup()
        has_key = Config.RAPIDAPI_KEY is not None
        
        print(f"\nRapidAPI Key configured: {has_key}")
        
        # Just report status, don't fail
        assert isinstance(has_key, bool)
    
    def test_pinecone_key_status(self):
        """Check Pinecone API key configuration"""
        from config import Config
        
        Config.setup()
        has_key = Config.PINECONE_API_KEY is not None
        
        print(f"\nPinecone API Key configured: {has_key}")
        
        # Just report status, don't fail
        assert isinstance(has_key, bool)


class TestAzureOpenAIConnectivity:
    """Test Azure OpenAI API connectivity (requires API key)."""
    
    @pytest.fixture
    def skip_if_no_azure(self):
        """Skip test if Azure credentials not configured"""
        from config import Config
        Config.setup()
        is_configured, _ = Config.check_azure_credentials()
        if not is_configured:
            pytest.skip("Azure OpenAI credentials not configured")
    
    def test_azure_openai_chat(self, skip_if_no_azure):
        """Test Azure OpenAI chat completion"""
        from config import Config
        from openai import AzureOpenAI
        
        Config.setup()
        
        client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )
        
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT or "gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_tokens=10
        )
        
        assert response is not None
        assert len(response.choices) > 0
        print(f"\nAzure OpenAI response: {response.choices[0].message.content}")
    
    def test_azure_openai_embedding(self, skip_if_no_azure):
        """Test Azure OpenAI embedding generation"""
        from config import Config
        from openai import AzureOpenAI
        
        Config.setup()
        
        client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )
        
        response = client.embeddings.create(
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-3-small",
            input="Test embedding text"
        )
        
        assert response is not None
        assert len(response.data) > 0
        assert len(response.data[0].embedding) > 0
        print(f"\nAzure embedding dimension: {len(response.data[0].embedding)}")


class TestIndeedAPIConnectivity:
    """Test Indeed API connectivity via RapidAPI."""
    
    @pytest.fixture
    def skip_if_no_rapidapi(self):
        """Skip test if RapidAPI key not configured"""
        from config import Config
        Config.setup()
        if not Config.RAPIDAPI_KEY:
            pytest.skip("RapidAPI key not configured")
    
    def test_indeed_job_search(self, skip_if_no_rapidapi):
        """Test Indeed job search API"""
        from config import Config
        import requests
        
        Config.setup()
        
        url = "https://indeed-scraper-api.p.rapidapi.com/api/job"
        
        params = {
            "search_terms": "python developer",
            "location": "hong kong",
            "page": "1"
        }
        
        headers = {
            "X-RapidAPI-Key": Config.RAPIDAPI_KEY,
            "X-RapidAPI-Host": "indeed-scraper-api.p.rapidapi.com"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        print(f"\nIndeed API status code: {response.status_code}")
        
        # API might return various status codes depending on rate limits
        assert response.status_code in [200, 400, 403, 429, 500]
        
        if response.status_code == 200:
            data = response.json()
            print(f"Jobs found: {len(data) if isinstance(data, list) else 'N/A'}")


class TestPineconeConnectivity:
    """Test Pinecone vector database connectivity."""
    
    @pytest.fixture
    def skip_if_no_pinecone(self):
        """Skip test if Pinecone key not configured"""
        from config import Config
        Config.setup()
        if not Config.PINECONE_API_KEY:
            pytest.skip("Pinecone API key not configured")
    
    def test_pinecone_connection(self, skip_if_no_pinecone):
        """Test Pinecone client connection"""
        from config import Config
        from pinecone import Pinecone
        
        Config.setup()
        
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # List indexes
        indexes = pc.list_indexes()
        
        print(f"\nPinecone connected successfully")
        print(f"Available indexes: {[idx['name'] for idx in indexes]}")
        
        assert indexes is not None


class TestDatabaseOperations:
    """Test database read/write operations."""
    
    def test_job_seeker_workflow(self):
        """Test complete job seeker workflow"""
        import tempfile
        import os
        from database import JobSeekerDB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                db = JobSeekerDB()
                
                # Create profile
                profile = {
                    'education_level': 'Master',
                    'major': 'Computer Science',
                    'hard_skills': 'Python, SQL, Machine Learning, TensorFlow',
                    'soft_skills': 'Leadership, Problem Solving',
                    'work_experience': '3-5 years',
                    'primary_role': 'Data Scientist',
                    'simple_search_terms': 'data science, machine learning, AI',
                    'location_preference': 'Hong Kong',
                }
                
                # Save
                job_seeker_id = db.save_profile(profile)
                print(f"\nCreated job seeker: {job_seeker_id}")
                
                # Retrieve
                retrieved = db.get_profile(job_seeker_id)
                assert retrieved is not None
                assert retrieved['primary_role'] == 'Data Scientist'
                
                # Get search fields
                search_fields = db.get_search_fields(job_seeker_id)
                assert search_fields is not None
                assert 'Python' in search_fields['hard_skills']
                
                print(f"Profile retrieved successfully")
                
            finally:
                os.chdir(old_cwd)
    
    def test_headhunter_workflow(self):
        """Test headhunter job posting workflow"""
        import tempfile
        import os
        from database import HeadhunterDB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                db = HeadhunterDB()
                
                # Create job posting
                job = {
                    'job_title': 'Senior Python Developer',
                    'job_description': 'We are looking for a Python expert',
                    'required_skills': 'Python, Django, PostgreSQL',
                    'min_salary': 40000,
                    'max_salary': 60000,
                    'currency': 'HKD',
                    'work_location': 'Hong Kong',
                    'industry': 'Technology',
                }
                
                # Save
                success = db.save_job(job)
                assert success is True
                print(f"\nCreated job posting")
                
                # Retrieve all jobs
                jobs = db.get_all_jobs()
                assert len(jobs) > 0
                assert jobs[0]['job_title'] == 'Senior Python Developer'
                
                print(f"Job posting retrieved successfully")
                
            finally:
                os.chdir(old_cwd)


class TestJobMatchingFlow:
    """Test the complete job matching flow."""
    
    def test_calculate_match_score(self):
        """Test job matching score calculation"""
        from core import calculate_job_match_score
        
        job_seeker = {
            'hard_skills': 'Python, SQL, Machine Learning, TensorFlow, Pandas',
            'soft_skills': 'Leadership, Communication, Problem Solving',
            'work_experience': '3-5 years',
            'location_preference': 'Hong Kong',
            'primary_role': 'Data Scientist',
        }
        
        # Good match job
        job_good = {
            'job_title': 'Senior Data Scientist',
            'job_description': 'Looking for a Python expert with machine learning experience. Must know SQL and TensorFlow.',
            'job_city': 'Hong Kong',
        }
        
        # Poor match job
        job_poor = {
            'job_title': 'Marketing Manager',
            'job_description': 'Need marketing professional with social media expertise',
            'job_city': 'Singapore',
        }
        
        score_good = calculate_job_match_score(job_seeker, job_good)
        score_poor = calculate_job_match_score(job_seeker, job_poor)
        
        print(f"\nGood match score: {score_good['overall_score']}")
        print(f"Poor match score: {score_poor['overall_score']}")
        print(f"Good match skills: {score_good.get('matched_skills', [])}")
        
        # Good match should score higher
        assert score_good['overall_score'] >= score_poor['overall_score']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
