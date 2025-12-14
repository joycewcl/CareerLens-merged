# 🔍 CareerLens - AI Career Intelligence Platform

An AI-powered career intelligence platform built with Streamlit, combining job matching, resume tailoring, and AI interview preparation.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## 🚀 Features

### Core Pages
- **🏠 Job Seeker**: Upload CV, GPT-4 analysis, auto-fill profile forms
- **💼 Job Match**: AI-matched positions with Pinecone vector search
- **🤖 AI Interview**: Mock interviews and skill assessment
- **📊 Market Dashboard**: CareerLens modular dashboard view
- **🎯 Recruiter**: Post jobs and manage recruitment positions
- **🔍 Recruitment Match**: Match candidates to job openings

### CareerLens Features
- **Market Positioning Dashboard**: See your match score, estimated salary, and skill gaps
- **Resume Tailoring**: Generate ATS-optimized resumes tailored to specific jobs
- **Multi-format Export**: Download resumes as DOCX, PDF, or TXT
- **Industry Filtering**: Filter jobs by 15+ industries/domains
- **Salary Filtering**: Filter by minimum salary expectation
- **Indeed Job Source**: Alternative to LinkedIn via IndeedScraperAPI
- **Token Tracking**: Monitor API usage and costs
- **Enhanced Profile Extraction**: Two-pass verification for accuracy
- **Advanced Visualizations**: Match score breakdowns, skill matrices, radar charts

## 📁 Project Structure

```
├── streamlit_app.py      # Main entry point - unified multi-page application
├── backend.py            # Backend services (ResumeParser, GPT4JobRoleDetector, JobMatcher, etc.)
├── config.py             # Configuration settings
├── database.py           # Database operations (JobSeekerDB, HeadhunterDB)
├── modules/              # Modular components for Market Dashboard
│   ├── analysis/         # Match analysis
│   ├── resume_generator/ # Resume generation & formatting
│   ├── resume_upload/    # File extraction & profile parsing
│   ├── semantic_search/  # Embeddings, cache, job search
│   ├── ui/               # Dashboard UI components
│   └── utils/            # API clients, config, helpers, validation
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python version for deployment
└── *.db                  # SQLite databases (generated)
```

## 🔧 How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install all necessary dependencies including:
   - `streamlit` - Web framework
   - `pandas`, `numpy` - Data processing
   - `matplotlib`, `plotly` - Visualizations
   - `openai` - AI integration
   - And all other required packages

   **Verify installation:**
   ```bash
   python verify_setup.py
   ```
   
   This script checks that all dependencies are installed correctly.

2. **Configure secrets** (create `.streamlit/secrets.toml`):
   ```toml
   AZURE_OPENAI_API_KEY = "your-key"
   AZURE_OPENAI_ENDPOINT = "your-endpoint"
   PINECONE_API_KEY = "your-key"
   RAPIDAPI_KEY = "your-key"
   ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

### Troubleshooting

**Import Errors (ModuleNotFoundError):**
If you encounter import errors for `streamlit`, `pandas`, `matplotlib`, `plotly`, or `numpy`, ensure you've run:
```bash
pip install -r requirements.txt
```

**Python Version:**
Verify you're using Python 3.8+:
```bash
python --version
```

**API Configuration Errors:**
If you encounter 404, 401, or 403 errors when using the application:

1. **404 Not Found:** This usually means the API endpoint is incorrect or unavailable
   - For Azure OpenAI: Check your `AZURE_OPENAI_ENDPOINT` in `.streamlit/secrets.toml`
   - Ensure the endpoint URL is in the format: `https://YOUR-RESOURCE.openai.azure.com`
   - Verify the deployment names match your Azure OpenAI deployments

2. **401 Unauthorized:** Your API key is invalid or expired
   - Update `AZURE_OPENAI_API_KEY` in `.streamlit/secrets.toml`
   - Update `RAPIDAPI_KEY` for job search features
   - Check that your API keys are active and have not expired

3. **403 Forbidden:** Your API key doesn't have access to the requested resource
   - Verify your Azure OpenAI subscription has access to the required models
   - Check your RapidAPI subscription includes LinkedIn and Indeed job search APIs

4. **429 Rate Limit:** You've exceeded the API rate limits
   - Wait a few minutes before trying again
   - Consider upgrading your API subscription for higher limits

**Missing Secrets File:**
Create `.streamlit/secrets.toml` with your API credentials:
```toml
AZURE_OPENAI_API_KEY = "your-azure-openai-key"
AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE.openai.azure.com"
PINECONE_API_KEY = "your-pinecone-key"
RAPIDAPI_KEY = "your-rapidapi-key"
```

**Job Search Not Working:**
- Verify your `RAPIDAPI_KEY` is configured and has an active subscription
- Check the RapidAPI dashboard for API usage and limits
- Try the alternative job source (LinkedIn vs Indeed) in the app settings

## 📦 Key Dependencies

- `streamlit` - Web framework
- `pinecone-client` - Vector database for semantic search
- `sentence-transformers` - Embedding model
- `openai` - Azure OpenAI API
- `tiktoken` - Token counting
- `matplotlib`, `plotly` - Visualizations
- `reportlab` - PDF generation
- `python-docx` - DOCX generation
- `PyPDF2` - PDF parsing

## 🔍 Setup Verification

After installing dependencies, run the verification script to ensure everything is configured correctly:

```bash
python verify_setup.py
```

This script will:
- ✅ Check Python version compatibility
- ✅ Verify all required modules are installed
- ✅ Confirm application files exist
- ✅ Test for syntax errors in the main application

If all checks pass, you're ready to run the application!

## ⚡ Streamlit Cloud Optimizations

The application includes several optimizations for stable deployment:
- Environment variables for reduced logging and telemetry
- Increased recursion limit (3000) for complex operations
- SQLite temp directory configuration
- WebSocket stability mechanisms
- Memory cleanup utilities

## 📝 License

MIT License
