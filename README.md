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

## ⚡ Streamlit Cloud Optimizations

The application includes several optimizations for stable deployment:
- Environment variables for reduced logging and telemetry
- Increased recursion limit (3000) for complex operations
- SQLite temp directory configuration
- WebSocket stability mechanisms
- Memory cleanup utilities

## 📝 License

MIT License
