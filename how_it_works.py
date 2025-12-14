"""
How It Works - Technical Deep Dive Page
Explains the AI-powered semantic job matching technology behind CareerLens
"""
import streamlit as st


def render_how_it_works_page():
    """Render the How It Works page with technical details"""
    
    st.header("🧠 How It Works: AI-Powered Semantic Job Matching")
    
    st.markdown("""
    This application uses cutting-edge AI and machine learning techniques to match your resume with the most relevant job opportunities. 
    Here's a deep dive into the technology behind the magic ✨
    """)
    
    st.divider()
    
    # Overview
    st.markdown("## 🎯 High-Level Overview")
    
    st.markdown("""
    <div class="how-it-works-section">
    <h3>Traditional vs. Semantic Search</h3>
    
    <strong>❌ Traditional Keyword Matching:</strong>
    <ul>
        <li>Looks for exact keyword matches</li>
        <li>Misses synonyms and related concepts</li>
        <li>"Python Developer" won't match "Python Engineer"</li>
        <li>Limited understanding of context</li>
    </ul>
    
    <strong>✅ Our Semantic AI Approach:</strong>
    <ul>
        <li>Understands the <em>meaning</em> behind text</li>
        <li>Recognizes synonyms and related concepts</li>
        <li>"Machine Learning Engineer" matches "ML Scientist", "AI Researcher"</li>
        <li>Context-aware: understands seniority, domain, and skills</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # The Pipeline
    st.markdown("## 🔄 The Complete AI Pipeline")
    
    # Step 1
    st.markdown("""
    <div class="step-box">
    <h3>Step 1: 🤖 GPT-4 Resume Analysis</h3>
    <p><strong>Technology:</strong> Azure OpenAI GPT-4 Turbo</p>
    
    <p><strong>What Happens:</strong></p>
    <ul>
        <li>Your resume text is sent to GPT-4 with specialized prompts</li>
        <li>GPT-4 extracts ALL skills: technical (Python, AWS), soft (Leadership, Communication), and domain-specific (Machine Learning, Finance)</li>
        <li>Identifies your primary job role and seniority level (Junior, Mid, Senior, Lead)</li>
        <li>Detects core strengths and target industries</li>
        <li>Suggests alternative roles you might be qualified for</li>
    </ul>
    
    <p><strong>Why GPT-4?</strong></p>
    <p>GPT-4 understands context and nuance that simple parsers miss. It can infer skills from descriptions like 
    "led team of 5 engineers" → extracts "Leadership", "Team Management", "Engineering Management"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 2
    st.markdown("""
    <div class="step-box">
    <h3>Step 2: 🔢 Vector Embeddings with Azure OpenAI</h3>
    <p><strong>Technology:</strong> Azure OpenAI Embeddings (text-embedding-ada-002)</p>
    
    <p><strong>What Happens:</strong></p>
    <ul>
        <li>Your resume is converted into a 1536-dimensional vector (a list of 1536 numbers)</li>
        <li>Each dimension captures different semantic aspects: skills, experience, domain, tone, etc.</li>
        <li>Similar resumes will have similar vectors (close together in 1536D space)</li>
        <li>Job descriptions are also converted into 1536D vectors</li>
    </ul>
    
    <p><strong>Vector Composition:</strong></p>
    <p>We create a weighted composite vector to emphasize important aspects:</p>
    <ul>
        <li><strong>Enhanced weighting</strong> on key skills and experience</li>
        <li><strong>Context preservation</strong> from full resume text</li>
        <li><strong>Skill emphasis</strong> on technical requirements</li>
    </ul>
    
    <p><strong>Example Vector (simplified to 3D):</strong></p>
    <code>
    Resume: [0.82, -0.45, 0.91]<br>
    Job A: [0.85, -0.42, 0.88]  ← Close! Similar meaning<br>
    Job B: [0.12, 0.78, -0.33]  ← Far! Different meaning
    </code>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 3
    st.markdown("""
    <div class="step-box">
    <h3>Step 3: 📊 Semantic Vector Search</h3>
    <p><strong>Technology:</strong> In-Memory Vector Database with Cosine Similarity</p>
    
    <p><strong>What Happens:</strong></p>
    <ul>
        <li>Your resume vector is compared against job vectors in our database</li>
        <li>Contains jobs from LinkedIn and Indeed via RapidAPI</li>
        <li>Performs ultra-fast cosine similarity search (milliseconds!)</li>
        <li>Returns top K most similar jobs with similarity scores</li>
    </ul>
    
    <p><strong>Cosine Similarity Explained:</strong></p>
    <p>Cosine similarity measures the angle between two vectors in high-dimensional space:</p>
    
    <ul>
        <li><strong>1.0</strong> = Perfect alignment (0° angle, identical meaning)</li>
        <li><strong>0.8-1.0</strong> = Very similar (small angle, strong match) 🎯</li>
        <li><strong>0.6-0.8</strong> = Related concepts (moderate angle, good match) 👍</li>
        <li><strong>0.4-0.6</strong> = Some overlap (larger angle, fair match) ⚠️</li>
        <li><strong>< 0.4</strong> = Different contexts (large angle, weak match) ❌</li>
    </ul>
    
    <p><strong>Visual Example:</strong></p>
    <pre>
    Resume Vector: →
    
    Job A: ↗ (angle ≈ 20°, cosine = 0.94) ✅ Excellent match
    Job B: → (angle ≈ 45°, cosine = 0.71) 👍 Good match  
    Job C: ↓ (angle ≈ 90°, cosine = 0.00) ❌ No match
    </pre>
    
    <p><strong>Why This Approach?</strong></p>
    <ul>
        <li>Fast in-memory processing for real-time results</li>
        <li>No external database dependencies</li>
        <li>Industry-standard cosine similarity for semantic matching</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 4
    st.markdown("""
    <div class="step-box">
    <h3>Step 4: 🎯 Multi-Criteria Scoring & Ranking</h3>
    <p><strong>Technology:</strong> Custom weighted scoring algorithm</p>
    
    <p><strong>What Happens:</strong></p>
    <p>Raw cosine similarity is great, but we combine it with other signals for better accuracy:</p>
    
    <h4>Combined Score Formula:</h4>
    <pre>
    Combined Score = 
        0.60 × Semantic Score +      (Cosine similarity from vector search)
        0.40 × Skill Match %         (Fuzzy skill matching)
    </pre>
    
    <h4>Component Breakdown:</h4>
    
    <p><strong>1. Semantic Score (60% weight):</strong></p>
    <ul>
        <li>Direct cosine similarity score from vector comparison</li>
        <li>Captures overall contextual alignment</li>
        <li>Most important signal for finding relevant jobs</li>
    </ul>
    
    <p><strong>2. Skill Match (40% weight):</strong></p>
    <ul>
        <li>Fuzzy string matching with skill synonyms</li>
        <li>Formula: (Matched Skills / Total Required Skills) × 100</li>
        <li>Examples of matches:
            <ul>
                <li>"Python" matches "Py", "Python3", "Python Programming"</li>
                <li>"JavaScript" matches "JS", "Node.js", "ECMAScript"</li>
                <li>"ML" matches "Machine Learning", "ML Engineering"</li>
            </ul>
        </li>
        <li>Uses fuzzy matching for typo tolerance</li>
    </ul>
    
    <h4>Match Categories:</h4>
    <ul>
        <li>🟢 <strong>Excellent (80-100%):</strong> Apply immediately! Strong fit across all criteria</li>
        <li>🟢 <strong>Very Good (65-79%):</strong> Great match, definitely worth applying</li>
        <li>🟡 <strong>Good (50-64%):</strong> Solid fit, consider applying</li>
        <li>🟠 <strong>Fair (35-49%):</strong> Stretch role, but possible if motivated</li>
        <li>🔴 <strong>Potential (0-34%):</strong> Growth opportunity, major skill gap</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Technology Stack
    st.markdown("## 🛠️ Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### AI & Machine Learning
        <div>
            <span class="tech-badge">🤖 Azure OpenAI GPT-4</span>
            <span class="tech-badge">🔢 Azure OpenAI Embeddings</span>
            <span class="tech-badge">📊 Vector Similarity Search</span>
            <span class="tech-badge">🔍 Cosine Similarity</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### Natural Language Processing
        <div>
            <span class="tech-badge">📝 PDF/DOCX Parsing</span>
            <span class="tech-badge">🔤 Fuzzy String Matching</span>
            <span class="tech-badge">📚 Skill Synonym Expansion</span>
            <span class="tech-badge">🧠 Context-Aware Extraction</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Data Sources
        <div>
            <span class="tech-badge">💼 LinkedIn Jobs API</span>
            <span class="tech-badge">🌐 RapidAPI</span>
            <span class="tech-badge">📊 Indeed API</span>
            <span class="tech-badge">📄 Resume Upload</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### Application Framework
        <div>
            <span class="tech-badge">🎨 Streamlit</span>
            <span class="tech-badge">🐍 Python 3.11</span>
            <span class="tech-badge">🔐 Azure Key Vault</span>
            <span class="tech-badge">☁️ Cloud-Native</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Real Example
    st.markdown("## 📚 Real-World Example")
    
    st.markdown("""
    <div class="how-it-works-section">
    <h3>How a Match is Found</h3>
    
    <p><strong>Your Resume Says:</strong></p>
    <blockquote>
    "Experienced Python developer with 5 years building ML pipelines. Skilled in TensorFlow, AWS, Docker, and data engineering. 
    Led team of 3 engineers on recommendation system."
    </blockquote>
    
    <p><strong>Step-by-Step Processing:</strong></p>
    
    <ol>
        <li><strong>GPT-4 Extracts:</strong>
            <ul>
                <li>Primary Role: "Machine Learning Engineer"</li>
                <li>Seniority: "Mid-Level" (5 years experience, team lead)</li>
                <li>Skills: Python, TensorFlow, AWS, Docker, Data Engineering, ML Pipelines, Recommendation Systems, Leadership</li>
                <li>Industries: Tech, E-commerce, Cloud Computing</li>
            </ul>
        </li>
        
        <li><strong>Azure OpenAI Creates Vector:</strong>
            <ul>
                <li>1536-dimensional vector capturing semantic meaning</li>
                <li>Optimized representation of your experience and skills</li>
            </ul>
        </li>
        
        <li><strong>Vector Search Finds:</strong>
            <ul>
                <li><strong>Job A:</strong> "Senior ML Engineer - Recommendation Systems" → Cosine: 0.91 ✅</li>
                <li><strong>Job B:</strong> "Python Backend Developer" → Cosine: 0.68 👍</li>
                <li><strong>Job C:</strong> "Data Analyst" → Cosine: 0.52 ⚠️</li>
                <li><strong>Job D:</strong> "Frontend React Developer" → Cosine: 0.23 ❌</li>
            </ul>
        </li>
        
        <li><strong>Multi-Criteria Scoring for Job A:</strong>
            <ul>
                <li>Semantic Score: 91% (cosine 0.91)</li>
                <li>Skill Match: 87.5% (7/8 skills matched: Python, TensorFlow, AWS, Docker, ML, Recommender Systems, Leadership)</li>
                <li><strong>Combined: 0.60×91 + 0.40×87.5 = 89.6% 🎯</strong></li>
            </ul>
        </li>
    </ol>
    
    <p><strong>Result:</strong> Job A appears as an "Excellent Match" (89.6%) at the top of your results!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Why It Works
    st.markdown("## 💡 Why This Approach Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Advantages
        
        - **Context-Aware**: Understands meaning, not just keywords
        - **Synonym Recognition**: "ML" = "Machine Learning" = "AI"
        - **Skill Transfer**: Recognizes transferable skills across domains
        - **Balanced Scoring**: Combines semantic + explicit skill matching
        - **Fast Processing**: Searches jobs in milliseconds
        - **Personalized**: Adapts to your unique background
        - **Up-to-Date**: Uses latest GPT-4 and embedding models
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Use Cases
        
        - **Career Transitions**: Find roles leveraging transferable skills
        - **Skill Gap Analysis**: See what skills you're missing
        - **Market Research**: Understand job market for your profile
        - **Confidence Boost**: Get data-driven match scores
        - **Time Savings**: No manual job filtering needed
        - **Hidden Gems**: Discover roles you might have missed
        - **Strategic Applications**: Focus on highest-match jobs
        """)
    
    st.divider()
    
    # FAQs
    st.markdown("## ❓ Frequently Asked Questions")
    
    with st.expander("Why is semantic similarity 60% of the score?"):
        st.markdown("""
        Semantic similarity is the primary indicator of job relevance because it captures the overall context 
        and meaning of your experience. We weight it at 60% because:
        
        1. It understands transferable skills and experience patterns
        2. It captures role alignment beyond specific keywords
        3. It recognizes industry and domain expertise
        
        The 40% skill match ensures specific technical requirements are met, providing a balanced approach.
        """)
    
    with st.expander("What does a cosine score of 0.75 actually mean?"):
        st.markdown("""
        A cosine score of 0.75 means:
        - The angle between vectors is ~41° (arccos(0.75))
        - In practical terms: **"Good contextual match"**
        - The job and resume discuss related concepts and domains
        - Not perfect alignment, but strong relevance
        
        **Interpretation Guide**:
        - **0.90-1.00**: Almost identical semantic meaning
        - **0.80-0.89**: Very similar context and domain
        - **0.70-0.79**: Related field with overlapping concepts
        - **0.60-0.69**: Some shared domain knowledge
        - **< 0.60**: Different contexts or domains
        """)
    
    with st.expander("How does fuzzy skill matching work?"):
        st.markdown("""
        Fuzzy matching accounts for:
        1. **Typos/Variations**: "Tensorflow" vs "TensorFlow"
        2. **Abbreviations**: "ML" = "Machine Learning", "JS" = "JavaScript"
        3. **Synonyms**: "Python" = "Py" = "Python3"
        4. **Related Skills**: "Node.js" contains "JavaScript"
        
        We use fuzzy string matching to measure similarity. If two skills are within 
        a threshold similarity, they're considered a match.
        
        **Example**: Your resume says "React.js" → Job requires "ReactJS" → Match! (90% similar)
        """)
    
    with st.expander("Can I trust the AI's skill extraction?"):
        st.markdown("""
        GPT-4 is highly accurate but not perfect. We've found ~95% accuracy in skill extraction through testing.
        
        **Tips for best results**:
        - Use clear, standard skill names in your resume
        - Include both technical and soft skills explicitly
        - Mention tools, frameworks, and technologies by name
        - Use industry-standard terminology
        
        **Review**: Always check the extracted profile to verify skills. You can 
        adjust your resume if important skills are missed.
        """)
    
    with st.expander("Why do I see jobs with low match scores?"):
        st.markdown("""
        We show a range of matches because:
        
        1. **Stretch Roles**: Sometimes a "Fair Match" (35-49%) could be a growth opportunity
        2. **Career Transitions**: Low scores might highlight what skills you need to develop
        3. **Hidden Opportunities**: You might be qualified for aspects the AI didn't fully capture
        4. **Market Awareness**: See what else is out there in your field
        
        **Best Practice**: Focus on "Good" (50%+) matches for active applications, but review lower 
        matches for career planning and skill development insights.
        """)
    
    with st.expander("How does CareerLens compare to traditional job search?"):
        st.markdown("""
        **Traditional Job Search**:
        - Manual keyword searches on job boards
        - Limited to exact keyword matches
        - Time-consuming to review hundreds of jobs
        - Miss relevant opportunities due to different terminology
        
        **CareerLens Approach**:
        - AI understands the meaning of your experience
        - Finds jobs even with different terminology
        - Ranked results save you time
        - Data-driven match scores give you confidence
        - Skill gap analysis helps you grow
        """)
    
    st.divider()
    
    # Limitations
    st.markdown("## ⚠️ Known Limitations")
    
    st.warning("""
    **This system is a powerful tool, but has limitations:**
    
    - **Resume Quality**: Poor/unclear resumes lead to poor extraction
    - **Job Description Quality**: Incomplete job posts reduce match accuracy  
    - **Salary/Benefits**: We don't always have complete compensation data
    - **Company Culture**: Can't assess culture fit or work environment
    - **Application Status**: Doesn't know if you already applied or were rejected
    - **Dynamic Market**: Job postings can be outdated or already filled
    - **Bias Potential**: AI models can inherit biases from training data
    - **API Limitations**: Job search APIs may have rate limits
    
    **Recommendation**: Use this as a *starting point* for your job search, not the only filter. 
    Always review jobs yourself and apply human judgment!
    """)
    
    st.divider()
    
    # Call to Action
    st.markdown("## 🚀 Ready to Find Your Perfect Match?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 Go to Market Dashboard", use_container_width=True, type="primary"):
            st.session_state.current_page = "market_dashboard"
            st.rerun()
        
        if st.button("🏠 Back to Job Seeker", use_container_width=True):
            st.session_state.current_page = "main"
            st.rerun()


if __name__ == "__main__":
    render_how_it_works_page()
