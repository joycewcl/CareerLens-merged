"""File extraction functions for resume upload"""
import streamlit as st
import PyPDF2
from docx import Document


def extract_text_from_resume(uploaded_file):
    """Extract text from uploaded resume file (PDF, DOCX, or TXT)"""
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            uploaded_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            # Check if we got any text
            if not text or len(text.strip()) < 50:
                st.warning(
                    "⚠️ Could not extract text from PDF. This may happen if:\n"
                    "• The PDF is scanned/image-based\n"
                    "• The PDF is corrupted or password-protected\n\n"
                    "Please try uploading a DOCX file or a PDF with selectable text."
                )
                return None
            return text
        
        elif file_type == 'docx':
            uploaded_file.seek(0)
            doc = Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Check if we got any text
            if not text or len(text.strip()) < 50:
                st.warning(
                    "⚠️ Could not extract sufficient text from DOCX. "
                    "The document may be mostly empty or use non-standard formatting."
                )
                return None
            return text
        
        elif file_type == 'txt':
            uploaded_file.seek(0)
            text = str(uploaded_file.read(), "utf-8")
            return text
        
        else:
            st.error(f"Unsupported file type: {file_type}. Please upload PDF, DOCX, or TXT.")
            return None
            
    except Exception as e:
        st.error(f"Error extracting text from resume: {e}")
        return None
