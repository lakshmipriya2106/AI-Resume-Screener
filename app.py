import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI ----------------
st.title("AI Resume Screening System")

resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd = st.text_area("Enter Job Description")

# ---------------- FUNCTIONS ----------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# ---------------- MAIN LOGIC ----------------
if st.button("Analyze"):
    if resume is not None and jd.strip() != "":
        
        # Extract text
        resume_text = extract_text(resume)

        # Clean text
        resume_text = clean_text(resume_text)
        jd_text = clean_text(jd)

        # Convert text to numbers (TF-IDF)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, jd_text])

        # Compare using cosine similarity
        score = cosine_similarity(vectors[0], vectors[1])

        # Show result
        match_percentage = score[0][0] * 100
        st.subheader(f"Match Score: {match_percentage:.2f}%")

        # Result message
        if match_percentage > 70:
            st.success("Good Match ✅")
        elif match_percentage > 40:
            st.warning("Average Match ⚠️")
        else:
            st.error("Poor Match ❌")

    else:
        st.warning("Please upload resume and enter job description")