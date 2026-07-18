import streamlit as st
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# Predefined list of known technical/professional skills.
# Only words from THIS list will ever be detected as "skills" —
# generic English words are never matched, no matter what.
SKILLS_DB = {
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "r",
    "html", "css", "php", "go", "rust", "kotlin", "swift",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data analysis", "data science", "data engineering",
    "scikit-learn", "tensorflow", "pytorch", "keras", "pandas", "numpy",
    "matplotlib", "seaborn", "opencv",
    "aws", "azure", "gcp", "cloud computing", "docker", "kubernetes",
    "git", "github", "ci/cd", "linux",
    "agile", "scrum", "project management",
    "rest api", "api development", "flask", "django", "fastapi",
    "streamlit", "react", "node.js", "angular", "vue",
    "mysql", "postgresql", "mongodb", "database management",
    "excel", "power bi", "tableau",
    "communication", "problem-solving", "teamwork", "leadership",
    "model deployment", "mlops", "data preprocessing", "etl",
}


def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_skills(text):
    """Detect only known skills (from SKILLS_DB) present in the text."""
    text_lower = text.lower()
    found_skills = set()
    for skill in SKILLS_DB:
        # Match whole skill phrases (handles multi-word skills too)
        if skill in text_lower:
            found_skills.add(skill)
    return found_skills


def ats_checker(resume_text, job_description):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    if not job_skills:
        match_score = 0.0
    else:
        matched_skills = resume_skills & job_skills
        match_score = (len(matched_skills) / len(job_skills)) * 100

    missing_skills = job_skills - resume_skills

    return {
        "match_score": round(match_score, 2),
        "matched_skills": resume_skills & job_skills,
        "missing_keywords": missing_skills
    }


# ---------- Streamlit UI ----------
st.set_page_config(page_title="ATS Resume Checker", page_icon="📄")

st.title("📄 ATS Resume Checker")
st.write("Upload your resume and paste a job description to see how well they match, "
         "and which important keywords might be missing from your resume.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the job description here:", height=200)

if st.button("Check Match"):
    if uploaded_file is None:
        st.warning("Please upload your resume PDF first!")
    elif job_description.strip() == "":
        st.warning("Please paste a job description!")
    else:
        resume_text = extract_text_from_pdf(uploaded_file)
        result = ats_checker(resume_text, job_description)

        st.subheader(f"Match Score: {result['match_score']}%")

        if result['match_score'] >= 70:
            st.success("Great match! Your resume aligns well with this job description.")
        elif result['match_score'] >= 40:
            st.warning("Moderate match. Consider adding more relevant keywords.")
        else:
            st.error("Low match. Your resume may need significant keyword adjustments.")

        st.write("### Matched Skills")
        if result['matched_skills']:
            st.write(", ".join(sorted(result['matched_skills'])))
        else:
            st.write("No matching skills found.")

        st.write("### Missing Skills")
        if result['missing_keywords']:
            st.write(", ".join(sorted(result['missing_keywords'])))
        else:
            st.write("No major skills missing — nice work!")

st.markdown("---")
st.caption("Built with scikit-learn (TF-IDF + Cosine Similarity) + NLTK + Streamlit | No external API used")
