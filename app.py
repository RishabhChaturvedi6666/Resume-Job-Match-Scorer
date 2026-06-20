import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag 

# 1. Download NLTK resources efficiently using Streamlit cache so they only download once
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    except Exception:
        pass

download_nltk_resources()

# 2. Page Configuration setup
st.set_page_config(page_title="Resume Job Match Scorer", page_icon="📄", layout="wide")

st.title("📄 Resume Job Match Scorer")
st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match!  
This tool uses **TF-IDF + Cosine Similarity** and **NLP Keyword Extraction** to analyze your resume against job requirements.
""")

with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you:
    - Measure how your resume matches a job description.
    - Identify important job keywords.
    - Improve your resume based on missing terms.
    """)
    
    st.header("How It Works")
    st.write("""
    1. Upload your resume (PDF).
    2. Paste the job description.
    3. Click **Analyze Match**.
    4. Review score & suggestions.
    """)

# 3. Helper Processing Functions
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
    except Exception:
        # Fallback simple whitespace split if NLTK data encounters an issue
        words = text.split()
        stop_words = set()
    return " ".join([word for word in words if word not in stop_words])

def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))  
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed

def extract_keywords(text, num_keywords=10):
    try:
        words = word_tokenize(text)
    except Exception:
        words = text.split()
        
    words = [w for w in words if len(w) > 2]
    try:
        tagged_words = pos_tag(words)
        # Filters for nouns (NN, NNS, NNP, NNPS) and adjectives (JJ, JJR, JJS)
        nouns = [w for w, pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
        word_freq = Counter(nouns)
        return [word for word, freq in word_freq.most_common(num_keywords)]
    except Exception:
        # Graceful fallback if POS tagger is unavailable
        word_freq = Counter(words)
        return [word for word, freq in word_freq.most_common(num_keywords)]

# 4. Main App logic
def main():
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
    job_description = st.text_area("Paste the job description", height=200)

    if st.button("Analyze Match"):
        if not uploaded_file:
            st.warning("Please upload your resume")
            return
        if not job_description:
            st.warning("Please paste the job description")
            return
        
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Could not extract text from PDF. Please try another file.")
                return 
            
            # Perform NLP operations
            similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)
            job_keywords = extract_keywords(job_processed, 12)
            resume_keywords = extract_keywords(resume_processed, 12)
            
            # Find what keywords are missing
            missing_keywords = [kw for kw in job_keywords if kw not in resume_keywords]

            # Display Results UI
            st.write("---")
            st.subheader("Analysis Results")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Match Score", f"{similarity_score:.2f}%")
            
            with col2:
                # Built-in Matplotlib chart horizontal progress bar
                fig, ax = plt.subplots(figsize=(6, 0.6))
                colors = ['#ff4b4b', '#ffa726', '#0f9d58']
                color_index = min(int(similarity_score // 33), 2)
                ax.barh([0], [similarity_score], color=colors[color_index])
                ax.set_xlim(0, 100)
                ax.set_xlabel("Match Percentage")
                ax.set_yticks([])
                ax.set_title("Visual Match Progress")
                st.pyplot(fig)

            # Conditional formatting response
            if similarity_score < 40:
                st.warning("⚠️ **Low Match:** Consider tailoring your resume more closely to include core requirements.")
            elif similarity_score < 70:
                st.info("💡 **Good Match:** Your resume aligns fairly well, but could use more specific keywords.")
            else:
                st.success("🔥 **Excellent Match!** Your resume strongly aligns with this job description.")

            # Analytics block for Keyword breakdown
            st.write("---")
            st.subheader("🔍 Key Term Insights")
            col3, col4 = st.columns(2)
            
            with col3:
                st.write("**Top Keywords in Job Description:**")
                st.caption(", ".join(job_keywords) if job_keywords else "None detected")
                
            with col4:
                st.write("**Missing Crucial Keywords:**")
                if missing_keywords:
                    st.error(", ".join(missing_keywords[:6]))
                else:
                    st.success("Your resume captures the essential context words!")

if __name__ == "__main__":
    main()                         


        