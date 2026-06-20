# 📄 Resume & Job Description Match Scorer

An intelligent, NLP-driven web application designed to help job seekers analyze how well their resumes align with specific job descriptions. Built using **Python** and **Streamlit**, this tool automates the process of skill-matching, text vectorization, and keyword optimization to help candidates bypass Applicant Tracking Systems (ATS).

---

## 🚀 Features
- **PDF Text Extraction:** Parses uploaded resume files smoothly and securely using `PyPDF2`.
- **Text Preprocessing Pipeline:** Automated text cleaning, lowercasing, regex filtering, tokenization, and stop-word removal powered by `NLTK`.
- **Algorithmic Match Scoring:** Leverages **TF-IDF Vectorization** and **Cosine Similarity** via `scikit-learn` to calculate a precise contextual match percentage.
- **NLP Keyword Analytics:** Employs Part-of-Speech (POS) tagging to isolate key technical competencies, nouns, and adjectives, dynamically identifying critical missing skills.
- **Dynamic Visual Dashboard:** Features an intuitive user interface complete with real-time feedback, status indicators, and a horizontal progress gauge chart powered by `Matplotlib`.

---

## 🛠️ Tech Stack
- **Frontend / UI Framework:** Streamlit
- **Natural Language Processing:** NLTK (Natural Language Toolkit)
- **Machine Learning & Analytics:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)
- **PDF Parsing:** PyPDF2
- **Data Visualization:** Matplotlib
- **Core Processing:** Python, Re (Regex), Collections (`Counter`)

---

## 📦 Installation & Local Setup

Follow these simple steps to run the application locally on your machine:

### 1. Clone or Download the Project
Ensure you have all the project files (`app.py` and `requirements.txt`) inside your local project directory.

### 2. Install Dependencies
Open your terminal inside the project directory and run the following command to install all necessary Python packages:
```bash
pip install -r requirements.txt
