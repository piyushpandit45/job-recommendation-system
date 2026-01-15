import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Job Recommendation System")
st.write("Recommend jobs based on **Skills, Education, Experience & Domain**")

# ------------------ Load Data ------------------
df = pd.read_csv("cleaned_data.csv")


# ------------------ NLP Functions ------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

for col in ["skills", "education", "domain"]:
    df[col] = df[col].apply(clean_text)

df["combined_text"] = df["skills"] + " " + df["education"] + " " + df["domain"]

# ------------------ TF-IDF ------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_text"])

# ------------------ UI Inputs ------------------
st.subheader("🔍 Enter Your Details")

user_skills = st.text_input("🛠 Skills (comma separated)", "python, machine learning")
user_education = st.selectbox(
    "🎓 Education",
    ["B.Tech", "BCA", "MCA", "MBA", "B.Sc", "M.Sc", "Any Graduate"]
)
user_experience = st.slider("📈 Experience (Years)", 0, 15, 1)
user_domain = st.selectbox(
    "💻 Domain",
    sorted(df["domain"].unique())
)

# ------------------ Recommendation Logic ------------------
def recommend_jobs(user_text, top_n=5):
    user_vector = tfidf.transform([user_text])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][["job_role", "domain"]]

# ------------------ Button ------------------
if st.button("🚀 Recommend Jobs"):
    user_text = clean_text(
        user_skills + " " +
        user_education + " " +
        user_domain
    )

    results = recommend_jobs(user_text)

    st.subheader("✅ Recommended Jobs")
    st.table(results.reset_index(drop=True))
