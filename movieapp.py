import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page Config ----------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    color: white;
    box-shadow: 0 0 10px rgba(255,0,0,0.15);
}
.title {
    font-size: 46px;
    font-weight: bold;
    color: #ff4b4b;
}
.subtitle {
    font-size: 20px;
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
df = pd.read_csv("movies.csv")

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movie(movie_name, top_n=5):
    index = df[df['title'] == movie_name].index[0]
    scores = list(enumerate(cosine_sim[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n+1]
    movie_indexes = [i[0] for i in scores]
    return df['title'].iloc[movie_indexes]

# ---------- UI ----------
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart, content-based movie suggestions</div>', unsafe_allow_html=True)

st.write("")

st.sidebar.header("🎯 Choose a Movie")
selected_movie = st.sidebar.selectbox(
    "Select a movie you like",
    df['title'].values
)

if st.sidebar.button("🔍 Recommend"):
    st.success(f"Movies similar to **{selected_movie}**")

    recommendations = recommend_movie(selected_movie)

    for movie in recommendations:
        st.markdown(f"""
        <div class="card">
            <h3>🎥 {movie}</h3>
            <p>Based on content similarity</p>
        </div>
        """, unsafe_allow_html=True)
