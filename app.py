import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Convert text to numbers using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movie(movie_name, top_n=5):
    if movie_name not in df['title'].values:
        return ["Movie not found in dataset"]

    index = df[df['title'] == movie_name].index[0]
    scores = list(enumerate(cosine_sim[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n+1]

    movie_indexes = [i[0] for i in scores]
    return df['title'].iloc[movie_indexes].tolist()

# Test
if __name__ == "__main__":
    movie = "Inception"
    print(f"Recommended movies for {movie}:")
    for m in recommend_movie(movie):
        print("-", m)
