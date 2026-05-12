import streamlit as st
import pickle
import pandas as pd
import requests

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── TMDB API (free key — get yours at themoviedb.org) ─────────────────────
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"   # public demo key
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Main container */
.block-container {
    padding: 2rem 3rem;
    max-width: 1300px;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f093fb, #f5576c, #fda085);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    color: #aaa;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Select box */
.stSelectbox label { color: #ddd !important; font-size: 1rem; }
.stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 0.7rem 2rem;
    border-radius: 50px;
    border: none;
    box-shadow: 0 4px 20px rgba(240,147,251,0.4);
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(240,147,251,0.7);
    transform: translateY(-2px);
}

/* Movie card */
.movie-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.movie-card:hover {
    border-color: rgba(240,147,251,0.5);
    background: rgba(255,255,255,0.09);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.movie-rank {
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f093fb, #f5576c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.movie-title-card {
    font-size: 1.15rem;
    font-weight: 700;
    color: #fff;
    margin: 0.3rem 0;
}
.movie-meta {
    color: #bbb;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
}
.genre-tag {
    display: inline-block;
    background: rgba(240,147,251,0.2);
    color: #f093fb;
    border: 1px solid rgba(240,147,251,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px 3px 2px 0;
}
.rating-badge {
    background: rgba(253,160,133,0.2);
    color: #fda085;
    border: 1px solid rgba(253,160,133,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 600;
}
.match-badge {
    background: rgba(87,240,133,0.15);
    color: #57f085;
    border: 1px solid rgba(87,240,133,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 600;
}
.overview-text {
    color: #ccc;
    font-size: 0.85rem;
    line-height: 1.5;
    margin-top: 0.4rem;
}
.section-header {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 800;
    color: #fff;
    margin: 1.5rem 0 1rem 0;
}
hr-custom { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    movie_dict = pickle.load(open('models/movie_dict.pkl', 'rb'))
    movies     = pd.DataFrame(movie_dict)
    sim        = pickle.load(open('models/similarity.pkl', 'rb'))
    return movies, sim

movies, similarity = load_data()


# ── TMDB poster fetch ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    try:
        url  = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        path = data.get('poster_path')
        return f"{TMDB_IMG_BASE}{path}" if path else None
    except:
        return None


# ── Recommend function ─────────────────────────────────────────────────────
def recommend(movie_title, n=5):
    matches = movies[movies['title'] == movie_title]
    if matches.empty:
        return []

    movie_index = matches.index[0]
    distances   = similarity[movie_index].copy()

    # Small rating boost
    norm_rating = movies['vote_average'] / 10.0
    distances   = distances * (0.85 + 0.15 * norm_rating)

    ranked = sorted(
        [(i, score) for i, score in enumerate(distances) if i != movie_index],
        key=lambda x: x[1], reverse=True
    )

    results = []
    for idx, score in ranked:
        row = movies.iloc[idx]
        if row.get('vote_count', 0) < 50:
            continue
        results.append({
            'title'         : row['title'],
            'year'          : int(row.get('year', 0)) if row.get('year', 0) else 'N/A',
            'genres'        : row.get('genres_str', ''),
            'rating'        : round(float(row.get('vote_average', 0)), 1),
            'overview'      : row.get('overview_short', ''),
            'movie_id'      : int(row.get('movie_id', 0)),
            'similarity_pct': round(float(score) * 100, 1)
        })
        if len(results) == n:
            break
    return results


# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>Movie Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Pick a movie — get 5 hand-picked recommendations instantly</div>",
            unsafe_allow_html=True)

st.markdown("---")

col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    selected_movie = st.selectbox(
        "Select a movie you like:",
        sorted(movies['title'].dropna().unique()),
        index=None,
        placeholder="Search or select a movie..."
    )
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_btn = st.button("Get Recommendations")

# ── Results ────────────────────────────────────────────────────────────────
if recommend_btn and selected_movie:
    recs = recommend(selected_movie)

    if not recs:
        st.warning("Could not find recommendations. Try another movie.")
    else:
        st.markdown(f"<div class='section-header'>Because you liked <span style='color:#f093fb'>{selected_movie}</span></div>",
                    unsafe_allow_html=True)
        st.markdown("---")

        for i, movie in enumerate(recs, 1):
            poster_url = fetch_poster(movie['movie_id'])

            with st.container():
                st.markdown("<div class='movie-card'>", unsafe_allow_html=True)

                img_col, info_col = st.columns([1, 4])

                with img_col:
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='background:rgba(255,255,255,0.05);border-radius:8px;"
                            "height:160px;display:flex;align-items:center;justify-content:center;"
                            "color:#666;font-size:2rem;'>🎬</div>",
                            unsafe_allow_html=True
                        )

                with info_col:
                    genres_html = "".join(
                        f"<span class='genre-tag'>{g.strip()}</span>"
                        for g in movie['genres'].split(',') if g.strip()
                    )
                    st.markdown(f"""
                        <div class='movie-rank'>#{i}</div>
                        <div class='movie-title-card'>{movie['title']}</div>
                        <div class='movie-meta'>{movie['year']}</div>
                        <div style='margin-bottom:6px'>
                            {genres_html}
                            <span class='rating-badge'>⭐ {movie['rating']}</span>
                            <span class='match-badge'>Match {movie['similarity_pct']}%</span>
                        </div>
                        <div class='overview-text'>{movie['overview']}</div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("")

elif recommend_btn and not selected_movie:
    st.info("Please select a movie first.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555;font-size:0.8rem'>"
    "Built with TF-IDF · Cosine Similarity · TMDB API · Streamlit"
    "</p>",
    unsafe_allow_html=True
)
