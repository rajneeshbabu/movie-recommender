# Movie Recommender System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange)](https://scikit-learn.org)
[![TMDB](https://img.shields.io/badge/TMDB-API-brightgreen)](https://themoviedb.org)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-blueviolet)](https://rajneeshbabu.github.io/movie-recommender)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A content-based movie recommendation system. Pick any movie from 4,796 titles and instantly get 5 similar recommendations with posters, ratings, genres, and match scores.

**Live Demo**: [rajneeshbabu.github.io/movie-recommender](https://rajneeshbabu.github.io/movie-recommender)

---

## How It Works

```
User selects a movie
        |
        v
Pre-computed TF-IDF similarity scores (Dataset/data.zip)
        |
        v
Top 5 most similar movies ranked by cosine similarity + rating boost
        |
        v
Movie posters fetched live from TMDB API
        |
        v
Results shown instantly in browser — no backend needed
```

### Why content-based filtering?
The model builds a rich text profile for each movie from its overview, genres, keywords, top 5 cast members, and director. It uses **TF-IDF vectorization** and **cosine similarity** to find movies with the most similar profiles.

**Feature weighting:**
- Director → weighted 3× (strongest signal)
- Cast → weighted 2×
- Genres, keywords, overview → 1×

---

## Project Structure

```
movie-recommender/
├── index.html                    # static webpage — works on GitHub Pages
├── app.py                        # Streamlit local app
├── requirements.txt
├── movie_recommender.ipynb       # full training notebook
├── tmdb_5000_movies.csv          # TMDB dataset — movies
├── tmdb_5000_credits.csv         # TMDB dataset — cast & crew
├── README.md
│
├── Dataset/
│   └── data.zip                  # pre-computed top-5 recs for 4,796 movies (2.8 MB)
│
└── models/
    ├── models.zip                # movie_dict.pkl + movie_list.pkl + plots (1.7 MB)
       ├── movie_dict.pkl            # enriched movie dataframe
       ├── movie_list.pkl            # list of all movie titles
       ├── eda_overview.png          # EDA plots
       ├── genre_distribution.png
       └── similarity.pkl            # cosine similarity matrix — 177 MB, not in git
```

> **Note**: `models/similarity.pkl` (177 MB) exceeds GitHub's file limit and is excluded from git.
> Run `movie_recommender.ipynb` locally to regenerate it.

---

## Quick Start

### Option 1 — Open the live webpage (no setup needed)
Visit: **[rajneeshbabu.github.io/movie-recommender](https://rajneeshbabu.github.io/movie-recommender)**

### Option 2 — Run locally with Streamlit

```bash
# 1. Clone the repo
git clone https://github.com/rajneeshbabu/movie-recommender.git
cd movie-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook to generate models/similarity.pkl
jupyter notebook movie_recommender.ipynb
# Run all cells — saves pkl files to models/

# 4. Run the Streamlit app
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Dataset

- **Source**: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) on Kaggle
- **Movies**: 4,796 after cleaning
- **Features used**: overview, genres, keywords, cast (top 5), director
- **Vectorizer**: TF-IDF (10,000 features, unigrams + bigrams)
- **Similarity**: Cosine similarity with rating boost

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | TF-IDF + Cosine Similarity (scikit-learn) |
| NLP | NLTK Porter Stemmer |
| Data | TMDB 5000 dataset (Kaggle) |
| Posters | TMDB API |
| Web page | Pure HTML / CSS / JavaScript + JSZip |
| Local app | Streamlit |
| Hosting | GitHub Pages (static) |

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with scikit-learn · NLTK · TMDB API · GitHub Pages*
