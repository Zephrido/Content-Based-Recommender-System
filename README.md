# Movie Recommender System

 movie recommendation web app built with **Streamlit**.  
Supports search by title, keyword, cast, director, or description, with genre filtering, and returns recommendations using a combination of **TF-IDF**, **SBERT semantic embeddings**, and **fuzzy matching**.

---

## Features

- Intelligent Recommendations: Combines TF-IDF, fuzzy title matching, and SBERT semantic search for accurate movie suggestions.
- Flexible Search: Query by movie title, keywords, genres, cast, director, or description.
- Genre Filtering: Filter recommendations by one or more genres.
- Movie Detail*: Displays movie posters, IMDb ratings, and direct links to IMDb pages.
- Modern UI: Dark-themed, responsive interface with custom CSS styling.
- Efficient: Caches data and embeddings for fast performance.

---

## 🗂 Requirements

- Python 3.8+
- Streamlit
- TensorFlow
- sentence-transformers
- scikit-learn
- pandas
- numpy
- requests
- fuzzywuzzy
- python-Levenshtein (recommended for fuzzywuzzy speed)

You also need the following datasets (place in your project directory):
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

You need API keys for:
- [TMDB API](https://developers.themoviedb.org/3/getting-started/introduction)
- [OMDb API](https://www.omdbapi.com/apikey.aspx)




