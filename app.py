import os
os.environ["USE_TF"] = "0"  # Force sentence-transformers to use PyTorch only
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sentence_transformers import SentenceTransformer, util

import numpy as np
import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process


# ─── Page config & CSS ─────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("""
<style>
  body, .stApp {
      background-color: #1F1F1F !important;
      color: #FFFFFF !important;
  }
  .stTextInput > div > input {
      background-color: #262730;
      color: #FFFFFF;
      border-radius: 5px;
      padding: 8px; width: 100%;
      border: 1px solid #444444;
  }
  label, .stTextInput label, .st-b6, .st-cg, .st-dw, .st-cy {
      color: #FFFFFF !important;
      font-weight: 500 !important;
  }
  div.stButton > button {
      background-color: transparent;
      color: #FF007F;
      border: 1px solid #FF007F;
      border-radius: 5px;
      padding: 10px; margin: 5px; width: 100%; transition: 0.3s ease;
  }
  div.stButton > button:hover {
      background-color: #FF007F; color: white; cursor: pointer;
  }
  .movie-title {
      color: #FFFFFF; text-align: center; margin-top: 5px; font-size: 16px;
  }
  .movie-title a {
      color: #FFFFFF !important; text-decoration: none !important;
  }
  .movie-title a:hover {
      color: #FF007F !important; text-decoration: underline !important;
  }
  .movie-meta {
      display: flex; justify-content: center; align-items: center;
      margin-bottom: 8px;
  }
  .movie-meta .star {
      font-size: 18px; color: #FFD700; margin-right: 4px;
  }
  .movie-meta .rating {
      font-size: 14px; color: #FFFFFF; font-weight: bold;
  }

  /* INFO/WARNING/ERROR BOXES */
  .stAlert, .stAlert>div {
      background-color: #23262F !important;
      color: #FFD700 !important;
      font-weight: 500;
      border-radius: 7px;
      border: 1px solid #FF007F;
  }
  .stAlert[data-testid="stAlertInfo"], .stAlert[data-testid="stAlertInfo"]>div {
      background-color: #23262F !important;
      color: #63b3ed !important;
      border: 1px solid #63b3ed;
  }
  .stAlert[data-testid="stAlertWarning"], .stAlert[data-testid="stAlertWarning"]>div {
      background-color: #332d1a !important;
      color: #FFD700 !important;
      border: 1px solid #FFD700;
  }
  .stAlert[data-testid="stAlertError"], .stAlert[data-testid="stAlertError"]>div {
      background-color: #330d1a !important;
      color: #ff4b4b !important;
      border: 1px solid #ff4b4b;
  }
</style>
""", unsafe_allow_html=True)


# ─── API KEYS ───────────────────────────────────────────────────────────────────
TMDB_API_KEY    = "8265bd1679663a7ea12ac168da84d2e8"
OMDB_API_KEY    = "6c765ab5"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ─── Fetch helpers ───────────────────────────────────────────────────────────────
def fetch_poster_and_imdb(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    resp = requests.get(url)
    if resp.status_code == 200:
        d = resp.json()
        poster = d.get("poster_path")
        imdb   = d.get("imdb_id")
        return (TMDB_IMAGE_BASE + poster if poster else "", imdb)
    return ("", None)

def fetch_imdb_rating(imdb_id):
    if not imdb_id:
        return None
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
    try:
        data = requests.get(url).json()
        return data.get("imdbRating")
    except:
        return None

# ─── Data prep helpers ──────────────────────────────────────────────────────────
def parse_names(txt, top_n=None):
    try:
        arr = ast.literal_eval(txt)
        names = [i["name"] for i in arr]
        return names[:top_n] if top_n else names
    except:
        return []

def get_director(txt):
    try:
        arr = ast.literal_eval(txt)
        return [i["name"] for i in arr if i.get("job") == "Director"]
    except:
        return []

def join_names(lst):
    return [n.replace(" ", "_") for n in lst]

# ─── Load & vectorize ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    m = pd.read_csv("tmdb_5000_movies.csv")
    c = pd.read_csv("tmdb_5000_credits.csv")
    c.rename(columns={"movie_id": "id"}, inplace=True)
    df = m.merge(c, on="id")

    df["keywords"] = df["keywords"].apply(parse_names).apply(join_names)
    df["genres"]   = df["genres"].apply(parse_names).apply(join_names)
    df["cast"]     = df["cast"].apply(lambda x: join_names(parse_names(x, 3)))
    df["crew"]     = df["crew"].apply(lambda x: join_names(get_director(x)))

    df["overview"] = df["overview"].fillna("")
    df["tags"]     = df["keywords"] + df["genres"] + df["cast"] + df["crew"] + \
                     df["overview"].apply(lambda x: x.split())
    df["tags"]     = df["tags"].apply(lambda x: " ".join(x).lower())

    # ── TF-IDF ─────────────────────────────
    cv   = TfidfVectorizer(max_features=5000, stop_words="english")
    vecs = cv.fit_transform(df["tags"])

    # ── SBERT Embeddings ─────────────────────────────
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    emb   = sbert.encode(df["tags"].tolist(), convert_to_tensor=True, show_progress_bar=False)

    return df, cv, vecs, emb

df, cv, vectors, sbert_embs = load_data()

# ─── Genre list for UI ───────────────────────────────
genre_list = sorted({g.replace("_", " ") for genre_arr in df["genres"] for g in genre_arr})

# ─── Recommendation logic with genre filtering ────────
def recommend_movies(q, selected_genres):
    query = q.lower().strip()
    q_us  = query.replace(" ", "_")

    if selected_genres:
        underscored_genres = [g.replace(" ", "_") for g in selected_genres]
        genre_mask = df["genres"].apply(lambda gs: any(g in gs for g in underscored_genres))
        genre_indices = np.where(genre_mask.values)[0]
        filtered_df = df.iloc[genre_indices]
        filtered_vectors = vectors[genre_indices]
        filtered_embs = sbert_embs[genre_indices]
    else:
        filtered_df = df
        filtered_vectors = vectors
        filtered_embs = sbert_embs

    if filtered_df.empty:
        st.warning("No movies found with the selected genres!")
        return []

    best, score = process.extractOne(query, filtered_df["title_x"].str.lower().values)
    if score == 100:
        st.success(f"🎬 Found Exact Title: **{best}**")
        idx = filtered_df[filtered_df["title_x"].str.lower() == best].index[0]
        vec = filtered_vectors[filtered_df.index.get_loc(idx)]
        sim = cosine_similarity(vec, filtered_vectors).flatten()
        ids = np.argsort(sim)[::-1][:8]
        res = filtered_df.iloc[ids]
    else:
        st.info(f"🔍 Not exact title. Searching keywords, genres, cast & directors for **{query}**")
        k = filtered_df[filtered_df["keywords"].apply(
            lambda arr: any(
                query == kw.lower().replace("_", " ") or q_us == kw.lower()
                for kw in arr
            )
        )]
        g = filtered_df[filtered_df["genres"].apply(
            lambda arr: any(
                query == gr.lower().replace("_", " ") or q_us == gr.lower()
                for gr in arr
            )
        )]
        c = filtered_df[filtered_df["cast"].apply(
            lambda arr: any(
                query == actor.lower().replace("_", " ") or q_us == actor.lower()
                for actor in arr
            )
        )]
        d = filtered_df[filtered_df["crew"].apply(
            lambda arr: any(
                query == director.lower().replace("_", " ") or q_us == director.lower()
                for director in arr
            )
        )]
        o = filtered_df[filtered_df["overview"].str.lower().str.contains(query)]
        res = pd.concat([k, g, c, d, o])

        for col in ["keywords", "genres", "cast", "crew"]:
            res[col] = res[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        res = res.drop_duplicates()

        if res.empty:
            st.warning("❌ No exact/keyword/cast/director matches—using SBERT semantic search.")
            sbert = SentenceTransformer('all-MiniLM-L6-v2')
            q_emb = sbert.encode([query], convert_to_tensor=True)
            sims  = util.pytorch_cos_sim(q_emb, filtered_embs)[0].cpu().numpy()
            ids   = np.argsort(sims)[::-1][:8]
            res   = filtered_df.iloc[ids]

    out = []
    for _, r in res.iterrows():
        pu, iid = fetch_poster_and_imdb(r["id"])
        rt      = fetch_imdb_rating(iid)
        out.append({
            "title"      : r["title_x"],
            "poster_url" : pu,
            "imdb_id"    : iid,
            "imdb_rating": rt
        })
        if len(out) == 8:
            break
    return out

# UI
st.markdown("#### 🔎 Search by title, keyword, cast, or description")
query = st.text_input("Movie search", "")

selected_genres = st.multiselect(
    "🎬 Filter by Genres (optional)", genre_list
)


if st.button("Recommend") and query:
    recs = recommend_movies(query, selected_genres)
    if recs:
        cols = st.columns(len(recs))
        for i, col in enumerate(cols):
            pu  = recs[i]["poster_url"]
            iid = recs[i]["imdb_id"]
            tit = recs[i]["title"]
            rt  = recs[i]["imdb_rating"]

            if pu and iid:
                col.markdown(f"[![Poster]({pu})](https://www.imdb.com/title/{iid})", unsafe_allow_html=True)
            elif pu:
                col.image(pu, use_container_width=True)

            if iid:
                col.markdown(
                    f"<p class='movie-title'><a href='https://www.imdb.com/title/{iid}' target='_blank'>{tit}</a></p>",
                    unsafe_allow_html=True
                )
            else:
                col.markdown(f"<p class='movie-title'>{tit}</p>", unsafe_allow_html=True)

            if rt:
                col.markdown(
                    f"<div class='movie-meta'><span class='star'>★</span><span class='rating'>IMDb: {rt}</span></div>",
                    unsafe_allow_html=True
                )
    else:
        st.warning("No recommendations found.")


