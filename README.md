# Mood-Based Music Search

A FastAPI web app that recommends songs from natural-language mood and situation queries. Instead of searching only by title, artist, or genre, users can type prompts like:

```text
songs for walking home alone at night
sad but energetic songs
annoyed but intense songs
```

The app retrieves songs from a Spotify lyrics/audio-feature dataset, ranks them with either semantic NLP search or a keyword baseline, and explains why each result matched.

## Why I Built This

Music search is often emotional and contextual. A user may not know the exact song they want, but they know the mood they are looking for. This project explores whether semantic search can improve mood-based music retrieval compared with a simpler keyword-based approach.

The project is built as an NLP retrieval experiment, not just a playlist UI. It includes:

- two search methods
- grounded result explanations
- optional filters
- an evaluation workflow using precision@5

## Features

- Natural-language mood search
- Semantic search using `sentence-transformers`
- `all-MiniLM-L6-v2` sentence embeddings
- Keyword baseline for comparison
- Mood tags generated from lyrics and audio features
- Song ranking using semantic similarity, mood overlap, and audio-feature evidence
- "Why This Matched" explanations for each recommendation
- Optional filters for genre, explicit status, energy, valence, popularity, and decade
- Evaluation page for labeling top-5 results
- Precision@5 scoring script
- JSON API endpoints for search and method comparison

## Tech Stack

- **Backend:** Python, FastAPI
- **Templates:** Jinja2
- **NLP:** Sentence Transformers, `all-MiniLM-L6-v2`
- **Evaluation:** precision@5
- **Data:** Spotify songs dataset with lyrics, genres, popularity, and audio features

## How It Works

Each song is converted into a searchable text document made from:

- lyrics excerpt
- generated mood tags
- theme summary
- genre and niche genre information
- audio feature descriptions
- metadata such as popularity and release year

The semantic search pipeline is:

```text
user mood query
→ expanded query concepts
→ sentence embedding
→ compare against song document embeddings
→ rank by cosine similarity
→ apply small mood/audio reranking bonuses
→ return top results with explanations
```

The keyword baseline uses direct word overlap, mood synonym expansion, mood tag matches, and small audio-profile bonuses. This gives the project a fair comparison point for evaluating whether semantic search actually improves retrieval.

## Semantic Search

The semantic search uses `all-MiniLM-L6-v2`, a compact Sentence Transformer model. It converts both the user query and each song document into embeddings, which are numerical vectors representing meaning.

The app compares the query embedding with each song embedding using cosine similarity. Songs with embeddings closest to the query are ranked higher.

Confidence is shown as a relative display value:

```text
song score / highest score for that query
```

So confidence is not a probability. It shows how strong a result is compared with the top-ranked result for the same search.

## Evaluation

The `/evaluate` page is used to label whether the top five results for fixed test queries are relevant.

Precision@5 is calculated as:

```text
relevant results in top 5 / 5
```

This allows semantic NLP search and the keyword baseline to be compared on the same queries.

## Project Structure

```text
app/
  main.py                  FastAPI routes and page rendering
  retrieval.py             keyword baseline, shared scoring, explanations
  nlp_retrieval.py         semantic search and embedding cache logic
  import_spotify_full.py   dataset import/preprocessing script
  score_evaluation.py      precision@5 scoring script
  data/
    songs_imported.json
    evaluation_queries.json
    evaluation_labels.json
  templates/
    index.html
    method.html
    evaluate.html
  static/
    styles.css
README.md
report.md
requirements.txt
```

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the server:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

For development with reload:

```bash
python -m uvicorn app.main:app --reload
```

The first semantic search may take longer because the sentence-transformer model and embeddings need to load.

## Rebuild the Dataset

If you have the original Kaggle dataset folder, replace `<DATASET_FOLDER>` with the path to that folder:

```bash
python -m app.import_spotify_full "<DATASET_FOLDER>" --limit 1500 --max-rows 75000 --min-popularity 20
```

This creates:

```text
app/data/songs_imported.json
```

The app may also generate:

```text
app/data/embedding_cache.json
```

That file is a generated cache and does not need to be committed.

## API Endpoints

```text
GET /api/search?q=<query>&method=semantic
GET /api/search?q=<query>&method=keyword
GET /api/compare?q=<query>
```

Example:

```text
/api/search?q=sad%20but%20energetic%20songs&method=semantic
```

## What I Learned

This project helped me practice:

- building a complete FastAPI app
- using sentence embeddings for semantic retrieval
- designing a keyword baseline for comparison
- using cosine similarity for ranking
- combining lyrics and audio features as retrieval evidence
- evaluating search quality with precision@5
- explaining model outputs in a user-facing interface

## Future Improvements

- Add more labeled evaluation queries
- Support graded relevance labels instead of only relevant/not relevant
- Improve reranking with stronger audio-feature weighting
- Add result diversity to avoid repetitive recommendations
- Try a music-specific embedding or emotion classification model
- Add a fuller RAG chatbot layer for conversational music recommendations

## References

- Reimers, N. and Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." arXiv, 2019. https://arxiv.org/abs/1908.10084
- Hugging Face model card: `sentence-transformers/all-MiniLM-L6-v2`. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- DataStax Vibe Check inspiration: https://github.com/datastax/vibe-check
