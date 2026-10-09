## Demo

[![Mood-Based Music Search Demo](https://img.youtube.com/vi/Hd8nCgaBpWI/maxresdefault.jpg)](https://youtu.be/Hd8nCgaBpWI)

# Mood-Based Music Search

A FastAPI web app that recommends songs based on natural-language mood and situation queries. Instead of searching by title or artist, describe what you're looking for:

- "Songs for walking home alone at night"
- "Sad but energetic songs"
- "Annoyed but intense songs"

## Features

- Natural-language song search using semantic NLP
- Keyword-based search for comparison
- Recommendations with explanations of why each song matched
- Optional filters for genre, energy, popularity, decade, and more
- Evaluation workflow to compare search quality using precision@5

## How It Works

The app uses the `all-MiniLM-L6-v2` Sentence Transformer model to convert search queries and song information into embeddings. Songs are ranked by semantic similarity, with additional signals from mood tags and audio features.

A keyword-based baseline provides a comparison point for evaluating whether semantic search improves results.

## Tech Stack

- **Backend:** Python, FastAPI
- **Frontend:** Jinja2, HTML, CSS
- **NLP:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Data:** Song lyrics, genres, and audio features
- **Evaluation:** Precision@5

## Run Locally

1. Clone the repository and navigate to the project directory.
2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the server:

   ```bash
   python -m uvicorn app.main:app --reload
   ```

4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

The first semantic search may take longer while the model and song embeddings load.

## Dataset

The app uses a preprocessed song dataset stored in `app/data/songs_imported.json`.

To rebuild it, download the original dataset and run:

```bash
python -m app.import_spotify_full "<DATASET_FOLDER>" --limit 1500 --max-rows 75000 --min-popularity 20
```

Replace `<DATASET_FOLDER>` with the path to your downloaded dataset.
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

## References

- [Sentence Transformers model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Dataset](https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres)
