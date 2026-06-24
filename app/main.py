from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import random
from pathlib import Path

from app.nlp_retrieval import build_nlp_index
from app.retrieval import format_ranked_results, load_songs, search_songs

app = FastAPI(title="Mood-Based Music Recommendation")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
EVALUATION_QUERY_PATH = Path(__file__).resolve().parent / "data" / "evaluation_queries.json"
EVALUATION_LABEL_PATH = Path(__file__).resolve().parent / "data" / "evaluation_labels.json"

SONGS = load_songs()
NLP_INDEX, NLP_BACKEND = build_nlp_index(SONGS)

RECOMMENDED_SEARCHES = [
    "romantic songs at night",
    "scared songs for walking alone",
    "frustrated songs after an argument",
    "annoyed songs when everything is irritating",
    "hopeful songs about getting better",
    "melancholy songs for a rainy evening",
    "joyful songs for celebrating",
    "content peaceful songs for relaxing",
    "grief songs about missing someone",
    "anxious songs for overthinking",
    "disgusted songs about betrayal",
    "surprised songs with high energy",
    "dreamy songs for late night driving",
    "confident songs before going out",
    "reckless songs for a chaotic mood",
    "lonely acoustic songs",
    "dark intense songs",
    "playful upbeat songs",
    "guilty songs about regret",
    "empowered songs about moving on",
    "sensual R&B songs",
    "bored restless songs",
    "dread-filled songs with a dark mood",
    "grateful songs about being lucky",
    "sad but danceable songs",
]

SPJIV_RESULT = {
    "title": "CNF",
    "artist": "spjiv",
    "display_artist": "spjiv",
    "genre": "",
    "mood_tags": ["easter-egg"],
    "theme_summary": "silverio jimenez.",
    "short_theme_summary": "A hidden playable track that appears only when spjiv is mentioned.",
    "audio_profile": "",
    "audio_url": "/static/audio/cnf.wav",
    "score": 1.0,
    "confidence": 1.0,
    "retrieval_method": "easter egg",
    "match_reasons": ["Triggered by the search term spjiv."],
    "explanation": "CNF by spjiv. Press play to listen.",
}


def genre_options() -> list[str]:
    return sorted(
        {
            song.get("genre", "")
            for song in SONGS
            if song.get("genre") and song.get("genre") != "unknown"
        }
    )


def decade_options() -> list[str]:
    decades = set()
    for song in SONGS:
        year = str(song.get("year", ""))
        if len(year) >= 4 and year[:4].isdigit():
            decades.add(f"{year[:3]}0s")
    return sorted(decades, reverse=True)


def matches_filters(song: dict, filters: dict) -> bool:
    audio = song.get("audio_features", {})
    year = str(song.get("year", ""))

    if filters["genre"] and song.get("genre") != filters["genre"]:
        return False
    if filters["explicit"] == "yes" and not song.get("explicit"):
        return False
    if filters["explicit"] == "no" and song.get("explicit"):
        return False
    if filters["popularity"] == "mainstream" and song.get("popularity", 0) < 55:
        return False
    if filters["popularity"] == "hidden" and song.get("popularity", 0) > 45:
        return False
    if filters["decade"] and not year.startswith(filters["decade"][:3]):
        return False

    energy = audio.get("energy", 0)
    valence = audio.get("valence", 0)
    if filters["energy"] == "high" and energy < 0.7:
        return False
    if filters["energy"] == "low" and energy > 0.45:
        return False
    if filters["valence"] == "bright" and valence < 0.65:
        return False
    if filters["valence"] == "dark" and valence > 0.4:
        return False
    return True


def apply_filters(results: list[dict], filters: dict, limit: int, offset: int) -> list[dict]:
    filtered = [result for result in results if matches_filters(result, filters)]
    return filtered[offset : offset + limit]


def run_search(query: str, method: str, limit: int = 5, offset: int = 0, filters: dict | None = None) -> list[dict]:
    if "spjiv" in query.lower():
        return [SPJIV_RESULT]
    filters = filters or empty_filters()

    if method == "keyword":
        results = search_songs(query, SONGS, limit=len(SONGS), offset=0)
        return apply_filters(results, filters, limit, offset)

    ranked = NLP_INDEX.search(query, limit=len(SONGS))
    retrieval_method = f"semantic NLP retrieval ({NLP_BACKEND})"
    results = format_ranked_results(ranked, len(ranked), retrieval_method, offset=0)
    return apply_filters(results, filters, limit, offset)


def empty_filters() -> dict:
    return {
        "genre": "",
        "explicit": "any",
        "energy": "any",
        "valence": "any",
        "popularity": "any",
        "decade": "",
    }


def recommended_searches(seed: int = 0, count: int = 5) -> list[str]:
    picker = random.Random(seed)
    return picker.sample(RECOMMENDED_SEARCHES, count)


def load_evaluation_queries() -> list[str]:
    with EVALUATION_QUERY_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def load_evaluation_labels() -> dict:
    if not EVALUATION_LABEL_PATH.exists():
        return {}
    with EVALUATION_LABEL_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def save_evaluation_labels(labels: dict) -> None:
    with EVALUATION_LABEL_PATH.open("w", encoding="utf-8") as file:
        json.dump(labels, file, indent=2, ensure_ascii=True)


def precision_at_5(relevant_titles: list[str]) -> float:
    return round(len(relevant_titles[:5]) / 5, 2)


@app.get("/", response_class=HTMLResponse)
def home(request: Request, rec_seed: int = Query(default=0, ge=0)) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": "",
            "method": "semantic",
            "backend": NLP_BACKEND,
            "rec_seed": rec_seed,
            "next_rec_seed": rec_seed + 1,
            "recommended_searches": recommended_searches(rec_seed),
            "filters": empty_filters(),
            "genres": genre_options(),
            "decades": decade_options(),
            "results": [],
        },
    )


@app.get("/search", response_class=HTMLResponse)
def search_page(
    request: Request,
    q: str = Query(default="", min_length=0),
    method: str = Query(default="semantic", pattern="^(semantic|keyword)$"),
    offset: int = Query(default=0, ge=0),
    rec_seed: int = Query(default=0, ge=0),
    genre: str = Query(default=""),
    explicit: str = Query(default="any", pattern="^(any|yes|no)$"),
    energy: str = Query(default="any", pattern="^(any|high|low)$"),
    valence: str = Query(default="any", pattern="^(any|bright|dark)$"),
    popularity: str = Query(default="any", pattern="^(any|mainstream|hidden)$"),
    decade: str = Query(default=""),
) -> HTMLResponse:
    filters = {
        "genre": genre,
        "explicit": explicit,
        "energy": energy,
        "valence": valence,
        "popularity": popularity,
        "decade": decade,
    }
    results = run_search(q, method, limit=5, offset=offset, filters=filters) if q.strip() else []
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "examples": [],
            "query": q,
            "method": method,
            "backend": NLP_BACKEND,
            "offset": offset,
            "next_offset": offset + 5,
            "rec_seed": rec_seed,
            "next_rec_seed": rec_seed + 1,
            "recommended_searches": recommended_searches(rec_seed),
            "filters": filters,
            "genres": genre_options(),
            "decades": decade_options(),
            "results": results,
        },
    )


@app.get("/method", response_class=HTMLResponse)
def method_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "method.html",
        {
            "request": request,
            "backend": NLP_BACKEND,
        },
    )


@app.get("/evaluate", response_class=HTMLResponse)
def evaluation_page(request: Request) -> HTMLResponse:
    rows = []
    labels = load_evaluation_labels()
    for query in load_evaluation_queries():
        query_labels = labels.get(query, {"semantic_relevant": [], "keyword_relevant": []})
        semantic_relevant = query_labels.get("semantic_relevant", [])
        keyword_relevant = query_labels.get("keyword_relevant", [])
        rows.append(
            {
                "query": query,
                "semantic_results": run_search(query, "semantic", limit=5),
                "keyword_results": run_search(query, "keyword", limit=5),
                "semantic_relevant": semantic_relevant,
                "keyword_relevant": keyword_relevant,
                "semantic_precision": precision_at_5(semantic_relevant),
                "keyword_precision": precision_at_5(keyword_relevant),
            }
        )

    semantic_average = round(sum(row["semantic_precision"] for row in rows) / len(rows), 2) if rows else 0
    keyword_average = round(sum(row["keyword_precision"] for row in rows) / len(rows), 2) if rows else 0

    return templates.TemplateResponse(
        "evaluate.html",
        {
            "request": request,
            "rows": rows,
            "backend": NLP_BACKEND,
            "semantic_average": semantic_average,
            "keyword_average": keyword_average,
        },
    )


@app.post("/evaluate")
async def save_evaluation(request: Request):
    form = await request.form()
    labels = {}

    for query in load_evaluation_queries():
        semantic_key = f"semantic::{query}"
        keyword_key = f"keyword::{query}"
        labels[query] = {
            "semantic_relevant": form.getlist(semantic_key),
            "keyword_relevant": form.getlist(keyword_key),
        }

    save_evaluation_labels(labels)
    return RedirectResponse(url="/evaluate", status_code=303)


@app.get("/api/search")
def search_api(
    q: str = Query(..., min_length=1),
    method: str = Query(default="semantic", pattern="^(semantic|keyword)$"),
    limit: int = Query(default=5, ge=1, le=20),
    offset: int = Query(default=0, ge=0),
    genre: str = Query(default=""),
    explicit: str = Query(default="any", pattern="^(any|yes|no)$"),
    energy: str = Query(default="any", pattern="^(any|high|low)$"),
    valence: str = Query(default="any", pattern="^(any|bright|dark)$"),
    popularity: str = Query(default="any", pattern="^(any|mainstream|hidden)$"),
    decade: str = Query(default=""),
):
    filters = {
        "genre": genre,
        "explicit": explicit,
        "energy": energy,
        "valence": valence,
        "popularity": popularity,
        "decade": decade,
    }
    return {
        "query": q,
        "method": method,
        "backend": NLP_BACKEND if method == "semantic" else "keyword",
        "offset": offset,
        "filters": filters,
        "results": run_search(q, method, limit=limit, offset=offset, filters=filters),
    }


@app.get("/api/compare")
def compare_api(q: str = Query(..., min_length=1), limit: int = Query(default=5, ge=1, le=20)):
    return {
        "query": q,
        "semantic_backend": NLP_BACKEND,
        "semantic_results": run_search(q, "semantic", limit=limit),
        "keyword_results": run_search(q, "keyword", limit=limit),
    }
