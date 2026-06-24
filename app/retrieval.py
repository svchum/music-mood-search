import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
IMPORTED_DATA_PATH = DATA_DIR / "songs_imported.json"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "but",
    "by",
    "day",
    "for",
    "i",
    "in",
    "is",
    "me",
    "music",
    "of",
    "on",
    "song",
    "songs",
    "that",
    "the",
    "to",
    "with",
}

MOOD_SYNONYMS = {
    "rainy": {"melancholy", "soft", "reflective", "intimate", "moody"},
    "study": {"calm", "focused", "soft", "ambient", "relaxed"},
    "workout": {"energetic", "intense", "confident", "driving", "hype"},
    "sad": {"sad", "melancholy", "heartbreak", "bittersweet", "lonely"},
    "happy": {"happy", "joyful", "bright", "upbeat", "playful"},
    "nostalgic": {"nostalgic", "reflective", "bittersweet", "warm"},
    "angry": {"angry", "intense", "defiant", "aggressive", "frustrated"},
    "rage": {"furious", "angry", "frustrated", "intense"},
    "furious": {"furious", "angry", "frustrated", "rage"},
    "mad": {"angry", "frustrated", "annoyed", "intense"},
    "pissed": {"frustrated", "angry", "rage", "aggressive", "fed-up"},
    "off": {"frustrated", "angry", "annoyed"},
    "annoyed": {"annoyed", "irritated", "fed-up", "bothered"},
    "irritated": {"annoyed", "irritated", "fed-up", "tense"},
    "fed": {"fed-up", "annoyed", "frustrated"},
    "resentful": {"resentful", "bitter", "betrayed", "angry"},
    "bitter": {"resentful", "heartbreak", "sad", "angry"},
    "defiant": {"defiant", "rebellious", "confident", "angry"},
    "anxious": {"anxious", "nervous", "worried", "tense"},
    "nervous": {"anxious", "nervous", "worried", "tense"},
    "fear": {"fearful", "dread", "anxious", "uneasy"},
    "fearful": {"fearful", "scared", "dread", "anxious"},
    "scared": {"fearful", "scared", "dread", "anxious", "uneasy"},
    "terrified": {"fearful", "dread", "terror", "anxious"},
    "terror": {"dread", "fearful", "dark", "anxious"},
    "dread": {"dread", "fearful", "uneasy", "dark"},
    "uneasy": {"uneasy", "anxious", "dread", "tense"},
    "paranoid": {"uneasy", "anxious", "fearful"},
    "disgust": {"disgusted", "repulsed", "dark", "frustrated"},
    "disgusted": {"disgusted", "repulsed", "ashamed", "dark"},
    "gross": {"disgusted", "repulsed"},
    "repulsed": {"disgusted", "repulsed"},
    "jealous": {"jealous", "possessive", "insecure", "heartbreak"},
    "guilty": {"guilty", "regretful", "sorry", "ashamed"},
    "ashamed": {"ashamed", "guilty", "regretful"},
    "bored": {"bored", "restless", "tired", "detached"},
    "hopeful": {"hopeful", "healing", "optimistic", "warm"},
    "hopeless": {"hopeless", "grief", "sad", "melancholy"},
    "despair": {"hopeless", "grief", "sad", "dark"},
    "grief": {"grief", "sad", "hopeless", "heartbreak"},
    "sorrow": {"sad", "grief", "melancholy"},
    "joy": {"joyful", "happy", "upbeat", "playful"},
    "joyful": {"joyful", "happy", "upbeat", "playful"},
    "content": {"content", "calm", "happy", "peaceful"},
    "satisfied": {"content", "happy", "peaceful"},
    "surprise": {"surprised", "amazed", "playful", "bright"},
    "surprised": {"surprised", "amazed", "playful"},
    "amazed": {"amazed", "surprised", "dreamy", "joyful"},
    "astonished": {"amazed", "surprised"},
    "grateful": {"grateful", "thankful", "warm", "hopeful"},
    "thankful": {"grateful", "thankful", "warm"},
    "empowered": {"empowered", "confident", "strong", "defiant"},
    "playful": {"playful", "happy", "bright", "upbeat"},
    "dreamy": {"dreamy", "soft", "ambient", "romantic"},
    "sensual": {"sensual", "romantic", "intimate", "warm"},
    "reflective": {"reflective", "nostalgic", "melancholy", "thoughtful"},
    "reckless": {"reckless", "wild", "energetic", "intense"},
    "dark": {"dark", "moody", "melancholy", "intense"},
    "melancholy": {"melancholy", "sad", "reflective", "moody"},
    "calm": {"calm", "peaceful", "soft", "relaxed", "gentle"},
    "lonely": {"lonely", "isolated", "intimate", "reflective"},
    "upbeat": {"upbeat", "energetic", "bright", "danceable"},
    "alone": {"lonely", "isolated", "intimate", "reflective"},
    "home": {"intimate", "calm", "warm", "soft"},
    "night": {"late-night", "reflective", "lonely", "restless", "moody"},
    "nighttime": {"late-night", "reflective", "lonely", "restless", "moody"},
    "peaceful": {"peaceful", "calm", "gentle", "soft", "relaxed"},
    "focus": {"focused", "calm", "ambient", "soft"},
    "hype": {"hype", "confident", "energetic", "intense"},
    "breakup": {"heartbreak", "sad", "bittersweet", "intense"},
}


def load_songs() -> list[dict]:
    if not IMPORTED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing app/data/songs_imported.json. Run app.import_spotify_full to build the project dataset."
        )

    with IMPORTED_DATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS]


def song_document(song: dict) -> str:
    parts = [
        song["title"],
        song["artist"],
        "" if song.get("genre") == "unknown" else song.get("genre", ""),
        " ".join(song["mood_tags"]),
        song["theme_summary"],
        song.get("lyrics_excerpt", ""),
        song.get("audio_profile", ""),
    ]
    return " ".join(parts)


def semantic_song_document(song: dict) -> str:
    """Mood-focused document for embeddings; avoids artist-name keyword leakage."""
    niche_genres = " ".join(song.get("niche_genres", []))
    audio_features = song.get("audio_features", {})
    feature_text = ""
    if audio_features:
        feature_text = (
            f"audio mood valence {audio_features.get('valence', 0):.2f} "
            f"energy {audio_features.get('energy', 0):.2f} "
            f"danceability {audio_features.get('danceability', 0):.2f} "
            f"acousticness {audio_features.get('acousticness', 0):.2f} "
            f"tempo {audio_features.get('tempo', 0):.0f}"
        )

    parts = [
        "" if song.get("genre") == "unknown" else song.get("genre", ""),
        niche_genres,
        " ".join(song["mood_tags"]),
        song["theme_summary"],
        feature_text,
        song.get("lyrics_excerpt", ""),
    ]
    return " ".join(parts)


def expand_query_terms(query: str) -> set[str]:
    terms = set(tokenize(query))
    for term in list(terms):
        terms.update(MOOD_SYNONYMS.get(term, set()))
    return terms


def score_song(query_terms: set[str], song: dict) -> tuple[float, list[str]]:
    document_terms = set(tokenize(song_document(song)))
    mood_terms = set(token.lower() for token in song["mood_tags"])

    document_matches = query_terms & document_terms
    mood_matches = query_terms & mood_terms

    score = len(document_matches) + (1.75 * len(mood_matches))

    audio_profile = song.get("audio_profile", "").lower()
    if "energetic" in query_terms and "high energy" in audio_profile:
        score += 1
    if "calm" in query_terms and ("low energy" in audio_profile or "relaxed" in audio_profile):
        score += 1
    if "danceable" in query_terms and "danceable" in audio_profile:
        score += 1

    reasons = []
    if mood_matches:
        reasons.append("matching mood tags: " + ", ".join(sorted(mood_matches)))
    if document_matches - mood_matches:
        reasons.append("matching description terms: " + ", ".join(sorted(document_matches - mood_matches)))
    if not reasons:
        reasons.append("closest available match in the current song dataset")

    return score, reasons


def explain_match(song: dict, reasons: list[str], retrieval_method: str = "keyword baseline") -> str:
    evidence = []
    audio_features = song.get("audio_features", {})
    if audio_features:
        valence = audio_features.get("valence", 0)
        energy = audio_features.get("energy", 0)
        danceability = audio_features.get("danceability", 0)
        tempo = audio_features.get("tempo", 0)

        if valence <= 0.35:
            evidence.append("low valence gives it a darker emotional tone")
        elif valence >= 0.7:
            evidence.append("high valence gives it a brighter tone")

        if energy >= 0.75:
            evidence.append("high energy makes it feel intense")
        elif energy <= 0.4:
            evidence.append("low energy makes it feel subdued")

        if danceability >= 0.7:
            evidence.append("strong danceability adds movement")
        if tempo >= 140:
            evidence.append("a fast tempo adds urgency")
        elif tempo and tempo <= 85:
            evidence.append("a slow tempo makes it feel more restrained")

    if song.get("mood_tags"):
        evidence.append("tags include " + ", ".join(song["mood_tags"][:3]))
    if reasons:
        evidence.append(reasons[0].rstrip("."))

    return "This fits because " + "; ".join(evidence[:4]) + "."


def shorten_text(text: str, max_length: int = 150) -> str:
    if len(text) <= max_length:
        return text

    trimmed = text[: max_length - 3].rsplit(" ", 1)[0]
    return f"{trimmed}..."


def format_ranked_results(
    ranked: list[tuple[float, dict, list[str]]],
    limit: int,
    retrieval_method: str,
    offset: int = 0,
) -> list[dict]:
    if not ranked:
        return []

    max_score = max(score for score, _, _ in ranked) or 1
    results = []
    for score, song, reasons in ranked[offset : offset + limit]:
        confidence = round(score / max_score, 2) if score else 0.0
        public_song = {key: value for key, value in song.items() if key != "lyrics_excerpt"}
        public_song["display_artist"] = display_artist(public_song)
        results.append(
            {
                **public_song,
                "score": round(score, 3),
                "confidence": confidence,
                "retrieval_method": retrieval_method,
                "short_theme_summary": shorten_text(song["theme_summary"], 130),
                "match_reasons": reasons,
                "explanation": explain_match(song, reasons, retrieval_method),
            }
        )
    return results


def display_artist(song: dict) -> str:
    genre = song.get("genre", "")
    if genre and genre != "unknown":
        return f"{song['artist']} / {genre}"
    return song["artist"]


def search_songs_keyword(query: str, songs: list[dict], limit: int = 5, offset: int = 0) -> list[dict]:
    query_terms = expand_query_terms(query)
    ranked = []

    for song in songs:
        score, reasons = score_song(query_terms, song)
        if score > 0:
            ranked.append((score, song, reasons))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        ranked = [(0.0, song, ["closest available match in the current song dataset"]) for song in songs[:limit]]

    return format_ranked_results(ranked, limit, "keyword baseline", offset=offset)


def search_songs(query: str, songs: list[dict], limit: int = 5, offset: int = 0) -> list[dict]:
    return search_songs_keyword(query, songs, limit, offset=offset)
