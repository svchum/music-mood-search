import argparse
import ast
import csv
import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_PATH = DATA_DIR / "songs_imported.json"
DEFAULT_CSV_NAME = "songs.csv"

EMOTION_LEXICON = {
    "dread": {"dread", "doom", "doomed", "terror", "horror", "haunt", "haunted"},
    "fearful": {"scared", "fear", "afraid", "terrified", "frightened", "panic", "shaking"},
    "uneasy": {"uneasy", "paranoid", "nervous", "tension", "edge", "unsafe"},
    "disgusted": {"disgust", "disgusted", "gross", "filthy", "sickened", "repulsed", "rotten"},
    "ashamed": {"ashamed", "shame", "dirty", "hide", "embarrassed"},
    "frustrated": {"pissed", "fuck", "fucking", "bullshit", "rage", "hate", "revenge", "fed", "sick"},
    "annoyed": {"annoyed", "irritated", "bother", "tired", "sick", "enough", "nagging"},
    "furious": {"furious", "rage", "raging", "wrath", "explode", "violent"},
    "resentful": {"resent", "bitter", "blame", "grudge", "betray", "betrayed", "unfair", "used"},
    "defiant": {"rebel", "fight", "stand", "never", "won't", "against", "resist", "refuse"},
    "anxious": {"worry", "worried", "anxious", "nervous", "fear", "afraid", "panic", "shaking"},
    "jealous": {"jealous", "envy", "mine", "another", "someone", "cheat", "cheating"},
    "guilty": {"guilty", "shame", "sorry", "forgive", "regret", "wrong", "apologize"},
    "heartbreak": {"goodbye", "leave", "left", "miss", "lost", "alone", "broken"},
    "romantic": {"love", "baby", "kiss", "hold", "lover", "darling", "touch"},
    "lonely": {"lonely", "alone", "empty", "dark", "night"},
    "hopeful": {"hope", "better", "tomorrow", "future", "believe", "rise"},
    "nostalgic": {"remember", "memory", "yesterday", "old", "back", "home"},
    "sad": {"sad", "cry", "tears", "hurt", "pain", "blue", "sorry"},
    "grief": {"grief", "mourn", "mourning", "funeral", "gone", "loss"},
    "hopeless": {"hopeless", "helpless", "nothing", "empty", "lost", "despair"},
    "content": {"content", "satisfied", "easy", "peaceful", "fine", "okay"},
    "amazed": {"amazed", "amazing", "wonder", "astonished", "miracle", "magic"},
    "surprised": {"surprise", "surprised", "unexpected", "suddenly", "shock", "shocked"},
    "grateful": {"thank", "thanks", "grateful", "blessed", "lucky", "appreciate"},
    "empowered": {"power", "strong", "winner", "shine", "boss", "fearless", "unstoppable"},
    "playful": {"play", "fun", "party", "tease", "laugh", "smile", "silly"},
    "dreamy": {"dream", "float", "cloud", "moon", "stars", "sleep", "haze"},
    "sensual": {"body", "touch", "skin", "bed", "desire", "want", "heat"},
    "reflective": {"think", "thought", "mind", "remember", "wonder", "question"},
    "isolated": {"alone", "distance", "silent", "gone", "empty", "nobody"},
    "reckless": {"wild", "danger", "risk", "crazy", "drunk", "crash"},
    "dark": {"dark", "shadow", "devil", "death", "blood", "haunt"},
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_list(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value or "[]")
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in parsed]


def audio_moods(row: dict) -> list[str]:
    valence = to_float(row.get("valence"))
    energy = to_float(row.get("energy"))
    danceability = to_float(row.get("danceability"))
    acousticness = to_float(row.get("acousticness"))
    instrumentalness = to_float(row.get("instrumentalness"))
    liveness = to_float(row.get("liveness"))
    speechiness = to_float(row.get("speechiness"))
    tempo = to_float(row.get("tempo"))
    mode = to_int(row.get("mode"), 1)
    tags = []

    if valence >= 0.7:
        tags.append("happy")
    elif valence <= 0.3:
        tags.append("melancholy")

    if energy >= 0.75:
        tags.append("intense")
    elif energy <= 0.35:
        tags.append("calm")

    if danceability >= 0.72:
        tags.append("danceable")
    if acousticness >= 0.65:
        tags.append("acoustic")
    if instrumentalness >= 0.65:
        tags.append("instrumental")
    if speechiness >= 0.28:
        tags.append("wordy")
    if tempo >= 140:
        tags.append("fast")
    elif tempo and tempo <= 85:
        tags.append("slow")
    if mode == 0 and valence <= 0.45:
        tags.append("minor-key")

    if energy >= 0.7 and valence <= 0.35:
        tags.append("angry")
    if energy >= 0.82 and valence <= 0.32:
        tags.append("furious")
    if energy >= 0.65 and valence <= 0.45:
        tags.append("frustrated")
    if energy <= 0.45 and valence <= 0.4:
        tags.append("sad")
    if energy <= 0.35 and valence <= 0.25:
        tags.append("hopeless")
    if energy <= 0.42 and valence <= 0.35 and mode == 0:
        tags.append("dread")
    if speechiness >= 0.32 and valence <= 0.4:
        tags.append("uneasy")
    if valence >= 0.72 and energy <= 0.55:
        tags.append("content")
    if valence >= 0.78 and energy >= 0.65:
        tags.append("joyful")
    if liveness >= 0.55 and energy >= 0.65:
        tags.append("surprised")
    if acousticness >= 0.55 and energy <= 0.5:
        tags.append("study")
    if danceability >= 0.7 and energy >= 0.6:
        tags.append("upbeat")

    return tags


def lyric_moods(lyrics: str) -> list[str]:
    lowered = lyrics.lower()
    scored = []
    for mood, words in EMOTION_LEXICON.items():
        score = sum(1 for word in words if re.search(rf"\b{re.escape(word)}\b", lowered))
        if score:
            scored.append((score, mood))
    scored.sort(reverse=True)
    return [mood for _, mood in scored[:4]]


def unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def feature_summary(row: dict) -> str:
    return (
        f"valence {to_float(row.get('valence')):.2f}, "
        f"energy {to_float(row.get('energy')):.2f}, "
        f"danceability {to_float(row.get('danceability')):.2f}, "
        f"acousticness {to_float(row.get('acousticness')):.2f}, "
        f"tempo {to_float(row.get('tempo')):.0f} BPM"
    )


def theme_summary(row: dict, mood_tags: list[str]) -> str:
    genre = clean_text(row.get("genre", "unknown"))
    year = clean_text(row.get("year", "unknown year"))
    tags = ", ".join(mood_tags[:4]) if mood_tags else "general mood"
    return (
        f"a {year} {genre} track with {tags} cues from lyrics and Spotify audio features "
        f"({feature_summary(row)})."
    )


def build_record(row: dict) -> dict:
    lyrics = clean_text(row.get("lyrics", ""))
    artists = parse_list(row.get("artists", "[]"))
    niche_genres = parse_list(row.get("niche_genres", "[]"))
    mood_tags = unique(audio_moods(row) + lyric_moods(lyrics))[:7]
    if not mood_tags:
        mood_tags = ["general"]

    return {
        "title": clean_text(row.get("name", "Unknown title")),
        "artist": ", ".join(artists) or "Unknown artist",
        "genre": clean_text(row.get("genre", "unknown")),
        "niche_genres": niche_genres[:5],
        "album_name": clean_text(row.get("album_name", "")),
        "year": clean_text(row.get("year", "")),
        "explicit": clean_text(row.get("explicit", "False")).lower() == "true",
        "popularity": to_int(row.get("popularity")),
        "mood_tags": mood_tags,
        "theme_summary": theme_summary(row, mood_tags),
        "audio_profile": feature_summary(row),
        "lyrics_excerpt": " ".join(lyrics.split()[:140]),
        "spotify_id": clean_text(row.get("id", "")),
        "audio_features": {
            "danceability": to_float(row.get("danceability")),
            "energy": to_float(row.get("energy")),
            "valence": to_float(row.get("valence")),
            "tempo": to_float(row.get("tempo")),
            "acousticness": to_float(row.get("acousticness")),
            "instrumentalness": to_float(row.get("instrumentalness")),
            "speechiness": to_float(row.get("speechiness")),
        },
    }


def resolve_csv_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    direct = input_path / DEFAULT_CSV_NAME
    if direct.exists():
        return direct
    matches = sorted(input_path.rglob(DEFAULT_CSV_NAME))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {DEFAULT_CSV_NAME}")


def import_dataset(
    input_path: Path,
    output_path: Path,
    limit: int,
    seed: int,
    max_rows: int,
    min_popularity: int,
) -> None:
    input_path = resolve_csv_path(input_path)
    print(f"Reading dataset from {input_path}")
    candidates = []

    with input_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            if not clean_text(row.get("name", "")):
                continue
            if not clean_text(row.get("lyrics", "")):
                continue
            if to_int(row.get("popularity")) < min_popularity:
                continue
            candidates.append(row)

    random.Random(seed).shuffle(candidates)
    selected_rows = candidates[:limit]
    selected = [build_record(row) for row in selected_rows]
    selected.sort(key=lambda record: record["popularity"], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2, ensure_ascii=True)

    print(f"Scanned up to {max_rows} rows and found {len(candidates)} candidates")
    print(f"Imported {len(selected)} songs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Spotify songs with audio features, lyrics, and genres.")
    parser.add_argument("csv_path", type=Path, help="Path to songs.csv or a folder containing it")
    parser.add_argument("--limit", type=int, default=1500, help="Number of songs to import")
    parser.add_argument("--seed", type=int, default=53, help="Random sampling seed")
    parser.add_argument("--max-rows", type=int, default=75000, help="Maximum CSV rows to scan")
    parser.add_argument("--min-popularity", type=int, default=20, help="Minimum Spotify popularity score")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON path")
    args = parser.parse_args()

    import_dataset(args.csv_path, args.output, args.limit, args.seed, args.max_rows, args.min_popularity)


if __name__ == "__main__":
    main()
