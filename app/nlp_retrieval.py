import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from app.retrieval import expand_query_terms, semantic_song_document, song_document, tokenize

DATA_DIR = Path(__file__).resolve().parent / "data"
EMBEDDING_CACHE_PATH = DATA_DIR / "embedding_cache.json"


class TfidfIndex:
    """Dependency-free NLP retrieval index used when transformer embeddings are unavailable."""

    def __init__(self, songs: list[dict]):
        self.songs = songs
        self.document_tokens = [tokenize(semantic_song_document(song)) for song in songs]
        self.idf = self._build_idf(self.document_tokens)
        self.document_vectors = [self._vectorize(tokens) for tokens in self.document_tokens]

    def search(self, query: str, limit: int = 5) -> list[tuple[float, dict, list[str]]]:
        query_tokens = sorted(expand_query_terms(query))
        query_vector = self._vectorize(query_tokens)
        ranked = []

        for song, document_vector, document_tokens in zip(self.songs, self.document_vectors, self.document_tokens):
            score = cosine_similarity(query_vector, document_vector)
            if score > 0:
                reasons = self._reasons(query_tokens, document_tokens, song)
                ranked.append((score, song, reasons))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:limit]

    @staticmethod
    def _build_idf(document_tokens: list[list[str]]) -> dict[str, float]:
        document_count = len(document_tokens)
        document_frequency = Counter()
        for tokens in document_tokens:
            document_frequency.update(set(tokens))

        return {
            term: math.log((1 + document_count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        if not counts:
            return {}

        total = sum(counts.values())
        return {
            term: (count / total) * self.idf.get(term, 0)
            for term, count in counts.items()
            if term in self.idf
        }

    @staticmethod
    def _reasons(query_tokens: list[str], document_tokens: list[str], song: dict) -> list[str]:
        overlaps = sorted(set(query_tokens) & set(document_tokens))
        mood_overlaps = sorted(set(overlaps) & {tag.lower() for tag in song["mood_tags"]})
        reasons = []

        if mood_overlaps:
            reasons.append("semantic document overlap with mood terms: " + ", ".join(mood_overlaps))
        if overlaps:
            reasons.append("TF-IDF matched important terms: " + ", ".join(overlaps[:6]))
        if not reasons:
            reasons.append("nearest document in the NLP retrieval index")
        return reasons


class SentenceTransformerIndex:
    """Transformer embedding index. Requires sentence-transformers to be installed."""

    def __init__(self, songs: list[dict], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.songs = songs
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents = [semantic_song_document(song) for song in songs]
        self.document_embeddings = self._load_or_build_embeddings()

    def search(self, query: str, limit: int = 5) -> list[tuple[float, dict, list[str]]]:
        query_terms = expand_query_terms(query)
        expanded_query = f"{query}. Mood concepts: {' '.join(sorted(query_terms))}."
        query_embedding = self.model.encode([expanded_query], normalize_embeddings=True)[0]
        ranked = []

        for song, embedding in zip(self.songs, self.document_embeddings):
            reasons = self._reasons(query_terms, song)
            base_score = float(sum(a * b for a, b in zip(query_embedding, embedding)))
            score = base_score + self._rerank_bonus(query_terms, song)
            ranked.append(
                (
                    score,
                    song,
                    reasons,
                )
            )

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:limit]

    def _rerank_bonus(self, query_terms: set[str], song: dict) -> float:
        mood_terms = {tag.lower() for tag in song["mood_tags"]}
        document_terms = set(tokenize(semantic_song_document(song)))
        mood_overlap = query_terms & mood_terms
        document_overlap = query_terms & document_terms
        return (0.055 * len(mood_overlap)) + (0.012 * min(len(document_overlap), 6))

    def _reasons(self, query_terms: set[str], song: dict) -> list[str]:
        mood_terms = {tag.lower() for tag in song["mood_tags"]}
        document_terms = set(tokenize(semantic_song_document(song)))
        mood_overlap = sorted(query_terms & mood_terms)
        document_overlap = sorted((query_terms & document_terms) - set(mood_overlap))
        reasons = []

        if mood_overlap:
            reasons.append("Mood tags match: " + ", ".join(mood_overlap[:6]))
        if document_overlap:
            reasons.append("Text concepts match: " + ", ".join(document_overlap[:6]))
        if not reasons:
            reasons.append("Closest semantic match in the song index.")
        return reasons

    def _load_or_build_embeddings(self):
        dataset_hash = self._dataset_hash()
        cached = self._load_cache(dataset_hash)
        if cached is not None:
            return cached

        embeddings = self.model.encode(
            self.documents,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self._save_cache(dataset_hash, embeddings)
        return embeddings

    def _dataset_hash(self) -> str:
        payload = json.dumps(self.documents, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self, dataset_hash: str):
        if not EMBEDDING_CACHE_PATH.exists():
            return None

        try:
            with EMBEDDING_CACHE_PATH.open(encoding="utf-8") as file:
                cache = json.load(file)
        except json.JSONDecodeError:
            return None

        if cache.get("model_name") != self.model_name:
            return None
        if cache.get("dataset_hash") != dataset_hash:
            return None
        if len(cache.get("embeddings", [])) != len(self.documents):
            return None

        return cache["embeddings"]

    def _save_cache(self, dataset_hash: str, embeddings) -> None:
        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "model_name": self.model_name,
            "dataset_hash": dataset_hash,
            "count": len(self.documents),
            "embeddings": [list(map(float, embedding)) for embedding in embeddings],
        }
        with EMBEDDING_CACHE_PATH.open("w", encoding="utf-8") as file:
            json.dump(cache, file)


def build_nlp_index(songs: list[dict]):
    try:
        index = SentenceTransformerIndex(songs)
        return index, f"sentence-transformers / {index.model_name}"
    except Exception:
        return TfidfIndex(songs), "tf-idf + mood concepts"


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    shared_terms = set(left) & set(right)
    dot_product = sum(left[term] * right[term] for term in shared_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)
