import json
from pathlib import Path

from app.main import NLP_BACKEND, run_search

QUERY_PATH = Path(__file__).resolve().parent / "data" / "evaluation_queries.json"


def load_queries() -> list[str]:
    with QUERY_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    print(f"Semantic backend: {NLP_BACKEND}")
    print("Use this output to manually judge top-5 relevance for precision@5.\n")

    for query in load_queries():
        semantic_titles = [result["title"] for result in run_search(query, "semantic", limit=5)]
        keyword_titles = [result["title"] for result in run_search(query, "keyword", limit=5)]

        print(f"Query: {query}")
        print(f"  Semantic: {semantic_titles}")
        print(f"  Keyword:  {keyword_titles}")
        print()


if __name__ == "__main__":
    main()
