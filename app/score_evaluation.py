import json
from pathlib import Path

LABEL_PATH = Path(__file__).resolve().parent / "data" / "evaluation_labels.json"


def precision_at_5(relevant_titles: list[str]) -> float:
    return len(relevant_titles[:5]) / 5


def main() -> None:
    with LABEL_PATH.open(encoding="utf-8") as file:
        labels = json.load(file)

    semantic_scores = []
    keyword_scores = []

    for query, query_labels in labels.items():
        semantic_precision = precision_at_5(query_labels.get("semantic_relevant", []))
        keyword_precision = precision_at_5(query_labels.get("keyword_relevant", []))
        semantic_scores.append(semantic_precision)
        keyword_scores.append(keyword_precision)

        print(f"Query: {query}")
        print(f"  Semantic precision@5: {semantic_precision:.2f}")
        print(f"  Keyword precision@5:  {keyword_precision:.2f}")
        print()

    if semantic_scores:
        print(f"Average semantic precision@5: {sum(semantic_scores) / len(semantic_scores):.2f}")
        print(f"Average keyword precision@5:  {sum(keyword_scores) / len(keyword_scores):.2f}")


if __name__ == "__main__":
    main()
