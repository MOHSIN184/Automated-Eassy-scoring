"""Evidence-based feedback derived from measured writing features."""

from __future__ import annotations

from typing import Any


def _position(value: float, quantiles: dict[str, float]) -> str:
    if value <= quantiles.get("q25", value - 1):
        return "low"
    if value >= quantiles.get("q75", value + 1):
        return "high"
    return "typical"


def generate_feedback(
    features: dict[str, float], feature_quantiles: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Generate strengths, weaknesses, and quality summaries from observed values."""

    def position(name: str) -> str:
        return _position(features[name], feature_quantiles.get(name, {}))

    strengths: list[str] = []
    weaknesses: list[str] = []

    if position("word_count") == "high":
        strengths.append("The response is well developed in length.")
    elif position("word_count") == "low":
        weaknesses.append("Develop the main claim with more specific evidence and explanation.")
    if position("type_token_ratio") == "high" or position("root_type_token_ratio") == "high":
        strengths.append("The essay uses a varied vocabulary with limited repetition.")
    if position("repetition_ratio") == "high":
        weaknesses.append("Replace repeated words or ideas with more precise alternatives.")
    if position("grammar_errors_per_100_words") == "low":
        strengths.append("The measured rate of grammar issues is low.")
    elif position("grammar_errors_per_100_words") == "high":
        weaknesses.append("Review sentence boundaries, agreement, and grammar conventions.")
    if position("spelling_errors_per_100_words") == "high":
        weaknesses.append("Proofread spelling, especially uncommon or content-specific words.")
    elif position("spelling_errors_per_100_words") == "low":
        strengths.append("Spelling is generally controlled.")
    if features["paragraph_count"] >= 3 and position("avg_paragraph_words") != "high":
        strengths.append("Paragraphing gives the response a visible organizational structure.")
    elif features["paragraph_count"] <= 1 and features["word_count"] > 120:
        weaknesses.append(
            "Separate the introduction, supporting ideas, and conclusion into paragraphs."
        )
    if position("transitions_per_100_words") == "high":
        strengths.append("Transitions help connect ideas across the response.")
    elif position("transitions_per_100_words") == "low" and features["sentence_count"] >= 5:
        weaknesses.append("Use more explicit transitions to clarify relationships between ideas.")
    if position("sentence_length_std") == "low" and features["sentence_count"] >= 4:
        weaknesses.append("Vary sentence length and structure to improve rhythm and emphasis.")
    elif position("complex_sentence_ratio") == "high":
        strengths.append("The response uses complex sentence structures.")
    if features["flesch_reading_ease"] < 25 and features["avg_sentence_length"] > 24:
        weaknesses.append(
            "Shorten or divide dense sentences to make the argument easier to follow."
        )

    if not strengths:
        best = min(
            ("Vocabulary is reasonably varied.", abs(features["type_token_ratio"] - 0.5)),
            (
                "The response shows a foundation for organized development.",
                abs(features["paragraph_count"] - 2),
            ),
            key=lambda item: item[1],
        )[0]
        strengths.append(best)
    if not weaknesses:
        weaknesses.append(
            "A final revision for precision, transitions, and sentence variety "
            "could strengthen the response."
        )

    grammar_rate = features["grammar_errors_per_100_words"]
    return {
        "strengths": strengths[:4],
        "weaknesses": weaknesses[:4],
        "quality": {
            "grammar": "strong"
            if grammar_rate <= 2
            else "developing"
            if grammar_rate <= 6
            else "needs attention",
            "vocabulary": position("type_token_ratio"),
            "readability": "accessible"
            if 40 <= features["flesch_reading_ease"] <= 80
            else "dense or atypical",
            "organization": "structured" if features["paragraph_count"] >= 3 else "limited",
        },
    }
