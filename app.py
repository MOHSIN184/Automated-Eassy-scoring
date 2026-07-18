"""Responsive Streamlit dashboard for essay scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Keep imports and runtime files anchored to the application directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import AESPipeline, ArtifactError  # noqa: E402

st.set_page_config(page_title="Essay Scoring Studio", page_icon="✍️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(145deg, #f7f9fc 0%, #eef3ff 100%);}
    [data-testid="stMetric"] {
        background: white; border: 1px solid #dbe4f0;
        padding: 1rem; border-radius: 14px;
    }
    .feedback {
        background: white; border-left: 5px solid #4968db;
        padding: 1rem 1.25rem; border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline() -> AESPipeline:
    return AESPipeline()


def readability_label(score: float) -> str:
    """Translate a Flesch Reading Ease score into plain language."""
    if score >= 90:
        return "Very easy"
    if score >= 80:
        return "Easy"
    if score >= 70:
        return "Fairly easy"
    if score >= 60:
        return "Standard"
    if score >= 50:
        return "Fairly difficult"
    if score >= 30:
        return "Difficult"
    return "Very difficult"


def rate_label(value: float, low: float, high: float) -> str:
    """Give an issue frequency a compact qualitative label."""
    if value <= low:
        return "Low"
    if value <= high:
        return "Moderate"
    return "High"


st.title("✍️ Essay Scoring Studio")
st.caption("Hybrid transformer and linguistic-feature analysis for formative writing feedback")

with st.sidebar:
    st.header("Optional assignment context")
    prompt_name = st.text_input("Prompt name")
    assignment = st.text_area("Assignment", height=120)
    source_text = st.text_area("Source text", height=180)

essay = st.text_area("Student essay", height=330, placeholder="Paste the original essay here…")
score_clicked = st.button("Score essay", type="primary", use_container_width=True)

if score_clicked:
    if not essay.strip():
        st.warning("Enter an essay before scoring.")
        st.stop()
    try:
        pipeline = load_pipeline()
        with st.spinner("Analyzing writing quality…"):
            result = pipeline.predict(
                essay,
                prompt_name=prompt_name,
                assignment=assignment,
                source_texts=[source_text] if source_text.strip() else [],
            )
    except ArtifactError as exc:
        st.error(str(exc))
        st.info(
            "Restore the required files under `checkpoints/best_model`, then restart Streamlit."
        )
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    stats = result["statistics"]
    score_min, score_max = pipeline.metadata["score_min"], pipeline.metadata["score_max"]
    first, second, third, fourth = st.columns(4)
    first.metric("Predicted score", f"{result['score']:.2f} / {score_max:g}")
    second.metric("Confidence", f"{result['confidence']:.0%}")
    third.metric("Words", f"{stats['word_count']:.0f}")
    fourth.metric("Sentences", f"{stats['sentence_count']:.0f}")
    if pipeline.metadata.get("model_type") == "quick_baseline":
        st.info("Running the CPU quick baseline.")
    st.progress(
        max(0.0, min(1.0, (result["score"] - score_min) / (score_max - score_min))),
        text="Position on the scoring scale",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Writing profile")
        readability = stats["flesch_reading_ease"]
        vocabulary_percent = stats["type_token_ratio"] * 100
        vocabulary_label = (
            "Highly varied"
            if vocabulary_percent >= 60
            else "Varied"
            if vocabulary_percent >= 45
            else "Limited"
        )
        grammar_rate = stats["grammar_errors_per_100_words"]
        spelling_rate = stats["spelling_errors_per_100_words"]
        transition_rate = stats["transitions_per_100_words"]
        profile = pd.DataFrame(
            {
                "Metric": [
                    "Vocabulary richness",
                    "Readability",
                    "Possible grammar issues",
                    "Possible spelling issues",
                    "Explicit transitions",
                ],
                "Value": [
                    f"{vocabulary_label} ({vocabulary_percent:.1f}% unique words)",
                    f"{readability_label(readability)} (Flesch {readability:.1f})",
                    f"{rate_label(grammar_rate, 2, 6)} ({grammar_rate:.1f} per 100 words)",
                    f"{rate_label(spelling_rate, 2, 5)} ({spelling_rate:.1f} per 100 words)",
                    (
                        "None detected"
                        if transition_rate == 0
                        else f"{transition_rate:.1f} per 100 words"
                    ),
                ],
            }
        ).set_index("Metric")
        st.dataframe(profile, use_container_width=True)
        st.caption(
            "Flesch Reading Ease usually ranges from 0 (very difficult) to 100 "
            "(very easy); exceptionally dense writing can score below 0. "
            "Grammar, spelling, and transition figures are automated estimates, not definitive errors."
        )
    with right:
        st.subheader("Local feature importance")
        importance = pd.Series(result["feature_importance"], name="Influence").sort_values()
        st.bar_chart(importance, horizontal=True)

    strengths, weaknesses = st.columns(2)
    with strengths:
        st.subheader("Strengths")
        st.markdown("\n".join(f"- {item}" for item in result["feedback"]["strengths"]))
    with weaknesses:
        st.subheader("Needs improvement")
        st.markdown("\n".join(f"- {item}" for item in result["feedback"]["weaknesses"]))

    with st.expander("All writing statistics"):
        st.json(stats)
