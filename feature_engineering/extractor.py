"""Interpretable linguistic and surface feature extraction."""

from __future__ import annotations

import logging
import math
import re
import string
from collections import Counter
from typing import Any

import numpy as np

from preprocessing.text import normalize_text

LOGGER = logging.getLogger(__name__)
_TEXTSTAT_USABLE: bool | None = None
WORD_RE = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TRANSITIONS = {
    "however",
    "therefore",
    "furthermore",
    "moreover",
    "consequently",
    "meanwhile",
    "firstly",
    "secondly",
    "finally",
    "although",
    "because",
    "similarly",
    "instead",
    "otherwise",
    "nevertheless",
    "additionally",
    "conclusion",
    "overall",
}

FEATURE_NAMES = [
    "word_count",
    "character_count",
    "paragraph_count",
    "sentence_count",
    "avg_sentence_length",
    "sentence_length_std",
    "avg_word_length",
    "unique_words",
    "type_token_ratio",
    "root_type_token_ratio",
    "hapax_ratio",
    "repetition_ratio",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "grammar_error_count",
    "grammar_errors_per_100_words",
    "spelling_error_count",
    "spelling_errors_per_100_words",
    "passive_voice_ratio",
    "transition_count",
    "transitions_per_100_words",
    "comma_count",
    "semicolon_count",
    "colon_count",
    "question_count",
    "exclamation_count",
    "punctuation_per_100_words",
    "complex_sentence_ratio",
    "avg_paragraph_words",
    "paragraph_length_std",
]


class WritingFeatureExtractor:
    """Extract deterministic features, with graceful fallbacks for optional NLP tools."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_grammar: bool = True,
        enable_spelling: bool = True,
        language: str = "en-US",
    ) -> None:
        self.spacy_model = spacy_model
        self.enable_grammar = enable_grammar
        self.enable_spelling = enable_spelling
        self.language = language
        self._nlp: Any = None
        self._language_tool: Any = None
        self._spellchecker: Any = None

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()

    def _load_spacy(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy

            try:
                self._nlp = spacy.load(self.spacy_model, disable=["ner"])
            except OSError:
                LOGGER.warning(
                    "spaCy model %s unavailable; using rule-based English pipeline",
                    self.spacy_model,
                )
                self._nlp = spacy.blank("en")
                self._nlp.add_pipe("sentencizer")
        except ImportError:
            LOGGER.warning("spaCy unavailable; regex sentence analysis will be used")
            self._nlp = False
        return self._nlp

    def _grammar_errors(self, text: str) -> int:
        if not self.enable_grammar:
            return 0
        if self._language_tool is None:
            try:
                import language_tool_python

                self._language_tool = language_tool_python.LanguageTool(self.language)
            except Exception as exc:
                LOGGER.warning(
                    "LanguageTool unavailable; using rule-based grammar estimates: %s", exc
                )
                self._language_tool = False
        if not self._language_tool:
            return self._rule_based_grammar_errors(text)
        try:
            return len(self._language_tool.check(text))
        except Exception as exc:
            LOGGER.warning("Grammar check failed; using rule-based estimates: %s", exc)
            return self._rule_based_grammar_errors(text)

    @staticmethod
    def _rule_based_grammar_errors(text: str) -> int:
        """Estimate common mechanical errors when Java LanguageTool is unavailable."""
        errors = 0
        errors += len(re.findall(r"\b([A-Za-z]+)\s+\1\b", text, flags=re.IGNORECASE))
        errors += len(re.findall(r"\bi\b", text))
        errors += len(re.findall(r"\s+[,.!?;:]", text))
        errors += len(re.findall(r"[!?.,]{2,}", text))
        errors += len(re.findall(r"\b(?:he|she|it)\s+(?:are|were|have|do)\b", text, re.IGNORECASE))
        errors += len(re.findall(r"\b(?:they|we|you)\s+(?:is|was|has|does)\b", text, re.IGNORECASE))
        errors += len(re.findall(r"\ba\s+[aeiou]\w*\b", text, re.IGNORECASE))
        errors += len(re.findall(r"\ban\s+[^aeiou\W]\w*\b", text, re.IGNORECASE))
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        for paragraph in paragraphs:
            if paragraph and paragraph[-1] not in ".!?\"'”’":
                errors += 1
        sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
        for sentence in sentences:
            words = WORD_RE.findall(sentence)
            if words and words[0][0].islower():
                errors += 1
            if len(words) > 45:
                errors += 1
        return errors

    def _spelling_errors(self, words: list[str]) -> int:
        if not self.enable_spelling:
            return 0
        if self._spellchecker is None:
            try:
                from spellchecker import SpellChecker

                self._spellchecker = SpellChecker(distance=1)
            except ImportError:
                LOGGER.warning("pyspellchecker unavailable; spelling features set to zero")
                self._spellchecker = False
        if not self._spellchecker:
            return 0
        alphabetic = [word.lower() for word in words if len(word) > 1]
        return len(self._spellchecker.unknown(alphabetic))

    @staticmethod
    def _readability(text: str) -> tuple[float, float]:
        global _TEXTSTAT_USABLE
        if _TEXTSTAT_USABLE is not False:
            try:
                import nltk
                import textstat

                nltk.data.find("corpora/cmudict")
                result = (
                    float(textstat.flesch_reading_ease(text)),
                    float(textstat.flesch_kincaid_grade(text)),
                )
                _TEXTSTAT_USABLE = True
                return result
            except Exception:
                _TEXTSTAT_USABLE = False
                LOGGER.warning("textstat resources unavailable; using built-in Flesch formulas")
        words = WORD_RE.findall(text)
        sentences = [part for part in SENTENCE_RE.split(text) if part.strip()]
        if not words:
            return 0.0, 0.0
        sentence_count = max(len(sentences), 1)
        syllable_count = sum(WritingFeatureExtractor._estimate_syllables(word) for word in words)
        words_per_sentence = len(words) / sentence_count
        syllables_per_word = syllable_count / len(words)
        reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
        return float(reading_ease), float(max(grade, 0.0))

    @staticmethod
    def _estimate_syllables(word: str) -> int:
        """Estimate English syllables for the built-in readability fallback."""
        value = re.sub(r"[^a-z]", "", word.lower())
        if not value:
            return 1
        groups = len(re.findall(r"[aeiouy]+", value))
        if value.endswith("e") and not value.endswith(("le", "ye")) and groups > 1:
            groups -= 1
        if value.endswith("es") and len(value) > 3 and groups > 1:
            groups -= 1
        return max(groups, 1)

    def extract(self, text: str) -> dict[str, float]:
        """Extract all features from one essay without altering its mistakes."""
        text = normalize_text(text)
        words = WORD_RE.findall(text)
        lower_words = [word.lower() for word in words]
        word_count = len(words)
        safe_words = max(word_count, 1)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]

        nlp = self._load_spacy()
        doc = nlp(text) if nlp else None
        if doc is not None:
            sentences = [sent for sent in doc.sents if sent.text.strip()]
            sentence_lengths = [len(WORD_RE.findall(sent.text)) for sent in sentences]
            passive = sum(
                1
                for sent in sentences
                if any(token.dep_ in {"auxpass", "nsubjpass"} for token in sent)
            )
            complex_count = sum(
                1
                for sent in sentences
                if any(token.dep_ in {"advcl", "ccomp", "xcomp", "relcl"} for token in sent)
            )
        else:
            sentence_texts = [part for part in SENTENCE_RE.split(text) if part.strip()] or (
                [text] if text else []
            )
            sentence_lengths = [len(WORD_RE.findall(sent)) for sent in sentence_texts]
            passive = 0
            complex_count = sum(1 for sent in sentence_texts if len(re.findall(r"[,;:]", sent)) > 0)
        sentence_count = len(sentence_lengths)
        unique_count = len(set(lower_words))
        frequencies = Counter(lower_words)
        repeated_tokens = sum(max(count - 1, 0) for count in frequencies.values())
        hapax = sum(count == 1 for count in frequencies.values())
        paragraph_lengths = [len(WORD_RE.findall(part)) for part in paragraphs]
        grammar_errors = self._grammar_errors(text)
        spelling_errors = self._spelling_errors(words)
        transition_count = sum(word in TRANSITIONS for word in lower_words)
        punctuation_count = sum(char in string.punctuation for char in text)
        reading_ease, grade = self._readability(text)

        values = {
            "word_count": word_count,
            "character_count": len(text),
            "paragraph_count": len(paragraphs),
            "sentence_count": sentence_count,
            "avg_sentence_length": np.mean(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_std": np.std(sentence_lengths) if sentence_lengths else 0,
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "unique_words": unique_count,
            "type_token_ratio": unique_count / safe_words,
            "root_type_token_ratio": unique_count / math.sqrt(safe_words),
            "hapax_ratio": hapax / safe_words,
            "repetition_ratio": repeated_tokens / safe_words,
            "flesch_reading_ease": reading_ease,
            "flesch_kincaid_grade": grade,
            "grammar_error_count": grammar_errors,
            "grammar_errors_per_100_words": grammar_errors * 100 / safe_words,
            "spelling_error_count": spelling_errors,
            "spelling_errors_per_100_words": spelling_errors * 100 / safe_words,
            "passive_voice_ratio": passive / max(sentence_count, 1),
            "transition_count": transition_count,
            "transitions_per_100_words": transition_count * 100 / safe_words,
            "comma_count": text.count(","),
            "semicolon_count": text.count(";"),
            "colon_count": text.count(":"),
            "question_count": text.count("?"),
            "exclamation_count": text.count("!"),
            "punctuation_per_100_words": punctuation_count * 100 / safe_words,
            "complex_sentence_ratio": complex_count / max(sentence_count, 1),
            "avg_paragraph_words": np.mean(paragraph_lengths) if paragraph_lengths else 0,
            "paragraph_length_std": np.std(paragraph_lengths) if paragraph_lengths else 0,
        }
        return {name: float(values[name]) for name in FEATURE_NAMES}

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform essays into a stable ordered numeric matrix."""
        return np.asarray(
            [[features[name] for name in FEATURE_NAMES] for features in map(self.extract, texts)]
        )

    def close(self) -> None:
        """Release external LanguageTool resources."""
        if self._language_tool and hasattr(self._language_tool, "close"):
            self._language_tool.close()
