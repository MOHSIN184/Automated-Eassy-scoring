"""Artifact-backed inference pipeline for the Streamlit application."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from feature_engineering import WritingFeatureExtractor
from feature_engineering.feedback import generate_feedback
from preprocessing.text import compose_baseline_input, normalize_text
from utils.config import Config, load_config, resolve_path

LOGGER = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    """Raised when trained artifacts are absent or inconsistent."""


class AESPipeline:
    """Load the shipped baseline artifacts once and score essays on CPU."""

    def __init__(
        self, config_path: str | Path = "config.yaml", artifact_dir: str | Path | None = None
    ) -> None:
        self.config: Config = load_config(config_path)
        self.artifact_dir = (
            Path(artifact_dir).resolve()
            if artifact_dir
            else resolve_path(self.config, "paths.artifact_dir")
        )
        metadata_path = self.artifact_dir / "label_metadata.json"
        if not metadata_path.exists():
            raise ArtifactError(
                f"Model artifacts not found in {self.artifact_dir}. Run `python train.py` first."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model_type = str(self.metadata.get("model_type", ""))
        if self.model_type != "quick_baseline":
            raise ArtifactError(
                "The Streamlit application requires quick-baseline model artifacts."
            )
        self.feature_names = list(self.metadata["feature_names"])
        self.scaler = joblib.load(self.artifact_dir / "feature_scaler.joblib")
        self.vectorizer = joblib.load(self.artifact_dir / "text_vectorizer.joblib")
        self.model = joblib.load(self.artifact_dir / "baseline_model.joblib")
        feature_settings = self.metadata.get("feature_settings", {})
        self.extractor = WritingFeatureExtractor(
            spacy_model=str(self.config.require("features.spacy_model")),
            enable_grammar=bool(
                feature_settings.get(
                    "enable_grammar", self.config.get("features.enable_grammar", True)
                )
            ),
            enable_spelling=bool(
                feature_settings.get(
                    "enable_spelling", self.config.get("features.enable_spelling", True)
                )
            ),
            language=str(self.config.get("features.language_tool_language", "en-US")),
        )
        self.diagnostic_extractor = (
            WritingFeatureExtractor(
                spacy_model=str(self.config.require("features.spacy_model")),
                enable_grammar=True,
                enable_spelling=True,
                language=str(self.config.get("features.language_tool_language", "en-US")),
            )
            if self.model_type == "quick_baseline"
            else self.extractor
        )

    def predict(
        self,
        essay: str,
        prompt_name: str = "",
        assignment: str = "",
        source_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Score one essay and return confidence, statistics, and feature-driven feedback."""
        return self._predict_baseline(essay, prompt_name, assignment, source_texts or [])

    def _predict_baseline(
        self, essay: str, prompt_name: str, assignment: str, source_texts: list[str]
    ) -> dict[str, Any]:
        """Run the trained lightweight TF-IDF plus writing-feature baseline."""
        from scipy.sparse import csr_matrix, hstack

        essay = normalize_text(essay)
        if not essay:
            raise ValueError("Essay must contain text")
        model_features = self.extractor.extract(essay)
        raw_features = self.diagnostic_extractor.extract(essay)
        ordered = np.asarray(
            [[model_features[name] for name in self.feature_names]], dtype=np.float32
        )
        scaled = self.scaler.transform(ordered).astype(np.float32)
        model_text = compose_baseline_input(essay, prompt_name, assignment, source_texts)
        text_features = self.vectorizer.transform([model_text])
        combined = hstack([text_features, csr_matrix(scaled)], format="csr")
        prediction = float(self.model.predict(combined)[0])
        prediction = float(
            np.clip(prediction, self.metadata["score_min"], self.metadata["score_max"])
        )
        confidence, uncertainty = self._baseline_confidence(prediction, prompt_name)
        coefficients = np.asarray(self.model.coef_)[-len(self.feature_names) :]
        contributions = np.abs(coefficients * scaled[0])
        total = float(contributions.sum())
        normalized = contributions / total if total else contributions
        ranked = sorted(
            zip(self.feature_names, normalized.tolist()),  # noqa: B905 - Python 3.9 support
            key=lambda item: item[1],
            reverse=True,
        )
        return {
            "score": round(prediction, 3),
            "confidence": round(confidence, 3),
            "uncertainty": round(uncertainty, 4),
            "feedback": generate_feedback(raw_features, self.metadata.get("feature_quantiles", {})),
            "statistics": {name: round(value, 4) for name, value in raw_features.items()},
            "feature_importance": {name: round(float(value), 4) for name, value in ranked[:10]},
        }

    def _baseline_confidence(self, prediction: float, prompt_name: str) -> tuple[float, float]:
        """Look up held-out empirical confidence for a local prediction band."""
        calibration = self.metadata.get("confidence_calibration")
        if not calibration:
            uncertainty = float(self.metadata.get("validation_rmse", 1.0))
            score_range = float(self.metadata["score_max"] - self.metadata["score_min"])
            confidence = float(np.clip(1.0 - uncertainty / score_range, 0.0, 1.0))
            return confidence, uncertainty
        width = float(calibration["bin_width"])
        score_min = float(self.metadata["score_min"])
        bucket = int(np.floor((prediction - score_min) / width))
        prompt_key = f"{prompt_name}::{bucket}"
        group = calibration.get("by_prompt_bin", {}).get(prompt_key)
        if group is None:
            group = calibration.get("by_bin", {}).get(str(bucket))
        if group is None:
            group = calibration["overall"]
        return float(group["confidence"]), float(group["rmse"])

    def predict_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score a bounded collection of essays."""
        maximum = int(self.config.get("inference.max_batch_size", 32))
        if not items or len(items) > maximum:
            raise ValueError(f"Batch size must be between 1 and {maximum}")
        return [
            self.predict(
                essay=str(item.get("essay", "")),
                prompt_name=str(item.get("prompt_name", "")),
                assignment=str(item.get("assignment", "")),
                source_texts=list(item.get("source_texts", [])),
            )
            for item in items
        ]
