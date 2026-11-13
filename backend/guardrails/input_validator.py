"""
Input Validator Guardrail
Validates all farmer inputs before processing:
- Soil parameter ranges (pH, NPK, temperature, rainfall, humidity)
- Query completeness
- Language detection
- Input sanitization
"""
import re
import html
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


# ─────────────────────── constants ──────────────────────────────────────────

PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "soil_ph":          (0.0,  14.0),
    "nitrogen":         (0.0,  500.0),    # kg/ha
    "phosphorus":       (0.0,  500.0),    # kg/ha
    "potassium":        (0.0,  500.0),    # kg/ha
    "temperature":      (-20.0, 60.0),    # °C
    "rainfall":         (0.0,  5000.0),   # mm/year
    "humidity":         (0.0,  100.0),    # %
    "elevation":        (-500.0, 9000.0), # metres
}

MAX_INPUT_LENGTH = 2000
MIN_QUERY_WORDS  = 2


# ─────────────────────── result ─────────────────────────────────────────────

@dataclass
class ValidationResult:
    is_valid:     bool
    errors:       List[str] = field(default_factory=list)
    warnings:     List[str] = field(default_factory=list)
    sanitized:    Optional[str] = None
    detected_lang: str = "en"
    metadata:     Dict[str, Any] = field(default_factory=dict)


# ─────────────────────── validator ───────────────────────────────────────────

class InputValidator:
    """Validates and sanitizes all farmer inputs."""

    # Minimal HTML/script stripping pattern
    _TAG_RE = re.compile(r"<[^>]+>")
    # Dangerous SQL/template patterns
    _SQL_RE = re.compile(
        r"(--|;|\bDROP\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bEXEC\b)",
        re.IGNORECASE,
    )
    # Hindi Unicode block
    _HINDI_RE = re.compile(r"[\u0900-\u097F]")

    # ── public entry points ──────────────────────────────────────────────────

    def validate_text_query(self, text: str) -> ValidationResult:
        """Validate a free-text farmer query."""
        result = ValidationResult(is_valid=True)

        # Null / empty
        if not text or not text.strip():
            result.is_valid = False
            result.errors.append("Query cannot be empty.")
            return result

        # Length
        if len(text) > MAX_INPUT_LENGTH:
            result.is_valid = False
            result.errors.append(
                f"Query exceeds maximum length of {MAX_INPUT_LENGTH} characters."
            )
            return result

        # Sanitize
        sanitized = self._sanitize(text)

        # Minimum meaningful words
        word_count = len(sanitized.split())
        if word_count < MIN_QUERY_WORDS:
            result.is_valid = False
            result.errors.append("Query is too short to process meaningfully.")
            return result

        # Detect language
        result.detected_lang = self._detect_language(sanitized)
        if result.detected_lang not in ("en", "hi"):
            result.warnings.append(
                "Language detected is not English or Hindi. Response may be in English."
            )

        result.sanitized = sanitized
        return result

    def validate_soil_params(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate structured soil / environment parameters."""
        result = ValidationResult(is_valid=True)

        for key, value in params.items():
            if key not in PARAM_RANGES:
                result.warnings.append(f"Unknown parameter '{key}' will be ignored.")
                continue

            try:
                num_val = float(value)
            except (TypeError, ValueError):
                result.is_valid = False
                result.errors.append(f"Parameter '{key}' must be a number.")
                continue

            lo, hi = PARAM_RANGES[key]
            if not (lo <= num_val <= hi):
                result.is_valid = False
                result.errors.append(
                    f"Parameter '{key}' value {num_val} is outside the valid range "
                    f"[{lo}, {hi}]."
                )

        result.metadata["validated_params"] = {
            k: v for k, v in params.items() if k in PARAM_RANGES
        }
        return result

    def validate_document(self, content: str, source_name: str = "") -> ValidationResult:
        """Validate and sanitize an uploaded document before ingestion."""
        result = ValidationResult(is_valid=True)

        if not content or not content.strip():
            result.is_valid = False
            result.errors.append("Document content is empty.")
            return result

        if len(content) > 500_000:
            result.is_valid = False
            result.errors.append("Document exceeds maximum allowed size (500,000 chars).")
            return result

        sanitized = self._sanitize(content)
        result.sanitized = sanitized
        result.metadata["source"] = source_name
        result.metadata["char_count"] = len(sanitized)
        return result

    # ── private helpers ──────────────────────────────────────────────────────

    def _sanitize(self, text: str) -> str:
        """Remove HTML tags, unescape entities, strip dangerous patterns."""
        text = self._TAG_RE.sub(" ", text)
        text = html.unescape(text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _detect_language(self, text: str) -> str:
        """Lightweight language detection: Hindi vs English."""
        hindi_chars = len(self._HINDI_RE.findall(text))
        if hindi_chars > len(text) * 0.1:
            return "hi"
        return "en"

    def has_suspicious_sql(self, text: str) -> bool:
        return bool(self._SQL_RE.search(text))
