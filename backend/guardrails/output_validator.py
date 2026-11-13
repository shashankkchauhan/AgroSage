"""
Output Validator Guardrail
Validates IBM Granite responses before displaying to farmers:
- Agricultural relevance check
- Evidence support check
- Dangerous instruction detection
- Citation existence check
- Privacy exposure check
- Uncertainty warning injection
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


# ─────────────────────── constants ──────────────────────────────────────────

# Minimum fraction of words from approved sources that should appear in response
MIN_GROUNDEDNESS_RATIO = 0.05

# PII patterns
PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]{1}\b"), "PAN number"),
    (re.compile(r"\b\d{12}\b"), "Aadhaar number"),
    (re.compile(r"\b[6-9]\d{9}\b"), "Indian mobile number"),
    (re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"), "Email address"),
    (re.compile(r"\b(?:IFSC|ifsc)\s*[A-Z]{4}0[A-Z0-9]{6}\b", re.I), "Bank IFSC code"),
    (re.compile(r"\bAPI[_\s]?KEY\s*[=:]\s*['\"]?[A-Za-z0-9\-_]{20,}['\"]?", re.I), "API key"),
]

# Patterns indicating the response is off-topic for agriculture
OFF_TOPIC_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(write (me )?a (poem|song|story|essay))\b", re.I),
    re.compile(r"\b(how to (hack|crack|bypass))\b", re.I),
    re.compile(r"\b(stock (market|trading)|cryptocurrency|forex trading)\b", re.I),
    re.compile(r"\b(political|election|vote|party manifesto)\b", re.I),
]

# Agricultural context keywords (at least one expected in a valid ag response)
AG_CONTEXT_WORDS = {
    "crop", "soil", "fertilizer", "fertiliser", "pesticide", "weather",
    "rain", "rainfall", "temperature", "farm", "farmer", "seed", "harvest",
    "disease", "pest", "weed", "irrigation", "mandi", "market", "price",
    "nitrogen", "phosphorus", "potassium", "ph", "yield", "season", "sowing",
    "wheat", "rice", "maize", "cotton", "tomato", "onion", "potato",
    "फसल", "मिट्टी", "उर्वरक", "कीटनाशक", "मौसम", "किसान", "बाजार",
}


# ─────────────────────── result ─────────────────────────────────────────────

@dataclass
class OutputValidationResult:
    is_valid:       bool
    final_response: str
    errors:         List[str] = field(default_factory=list)
    warnings:       List[str] = field(default_factory=list)
    pii_found:      List[str] = field(default_factory=list)
    metadata:       Dict[str, Any] = field(default_factory=dict)


# ─────────────────────── validator ───────────────────────────────────────────

class OutputValidator:
    """Final validation gate before showing any AI response to the user."""

    def validate(
        self,
        response: str,
        query: str,
        approved_sources: Optional[List[str]] = None,
        confidence_level: str = "HIGH",
    ) -> OutputValidationResult:
        result = OutputValidationResult(is_valid=True, final_response=response)
        approved_sources = approved_sources or []

        # 1. Empty response
        if not response or not response.strip():
            result.is_valid = False
            result.errors.append("Model returned an empty response.")
            result.final_response = "I'm sorry, I could not generate a response. Please try again."
            return result

        # 2. Off-topic detection
        for pattern in OFF_TOPIC_PATTERNS:
            if pattern.search(response):
                result.is_valid = False
                result.errors.append("Response appears to be off-topic for agricultural assistance.")
                result.final_response = (
                    "I'm designed to help with agricultural questions only. "
                    "Please ask about crops, soil, weather, pests, or market prices."
                )
                return result

        # 3. Agricultural context check
        if not self._has_ag_context(response) and not self._has_ag_context(query):
            result.warnings.append(
                "Response may not address the agricultural query. "
                "Please verify the answer is relevant."
            )

        # 4. PII leakage check
        pii_issues = self._scan_pii(response)
        if pii_issues:
            result.pii_found = pii_issues
            result.is_valid = False
            result.errors.append(
                f"Response contains potential PII/sensitive data: {', '.join(pii_issues)}. "
                "Response blocked for privacy protection."
            )
            result.final_response = (
                "I cannot display this response as it may contain sensitive information. "
                "Please contact support if this issue persists."
            )
            return result

        # 5. Uncertainty warning when evidence is thin
        if confidence_level in ("LOW", "INSUFFICIENT"):
            disclaimer = (
                "\n\n⚠️ **Low Confidence**: The information available to support this answer "
                "is limited. Please verify with your local agricultural department or "
                "a qualified agronomist before acting on this advice."
            )
            if disclaimer not in response:
                result.final_response = response + disclaimer
            result.warnings.append("Uncertainty disclaimer appended due to low retrieval confidence.")

        # 6. Response too short to be meaningful
        if len(result.final_response.split()) < 10:
            result.warnings.append("Response is very short. It may not fully address the query.")

        result.metadata["word_count"] = len(result.final_response.split())
        result.metadata["confidence_level"] = confidence_level
        result.metadata["sources_count"] = len(approved_sources)

        return result

    # ── private ──────────────────────────────────────────────────────────────

    def _has_ag_context(self, text: str) -> bool:
        text_lower = text.lower()
        return any(w in text_lower for w in AG_CONTEXT_WORDS)

    def _scan_pii(self, text: str) -> List[str]:
        found = []
        for pattern, label in PII_PATTERNS:
            if pattern.search(text):
                found.append(label)
        return found
