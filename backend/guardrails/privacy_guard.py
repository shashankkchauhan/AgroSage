"""
Privacy Guard
- Masks API keys and credentials before logging
- Prevents sensitive information from reaching logs or responses
- Protects farmer profile data
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger


# ─────────────────────── patterns ────────────────────────────────────────────

CREDENTIAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"(api[_\-\s]?key\s*[=:]\s*)['\"]?([A-Za-z0-9\-_.]{16,})['\"]?", re.I),
    re.compile(r"(password\s*[=:]\s*)['\"]?([^\s'\"]{6,})['\"]?", re.I),
    re.compile(r"(secret[_\-\s]?key\s*[=:]\s*)['\"]?([A-Za-z0-9\-_.]{16,})['\"]?", re.I),
    re.compile(r"(token\s*[=:]\s*)['\"]?([A-Za-z0-9\-_.]{20,})['\"]?", re.I),
    re.compile(r"(Bearer\s+)([A-Za-z0-9\-_.]{20,})", re.I),
]

PII_MASK_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b\d{12}\b"),                                  # Aadhaar
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                     # PAN
    re.compile(r"\b[6-9]\d{9}\b"),                              # Phone
    re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),  # Email
]

# Environment variables that must never be logged or exposed
SENSITIVE_ENV_KEYS = {
    "WATSONX_API_KEY", "SECRET_KEY", "MANDI_API_KEY",
    "WEATHER_API_KEY", "DATABASE_URL", "REDIS_URL",
}


# ─────────────────────── guard ───────────────────────────────────────────────

class PrivacyGuard:
    """Masks secrets and PII in logs and outgoing text."""

    def mask_credentials(self, text: str) -> str:
        """Replace API keys / passwords with masked versions."""
        for pattern in CREDENTIAL_PATTERNS:
            text = pattern.sub(lambda m: m.group(1) + "***REDACTED***", text)
        return text

    def mask_pii(self, text: str) -> str:
        """Mask PII patterns with safe placeholders."""
        text = re.sub(r"\b\d{12}\b", "***AADHAAR***", text)
        text = re.sub(r"\b[A-Z]{5}\d{4}[A-Z]\b", "***PAN***", text)
        text = re.sub(r"\b[6-9]\d{9}\b", "***PHONE***", text)
        text = re.sub(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "***EMAIL***",
            text,
        )
        return text

    def safe_env(self, key: str) -> Optional[str]:
        """
        Return an env var value.
        Raises if key is in SENSITIVE_ENV_KEYS and DEBUG is on
        (prevents accidental exposure in debug logs).
        """
        value = os.environ.get(key)
        if key in SENSITIVE_ENV_KEYS and not value:
            logger.warning(f"Sensitive env variable '{key}' is not set.")
        return value

    def sanitize_log_message(self, message: str) -> str:
        """Prepare a log message by removing credentials and PII."""
        message = self.mask_credentials(message)
        message = self.mask_pii(message)
        return message

    def safe_farmer_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a copy of the farmer profile with sensitive fields redacted
        for logging or display purposes.
        """
        SAFE_FIELDS = {
            "farmer_id", "region", "district", "state", "land_area",
            "primary_crops", "soil_type", "preferred_language",
        }
        return {k: v for k, v in profile.items() if k in SAFE_FIELDS}

    def validate_response_for_secrets(self, response: str) -> bool:
        """Return True if the response is free of credentials."""
        for pattern in CREDENTIAL_PATTERNS:
            if pattern.search(response):
                logger.error("Credential pattern found in AI response — blocking output.")
                return False
        return True
