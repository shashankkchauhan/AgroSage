"""
Prompt Injection Detector
Detects and blocks attempts to override system instructions via user inputs
or retrieved documents.
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple
from loguru import logger


# ─────────────────────── injection patterns ──────────────────────────────────

# Each entry: (pattern, severity, description)
INJECTION_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # Classic ignore-previous instructions
    (re.compile(r"\bignore\b.{0,30}\b(previous|above|prior|system)\b.{0,30}\b(instructions?|prompt|rules?)\b", re.I),
     "HIGH", "Attempt to ignore system instructions"),

    # Role override
    (re.compile(r"\b(you are now|act as|pretend (to be|you are)|roleplay as)\b.{0,50}\b(a?n?\s*)?(unrestricted|jailbroken|dan|evil|hacker|root)\b", re.I),
     "HIGH", "Role override / jailbreak attempt"),

    # Forget / disregard
    (re.compile(r"\b(forget|disregard|override|bypass|disable)\b.{0,30}\b(all|your|previous|safety|guardrail|rule|filter|restriction|constraint)\b", re.I),
     "HIGH", "Instruction override attempt"),

    # System prompt exfiltration
    (re.compile(r"\b(print|reveal|show|output|repeat|tell me).{0,30}\b(system prompt|initial prompt|your (instructions?|rules?|configuration))\b", re.I),
     "HIGH", "System prompt extraction attempt"),

    # Indirect injection via document
    (re.compile(r"<\|.{0,30}\|>", re.I),
     "MEDIUM", "Special token injection in document"),

    # HTML/script injection
    (re.compile(r"<script[\s\S]*?>[\s\S]*?</script>", re.I),
     "HIGH", "Script tag injection"),

    # Prompt continuation tricks
    (re.compile(r"(---+|===+|\*\*\*+)\s*NEW INSTRUCTIONS?\s*(---+|===+|\*\*\*+)", re.I),
     "MEDIUM", "Delimiter-based instruction injection"),

    # Command execution tries
    (re.compile(r"\b(exec|eval|os\.system|subprocess|__import__)\s*\(", re.I),
     "HIGH", "Code execution injection"),

    # SQL injection
    (re.compile(r"(\bDROP\s+TABLE\b|\bDELETE\s+FROM\b|\bINSERT\s+INTO\b)", re.I),
     "HIGH", "SQL injection"),

    # Indirect: "as a developer mode"
    (re.compile(r"\b(developer mode|debug mode|admin mode|maintenance mode)\b", re.I),
     "MEDIUM", "Mode-switching attempt"),
]

# Patterns specific to injections inside RETRIEVED documents (stricter)
DOCUMENT_INJECTION_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"\[SYSTEM\]|\[INST\]|<\|system\|>|<\|user\|>|<\|assistant\|>", re.I),
     "HIGH", "LLM template token in retrieved document"),
    (re.compile(r"Human:\s|Assistant:\s|System:\s", re.I),
     "MEDIUM", "Conversation role injection in document"),
]


# ─────────────────────── result ─────────────────────────────────────────────

@dataclass
class InjectionResult:
    is_injection: bool
    severity: str = "NONE"           # NONE | LOW | MEDIUM | HIGH
    detections: List[str] = field(default_factory=list)
    sanitized_text: str = ""


# ─────────────────────── detector ────────────────────────────────────────────

class PromptInjectionDetector:
    """Detects prompt injection in user queries and retrieved documents."""

    def check_user_input(self, text: str) -> InjectionResult:
        """Check a user-supplied query for injection attempts."""
        return self._run_checks(text, INJECTION_PATTERNS)

    def check_retrieved_document(self, text: str) -> InjectionResult:
        """Check a document chunk from the vector store for embedded instructions."""
        result = self._run_checks(text, INJECTION_PATTERNS + DOCUMENT_INJECTION_PATTERNS)
        return result

    def sanitize_for_prompt(self, text: str) -> str:
        """
        Return a version of text safe to include in an LLM prompt.
        Strips known injection tokens and limits special characters.
        """
        # Remove LLM template tokens
        for pattern, _, _ in DOCUMENT_INJECTION_PATTERNS:
            text = pattern.sub("", text)
        # Remove HTML
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── private ──────────────────────────────────────────────────────────────

    def _run_checks(
        self,
        text: str,
        patterns: List[Tuple[re.Pattern, str, str]],
    ) -> InjectionResult:
        detections: List[str] = []
        max_severity = "NONE"
        severity_rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

        for pattern, severity, description in patterns:
            if pattern.search(text):
                detections.append(f"[{severity}] {description}")
                if severity_rank[severity] > severity_rank[max_severity]:
                    max_severity = severity
                logger.warning(f"Injection detected: {description} | text snippet: {text[:100]!r}")

        is_injection = len(detections) > 0
        sanitized = self.sanitize_for_prompt(text) if is_injection else text

        return InjectionResult(
            is_injection=is_injection,
            severity=max_severity,
            detections=detections,
            sanitized_text=sanitized,
        )
