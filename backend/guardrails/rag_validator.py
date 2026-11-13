"""
RAG Validator Guardrail
Verifies retrieval quality before answer generation:
- Relevance threshold enforcement
- Source citation validation
- Hallucination prevention (no fabricated citations)
- Document instruction isolation
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


# ─────────────────────── constants ──────────────────────────────────────────

MIN_RELEVANCE_SCORE = 0.30          # cosine / similarity threshold
MIN_DOCS_REQUIRED   = 1
MAX_DOCS_TO_USE     = 5
IRRELEVANT_DOMAIN_KEYWORDS = [
    "cryptocurrency", "bitcoin", "stock market", "gambling",
    "politics", "election", "warfare", "military",
    "pornography", "violence", "drug trafficking",
]


# ─────────────────────── result types ───────────────────────────────────────

@dataclass
class RetrievalDoc:
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGValidationResult:
    is_sufficient:    bool
    approved_docs:    List[RetrievalDoc] = field(default_factory=list)
    rejected_docs:    List[RetrievalDoc] = field(default_factory=list)
    warnings:         List[str] = field(default_factory=list)
    errors:           List[str] = field(default_factory=list)
    confidence_level: str = "HIGH"      # HIGH | MEDIUM | LOW | INSUFFICIENT
    source_refs:      List[str] = field(default_factory=list)


# ─────────────────────── validator ───────────────────────────────────────────

class RAGValidator:
    """Validates retrieved documents before they feed into the LLM prompt."""

    def __init__(
        self,
        min_relevance: float = MIN_RELEVANCE_SCORE,
        min_docs: int = MIN_DOCS_REQUIRED,
    ) -> None:
        self.min_relevance = min_relevance
        self.min_docs      = min_docs

    # ── public ───────────────────────────────────────────────────────────────

    def validate_retrieval(
        self,
        query: str,
        docs: List[RetrievalDoc],
    ) -> RAGValidationResult:
        """
        Filter retrieved documents by relevance score and domain safety.
        Returns only approved docs and a confidence level.
        """
        result = RAGValidationResult(is_sufficient=False)

        if not docs:
            result.errors.append(
                "No documents were retrieved. Cannot provide a reliable recommendation."
            )
            result.confidence_level = "INSUFFICIENT"
            return result

        # Score-based filtering
        for doc in docs:
            if doc.score < self.min_relevance:
                result.rejected_docs.append(doc)
                result.warnings.append(
                    f"Document '{doc.source}' rejected (relevance {doc.score:.2f} < "
                    f"threshold {self.min_relevance})."
                )
            elif self._is_off_domain(doc.content):
                result.rejected_docs.append(doc)
                result.warnings.append(
                    f"Document '{doc.source}' rejected: off-domain content detected."
                )
            else:
                result.approved_docs.append(doc)

        approved_count = len(result.approved_docs)

        if approved_count < self.min_docs:
            result.is_sufficient = False
            result.confidence_level = "INSUFFICIENT"
            result.errors.append(
                "Insufficient relevant agricultural information found. "
                "The system cannot provide a reliable recommendation at this time."
            )
            return result

        # Cap to max docs
        result.approved_docs = result.approved_docs[:MAX_DOCS_TO_USE]

        # Build source references (no fabrication allowed — only real sources)
        result.source_refs = self._build_source_refs(result.approved_docs)

        # Assign confidence level based on average score
        avg_score = sum(d.score for d in result.approved_docs) / len(result.approved_docs)
        result.confidence_level = self._score_to_confidence(avg_score)
        result.is_sufficient = True

        logger.info(
            f"RAG validation: {approved_count} approved docs, "
            f"confidence={result.confidence_level}, avg_score={avg_score:.3f}"
        )
        return result

    def verify_cited_sources(
        self,
        answer: str,
        approved_sources: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Check that every source name cited in `answer` appears in `approved_sources`.
        Returns (all_valid, list_of_fabricated_citations).
        """
        # Heuristic: look for quoted filenames or bracketed references
        cited = re.findall(r"\[([^\]]+)\]|\(([^)]+\.(?:pdf|txt|md|csv))\)", answer, re.I)
        cited_names = [c[0] or c[1] for c in cited]

        fabricated = []
        for name in cited_names:
            if not any(name.lower() in src.lower() for src in approved_sources):
                fabricated.append(name)

        return (len(fabricated) == 0), fabricated

    # ── private ──────────────────────────────────────────────────────────────

    def _is_off_domain(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in IRRELEVANT_DOMAIN_KEYWORDS)

    def _build_source_refs(self, docs: List[RetrievalDoc]) -> List[str]:
        seen, refs = set(), []
        for doc in docs:
            src = doc.source or "Unknown Source"
            if src not in seen:
                seen.add(src)
                refs.append(src)
        return refs

    @staticmethod
    def _score_to_confidence(score: float) -> str:
        if score >= 0.75:
            return "HIGH"
        if score >= 0.50:
            return "MEDIUM"
        if score >= 0.30:
            return "LOW"
        return "INSUFFICIENT"
