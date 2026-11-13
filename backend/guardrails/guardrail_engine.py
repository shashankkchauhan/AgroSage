"""
Guardrail Engine
Central orchestrator that runs all guardrail checks in the correct order:

  PRE-GENERATION:
    1. InputValidator      – validate & sanitize user input
    2. PromptInjectionDetector – block injection attempts
    3. AgriculturalSafetyChecker.check_query – flag banned chemicals in query
    4. RAGValidator        – filter retrieved documents

  POST-GENERATION:
    5. AgriculturalSafetyChecker.check_response – enforce safe recommendations
    6. OutputValidator     – relevance, PII, uncertainty
    7. PrivacyGuard        – final credential/PII sweep

All metrics are collected for the LLM Evaluation Dashboard.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from .input_validator import InputValidator, ValidationResult
from .prompt_injection_detector import PromptInjectionDetector, InjectionResult
from .rag_validator import RAGValidator, RAGValidationResult, RetrievalDoc
from .agricultural_safety import AgriculturalSafetyChecker, AgriSafetyResult
from .output_validator import OutputValidator, OutputValidationResult
from .privacy_guard import PrivacyGuard


# ─────────────────────── status enum ─────────────────────────────────────────

class GuardrailStatus(str, Enum):
    PASS    = "PASS"
    WARN    = "WARN"
    BLOCK   = "BLOCK"


# ─────────────────────── result ─────────────────────────────────────────────

@dataclass
class GuardrailResult:
    status:          GuardrailStatus
    final_query:     str
    final_response:  str
    errors:          List[str]           = field(default_factory=list)
    warnings:        List[str]           = field(default_factory=list)
    approved_docs:   List[RetrievalDoc]  = field(default_factory=list)
    source_refs:     List[str]           = field(default_factory=list)
    confidence_level: str                = "HIGH"
    detected_lang:   str                 = "en"
    requires_expert: bool                = False
    pii_found:       List[str]           = field(default_factory=list)
    injection_detected: bool             = False
    metrics:         Dict[str, Any]      = field(default_factory=dict)


# ─────────────────────── engine ──────────────────────────────────────────────

class GuardrailEngine:
    """
    Single entry point for all guardrail checks.
    Usage:
        engine = GuardrailEngine()

        # Before calling the LLM
        pre = engine.pre_generation(query, soil_params, retrieved_docs)
        if pre.status == GuardrailStatus.BLOCK:
            return pre.final_response  # blocked message

        # After getting LLM response
        result = engine.post_generation(pre, llm_response)
        return result.final_response
    """

    def __init__(self) -> None:
        self._input_val     = InputValidator()
        self._inject_det    = PromptInjectionDetector()
        self._rag_val       = RAGValidator()
        self._ag_safety     = AgriculturalSafetyChecker()
        self._output_val    = OutputValidator()
        self._privacy       = PrivacyGuard()

        # Metrics accumulator (across a session)
        self._total_requests         = 0
        self._blocked_requests       = 0
        self._injection_attempts     = 0
        self._guardrail_interventions = 0
        self._pii_blocks             = 0

    # ── PRE-GENERATION ────────────────────────────────────────────────────────

    def pre_generation(
        self,
        query: str,
        soil_params: Optional[Dict[str, Any]] = None,
        retrieved_docs: Optional[List[RetrievalDoc]] = None,
    ) -> GuardrailResult:
        """Run all pre-generation checks. Returns a GuardrailResult."""
        t0 = time.perf_counter()
        self._total_requests += 1
        errors: List[str]   = []
        warnings: List[str] = []

        # ── 1. Input validation ───────────────────────────────────────────────
        input_result: ValidationResult = self._input_val.validate_text_query(query)
        if not input_result.is_valid:
            self._blocked_requests += 1
            self._guardrail_interventions += 1
            return GuardrailResult(
                status=GuardrailStatus.BLOCK,
                final_query=query,
                final_response=self._block_message(input_result.errors),
                errors=input_result.errors,
            )
        query = input_result.sanitized or query
        warnings.extend(input_result.warnings)

        # ── 2. Soil / env param validation ───────────────────────────────────
        if soil_params:
            param_result = self._input_val.validate_soil_params(soil_params)
            if not param_result.is_valid:
                self._blocked_requests += 1
                self._guardrail_interventions += 1
                return GuardrailResult(
                    status=GuardrailStatus.BLOCK,
                    final_query=query,
                    final_response=self._block_message(param_result.errors),
                    errors=param_result.errors,
                )
            warnings.extend(param_result.warnings)

        # ── 3. Prompt injection detection ─────────────────────────────────────
        inject_result: InjectionResult = self._inject_det.check_user_input(query)
        if inject_result.is_injection and inject_result.severity == "HIGH":
            self._blocked_requests += 1
            self._injection_attempts += 1
            self._guardrail_interventions += 1
            return GuardrailResult(
                status=GuardrailStatus.BLOCK,
                final_query=query,
                final_response=(
                    "Your query contains content that cannot be processed. "
                    "Please ask a genuine agricultural question."
                ),
                errors=inject_result.detections,
                injection_detected=True,
            )
        if inject_result.is_injection:
            self._injection_attempts += 1
            self._guardrail_interventions += 1
            warnings.extend(inject_result.detections)
            query = inject_result.sanitized_text

        # ── 4. Agricultural safety check on query ─────────────────────────────
        ag_query_result: AgriSafetyResult = self._ag_safety.check_query(query)
        if not ag_query_result.is_safe:
            self._blocked_requests += 1
            self._guardrail_interventions += 1
            return GuardrailResult(
                status=GuardrailStatus.BLOCK,
                final_query=query,
                final_response=self._block_message(ag_query_result.errors),
                errors=ag_query_result.errors,
                warnings=warnings,
            )
        warnings.extend(ag_query_result.warnings)

        # ── 5. RAG retrieval validation ───────────────────────────────────────
        docs = retrieved_docs or []
        rag_result: RAGValidationResult = self._rag_val.validate_retrieval(query, docs)
        warnings.extend(rag_result.warnings)
        if not rag_result.is_sufficient:
            errors.extend(rag_result.errors)
            # Soft-block: allow continuation with disclaimer
            self._guardrail_interventions += 1

        elapsed = time.perf_counter() - t0
        status = GuardrailStatus.WARN if warnings else GuardrailStatus.PASS

        return GuardrailResult(
            status=status,
            final_query=query,
            final_response="",          # filled in post_generation
            errors=errors,
            warnings=warnings,
            approved_docs=rag_result.approved_docs,
            source_refs=rag_result.source_refs,
            confidence_level=rag_result.confidence_level,
            detected_lang=input_result.detected_lang,
            requires_expert=ag_query_result.requires_expert,
            injection_detected=inject_result.is_injection,
            metrics={"pre_latency_ms": round(elapsed * 1000, 2)},
        )

    # ── POST-GENERATION ───────────────────────────────────────────────────────

    def post_generation(
        self,
        pre_result: GuardrailResult,
        llm_response: str,
        query: str,
    ) -> GuardrailResult:
        """Run all post-generation checks on the LLM response."""
        t0 = time.perf_counter()
        errors   = list(pre_result.errors)
        warnings = list(pre_result.warnings)

        # ── 6. Agricultural safety check on response ──────────────────────────
        ag_resp_result: AgriSafetyResult = self._ag_safety.check_response(llm_response)
        if not ag_resp_result.is_safe:
            self._blocked_requests += 1
            self._guardrail_interventions += 1
            return GuardrailResult(
                status=GuardrailStatus.BLOCK,
                final_query=pre_result.final_query,
                final_response=self._block_message(ag_resp_result.errors),
                errors=errors + ag_resp_result.errors,
                warnings=warnings,
                confidence_level=pre_result.confidence_level,
                detected_lang=pre_result.detected_lang,
            )
        if ag_resp_result.modified_text:
            llm_response = ag_resp_result.modified_text
            self._guardrail_interventions += 1
        warnings.extend(ag_resp_result.warnings)
        requires_expert = pre_result.requires_expert or ag_resp_result.requires_expert

        # ── 7. Output validation ──────────────────────────────────────────────
        out_result: OutputValidationResult = self._output_val.validate(
            response=llm_response,
            query=query,
            approved_sources=pre_result.source_refs,
            confidence_level=pre_result.confidence_level,
        )
        if not out_result.is_valid:
            self._blocked_requests += 1
            self._pii_blocks += 1 if out_result.pii_found else 0
            self._guardrail_interventions += 1
            return GuardrailResult(
                status=GuardrailStatus.BLOCK,
                final_query=pre_result.final_query,
                final_response=out_result.final_response,
                errors=errors + out_result.errors,
                warnings=warnings,
                pii_found=out_result.pii_found,
                confidence_level=pre_result.confidence_level,
                detected_lang=pre_result.detected_lang,
            )
        warnings.extend(out_result.warnings)
        final_response = out_result.final_response

        # ── 8. Privacy final sweep ────────────────────────────────────────────
        if not self._privacy.validate_response_for_secrets(final_response):
            self._blocked_requests += 1
            self._guardrail_interventions += 1
            return GuardrailResult(
                status=GuardrailStatus.BLOCK,
                final_query=pre_result.final_query,
                final_response="Response blocked: contains sensitive credential data.",
                errors=errors + ["Credential data found in response."],
                warnings=warnings,
            )

        # Append educational disclaimer
        final_response += self._ag_safety.educational_disclaimer()

        elapsed = time.perf_counter() - t0
        status = GuardrailStatus.WARN if (warnings or errors) else GuardrailStatus.PASS

        metrics = {
            **pre_result.metrics,
            "post_latency_ms": round(elapsed * 1000, 2),
            "total_requests": self._total_requests,
            "blocked_requests": self._blocked_requests,
            "injection_attempts": self._injection_attempts,
            "guardrail_interventions": self._guardrail_interventions,
            "pii_blocks": self._pii_blocks,
        }

        return GuardrailResult(
            status=status,
            final_query=pre_result.final_query,
            final_response=final_response,
            errors=errors,
            warnings=warnings,
            approved_docs=pre_result.approved_docs,
            source_refs=pre_result.source_refs,
            confidence_level=pre_result.confidence_level,
            detected_lang=pre_result.detected_lang,
            requires_expert=requires_expert,
            injection_detected=pre_result.injection_detected,
            pii_found=out_result.pii_found,
            metrics=metrics,
        )

    # ── metrics ───────────────────────────────────────────────────────────────

    def get_session_metrics(self) -> Dict[str, Any]:
        return {
            "total_requests":          self._total_requests,
            "blocked_requests":        self._blocked_requests,
            "injection_attempts":      self._injection_attempts,
            "guardrail_interventions": self._guardrail_interventions,
            "pii_blocks":              self._pii_blocks,
            "block_rate": (
                round(self._blocked_requests / self._total_requests * 100, 2)
                if self._total_requests else 0.0
            ),
            "intervention_rate": (
                round(self._guardrail_interventions / self._total_requests * 100, 2)
                if self._total_requests else 0.0
            ),
        }

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _block_message(errors: List[str]) -> str:
        base = "Your request could not be processed due to the following:\n"
        return base + "\n".join(f"• {e}" for e in errors)
