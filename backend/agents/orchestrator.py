"""
AI Agent Orchestrator
Coordinates the multi-agent workflow for AgroSage:
  - Supervisor Agent: routes queries to specialist agents
  - Crop Advisor Agent
  - Soil Analyst Agent
  - Pest Control Agent
  - Market Price Agent
  - Weather Agent
  - Government Schemes Agent
Integrates GuardrailEngine at every step and feeds metrics to LLMEvaluator.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from ..services.granite_service import GraniteService, GraniteResponse
from ..services.rag_service import RAGService
from ..services.live_data_service import LiveDataService
from ..services.llm_evaluator import LLMEvaluator, RequestRecord
from ..guardrails.guardrail_engine import GuardrailEngine, GuardrailResult, GuardrailStatus
from ..guardrails.rag_validator import RetrievalDoc


# ─────────────────────── agent definitions ───────────────────────────────────

AGENTS = {
    "crop_advisor": {
        "name": "Crop Advisor",
        "keywords": ["crop", "cultivation", "sowing", "harvest", "variety", "season",
                     "फसल", "बुआई", "कटाई"],
        "system_prompt": (
            "You are an expert agricultural crop advisor for Indian farmers. "
            "Provide precise, evidence-based recommendations for crop selection, "
            "cultivation practices, and yield improvement. Always cite your sources. "
            "Do not guarantee yield outcomes. Recommend consulting local KVK for site-specific advice."
        ),
    },
    "soil_analyst": {
        "name": "Soil Analyst",
        "keywords": ["soil", "ph", "nitrogen", "phosphorus", "potassium", "npk",
                     "organic matter", "fertilizer", "मिट्टी", "उर्वरक"],
        "system_prompt": (
            "You are an expert soil scientist helping Indian farmers understand their soil. "
            "Interpret soil test results, recommend amendments and fertilizers based on "
            "scientific evidence. Always specify dosages as per registered guidelines. "
            "Advise farmers to get soil tested at the nearest Soil Testing Lab."
        ),
    },
    "pest_control": {
        "name": "Pest & Disease Specialist",
        "keywords": ["pest", "disease", "insect", "fungal", "blight", "wilt", "spray",
                     "pesticide", "कीट", "रोग", "कीटनाशक"],
        "system_prompt": (
            "You are an IPM (Integrated Pest Management) specialist for Indian agriculture. "
            "Recommend pest and disease management strategies prioritising biological, cultural, "
            "and chemical methods in that order. For chemical recommendations, always refer to "
            "the registered CIB&RC label. Include PPE instructions and pre-harvest intervals. "
            "Never recommend banned chemicals."
        ),
    },
    "market_analyst": {
        "name": "Market Price Analyst",
        "keywords": ["price", "mandi", "market", "rate", "sell", "buying", "selling",
                     "quintal", "भाव", "बाजार", "मंडी"],
        "system_prompt": (
            "You are an agricultural market analyst providing farmers with current market "
            "price information. Always specify whether prices are live, cached, or mock. "
            "Reference MSP from CACP for context. Do not predict future prices."
        ),
    },
    "weather_advisor": {
        "name": "Weather Advisor",
        "keywords": ["weather", "rain", "rainfall", "temperature", "forecast", "humidity",
                     "drought", "flood", "मौसम", "बारिश"],
        "system_prompt": (
            "You are a meteorological advisor for farmers. Provide weather-based crop "
            "advisories using IMD data. Clearly label data as live, cached, or mock. "
            "Recommend precautionary measures for adverse weather events."
        ),
    },
    "schemes_advisor": {
        "name": "Government Schemes Advisor",
        "keywords": ["scheme", "subsidy", "loan", "insurance", "pm-kisan", "pmfby",
                     "kcc", "government", "yojana", "योजना", "सब्सिडी"],
        "system_prompt": (
            "You are an advisor on Indian government agricultural schemes and subsidies. "
            "Provide accurate information about PM-KISAN, PMFBY, KCC, MSP, and other "
            "schemes. Always reference official government sources. Do not make claims "
            "about eligibility without asking for farmer details."
        ),
    },
}


# ─────────────────────── result ─────────────────────────────────────────────

@dataclass
class AgentResponse:
    agent_name:      str
    final_response:  str
    source_refs:     List[str]        = field(default_factory=list)
    confidence:      str              = "HIGH"
    detected_lang:   str              = "en"
    requires_expert: bool             = False
    guardrail_status: str             = "PASS"
    warnings:        List[str]        = field(default_factory=list)
    errors:          List[str]        = field(default_factory=list)
    live_data:       Optional[Dict]   = None
    metrics:         Dict[str, Any]   = field(default_factory=dict)
    granite_meta:    Dict[str, Any]   = field(default_factory=dict)


# ─────────────────────── orchestrator ────────────────────────────────────────

class AgentOrchestrator:
    """
    Supervisor + specialist multi-agent system.
    All calls are wrapped in pre/post guardrail checks.
    """

    def __init__(self) -> None:
        self._granite   = GraniteService()
        self._rag       = RAGService()
        self._live_data = LiveDataService()
        self._guardrails = GuardrailEngine()
        self._evaluator  = LLMEvaluator()

    # ── public ───────────────────────────────────────────────────────────────

    async def process(
        self,
        query: str,
        soil_params: Optional[Dict[str, Any]] = None,
        location: str = "Delhi",
        commodity: Optional[str] = None,
    ) -> AgentResponse:
        """Main entry point. Routes query → agent → guarded Granite call."""
        t0 = time.perf_counter()

        # 1. Route to specialist agent
        agent_key  = self._route(query)
        agent_cfg  = AGENTS[agent_key]
        logger.info(f"Routing query to agent: {agent_cfg['name']}")

        # 2. Retrieve relevant documents
        raw_docs   = self._rag.retrieve(query)
        retv_docs  = [
            RetrievalDoc(
                content=d["content"],
                source=d["source"],
                score=d["score"],
                metadata=d["metadata"],
            )
            for d in raw_docs
        ]

        # 3. PRE-GENERATION guardrails
        pre: GuardrailResult = self._guardrails.pre_generation(
            query=query,
            soil_params=soil_params,
            retrieved_docs=retv_docs,
        )
        if pre.status == GuardrailStatus.BLOCK:
            self._record_eval(query, pre.final_response, pre, GraniteResponse("", ""), t0, blocked=True)
            return AgentResponse(
                agent_name=agent_cfg["name"],
                final_response=pre.final_response,
                guardrail_status="BLOCK",
                errors=pre.errors,
                warnings=pre.warnings,
            )

        # 4. Fetch live data if relevant
        live_data = None
        if agent_key == "market_analyst":
            live_data = await self._live_data.get_mandi_prices(
                commodity or self._extract_commodity(query), location
            )
        elif agent_key == "weather_advisor":
            live_data = await self._live_data.get_weather(location)

        # 5. Build context from approved docs + live data
        context = self._build_context(pre.approved_docs, live_data)

        # 6. Call IBM Granite
        granite_resp: GraniteResponse = self._granite.generate(
            system_prompt=agent_cfg["system_prompt"],
            user_prompt=pre.final_query,
            context=context,
        )

        llm_text = granite_resp.text or (granite_resp.error or "No response generated.")

        # 7. POST-GENERATION guardrails
        post: GuardrailResult = self._guardrails.post_generation(
            pre_result=pre,
            llm_response=llm_text,
            query=pre.final_query,
        )

        # 8. Record evaluation metrics
        self._record_eval(query, post.final_response, post, granite_resp, t0,
                          blocked=post.status == GuardrailStatus.BLOCK)

        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        return AgentResponse(
            agent_name=agent_cfg["name"],
            final_response=post.final_response,
            source_refs=post.source_refs,
            confidence=post.confidence_level,
            detected_lang=post.detected_lang,
            requires_expert=post.requires_expert,
            guardrail_status=post.status.value,
            warnings=post.warnings,
            errors=post.errors,
            live_data=live_data,
            granite_meta={
                "model_id":      granite_resp.model_id,
                "input_tokens":  granite_resp.input_tokens,
                "output_tokens": granite_resp.output_tokens,
                "latency_ms":    granite_resp.latency_ms,
                "is_mock":       granite_resp.is_mock,
            },
            metrics={"total_ms": elapsed, **post.metrics},
        )

    def get_eval_metrics(self) -> Dict[str, Any]:
        return self._evaluator.to_dict()

    def get_eval_history(self) -> List[Dict[str, Any]]:
        return self._evaluator.history()

    def get_guardrail_metrics(self) -> Dict[str, Any]:
        return self._guardrails.get_session_metrics()

    # ── routing ───────────────────────────────────────────────────────────────

    def _route(self, query: str) -> str:
        q = query.lower()
        scores: Dict[str, int] = {}
        for key, cfg in AGENTS.items():
            scores[key] = sum(1 for kw in cfg["keywords"] if kw in q)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "crop_advisor"

    # ── context builder ───────────────────────────────────────────────────────

    def _build_context(
        self,
        docs: List[RetrievalDoc],
        live_data: Optional[Dict],
    ) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[Source {i}: {doc.source}]\n{doc.content}")
        if live_data and not live_data.get("is_mock"):
            parts.append(f"[Live Data]\n{live_data}")
        elif live_data and live_data.get("is_mock"):
            parts.append(f"[Mock Data - NOT real-time]\n{live_data}")
        return "\n\n".join(parts)

    def _extract_commodity(self, query: str) -> str:
        commodities = ["wheat", "rice", "tomato", "onion", "potato", "maize", "cotton", "soybean"]
        q = query.lower()
        return next((c for c in commodities if c in q), "wheat")

    # ── metrics recording ─────────────────────────────────────────────────────

    def _record_eval(
        self,
        query: str,
        response: str,
        result: GuardrailResult,
        granite: GraniteResponse,
        t0: float,
        blocked: bool = False,
    ) -> None:
        elapsed = (time.perf_counter() - t0) * 1000
        self._evaluator.record(RequestRecord(
            query=query,
            response=response,
            input_tokens=granite.input_tokens,
            output_tokens=granite.output_tokens,
            latency_ms=granite.latency_ms or elapsed,
            ttft_ms=granite.ttft_ms,
            is_blocked=blocked,
            is_injection=result.injection_detected,
            pii_leaked=bool(result.pii_found),
            guardrail_fired=result.status != GuardrailStatus.PASS,
            retrieval_scores=[d.score for d in result.approved_docs],
            num_docs=len(result.approved_docs),
            task_completed=not blocked,
            confidence=result.confidence_level,
        ))
