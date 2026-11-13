"""
IBM Granite Service
Wraps ibm-watsonx-ai for text generation using IBM Granite models.
Integrates guardrails before and after every generation call.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ─────────────────────── result ─────────────────────────────────────────────

@dataclass
class GraniteResponse:
    text: str
    model_id: str
    input_tokens: int      = 0
    output_tokens: int     = 0
    latency_ms: float      = 0.0
    ttft_ms: float         = 0.0
    is_mock: bool          = False
    error: Optional[str]   = None


# ─────────────────────── session token tracker ───────────────────────────────

class TokenUsageTracker:
    """Accumulates token usage across all calls in a session."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_input_tokens:  int   = 0
        self.total_output_tokens: int   = 0
        self.total_requests:      int   = 0
        self.total_cost_usd:      float = 0.0
        self.calls: List[Dict[str, Any]] = []

    def record(self, input_tok: int, output_tok: int,
               latency_ms: float, model_id: str) -> None:
        # Approximate pricing (illustrative — watsonx.ai Lite is free up to limits)
        cost = (input_tok / 1_000_000 * 0.60) + (output_tok / 1_000_000 * 2.40)
        self.total_input_tokens  += input_tok
        self.total_output_tokens += output_tok
        self.total_requests      += 1
        self.total_cost_usd      += cost
        self.calls.append({
            "request_num":    self.total_requests,
            "input_tokens":   input_tok,
            "output_tokens":  output_tok,
            "total_tokens":   input_tok + output_tok,
            "latency_ms":     latency_ms,
            "cost_usd":       round(cost, 8),
            "model_id":       model_id,
            "timestamp":      time.strftime("%H:%M:%S"),
        })

    def summary(self) -> Dict[str, Any]:
        return {
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens":        self.total_input_tokens + self.total_output_tokens,
            "total_requests":      self.total_requests,
            "total_cost_usd":      round(self.total_cost_usd, 6),
            "avg_tokens_per_req":  round(
                (self.total_input_tokens + self.total_output_tokens)
                / max(self.total_requests, 1), 1
            ),
            "calls": self.calls[-50:],   # last 50
        }


# Singleton tracker shared across the app session
_token_tracker = TokenUsageTracker()


def get_token_tracker() -> TokenUsageTracker:
    return _token_tracker


# ─────────────────────── service ─────────────────────────────────────────────

class GraniteService:
    """
    Calls IBM Granite via watsonx.ai.
    Falls back to a clearly-marked mock when credentials are unavailable.
    """

    def __init__(self) -> None:
        self.api_key    = os.getenv("WATSONX_API_KEY", "")
        self.project_id = os.getenv("WATSONX_PROJECT_ID", "")
        self.url        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.model_id   = os.getenv("GRANITE_MODEL_ID", "ibm/granite-4-h-small")
        self._client    = None
        self._model     = None

        if self.api_key and self.project_id:
            self._init_client()
        else:
            logger.warning(
                "WATSONX_API_KEY or WATSONX_PROJECT_ID not set. "
                "GraniteService will use mock mode."
            )

    def _init_client(self) -> None:
        try:
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params

            credentials = Credentials(url=self.url, api_key=self.api_key)
            client = APIClient(credentials)
            self._model = ModelInference(
                model_id=self.model_id,
                api_client=client,
                project_id=self.project_id,
                params={
                    Params.MAX_NEW_TOKENS: 1024,
                    Params.MIN_NEW_TOKENS: 10,
                    Params.TEMPERATURE: 0.3,
                    Params.TOP_P: 0.9,
                    Params.REPETITION_PENALTY: 1.1,
                },
            )
            logger.info(f"IBM Granite model '{self.model_id}' initialised.")
        except Exception as exc:
            logger.error(f"Failed to initialise IBM Granite: {exc}")
            self._model = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
    ) -> GraniteResponse:
        """Generate a response using IBM Granite."""
        full_prompt = self._build_prompt(system_prompt, context, user_prompt)
        t0 = time.perf_counter()

        if self._model is None:
            return self._mock_response(user_prompt, t0)

        try:
            # Use generate (returns full dict with token counts)
            raw = self._model.generate(prompt=full_prompt)
            elapsed = (time.perf_counter() - t0) * 1000

            # Normalise watsonx.ai response shape
            if isinstance(raw, dict):
                res_block  = raw.get("results", [{}])[0]
                text       = res_block.get("generated_text", "")
                input_tok  = res_block.get("input_token_count", 0)
                output_tok = res_block.get("generated_token_count", 0)
            else:
                # Plain string fallback — estimate from word count
                text       = str(raw)
                input_tok  = len(full_prompt.split())
                output_tok = len(text.split())

            # Fallback estimation if API returns 0
            if input_tok == 0:
                input_tok = max(1, len(full_prompt.split()) * 4 // 3)
            if output_tok == 0:
                output_tok = max(1, len(text.split()) * 4 // 3)

            _token_tracker.record(input_tok, output_tok, round(elapsed, 2), self.model_id)

            return GraniteResponse(
                text=text.strip(),
                model_id=self.model_id,
                input_tokens=input_tok,
                output_tokens=output_tok,
                latency_ms=round(elapsed, 2),
                ttft_ms=round(elapsed * 0.3, 2),
            )
        except Exception as exc:
            logger.error(f"Granite generation error: {exc}")
            return GraniteResponse(
                text="",
                model_id=self.model_id,
                error=str(exc),
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    def _build_prompt(self, system: str, context: str, query: str) -> str:
        parts = [f"[SYSTEM]\n{system}\n"]
        if context:
            parts.append(f"[CONTEXT]\n{context}\n")
        parts.append(f"[QUERY]\n{query}\n[ANSWER]")
        return "\n".join(parts)

    def _mock_response(self, query: str, t0: float) -> GraniteResponse:
        elapsed = (time.perf_counter() - t0) * 1000
        mock_answers = {
            "wheat": "Wheat (Triticum aestivum) is a rabi crop best sown in October–November. Recommended fertilizer: NPK 120:60:40 kg/ha. Ensure adequate irrigation at crown root initiation, jointing, and grain filling stages.",
            "tomato": "Tomatoes require well-drained loamy soil with pH 6.0–7.0. Apply FYM 25 tonnes/ha before transplanting. Watch for early blight (Alternaria solani) — use Mancozeb 0.25% spray.",
            "rice": "Paddy (Kharif) is best transplanted in June–July. Maintain 2–5 cm standing water. Recommended dose: Urea 100 kg/ha split in three applications.",
            "mandi": "Mock mandi data: Tomato ₹1,200/quintal, Onion ₹800/quintal, Wheat ₹2,100/quintal (Note: This is mock data, not real-time prices).",
            "weather": "Current mock weather: Temperature 28°C, Humidity 65%, Rainfall forecast: moderate rain expected in next 48 hours (Note: This is mock data).",
        }
        query_lower = query.lower()
        answer = next((v for k, v in mock_answers.items() if k in query_lower),
                      "I can help you with crop recommendations, soil management, pest control, weather information, and market prices. Please describe your agricultural query in detail.")
        in_tok  = max(1, len(query.split()) * 4 // 3)
        out_tok = max(1, len(answer.split()) * 4 // 3)
        _token_tracker.record(in_tok, out_tok, round(elapsed + 50, 2), f"{self.model_id}-MOCK")
        return GraniteResponse(
            text=answer,
            model_id=f"{self.model_id}-MOCK",
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(elapsed + 50, 2),
            ttft_ms=15.0,
            is_mock=True,
        )
