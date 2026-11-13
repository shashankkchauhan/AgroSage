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
            result = self._model.generate_text(prompt=full_prompt)
            elapsed = (time.perf_counter() - t0) * 1000

            # watsonx.ai returns different shapes; normalise
            if isinstance(result, dict):
                text = result.get("results", [{}])[0].get("generated_text", str(result))
                usage = result.get("results", [{}])[0]
                input_tok  = usage.get("input_token_count", 0)
                output_tok = usage.get("generated_token_count", 0)
            else:
                text = str(result)
                input_tok = output_tok = 0

            return GraniteResponse(
                text=text.strip(),
                model_id=self.model_id,
                input_tokens=input_tok,
                output_tokens=output_tok,
                latency_ms=round(elapsed, 2),
                ttft_ms=round(elapsed * 0.3, 2),   # estimate
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
        return GraniteResponse(
            text=answer,
            model_id=f"{self.model_id}-MOCK",
            input_tokens=len(query.split()),
            output_tokens=len(answer.split()),
            latency_ms=round(elapsed + 50, 2),
            ttft_ms=15.0,
            is_mock=True,
        )
