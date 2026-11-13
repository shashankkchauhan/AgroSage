"""
AgroSage FastAPI Application
All agricultural recommendation APIs with integrated guardrails.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from backend.agents.orchestrator import AgentOrchestrator

# ─────────────────────── singleton ───────────────────────────────────────────

_orchestrator: Optional[AgentOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    _orchestrator = AgentOrchestrator()
    logger.info("AgroSage backend started.")
    yield
    logger.info("AgroSage backend shut down.")


app = FastAPI(
    title="AgroSage API",
    version="1.0.0",
    description="AI-powered agricultural advisor backed by IBM Granite and RAG.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ─────────────────────── schemas ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:       str           = Field(..., min_length=2, max_length=2000)
    location:    str           = "Delhi"
    commodity:   Optional[str] = None
    language:    str           = "en"
    soil_params: Optional[Dict[str, float]] = None


class SoilAnalysisRequest(BaseModel):
    soil_ph:    float = Field(..., ge=0, le=14)
    nitrogen:   float = Field(..., ge=0, le=500)
    phosphorus: float = Field(..., ge=0, le=500)
    potassium:  float = Field(..., ge=0, le=500)
    location:   str   = "Delhi"


class AgentResponseOut(BaseModel):
    agent_name:       str
    response:         str
    sources:          List[str]
    confidence:       str
    language:         str
    requires_expert:  bool
    guardrail_status: str
    warnings:         List[str]
    errors:           List[str]
    is_mock:          bool
    metrics:          Dict[str, Any]


# ─────────────────────── endpoints ───────────────────────────────────────────

def get_orchestrator() -> AgentOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not ready.")
    return _orchestrator


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "AgroSage", "version": "1.0.0"}


@app.post("/api/v1/chat", response_model=AgentResponseOut)
async def chat(
    req: QueryRequest,
    orch: AgentOrchestrator = Depends(get_orchestrator),
):
    """Main chat endpoint. Routes to appropriate agent with full guardrail protection."""
    result = await orch.process(
        query=req.query,
        soil_params=req.soil_params,
        location=req.location,
        commodity=req.commodity,
    )
    return AgentResponseOut(
        agent_name=result.agent_name,
        response=result.final_response,
        sources=result.source_refs,
        confidence=result.confidence,
        language=result.detected_lang,
        requires_expert=result.requires_expert,
        guardrail_status=result.guardrail_status,
        warnings=result.warnings,
        errors=result.errors,
        is_mock=result.granite_meta.get("is_mock", True),
        metrics=result.metrics,
    )


@app.post("/api/v1/soil-analysis")
async def soil_analysis(
    req: SoilAnalysisRequest,
    orch: AgentOrchestrator = Depends(get_orchestrator),
):
    """Analyse soil parameters and get fertilizer recommendations."""
    query = (
        f"Soil analysis required: pH={req.soil_ph}, Nitrogen={req.nitrogen} kg/ha, "
        f"Phosphorus={req.phosphorus} kg/ha, Potassium={req.potassium} kg/ha. "
        f"Location: {req.location}. Provide fertilizer and amendment recommendations."
    )
    result = await orch.process(
        query=query,
        soil_params={
            "soil_ph": req.soil_ph,
            "nitrogen": req.nitrogen,
            "phosphorus": req.phosphorus,
            "potassium": req.potassium,
        },
        location=req.location,
    )
    return {
        "recommendation": result.final_response,
        "sources": result.source_refs,
        "guardrail_status": result.guardrail_status,
        "warnings": result.warnings,
    }


@app.get("/api/v1/weather/{location}")
async def get_weather(
    location: str,
    orch: AgentOrchestrator = Depends(get_orchestrator),
):
    """Get weather information for a location."""
    data = await orch._live_data.get_weather(location)
    return data


@app.get("/api/v1/mandi/{commodity}")
async def get_mandi(
    commodity: str,
    state: str = "Maharashtra",
    orch: AgentOrchestrator = Depends(get_orchestrator),
):
    """Get mandi prices for a commodity."""
    data = await orch._live_data.get_mandi_prices(commodity, state)
    return data


@app.get("/api/v1/metrics/llm")
async def llm_metrics(
    orch: AgentOrchestrator = Depends(get_orchestrator),
):
    """Get LLM evaluation metrics for the dashboard."""
    return {
        "metrics": orch.get_eval_metrics(),
        "guardrails": orch.get_guardrail_metrics(),
        "history": orch.get_eval_history(),
    }


@app.get("/api/v1/metrics/history")
async def metrics_history(
    orch: AgentOrchestrator = Depends(get_orchestrator),
):
    """Get time-series metric history."""
    return {"history": orch.get_eval_history()}
