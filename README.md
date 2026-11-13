# 🌾 AgroSage — AI-Powered Smart Farming Assistant

> **Problem Statement #9** — AI Agent for Smart Farming Advice  
> Powered by **IBM Granite** · **IBM watsonx.ai** · RAG · FastAPI · Streamlit

---

## Overview

AgroSage is an AI-powered agricultural assistant that provides **real-time, localized farming guidance** to small-scale farmers. It retrieves trusted information from agricultural departments, meteorological sources, and agri-tech platforms using **Retrieval-Augmented Generation (RAG)**, then generates safe, grounded advice through **IBM Granite** models.

### Key Features

- 💬 **Multi-lingual Chatbot** (English + Hindi) for farmer queries
- 🌾 **Specialist AI Agents** (Crop, Soil, Pest, Market, Weather, Schemes)
- 🔍 **RAG Pipeline** with ChromaDB + semantic embeddings
- ☁️ **Live Data** — weather (OpenWeatherMap) + mandi prices (data.gov.in)
- 📊 **LLM Evaluation Dashboard** — 50+ metrics across 6 categories
- 🛡️ **Comprehensive AI Guardrails** — 8-layer safety framework

---

## Project Structure

```
AgroSage/
├── backend/
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_validator.py          # Soil params, text validation
│   │   ├── prompt_injection_detector.py # Injection & jailbreak detection
│   │   ├── rag_validator.py            # Retrieval quality & citations
│   │   ├── agricultural_safety.py      # Chemical safety & disclaimers
│   │   ├── output_validator.py         # PII, relevance, uncertainty
│   │   ├── privacy_guard.py            # API key & PII masking
│   │   └── guardrail_engine.py         # Central orchestrator
│   ├── services/
│   │   ├── granite_service.py          # IBM Granite / watsonx.ai
│   │   ├── rag_service.py              # ChromaDB + embeddings
│   │   ├── live_data_service.py        # Weather + mandi APIs
│   │   └── llm_evaluator.py            # Metrics computation
│   ├── agents/
│   │   └── orchestrator.py             # Multi-agent coordinator
│   └── main.py                         # FastAPI application
├── frontend/
│   └── app.py                          # Streamlit UI
├── tests/
│   └── test_guardrails.py              # 73 automated tests
├── GUARDRAILS.md                       # Safety framework documentation
├── env.txt                             # Environment variable template
├── requirements.txt
└── run.py                              # Startup script
```

---

## Quick Start

### 1. Configure Environment

Copy `env.txt` to `.env` and fill in your credentials:

```bash
WATSONX_API_KEY=your_key_here
WATSONX_PROJECT_ID=your_project_id_here
WEATHER_API_KEY=your_openweathermap_key
MANDI_API_KEY=your_data_gov_in_key
```

> Without credentials, the app runs in **mock mode** — all functionality works with clearly-labelled sample data.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Option A: One command (starts both backend + frontend)
python run.py

# Option B: Separately
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

### 4. Access

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |

### 5. Run Tests

```bash
pytest tests/test_guardrails.py -v
# Result: 73 passed in ~1.4s
```

---

## LLM Evaluation Dashboard

The Streamlit dashboard includes **6 evaluation categories** with live metrics:

| Category | Metrics |
|----------|---------|
| **Accuracy & Quality** | Exact Match, F1, Precision, Recall, ROUGE-1/L, BLEU, Semantic Similarity |
| **RAG Evaluation** | Context Precision/Recall, Faithfulness, Answer Relevancy, Groundedness |
| **Agentic AI** | Task Completion, Tool Call Accuracy, Planning, Reasoning, Coordination |
| **Safety & Guardrails** | Hallucination Rate, Injection Resistance, PII Leakage, Policy Violations |
| **Performance & Cost** | Latency, TTFT, Tokens/sec, Cost/Request, Throughput, Error Rate |
| **Multi-Agent System** | Workflow Completion, Supervisor Accuracy, Inter-agent Consistency |

---

## Guardrails Framework

8 layers of Python-enforced safety (see [`GUARDRAILS.md`](GUARDRAILS.md)):

1. **Input Validation** — soil pH, NPK, temperature ranges + text sanitization
2. **Prompt Injection Detection** — 10 pattern categories, HIGH/MEDIUM severity
3. **RAG Accuracy** — relevance threshold, off-domain filtering, citation verification
4. **Agricultural Safety** — banned chemical list (endosulfan, DDT, etc.), dosage disclaimers
5. **Live Data** — MOCK/LIVE/CACHED/UNAVAILABLE labelling
6. **Output Validation** — PII scan, off-topic detection, uncertainty warnings
7. **Privacy Guard** — API key masking, Aadhaar/PAN/phone/email redaction
8. **Language Support** — Hindi & English with Unicode detection

---

## IBM Cloud Services Used

- **IBM Granite** (`ibm/granite-13b-instruct-v2`) — LLM generation
- **IBM watsonx.ai** — Model hosting and inference API
- **IBM Cloud Lite** compatible (free tier API key)

---

## Test Results

```
73 passed in 1.39s ✅

TestInputValidatorSoilParams     10 passed
TestPromptInjectionDetector      11 passed
TestRAGValidator                  9 passed
TestAgriculturalSafety           10 passed
TestLiveDataGuardrails            4 passed
TestPrivacyGuard                 10 passed
TestLanguageHandling              8 passed
TestOutputValidator               5 passed
TestGuardrailEngineIntegration    6 passed
```
