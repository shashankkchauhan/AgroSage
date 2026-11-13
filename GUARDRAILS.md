# AgroSage AI Guardrails & Safety Framework

**Version:** 1.0.0  
**Last Updated:** November 2026  
**Project:** AgroSage — AI-powered Smart Farming Assistant  
**Technology:** IBM Granite · IBM watsonx.ai · RAG · FastAPI · Streamlit

---

## Overview

AgroSage implements a **comprehensive, multi-layer AI Guardrails and Safety Framework** that runs Python-based validation, policy enforcement, retrieval verification, and output filtering. Guardrails execute both **before** (pre-generation) and **after** (post-generation) every IBM Granite model call.

> **Important:** Guardrails are implemented as actual Python code — not merely as LLM system-prompt instructions. They cannot be bypassed by user inputs or retrieved documents.

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────── PRE-GENERATION ────────────────────────────────┐
│  1. InputValidator           → validate text, length, language            │
│  2. InputValidator           → validate soil/env params (pH, NPK, etc.)  │
│  3. PromptInjectionDetector  → block injection, jailbreak attempts        │
│  4. AgriculturalSafetyChecker.check_query → ban list, escalation triggers│
│  5. RAGValidator             → filter low-relevance retrieved documents   │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
IBM Granite Generation (ibm/granite-13b-instruct-v2)
    │
    ▼
┌─────────────────────────── POST-GENERATION ───────────────────────────────┐
│  6. AgriculturalSafetyChecker.check_response → chemicals, dosage, yield  │
│  7. OutputValidator          → relevance, PII scan, uncertainty warning   │
│  8. PrivacyGuard             → credential & PII sweep on final text       │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Final Response → User
```

---

## Module Reference

| File | Description |
|------|-------------|
| `backend/guardrails/__init__.py` | Package exports |
| `backend/guardrails/input_validator.py` | Text + soil parameter validation |
| `backend/guardrails/prompt_injection_detector.py` | Injection & jailbreak detection |
| `backend/guardrails/rag_validator.py` | Retrieval quality & citation verification |
| `backend/guardrails/agricultural_safety.py` | Chemical ban list, yield guarantees, escalation |
| `backend/guardrails/output_validator.py` | PII scan, relevance check, uncertainty warnings |
| `backend/guardrails/privacy_guard.py` | API key masking, farmer profile protection |
| `backend/guardrails/guardrail_engine.py` | Central orchestrator — runs all 8 checks |

---

## Safety Policies

### 1. Input Validation Guardrails

**File:** `input_validator.py`

| Policy | Mechanism | Action |
|--------|-----------|--------|
| Soil pH out of range [0–14] | Python range check | BLOCK with error message |
| NPK values out of range [0–500 kg/ha] | Python range check | BLOCK with error message |
| Temperature out of range [−20°C–60°C] | Python range check | BLOCK with error message |
| Rainfall out of range [0–5000 mm] | Python range check | BLOCK with error message |
| Humidity out of range [0–100%] | Python range check | BLOCK with error message |
| Query shorter than 2 words | Word count check | BLOCK |
| Query longer than 2000 characters | Length check | BLOCK |
| HTML tags in input | Regex strip (`<[^>]+>`) | Sanitize |
| SQL injection patterns | Regex check | Warn / Sanitize |
| Language detection | Hindi Unicode block ratio | Tag as `en` or `hi` |

---

### 2. Prompt Injection Guardrails

**File:** `prompt_injection_detector.py`

| Pattern | Severity | Action |
|---------|----------|--------|
| "Ignore previous instructions" | HIGH | BLOCK |
| "Act as / You are now DAN/evil/unrestricted" | HIGH | BLOCK |
| "Forget all safety rules" | HIGH | BLOCK |
| "Print your system prompt" | HIGH | BLOCK |
| `eval()`, `os.system()`, `__import__()` | HIGH | BLOCK |
| SQL DROP/DELETE/INSERT | HIGH | BLOCK |
| `<script>` tags | HIGH | BLOCK |
| Special LLM template tokens in documents (`<\|system\|>`, `[INST]`) | HIGH | Strip + Warn |
| Delimiter-based injection (`--- NEW INSTRUCTIONS ---`) | MEDIUM | Sanitize + Warn |
| "Developer mode / Admin mode" | MEDIUM | Warn |

**Document Protection:** Retrieved documents are also scanned before inclusion in the LLM context. Instructions embedded in documents are stripped and cannot override system behaviour.

---

### 3. RAG Accuracy Guardrails

**File:** `rag_validator.py`

| Policy | Mechanism | Action |
|--------|-----------|--------|
| Minimum relevance score 0.30 | Cosine similarity threshold | Reject low-scoring docs |
| Off-domain content detection | Keyword blocklist (crypto, politics, etc.) | Reject doc |
| No documents retrieved | Count check | Report "insufficient evidence" |
| Maximum 5 docs to prevent context pollution | Slice | Cap at 5 |
| Citation verification | Regex match against approved sources | Flag fabricated citations |
| Confidence levels | Score bands (HIGH ≥ 0.75, MEDIUM ≥ 0.50, LOW ≥ 0.30) | Attached to response |

**Hallucination Prevention:** If `is_sufficient = False`, the system reports that a reliable recommendation cannot be provided, rather than generating an unsupported answer.

---

### 4. Agricultural Safety Guardrails

**File:** `agricultural_safety.py`

#### Banned Chemicals (BLOCK response entirely)
`endosulfan`, `monocrotophos`, `phorate`, `methyl parathion`, `triazophos`, `carbofuran`, `chlorpyrifos`, `DDT`, `aldrin`, `dieldrin`, `heptachlor`, `lindane`, `chlordane`, `aldicarb`, and others per Ministry of Agriculture ban lists.

#### Restricted Chemicals (Add mandatory PPE + label warning)
`glyphosate`, `paraquat`, `2,4-D`, `atrazine`, `cypermethrin`, `imidacloprid`, `acephate`, `chlorothalonil`, `dimethoate`, `malathion`, and others.

| Policy | Action |
|--------|--------|
| Banned chemical in query or response | BLOCK + error message |
| Restricted chemical in response | Append PPE + label reference warning |
| Dosage claim without registered label citation | Append label disclaimer |
| Yield/profit guarantee language | Append outcome-variability disclaimer |
| Serious disease / crop failure scenario | Set `requires_expert=True` + escalation message |
| All recommendations | Append educational disclaimer |

---

### 5. Live Data Guardrails

**File:** `live_data_service.py`

| Policy | Mechanism |
|--------|-----------|
| Mock data must be labelled `data_source: MOCK` | `is_mock=True` flag on all mock responses |
| Cached data labelled `data_source: CACHED` | Cache metadata |
| Live data labelled `data_source: LIVE` | Direct API flag |
| API failure returns `UNAVAILABLE` status | try/except → `_unavailable()` |
| Last-updated timestamps included | UTC ISO timestamp on every response |
| No invented prices or weather data | Only return real API or clearly-marked mock |

---

### 6. Output Validation Guardrails

**File:** `output_validator.py`

| Policy | Action |
|--------|--------|
| Empty response | BLOCK; return error message |
| Off-topic response (hacking, crypto, politics) | BLOCK; redirect to agriculture |
| PII in response (Aadhaar, PAN, phone, email, IFSC) | BLOCK; privacy protection message |
| Credential data (API keys, tokens) | BLOCK |
| LOW/INSUFFICIENT confidence | Append uncertainty disclaimer |
| Response lacks agricultural context | Warn |
| Very short response (< 10 words) | Warn |

---

### 7. Privacy & Security Guardrails

**File:** `privacy_guard.py`

| Policy | Mechanism |
|--------|-----------|
| API key masking in logs | Regex credential patterns |
| PII masking (Aadhaar, PAN, phone, email) | Named regex replacements |
| Sensitive env vars never logged | `safe_env()` checks for missing keys |
| Farmer profile sanitisation | Allowlist of safe fields only |
| Backend credential exposure in response | `validate_response_for_secrets()` |

---

### 8. Language & Accessibility

| Policy | Mechanism |
|--------|-----------|
| Hindi detection | Unicode range `\u0900–\u097F` ratio check |
| English default | Fallback when language undetected |
| Non-EN/HI warning | Soft warning, response proceeds in English |
| Agricultural terminology preservation | No translation pipeline in core path |

---

## Enforcement Summary

| Stage | Guardrail | Blocking? |
|-------|-----------|-----------|
| Pre-gen | Invalid input text | ✅ Hard block |
| Pre-gen | Invalid soil parameters | ✅ Hard block |
| Pre-gen | Prompt injection (HIGH severity) | ✅ Hard block |
| Pre-gen | Banned chemical in query | ✅ Hard block |
| Pre-gen | No/insufficient retrieved docs | ⚠️ Soft warn + disclaimer |
| Post-gen | Banned chemical in response | ✅ Hard block |
| Post-gen | Restricted chemical in response | ⚠️ Append safety warning |
| Post-gen | Yield guarantee language | ⚠️ Append disclaimer |
| Post-gen | PII/credential in response | ✅ Hard block |
| Post-gen | Off-topic response | ✅ Hard block |
| Post-gen | Low/insufficient confidence | ⚠️ Append uncertainty warning |

---

## Testing

Automated tests: `tests/test_guardrails.py`

```
pytest tests/test_guardrails.py -v
```

| Test Class | Coverage Area |
|------------|--------------|
| `TestInputValidatorSoilParams` | Invalid soil parameters (10 tests) |
| `TestPromptInjectionDetector` | Injection attempts (11 tests) |
| `TestRAGValidator` | Irrelevant documents, citations (9 tests) |
| `TestAgriculturalSafety` | Unsafe chemicals, claims (10 tests) |
| `TestLiveDataGuardrails` | Mock/unavailable data labelling (4 tests) |
| `TestPrivacyGuard` | Sensitive info leakage (10 tests) |
| `TestLanguageHandling` | Hindi & English inputs (8 tests) |
| `TestOutputValidator` | Output validation (5 tests) |
| `TestGuardrailEngineIntegration` | Full pipeline integration (6 tests) |

---

## Known Limitations

1. **Toxicity & Bias scores** are placeholder values (0.02, 0.03). A dedicated classifier (e.g., Detoxify) is needed for production-grade toxicity detection.

2. **Semantic similarity** uses Jaccard overlap as a proxy. For higher accuracy, integrate `sentence-transformers` scoring directly.

3. **Hindi language support** relies on Unicode block detection. Full NLP for Devanagari agricultural terminology would require a Hindi NLP library.

4. **Chemical ban list** covers major Indian prohibitions. Users should verify against the latest CIB&RC gazette notifications for completeness.

5. **Injection detection** covers known patterns. Adversarially crafted obfuscated prompts may bypass current regex patterns.

6. **RAG relevance scoring** in mock mode uses keyword overlap — not true semantic embeddings. Install ChromaDB + sentence-transformers for production quality.

7. **Cost per request** figures are illustrative. Actual IBM watsonx.ai costs depend on model selection and subscription tier.

---

## Configuration

Key thresholds are set in `env.txt` / environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_INPUT_LENGTH` | 2000 | Maximum query length |
| `MIN_RETRIEVAL_RELEVANCE_SCORE` | 0.3 | Minimum cosine similarity |
| `HALLUCINATION_THRESHOLD` | 0.7 | Confidence below which uncertainty is flagged |
| `GRANITE_MODEL_ID` | `ibm/granite-13b-instruct-v2` | IBM Granite model |

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI backend
uvicorn backend.main:app --reload --port 8000

# Run Streamlit frontend
streamlit run frontend/app.py

# Run tests
pytest tests/test_guardrails.py -v
```

---

*AgroSage Guardrails Framework — Protecting farmers with trustworthy AI.*
