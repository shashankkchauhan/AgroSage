"""
RAG Service
Retrieval-Augmented Generation pipeline using ChromaDB + sentence-transformers.
Integrates with guardrails for relevance validation and injection detection.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./backend/data/chroma_db")
COLLECTION_NAME = "agrosage_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


# ─────────────────────── mock knowledge base ─────────────────────────────────

MOCK_KNOWLEDGE_BASE = [
    {
        "id": "doc_wheat_1",
        "text": "Wheat cultivation guide: Wheat (Triticum aestivum) is a major rabi crop in India. Best sowing time is October to December. Recommended soil: loamy with pH 6.0–7.5. Fertilizer: 120:60:40 NPK kg/ha. Water requirement: 450–650 mm. Varieties: HD-2967, PBW-343, GW-322.",
        "source": "ICAR Wheat Production Guide 2023",
        "metadata": {"crop": "wheat", "category": "cultivation"},
    },
    {
        "id": "doc_rice_1",
        "text": "Rice cultivation: Paddy is the most important kharif crop. Transplanting: June–July. Soil: clayey loam, pH 5.5–6.5. Fertilizer: 100:50:50 kg/ha NPK. Key diseases: Blast (Pyricularia oryzae), Bacterial Blight (Xanthomonas oryzae). Use resistant varieties like Pusa Basmati 1121.",
        "source": "Ministry of Agriculture Rice Guide",
        "metadata": {"crop": "rice", "category": "cultivation"},
    },
    {
        "id": "doc_soil_1",
        "text": "Soil health management: Soil pH 6.5–7.5 is ideal for most crops. For acidic soils (pH < 6), apply agricultural lime @ 2–4 tonnes/ha. For alkaline soils (pH > 8), apply gypsum @ 2 tonnes/ha. Organic matter improves soil structure and water retention. Add FYM or compost @ 10–15 tonnes/ha.",
        "source": "NABARD Soil Health Card Guide",
        "metadata": {"topic": "soil", "category": "soil_management"},
    },
    {
        "id": "doc_pest_1",
        "text": "Integrated Pest Management (IPM): Use pheromone traps for monitoring. For aphids on wheat, spray Imidacloprid 17.8 SL @ 0.5 ml/L (follow registered label). Neem oil 5% effective against soft-bodied insects. Always wear PPE: gloves, mask, full-body cover. Pre-harvest interval must be observed.",
        "source": "CIB&RC Pesticide Application Guidelines",
        "metadata": {"topic": "pest_control", "category": "ipm"},
    },
    {
        "id": "doc_market_1",
        "text": "Minimum Support Price (MSP) 2023-24: Wheat ₹2,275/quintal, Paddy (common) ₹2,183/quintal, Maize ₹2,090/quintal, Cotton (medium) ₹6,620/quintal. MSP is announced by CACP and notified by Government of India. Farmers should sell at MSP or above at designated procurement centres.",
        "source": "CACP MSP Circular 2023-24",
        "metadata": {"topic": "market_prices", "category": "msp"},
    },
    {
        "id": "doc_tomato_1",
        "text": "Tomato cultivation: Warm-season crop, optimum temperature 21–27°C. Soil: well-drained loamy, pH 6.0–7.0. FYM: 25 tonnes/ha. NPK: 100:60:60 kg/ha. Common diseases: Early Blight (Alternaria solani) — spray Mancozeb 0.25%; Late Blight (Phytophthora infestans) — spray Metalaxyl+Mancozeb. Varieties: Pusa Ruby, Arka Vikas.",
        "source": "ICAR Vegetable Crop Production Manual",
        "metadata": {"crop": "tomato", "category": "cultivation"},
    },
    {
        "id": "doc_weather_1",
        "text": "Weather-based crop advisory: During excess rainfall (>150mm/week), drain fields to prevent waterlogging. Heat stress (>42°C) during wheat grain filling reduces yield — irrigate at 10-day intervals. Cold wave advisory: cover nursery beds with polythene. Hailstorm: apply potassium nitrate 1% spray post-hail.",
        "source": "IMD Agro-meteorological Advisory Services",
        "metadata": {"topic": "weather", "category": "advisory"},
    },
    {
        "id": "doc_govt_1",
        "text": "Government schemes for farmers: PM-KISAN provides ₹6,000/year direct benefit transfer. Pradhan Mantri Fasal Bima Yojana (PMFBY) offers crop insurance. Kisan Credit Card (KCC) for short-term credit. Soil Health Card scheme provides free soil testing. e-NAM platform for online agricultural marketing.",
        "source": "Ministry of Agriculture Government Schemes 2024",
        "metadata": {"topic": "government_schemes", "category": "policy"},
    },
]


# ─────────────────────── service ─────────────────────────────────────────────

class RAGService:
    """
    RAG pipeline: embed query → retrieve relevant chunks → return with scores.
    Uses ChromaDB for vector storage, falls back to in-memory mock.
    """

    def __init__(self) -> None:
        self._collection = None
        self._embedder   = None
        self._use_chroma = False
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._embedder = SentenceTransformer(EMBED_MODEL)
            self._use_chroma = True

            # Seed with mock KB if empty
            if self._collection.count() == 0:
                self._seed_collection()
            logger.info(f"ChromaDB initialised: {self._collection.count()} documents.")
        except Exception as exc:
            logger.warning(f"ChromaDB unavailable ({exc}). Using in-memory mock retrieval.")
            self._use_chroma = False

    def _seed_collection(self) -> None:
        texts    = [d["text"] for d in MOCK_KNOWLEDGE_BASE]
        ids      = [d["id"]   for d in MOCK_KNOWLEDGE_BASE]
        metas    = [{"source": d["source"], **d["metadata"]} for d in MOCK_KNOWLEDGE_BASE]
        embeds   = self._embedder.encode(texts).tolist()
        self._collection.add(documents=texts, ids=ids, metadatas=metas, embeddings=embeds)
        logger.info(f"Seeded {len(texts)} documents into ChromaDB.")

    # ── public ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Return top-k relevant document chunks with cosine similarity scores.
        Each item: {"content", "source", "score", "metadata", "is_mock"}
        """
        if self._use_chroma and self._embedder:
            return self._chroma_retrieve(query, top_k)
        return self._mock_retrieve(query, top_k)

    def ingest_document(self, text: str, source: str, metadata: Dict) -> bool:
        """Add a document to the vector store."""
        if not self._use_chroma:
            logger.warning("Cannot ingest: ChromaDB not available.")
            return False
        try:
            doc_id = f"user_{hash(text) & 0xFFFFFF}"
            embed  = self._embedder.encode([text]).tolist()
            self._collection.add(
                documents=[text], ids=[doc_id],
                metadatas=[{"source": source, **metadata}],
                embeddings=embed,
            )
            return True
        except Exception as exc:
            logger.error(f"Ingestion error: {exc}")
            return False

    # ── private ──────────────────────────────────────────────────────────────

    def _chroma_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_embed = self._embedder.encode([query]).tolist()
        results = self._collection.query(query_embeddings=q_embed, n_results=top_k)
        docs = []
        for i, doc in enumerate(results["documents"][0]):
            dist = results["distances"][0][i]  # cosine distance
            score = max(0.0, 1.0 - dist)        # convert to similarity
            meta  = results["metadatas"][0][i] if results["metadatas"] else {}
            docs.append({
                "content":  doc,
                "source":   meta.get("source", "Unknown"),
                "score":    round(score, 4),
                "metadata": meta,
                "is_mock":  False,
            })
        return docs

    def _mock_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based mock retrieval."""
        query_words = set(query.lower().split())
        scored = []
        for doc in MOCK_KNOWLEDGE_BASE:
            doc_words = set(doc["text"].lower().split())
            overlap   = len(query_words & doc_words)
            score     = min(1.0, overlap / max(len(query_words), 1) * 1.5)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_k]:
            results.append({
                "content":  doc["text"],
                "source":   doc["source"],
                "score":    round(score, 4),
                "metadata": doc["metadata"],
                "is_mock":  True,
            })
        return results
