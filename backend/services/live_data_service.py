"""
Live Data Service
Fetches real-time weather forecasts and mandi (market) prices.
Applies guardrails to never display mock data as real-time.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_BASE    = os.getenv("WEATHER_API_URL", "https://api.openweathermap.org/data/2.5")
MANDI_BASE      = os.getenv("MANDI_API_URL", "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070")
MANDI_API_KEY   = os.getenv("MANDI_API_KEY", "")

TIMEOUT = 8.0  # seconds


class LiveDataService:
    """
    Fetches weather and mandi prices from live APIs.
    Clearly labels all data as LIVE, CACHED, or MOCK.
    """

    def __init__(self) -> None:
        self._weather_cache: Dict[str, Any] = {}
        self._mandi_cache:   Dict[str, Any] = {}
        self._cache_ttl = 1800  # 30 minutes

    # ── Weather ──────────────────────────────────────────────────────────────

    async def get_weather(self, location: str) -> Dict[str, Any]:
        """Return weather data for a location."""
        cache_key = location.lower().strip()
        cached = self._get_cached(self._weather_cache, cache_key)
        if cached:
            return {**cached, "data_source": "CACHED", "is_mock": False}

        if not WEATHER_API_KEY:
            logger.warning("WEATHER_API_KEY not set — returning mock weather data.")
            return self._mock_weather(location)

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(
                    f"{WEATHER_BASE}/weather",
                    params={"q": location, "appid": WEATHER_API_KEY, "units": "metric"},
                )
                resp.raise_for_status()
                data = resp.json()
                result = {
                    "location":    data.get("name", location),
                    "temperature": data["main"]["temp"],
                    "humidity":    data["main"]["humidity"],
                    "rainfall":    data.get("rain", {}).get("1h", 0.0),
                    "description": data["weather"][0]["description"].title(),
                    "wind_speed":  data["wind"]["speed"],
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "data_source": "LIVE",
                    "is_mock":     False,
                }
                self._set_cache(self._weather_cache, cache_key, result)
                return result
        except httpx.HTTPStatusError as exc:
            logger.error(f"Weather API error {exc.response.status_code}: {exc}")
            return self._unavailable("weather", location)
        except Exception as exc:
            logger.error(f"Weather fetch error: {exc}")
            return self._unavailable("weather", location)

    # ── Mandi prices ─────────────────────────────────────────────────────────

    async def get_mandi_prices(
        self,
        commodity: str,
        state: str = "Maharashtra",
    ) -> Dict[str, Any]:
        """Return mandi prices for a commodity."""
        cache_key = f"{commodity}_{state}".lower()
        cached = self._get_cached(self._mandi_cache, cache_key)
        if cached:
            return {**cached, "data_source": "CACHED", "is_mock": False}

        if not MANDI_API_KEY:
            logger.warning("MANDI_API_KEY not set — returning mock mandi data.")
            return self._mock_mandi(commodity, state)

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(
                    MANDI_BASE,
                    params={
                        "api-key": MANDI_API_KEY,
                        "format":  "json",
                        "filters[commodity]": commodity.title(),
                        "filters[state]":     state.title(),
                        "limit":  5,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                records = data.get("records", [])
                if not records:
                    return self._unavailable("mandi", commodity)

                prices = [
                    {
                        "market":    r.get("market"),
                        "district":  r.get("district"),
                        "min_price": r.get("min_price"),
                        "max_price": r.get("max_price"),
                        "modal_price": r.get("modal_price"),
                        "date":      r.get("arrival_date"),
                    }
                    for r in records
                ]
                result = {
                    "commodity":   commodity.title(),
                    "state":       state.title(),
                    "prices":      prices,
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "data_source": "LIVE",
                    "is_mock":     False,
                }
                self._set_cache(self._mandi_cache, cache_key, result)
                return result
        except Exception as exc:
            logger.error(f"Mandi API error: {exc}")
            return self._unavailable("mandi", commodity)

    # ── mock / unavailable ───────────────────────────────────────────────────

    def _mock_weather(self, location: str) -> Dict[str, Any]:
        return {
            "location":    location,
            "temperature": 28.0,
            "humidity":    65.0,
            "rainfall":    0.0,
            "description": "Partly Cloudy",
            "wind_speed":  12.0,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "data_source": "MOCK",
            "is_mock":     True,
            "notice":      "⚠️ This is mock data. Live API key not configured.",
        }

    def _mock_mandi(self, commodity: str, state: str) -> Dict[str, Any]:
        mock_prices = {
            "tomato":  {"min": 800,  "max": 1500, "modal": 1200},
            "onion":   {"min": 600,  "max": 1200, "modal": 900},
            "wheat":   {"min": 2000, "max": 2300, "modal": 2150},
            "potato":  {"min": 500,  "max": 900,  "modal": 700},
            "rice":    {"min": 1800, "max": 2300, "modal": 2100},
            "maize":   {"min": 1800, "max": 2200, "modal": 2000},
            "cotton":  {"min": 5800, "max": 7000, "modal": 6400},
        }
        p = mock_prices.get(commodity.lower(), {"min": 1000, "max": 2000, "modal": 1500})
        return {
            "commodity":   commodity.title(),
            "state":       state,
            "prices": [{
                "market":      "Pune APMC (Mock)",
                "district":    "Pune",
                "min_price":   p["min"],
                "max_price":   p["max"],
                "modal_price": p["modal"],
                "date":        datetime.now(timezone.utc).strftime("%d-%m-%Y"),
            }],
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "data_source": "MOCK",
            "is_mock":     True,
            "notice":      "⚠️ This is mock data. Live API key not configured.",
        }

    def _unavailable(self, service: str, identifier: str) -> Dict[str, Any]:
        return {
            "status":    "UNAVAILABLE",
            "service":   service,
            "identifier": identifier,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "UNAVAILABLE",
            "is_mock":   False,
            "notice":    f"⚠️ {service.title()} data for '{identifier}' is currently unavailable. Please try again later.",
        }

    # ── cache helpers ─────────────────────────────────────────────────────────

    def _get_cached(self, cache: Dict, key: str) -> Optional[Dict]:
        entry = cache.get(key)
        if entry and (time.time() - entry["_cached_at"] < self._cache_ttl):
            return entry
        return None

    def _set_cache(self, cache: Dict, key: str, value: Dict) -> None:
        cache[key] = {**value, "_cached_at": time.time()}
