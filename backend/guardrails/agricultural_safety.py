"""
Agricultural Safety Guardrail
Enforces safe recommendations for pesticides, chemicals, yield guarantees,
disease diagnosis, and professional escalation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger


# ─────────────────────── banned / restricted chemicals ───────────────────────

# Chemicals banned/restricted in India (partial list; extend as needed)
BANNED_CHEMICALS: List[str] = [
    "monocrotophos", "phorate", "methyl parathion", "triazophos",
    "carbofuran", "chlorpyrifos", "endosulfan", "heptachlor",
    "lindane", "aldrin", "dieldrin", "chlordane", "ddt",
    "aldicarb", "captan", "dicofol", "dinocap",
    "mancozeb in excessive dose",
]

# Chemicals that require a licensed agronomist / mandatory label reference
RESTRICTED_CHEMICALS: List[str] = [
    "glyphosate", "paraquat", "2,4-d", "atrazine",
    "cypermethrin", "imidacloprid", "acephate",
    "chlorothalonil", "dimethoate", "malathion",
    "thiamethoxam", "deltamethrin",
]

# Phrases that imply guaranteed outcomes (must not appear in responses)
GUARANTEE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(guarantee|guaranteed|certain|definitely will|100%|sure to)\b.{0,60}\b(yield|profit|cure|recover|eliminate)\b", re.I),
    re.compile(r"\bwill (definitely|certainly|absolutely)\b.{0,40}\b(grow|cure|prevent|eliminate)\b", re.I),
    re.compile(r"\b(no risk|risk-free|fail-safe)\b", re.I),
]

# Phrases indicating a dosage claim without label citation
DOSAGE_PATTERN = re.compile(
    r"\b(\d[\d.]*\s*(ml|l|g|kg|oz|litre|liter|gram|kilogram))\b.{0,60}\b(per|\/)\s*(acre|hectare|litre|plant|tree)\b",
    re.I,
)

# Serious disease indicators that require expert escalation
ESCALATION_TRIGGERS: List[str] = [
    "complete crop failure", "total loss", "widespread disease",
    "unknown disease", "unidentified pathogen", "abnormal symptom",
    "rapid spread", "epidemic", "blights entire field",
]


# ─────────────────────── result ─────────────────────────────────────────────

@dataclass
class AgriSafetyResult:
    is_safe:       bool
    errors:        List[str] = field(default_factory=list)
    warnings:      List[str] = field(default_factory=list)
    requires_expert: bool = False
    modified_text: Optional[str] = None   # patched output if fixable


# ─────────────────────── checker ─────────────────────────────────────────────

class AgriculturalSafetyChecker:
    """
    Checks both user queries and AI-generated outputs for unsafe agricultural
    recommendations.
    """

    # ── check the user query ─────────────────────────────────────────────────

    def check_query(self, query: str) -> AgriSafetyResult:
        result = AgriSafetyResult(is_safe=True)
        query_lower = query.lower()

        for chem in BANNED_CHEMICALS:
            if chem in query_lower:
                result.is_safe = False
                result.errors.append(
                    f"Query references banned chemical '{chem}'. "
                    "This substance is prohibited for agricultural use in India."
                )

        for trigger in ESCALATION_TRIGGERS:
            if trigger in query_lower:
                result.requires_expert = True
                result.warnings.append(
                    f"Situation described ('{trigger}') may require consultation with "
                    "a qualified agricultural expert or Krishi Vigyan Kendra (KVK)."
                )

        return result

    # ── check the generated response ─────────────────────────────────────────

    def check_response(self, response: str) -> AgriSafetyResult:
        result = AgriSafetyResult(is_safe=True)
        modified = response

        # 1. Banned chemicals in response
        response_lower = response.lower()
        for chem in BANNED_CHEMICALS:
            if chem in response_lower:
                result.is_safe = False
                result.errors.append(
                    f"Response mentions banned chemical '{chem}'. "
                    "This recommendation has been blocked."
                )

        # 2. Restricted chemicals — add mandatory label warning
        for chem in RESTRICTED_CHEMICALS:
            if chem in response_lower:
                warning_text = (
                    f"\n\n⚠️ **Safety Notice**: {chem.title()} is a restricted pesticide. "
                    "Always follow the registered product label, wear appropriate PPE "
                    "(gloves, mask, protective clothing), and adhere to the pre-harvest "
                    "interval. Consult a licensed agronomist before application."
                )
                if warning_text not in modified:
                    modified += warning_text
                result.warnings.append(
                    f"Restricted chemical '{chem}' mentioned. Safety notice appended."
                )

        # 3. Dosage claims without label citation
        if DOSAGE_PATTERN.search(response):
            if "product label" not in response_lower and "registered label" not in response_lower:
                modified += (
                    "\n\n📋 **Important**: Any dosage mentioned is indicative only. "
                    "Always verify exact dosages on the registered product label before application."
                )
                result.warnings.append(
                    "Dosage claim detected without label citation. Disclaimer appended."
                )

        # 4. Yield / profit guarantees
        for pattern in GUARANTEE_PATTERNS:
            if pattern.search(response):
                result.warnings.append(
                    "Response contains outcome guarantee language. "
                    "Agricultural results depend on many uncontrollable factors."
                )
                # Soft-block: append disclaimer rather than hard block
                if "results may vary" not in response_lower:
                    modified += (
                        "\n\n📢 **Disclaimer**: Agricultural outcomes vary based on soil health, "
                        "weather, pest pressure, and farm management practices. "
                        "No guarantee of yield, profit, or disease recovery is implied."
                    )
                break

        # 5. Escalation triggers in response
        for trigger in ESCALATION_TRIGGERS:
            if trigger in response.lower():
                result.requires_expert = True
                result.warnings.append(
                    f"Response discusses '{trigger}'. Expert escalation recommended."
                )
                modified += (
                    "\n\n🚨 **Recommendation**: This situation appears serious. "
                    "Please contact your nearest Krishi Vigyan Kendra (KVK) or "
                    "State Agricultural Department for expert guidance immediately."
                )
                break

        result.modified_text = modified if modified != response else None
        return result

    # ── utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def educational_disclaimer() -> str:
        return (
            "\n\n*This information is provided for educational purposes only and does not "
            "constitute professional agricultural advice. For site-specific recommendations, "
            "consult a qualified agronomist or your local agricultural extension officer.*"
        )
