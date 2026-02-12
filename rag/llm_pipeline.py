"""LLM pipeline — Groq API (Llama 3.3 70B) with hallucination guard.

Supports:
  • Single-shot generate(prompt)
  • Multi-turn chat(messages)
  • Hallucination detection + automatic regeneration
  • Confidence-scored responses
"""

import os
import requests

from rag.hallucination import detect_hallucination, build_regeneration_prompt
from rag.confidence import compute_confidence, format_confidence_badge


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_MESSAGE = (
    "You are a world-class air quality intelligence analyst specializing in "
    "Delhi/NCR pollution. You combine deep expertise across atmospheric science, "
    "environmental epidemiology, public health policy, and Indian regulatory "
    "frameworks to deliver precise, actionable analysis.\n\n"

    "## YOUR EXPERTISE\n"
    "- **India NAQI Standard**: 6-pollutant index (PM2.5, PM10, NO2, SO2, CO, O3) "
    "with breakpoints from 0-500. AQI = max sub-index. CPCB operates the national "
    "monitoring network with 40+ stations in Delhi/NCR.\n"
    "- **Delhi Pollution Sources**: Vehicular emissions (40-50% of NOx), "
    "construction dust, industrial emissions (brick kilns, power plants), "
    "crop residue/stubble burning (Oct-Nov spike), road dust resuspension, "
    "biomass cooking fuel, landfill fires, and secondary aerosol formation.\n"
    "- **Seasonal Patterns**: Winter (Nov-Feb) worst due to temperature inversion, "
    "low wind speed, and stubble burning. Monsoon (Jul-Sep) cleanest due to "
    "washout. Post-Diwali (late Oct) annual spike. Summer dust storms (May-Jun).\n"
    "- **Health Science**: PM2.5 penetrates deep into alveoli; long-term exposure "
    "linked to COPD, IHD, stroke, lung cancer, and reduced life expectancy "
    "(estimated 6-10 year reduction in Delhi). Children, elderly, and those with "
    "pre-existing cardiopulmonary conditions are most vulnerable.\n"
    "- **Policy Context**: GRAP (Graded Response Action Plan) stages I-IV, "
    "NCAP (National Clean Air Programme) targets 40% PM reduction by 2026, "
    "BS-VI emission norms, odd-even vehicle rationing, construction bans, "
    "Supreme Court interventions, EPCA/CAQM directives.\n"
    "- **WHO Guidelines**: PM2.5 annual 5 ug/m3, 24h 15 ug/m3; PM10 annual "
    "15 ug/m3, 24h 45 ug/m3 -- Delhi routinely exceeds by 10-20x.\n"
    "- **Comparative Context**: Delhi vs Beijing, Lahore, Dhaka, Mumbai. "
    "IQAir rankings. Progress of peer cities.\n"
    "- **Measurement Science**: Gravimetric vs beta-attenuation vs optical "
    "sensors. CAAQMS vs manual stations. Data quality issues.\n\n"

    "## RESPONSE RULES\n"
    "1. **Answer ONLY what the user asked.** Do NOT dump a generic AQI report "
    "unless specifically requested. Match response depth to question complexity.\n"
    "2. You receive live AQI data and knowledge-base excerpts as context. "
    "Use them when relevant; otherwise rely on your own expertise.\n"
    "3. Include **clickable hyperlinks** to authoritative sources: "
    "[text](https://url). Primary sources: WHO, CPCB (cpcb.nic.in), "
    "IQAir, US EPA, PubMed, The Lancet, TERI, CSE (cseindia.org).\n"
    "4. Use **markdown** formatting for clarity: headers, bullets, bold, tables.\n"
    "5. Match tone to question: conversational for casual, technical for deep.\n"
    "6. Use full conversation history to answer follow-ups contextually.\n"
    "7. Cite knowledge-base excerpts as [Ref N] when used.\n"
    "8. Vary your response structure naturally. Never start every answer the same.\n"
    "9. When discussing numbers, always include units and comparison benchmarks.\n"
    "10. For health advice, be specific about who is at risk and what actions "
    "to take, including mask type (N95/KN95), purifier CADR requirements, "
    "and exposure duration limits."
)


class LLMPipeline:
    def __init__(self, groq_api_key: str = ""):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    # ── Single-shot ─────────────────────────────────────────────────
    def generate(self, prompt: str, max_length: int = 1024) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        return self._call_groq(messages, max_length)

    # ── Multi-turn chat ─────────────────────────────────────────────
    def chat(self, messages: list, max_length: int = 1024) -> str:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + messages
        return self._call_groq(messages, max_length)

    # ── Chat with hallucination guard ───────────────────────────────
    def chat_with_guard(
        self,
        messages: list,
        query: str,
        retrieved_chunks: list | None = None,
        intent: str = "general",
        max_length: int = 1500,
    ) -> dict:
        """Generate response with hallucination detection and confidence.

        Returns:
            {
                "answer": str,
                "confidence": dict,
                "hallucination_check": dict,
                "regenerated": bool,
            }
        """
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + messages

        answer = self._call_groq(messages, max_length)
        regenerated = False

        # Hallucination check
        chunks = retrieved_chunks or []
        hall_check = detect_hallucination(answer, chunks, query)

        if hall_check["should_regenerate"]:
            # Build a stricter prompt and retry
            regen_prompt = build_regeneration_prompt(query, chunks)
            regen_messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": regen_prompt},
            ]
            new_answer = self._call_groq(regen_messages, max_length)
            if not new_answer.startswith("["):
                answer = new_answer
                regenerated = True
                hall_check = detect_hallucination(answer, chunks, query)

        # Confidence scoring
        confidence = compute_confidence(answer, chunks, query, intent)

        return {
            "answer": answer,
            "confidence": confidence,
            "hallucination_check": hall_check,
            "regenerated": regenerated,
        }

    # ── Core Groq call ──────────────────────────────────────────────
    def _call_groq(self, messages: list, max_length: int) -> str:
        if not self.groq_api_key:
            return "[LLM unavailable — configure a Groq API key in the sidebar]"
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": max_length,
                "temperature": 0.4,
                "top_p": 0.9,
            }
            resp = requests.post(
                GROQ_API_URL, headers=headers, json=payload, timeout=30
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                err = resp.text[:300]
                return f"[Groq API error {resp.status_code}: {err}]"
        except requests.Timeout:
            return "[Groq API timeout after 30s]"
        except Exception as exc:
            return f"[Groq API error: {exc}]"


if __name__ == "__main__":
    p = LLMPipeline()
    print(p.generate("Explain why PM2.5 matters for Delhi residents."))
