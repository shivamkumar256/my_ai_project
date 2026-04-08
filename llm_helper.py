"""
llm_helper.py
=============
LLM Triage Helper — standalone module (no coupling to env internals).

Calls an LLM (Google Gemini by default) with a system prompt that
frames it as a medical triage assistant.  The result is used ONLY as a
*hint* to the human/agent; ground-truth triage decisions remain rule-based
via Patient.ideal_tier and the environment's severity score.

Usage
-----
>>> from ed_triage.llm_helper import analyze_complaint, set_api_key
>>> set_api_key("AIza...")           # or set GEMINI_API_KEY env var
>>> hint = analyze_complaint("chest pain with diaphoresis")
>>> print(hint)
{
  "suspected_condition": "Acute Coronary Syndrome",
  "suggested_tier": "icu",
  "confidence": 0.95,
  "red_flags": ["diaphoresis", "chest pain", "possible MI"]
}

If the LLM call fails (no key, network error, bad JSON), the module silently
falls back to a fast rule-based classifier that maps keyword patterns to tiers.
"""

from __future__ import annotations

import json
import os
import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level API key override (alternative to env var)
# ---------------------------------------------------------------------------

_api_key_override: Optional[str] = None


def set_api_key(key: str) -> None:
    """Set the Gemini API key programmatically (overrides GEMINI_API_KEY env var)."""
    global _api_key_override
    _api_key_override = key


def _get_api_key() -> Optional[str]:
    return _api_key_override or os.getenv("AIzaSyDJwjZ4QB5G8a7iCNu-yRyhbxcj6lRH61k")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a medical triage assistant in a busy emergency department. "
    "Given a patient's chief complaint, analyze it and return ONLY a single "
    "valid JSON object with exactly these keys:\n"
    "  suspected_condition  – most likely diagnosis (string)\n"
    "  suggested_tier       – one of: \"icu\", \"general\", \"observation\" (string)\n"
    "  confidence           – your confidence in the tier suggestion, 0.0–1.0 (float)\n"
    "  red_flags            – list of warning signs present in the complaint (list of strings)\n"
    "Do NOT include markdown, explanations, or any text outside the JSON object."
)

_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Rule-based fallback classifier
# ---------------------------------------------------------------------------

# Each entry: (keywords, condition, tier, confidence, red_flags)
_RULE_TABLE: List[tuple] = [
    # Critical – ICU
    (["cardiac arrest", "code blue", "no pulse"],
     "Cardiac Arrest", "icu", 0.98, ["pulselessness", "loss of consciousness"]),
    (["chest pain", "diaphoresis"],
     "Acute Coronary Syndrome", "icu", 0.92, ["chest pain", "diaphoresis"]),
    (["massive haemorrhage", "massive hemorrhage", "uncontrolled bleeding"],
     "Haemorrhagic Shock", "icu", 0.95, ["haemorrhage", "haemodynamic instability"]),
    (["respiratory failure", "can't breathe", "cannot breathe", "apnoea"],
     "Respiratory Failure", "icu", 0.93, ["respiratory arrest", "hypoxia"]),
    (["stroke", "facial droop", "arm weakness", "speech difficulty"],
     "Acute Ischaemic Stroke", "icu", 0.91, ["neurological deficits", "FAST positive"]),
    (["septic shock", "sepsis", "high fever", "hypotension"],
     "Sepsis / Septic Shock", "icu", 0.89, ["fever", "hypotension", "altered mental status"]),
    (["altered mental status", "loss of consciousness", "unconscious"],
     "Altered Consciousness", "icu", 0.87, ["GCS drop", "unresponsiveness"]),

    # Emergent – General
    (["chest pain"],
     "Possible Acute Coronary Syndrome", "general", 0.78, ["chest pain"]),
    (["abdominal pain severe", "acute abdomen"],
     "Acute Abdomen", "general", 0.80, ["peritoneal signs"]),
    (["asthma", "wheezing", "dyspnoea", "shortness of breath"],
     "Asthma / Bronchospasm", "general", 0.75, ["wheeze", "dyspnoea"]),
    (["hyperglycaemia", "high blood sugar", "diabetic"],
     "Hyperglycaemic Episode", "general", 0.72, ["hyperglycaemia"]),
    (["seizure", "convulsion"],
     "Seizure Disorder", "general", 0.80, ["seizure activity"]),
    (["head injury", "head trauma", "concussion"],
     "Traumatic Brain Injury", "general", 0.76, ["head trauma", "altered GCS"]),

    # Urgent / Less urgent – Observation
    (["sprained", "sprain"],
     "Musculoskeletal Sprain", "observation", 0.85, []),
    (["ear pain", "otitis"],
     "Otitis Media / Externa", "observation", 0.90, []),
    (["vomiting", "nausea"],
     "Gastroenteritis", "observation", 0.80, []),
    (["uti", "urinary tract infection", "dysuria"],
     "Urinary Tract Infection", "observation", 0.88, []),
    (["back pain"],
     "Musculoskeletal Back Pain", "observation", 0.82, []),
    (["rash", "skin rash"],
     "Dermatological Complaint", "observation", 0.85, []),
    (["sore throat", "cold", "cough", "flu", "fever"],
     "Upper Respiratory Infection", "observation", 0.80, []),
    (["headache", "migraine"],
     "Headache / Migraine", "observation", 0.78, []),
    (["medication refill", "minor abrasion", "abrasion", "laceration"],
     "Minor Complaint", "observation", 0.92, []),
]


def _rule_based_classify(complaint: str) -> Dict[str, Any]:
    """
    Fast keyword-based fallback when the LLM is unavailable.

    Returns the same dict structure as the LLM response.
    """
    text = complaint.lower()
    for keywords, condition, tier, confidence, red_flags in _RULE_TABLE:
        if any(kw in text for kw in keywords):
            return {
                "suspected_condition": condition,
                "suggested_tier": tier,
                "confidence": confidence,
                "red_flags": red_flags,
                "_source": "rule_based",
            }
    # Default: treat as minor observation case
    return {
        "suspected_condition": "Unspecified Complaint",
        "suggested_tier": "observation",
        "confidence": 0.40,
        "red_flags": [],
        "_source": "rule_based_default",
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Robustly parse JSON, stripping out any markdown code blocks."""
    if not text:
        raise ValueError("Empty response from LLM.")
    text = text.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    return json.loads(text)

def _call_gemini(complaint: str, api_key: str) -> Dict[str, Any]:
    """
    Call Google Gemini API synchronously.

    Raises exceptions on network / API errors so the caller can fall back.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "google-genai package not installed. Run: pip install google-genai"
        ) from e

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_MODEL,
        contents=f"Patient complaint: {complaint}",
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=256,
            response_mime_type="application/json",
        )
    )
    
    raw = response.text
    result = _parse_json_response(raw)
    result["_source"] = "llm"
    return result

def chat_with_patient_ai(
    patient_context: str,
    user_message: str,
    chat_history: List[Dict[str, str]],
    api_key: Optional[str] = None
) -> str:
    """
    Allows a conversational chat about a specific patient using Gemini.
    Returns the AI's response text.
    """
    effective_key = api_key or _get_api_key()
    if not effective_key:
        return "⚠️ Please provide a GEMINI_API_KEY to enable AI chat."
        
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "⚠️ google-genai package not installed."

    client = genai.Client(api_key=effective_key)
    
    system_instruction = (
        "You are an expert ED clinical assistant. You are helping a user (triage nurse or doctor) "
        "understand the status, vitals, and condition of a currently waiting patient. "
        "Keep your responses friendly, very concise, and highly clinical. "
        f"Here is the data for the patient you are discussing:\n{patient_context}"
    )
    
    formatted_contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        formatted_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    
    formatted_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    try:
        response = client.models.generate_content(
            model=_MODEL,
            contents=formatted_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            )
        )
        return response.text
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return f"⚠️ Chat processing failed: {e}"


def get_autonomous_action(state: 'EDState', api_key: Optional[str] = None) -> 'Action':
    """
    Acts as an autonomous LLM agent. Given the full environment state,
    it selects the best action for the current patient.
    """
    from ed_triage import Action
    effective_key = api_key or _get_api_key()
    if not effective_key:
        return Action.HOLD_QUEUE
        
    patient = state.current_patient
    if not patient:
        return Action.HOLD_QUEUE

    context = (
        f"Available Beds: ICU={state.available_beds.icu}, "
        f"General={state.available_beds.general}, "
        f"Observation={state.available_beds.observation}\n"
        f"Current Patient: {patient.name}, Age: {patient.age}, "
        f"Complaint: {patient.chief_complaint}\n"
        f"Vitals: BP {patient.vitals.bp_systolic}/{patient.vitals.bp_diastolic}, "
        f"HR {patient.vitals.hr}, SpO2 {patient.vitals.spo2}%, Temp {patient.vitals.temp}C\n"
        f"Severity Score: {patient.severity_score}, Ideal Tier: {patient.ideal_tier}\n"
    )
    
    system_instruction = (
        "You are an AI autonomous triage agent controlling an emergency department. "
        "Review the patient conditions and the available beds. "
        "Output ONLY valid JSON matching this schema: {'action': 'value'} "
        "where value MUST be one of: 'ASSIGN_ICU', 'ASSIGN_GENERAL', 'ASSIGN_OBSERVATION', 'HOLD_QUEUE', 'DISCHARGE'."
    )
    
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=effective_key)
        
        response = client.models.generate_content(
            model=_MODEL,
            contents=context,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json",
            )
        )
        data = _parse_json_response(response.text)
        action_str = data.get("action", "")
        
        # safely map the returned string to the enum Action equivalent 
        mapping = {
            "ASSIGN_ICU": Action.ASSIGN_ICU,
            "ASSIGN_GENERAL": Action.ASSIGN_GENERAL,
            "ASSIGN_OBSERVATION": Action.ASSIGN_OBSERVATION,
            "HOLD_QUEUE": Action.HOLD_QUEUE,
            "DISCHARGE": Action.DISCHARGE
        }
        return mapping.get(action_str, Action.HOLD_QUEUE)
        
    except Exception as e:
        logger.error(f"Autonomous agent failed: {e}")
        return Action.HOLD_QUEUE


def _validate_and_normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the returned dict has all required keys with correct types.
    Mutates *data* in-place and returns it.
    """
    valid_tiers = {"icu", "general", "observation"}

    # Normalise tier
    tier = str(data.get("suggested_tier", "observation")).strip().lower()
    data["suggested_tier"] = tier if tier in valid_tiers else "observation"

    # Clamp confidence
    try:
        conf = float(data.get("confidence", 0.5))
        data["confidence"] = round(max(0.0, min(1.0, conf)), 3)
    except (TypeError, ValueError):
        data["confidence"] = 0.5

    # Ensure red_flags is a list of strings
    flags = data.get("red_flags", [])
    if not isinstance(flags, list):
        flags = [str(flags)]
    data["red_flags"] = [str(f) for f in flags]

    # Ensure suspected_condition is a string
    data["suspected_condition"] = str(
        data.get("suspected_condition", "Unknown")
    )

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_complaint(
    complaint_text: str,
    api_key: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Analyse a patient's chief complaint and return a triage hint dict.

    This function is intentionally *decoupled* from the environment — it
    acts as a read-only external advisor.  The env never calls it; agents
    (or the UI) call it to enrich their decision-making.

    Parameters
    ----------
    complaint_text : str
        Free-text chief complaint as entered by the patient or clinician.
    api_key : str, optional
        Gemini API key.  Falls back to ``GEMINI_API_KEY`` env var, then to
        the key set via :func:`set_api_key`.
    use_llm : bool
        If ``False``, skip the LLM entirely and use the rule-based fallback.
        Useful for offline testing or when no API key is available.

    Returns
    -------
    dict with keys
        ``suspected_condition`` (str)
            Best-guess diagnosis label.
        ``suggested_tier`` (str)
            One of ``"icu"``, ``"general"``, ``"observation"``.
        ``confidence`` (float)
            Confidence in the tier suggestion, in [0.0, 1.0].
        ``red_flags`` (list[str])
            List of warning signs detected in the complaint.
        ``_source`` (str)
            Either ``"llm"`` or ``"rule_based"`` / ``"rule_based_default"``
            indicating whether the LLM or the fallback produced the result.
    """
    if not complaint_text or not complaint_text.strip():
        return {
            "suspected_condition": "No complaint provided",
            "suggested_tier": "observation",
            "confidence": 0.0,
            "red_flags": [],
            "_source": "empty_input",
        }

    effective_key = api_key or _get_api_key()

    if use_llm and effective_key:
        try:
            result = _call_gemini(complaint_text.strip(), effective_key)
            return _validate_and_normalise(result)
        except Exception as exc:
            logger.warning(
                "LLM triage helper failed (%s: %s); falling back to rule-based.",
                type(exc).__name__,
                exc,
            )

    # Fallback
    result = _rule_based_classify(complaint_text)
    return _validate_and_normalise(result)


# ---------------------------------------------------------------------------
# Tier badge helper (for UI use)
# ---------------------------------------------------------------------------

TIER_COLOURS: Dict[str, str] = {
    "icu": "#e74c3c",          # red
    "general": "#f39c12",      # amber
    "observation": "#27ae60",  # green
}

TIER_LABELS: Dict[str, str] = {
    "icu": "ICU",
    "general": "General",
    "observation": "Observation",
}


def hint_badge_html(hint: Dict[str, Any]) -> str:
    """
    Return a small HTML badge string suitable for embedding in Streamlit
    markdown renders.

    Example output:
        <span style="...">ICU 95%</span>
    """
    tier = hint.get("suggested_tier", "observation")
    colour = TIER_COLOURS.get(tier, "#888")
    label = TIER_LABELS.get(tier, tier.upper())
    conf = hint.get("confidence", 0.0)
    src = hint.get("_source", "")
    src_tag = " [LLM]" if src == "llm" else " [rule]"
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 8px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600;">'
        f"{label} {conf:.0%}{src_tag}</span>"
    )
