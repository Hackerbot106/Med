"""
Rules-based risk-flagging engine.

This module is intentionally NOT a diagnostic engine. It never outputs a
disease name or a treatment. It only:
  1. Detects "urgency signals" (red-flag keywords / vital sign thresholds)
     from patient-reported text and OCR-extracted lab values.
  2. Produces a risk TIER (emergency / high / medium / low) purely to help
     a human reviewer prioritise their queue.
  3. Always attaches the plain-language reasons behind the tier, so a
     reviewer can sanity-check (and override) the system's assessment.

Every tier and reason must be reviewed by a qualified health worker,
nurse, doctor, or medical officer. This module does not replace that
review.
"""

import re

# ---------------------------------------------------------------------------
# Symptom ontology: canonical symptom -> metadata
#   aliases: alternate phrasings/synonyms to match in free text
#   category: broad clinical category (used for follow-up question templates)
#   red_flag: True if this symptom ALONE should trigger an emergency tier
#   weight: contribution to the numeric risk score
# ---------------------------------------------------------------------------
SYMPTOM_ONTOLOGY = {
    "chest_pain": {
        "aliases": ["chest pain", "chest tightness", "chest pressure", "seene mein dard"],
        "category": "cardiac", "red_flag": False, "weight": 25,
    },
    "breathlessness": {
        "aliases": ["breathlessness", "difficulty breathing", "shortness of breath",
                    "can't breathe", "cannot breathe", "gasping", "saans lene mein takleef"],
        "category": "respiratory", "red_flag": False, "weight": 25,
    },
    "unconsciousness": {
        "aliases": ["unconscious", "unresponsive", "fainted", "passed out", "loss of consciousness"],
        "category": "neuro", "red_flag": True, "weight": 40,
    },
    "seizure": {
        "aliases": ["seizure", "fits", "convulsion", "convulsions"],
        "category": "neuro", "red_flag": True, "weight": 40,
    },
    "stroke_signs": {
        "aliases": ["face drooping", "slurred speech", "sudden weakness one side",
                    "one sided weakness", "cannot speak properly", "facial droop"],
        "category": "neuro", "red_flag": True, "weight": 40,
    },
    "severe_bleeding": {
        "aliases": ["heavy bleeding", "severe bleeding", "uncontrolled bleeding",
                    "bleeding heavily", "blood loss"],
        "category": "trauma", "red_flag": True, "weight": 38,
    },
    "blue_lips": {
        "aliases": ["blue lips", "bluish skin", "cyanosis", "turning blue"],
        "category": "respiratory", "red_flag": True, "weight": 40,
    },
    "severe_abdominal_pain": {
        "aliases": ["severe abdominal pain", "severe stomach pain", "unbearable stomach pain"],
        "category": "gi", "red_flag": False, "weight": 22,
    },
    "high_fever": {
        "aliases": ["high fever", "fever", "temperature", "bukhar", "jwara"],
        "category": "infection", "red_flag": False, "weight": 10,
    },
    "persistent_vomiting": {
        "aliases": ["vomiting repeatedly", "unable to keep anything down", "persistent vomiting",
                    "vomiting", "throwing up"],
        "category": "gi", "red_flag": False, "weight": 12,
    },
    "diarrhea": {
        "aliases": ["diarrhea", "diarrhoea", "loose motions", "loose stools"],
        "category": "gi", "red_flag": False, "weight": 8,
    },
    "dehydration_signs": {
        "aliases": ["not urinating", "very little urine", "dry mouth", "sunken eyes",
                    "extreme thirst", "dizziness on standing"],
        "category": "general", "red_flag": False, "weight": 15,
    },
    "cough": {
        "aliases": ["cough", "coughing", "khansi"],
        "category": "respiratory", "red_flag": False, "weight": 6,
    },
    "headache": {
        "aliases": ["headache", "head pain", "sar dard"],
        "category": "neuro", "red_flag": False, "weight": 6,
    },
    "severe_headache_sudden": {
        "aliases": ["worst headache of my life", "sudden severe headache", "thunderclap headache"],
        "category": "neuro", "red_flag": True, "weight": 35,
    },
    "body_ache": {
        "aliases": ["body ache", "body pain", "muscle ache", "joint pain"],
        "category": "general", "red_flag": False, "weight": 5,
    },
    "rash": {
        "aliases": ["rash", "skin rash", "red spots", "itching"],
        "category": "derm", "red_flag": False, "weight": 6,
    },
    "fatigue": {
        "aliases": ["fatigue", "weakness", "tiredness", "lethargy", "no energy"],
        "category": "general", "red_flag": False, "weight": 6,
    },
    "sore_throat": {
        "aliases": ["sore throat", "throat pain", "difficulty swallowing"],
        "category": "ent", "red_flag": False, "weight": 5,
    },
    "burning_urination": {
        "aliases": ["burning urination", "painful urination", "burning while urinating"],
        "category": "urinary", "red_flag": False, "weight": 8,
    },
    "pregnancy_bleeding": {
        "aliases": ["bleeding in pregnancy", "spotting during pregnancy", "pregnant and bleeding"],
        "category": "obstetric", "red_flag": True, "weight": 38,
    },
    "reduced_fetal_movement": {
        "aliases": ["baby not moving", "reduced fetal movement", "no fetal movement"],
        "category": "obstetric", "red_flag": True, "weight": 32,
    },
    "infant_poor_feeding": {
        "aliases": ["not feeding", "refusing feed", "poor feeding", "not waking to feed"],
        "category": "pediatric", "red_flag": False, "weight": 20,
    },
    "injury_fracture": {
        "aliases": ["fracture", "broken bone", "deformed limb", "cannot move limb"],
        "category": "trauma", "red_flag": False, "weight": 18,
    },
    "chest_injury": {
        "aliases": ["chest injury", "hit on chest", "rib injury"],
        "category": "trauma", "red_flag": False, "weight": 15,
    },
    "mental_health_crisis": {
        "aliases": ["wants to harm self", "talking about ending life", "severe distress",
                    "suicidal", "harm themselves"],
        "category": "mental_health", "red_flag": True, "weight": 40,
    },
    "occupational_exposure": {
        "aliases": ["chemical exposure", "gas leak exposure", "fumes exposure",
                    "inhaled fumes", "chemical splash"],
        "category": "occupational", "red_flag": False, "weight": 20,
    },
}

# Duration / severity phrase patterns
DURATION_PATTERN = re.compile(
    r"(\d+)\s*(hour|hours|hr|hrs|day|days|week|weeks|month|months)", re.IGNORECASE
)
SEVERITY_WORDS = {
    "mild": 1, "moderate": 2, "severe": 3, "unbearable": 4, "extreme": 4, "worst": 4,
}

# Vital sign extraction patterns -> (regex, canonical field)
VITALS_PATTERNS = {
    "temperature_f": re.compile(r"(\d{2,3}(?:\.\d)?)\s*(?:deg(?:ree)?s?\s*)?f\b", re.IGNORECASE),
    "temperature_c": re.compile(r"(\d{2,3}(?:\.\d)?)\s*(?:deg(?:ree)?s?\s*)?c\b", re.IGNORECASE),
    "blood_pressure": re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*hg)?", re.IGNORECASE),
    "heart_rate": re.compile(r"(?:hr|heart rate|pulse)\D{0,5}(\d{2,3})\b", re.IGNORECASE),
    # sp[o0]2 tolerates common OCR letter/digit confusion ("SpO2" read as "Sp02")
    "spo2": re.compile(r"(?:sp[o0]2|oxygen saturation|sats?)\D{0,5}(\d{2,3})\s*%?", re.IGNORECASE),
    "respiratory_rate": re.compile(r"(?:rr|respiratory rate|breaths? per minute)\D{0,5}(\d{1,2})\b", re.IGNORECASE),
}


def _severity_in_window(text_lower, idx, window=40):
    start = max(0, idx - window)
    snippet = text_lower[start:idx + window]
    best = 0
    for word, level in SEVERITY_WORDS.items():
        if word in snippet:
            best = max(best, level)
    return best


# Negation cues checked immediately before a matched symptom phrase, so
# "no vomiting" / "denies chest pain" / "without breathlessness" are NOT
# counted as the symptom being present. This is a documented heuristic
# (a fixed-width lookback window), not full clinical NLP - a reviewer
# should still read the raw text themselves.
NEGATION_CUES = [
    "no ", "not ", "denies ", "denied ", "without ", "never had ",
    "no history of ", "negative for ", "no signs of ", "no evidence of ", "n't ",
]
NEGATION_LOOKBACK = 18

# Compound-list negation ("does not have breathlessness or chest pain", "denies fever and
# cough"): a real, confirmed gap - a symptom directly preceded by an "or"/"and"/"nor" connector
# with nothing else in between can still be covered by a negation cue further back in the same
# clause, but the tight NEGATION_LOOKBACK above is too short to reach it once the connector's
# preceding item is itself a longer word (e.g. "breathlessness"). The check below only widens
# the search for this specific, narrowly-confirmed "bare connector directly before the symptom"
# pattern, and never crosses a sentence boundary or a contrastive word like "but"/"however" (so
# "has chest pain but denies breathlessness" still correctly leaves chest pain un-negated).
# Comma is deliberately NOT treated as a list connector here: unlike "or"/"and"/"nor", a comma is
# genuinely ambiguous between "another item in the same denied list" and "a new clause entirely"
# (e.g. "no history of fever, chest pain now present") - and wrongly marking a real symptom as
# negated is the more dangerous failure mode for a triage safety net, so this stays conservative.
NEGATION_LIST_LOOKBACK = 60
_LIST_CONNECTOR = re.compile(r"(?:\s+or\s+|\s+and\s+|\s+nor\s+)$")
_CLAUSE_BREAK = re.compile(r"[.;!?]|\bbut\b|\bhowever\b|\balthough\b|\bthough\b")


def _is_negated(text_lower, idx):
    start = max(0, idx - NEGATION_LOOKBACK)
    window = text_lower[start:idx]
    if any(window.endswith(cue) or cue in window for cue in NEGATION_CUES):
        return True
    m = _LIST_CONNECTOR.search(text_lower[:idx])
    if m:
        clause_start = 0
        for brk in _CLAUSE_BREAK.finditer(text_lower[:m.start()]):
            clause_start = brk.end()
        prior_window = text_lower[max(clause_start, m.start() - NEGATION_LIST_LOOKBACK):m.start()]
        if any(cue in prior_window for cue in NEGATION_CUES):
            return True
    return False


def detect_symptoms(raw_text):
    """Scan free text for known symptom phrases. Returns list of dicts:
    {canonical, matched_phrase, category, red_flag, severity_level}
    for symptoms reported as PRESENT. Explicitly negated mentions (e.g.
    "no breathlessness") are excluded from this list and instead available
    via detect_negated_symptoms() for transparency in the reviewer note.
    """
    text_lower = (raw_text or "").lower()
    found = []
    seen = set()
    for canonical, meta in SYMPTOM_ONTOLOGY.items():
        for alias in meta["aliases"]:
            idx = text_lower.find(alias.lower())
            if idx != -1 and canonical not in seen:
                if _is_negated(text_lower, idx):
                    continue
                seen.add(canonical)
                found.append({
                    "canonical": canonical,
                    "matched_phrase": alias,
                    "category": meta["category"],
                    "red_flag": meta["red_flag"],
                    "weight": meta["weight"],
                    "severity_level": _severity_in_window(text_lower, idx),
                })
                break
    return found


def detect_negated_symptoms(raw_text):
    """Returns canonical symptom names the patient explicitly denied, so a
    reviewer can see what was ruled out rather than just what's missing.
    """
    text_lower = (raw_text or "").lower()
    negated = []
    seen = set()
    for canonical, meta in SYMPTOM_ONTOLOGY.items():
        for alias in meta["aliases"]:
            idx = text_lower.find(alias.lower())
            if idx != -1 and canonical not in seen and _is_negated(text_lower, idx):
                seen.add(canonical)
                negated.append(canonical)
                break
    return negated


def extract_durations(raw_text):
    """Returns list of (value, unit) duration mentions found in text."""
    return [(int(m.group(1)), m.group(2).lower()) for m in DURATION_PATTERN.finditer(raw_text or "")]


# Physiologically plausible bounds for a real blood pressure reading. Real, confirmed bug fixed
# here: VITALS_PATTERNS["blood_pressure"] matches ANY "N/N" fragment in free text with no
# contextual requirement (the "mm hg" suffix is optional) - so a pain score ("12/10 unbearable"),
# a date ("follow up on 15/11"), or a test score ("18/20") was silently read as a blood pressure
# reading and then flagged "abnormal" by evaluate_vitals() below, adding 28 points to the risk
# score from a completely unrelated number. A systolic reading under 60 or over 260, a diastolic
# under 30 or over 150, or a systolic that isn't actually higher than the diastolic, is not a
# plausible real BP reading and is now rejected rather than accepted at face value.
BLOOD_PRESSURE_SYSTOLIC_RANGE = (60, 260)
BLOOD_PRESSURE_DIASTOLIC_RANGE = (30, 150)


def _is_plausible_blood_pressure(systolic, diastolic):
    return (
        BLOOD_PRESSURE_SYSTOLIC_RANGE[0] <= systolic <= BLOOD_PRESSURE_SYSTOLIC_RANGE[1]
        and BLOOD_PRESSURE_DIASTOLIC_RANGE[0] <= diastolic <= BLOOD_PRESSURE_DIASTOLIC_RANGE[1]
        and systolic > diastolic
    )


def extract_vitals(text):
    """Extract vital signs mentioned in free text or OCR'd lab/vitals text."""
    if not text:
        return {}
    vitals = {}
    for field, pattern in VITALS_PATTERNS.items():
        m = pattern.search(text)
        if not m:
            continue
        if field == "blood_pressure":
            systolic, diastolic = int(m.group(1)), int(m.group(2))
            if _is_plausible_blood_pressure(systolic, diastolic):
                vitals["blood_pressure"] = f"{systolic}/{diastolic}"
        else:
            vitals[field] = m.group(1)
    return vitals


# ---------------------------------------------------------------------------
# Vital sign danger thresholds (adult defaults; a real deployment would vary
# these by age band - flagged here as a documented simplification).
# ---------------------------------------------------------------------------
def evaluate_vitals(vitals):
    """Returns list of {reason, weight} for vitals outside safe ranges."""
    reasons = []
    try:
        if "temperature_f" in vitals and float(vitals["temperature_f"]) >= 103:
            reasons.append({"reason": f"High temperature reported ({vitals['temperature_f']} F)", "weight": 15})
        if "temperature_c" in vitals and float(vitals["temperature_c"]) >= 39.4:
            reasons.append({"reason": f"High temperature reported ({vitals['temperature_c']} C)", "weight": 15})
        if "spo2" in vitals and float(vitals["spo2"]) < 92:
            reasons.append({"reason": f"Low oxygen saturation reported (SpO2 {vitals['spo2']}%)", "weight": 35})
        if "heart_rate" in vitals:
            hr = float(vitals["heart_rate"])
            if hr >= 130 or hr <= 45:
                reasons.append({"reason": f"Abnormal heart rate reported ({int(hr)} bpm)", "weight": 25})
        if "respiratory_rate" in vitals:
            rr = float(vitals["respiratory_rate"])
            if rr >= 28 or rr <= 8:
                reasons.append({"reason": f"Abnormal respiratory rate reported ({int(rr)}/min)", "weight": 30})
        if "blood_pressure" in vitals:
            sys_bp = float(vitals["blood_pressure"].split("/")[0])
            if sys_bp >= 180 or sys_bp <= 90:
                reasons.append({"reason": f"Abnormal blood pressure reported ({vitals['blood_pressure']} mmHg)", "weight": 28})
    except (ValueError, IndexError, ZeroDivisionError):
        pass
    return reasons


# Combination rules: pairs/sets of canonical symptoms that together change
# urgency even if neither is individually a hard red flag.
#   COMBO_EMERGENCY  -> forces the "emergency" tier outright (recognised
#                       danger-sign patterns, e.g. WHO/IMCI-style combos).
#   COMBO_HIGH_BOOST -> adds score + a reason, but does NOT by itself force
#                       emergency; it nudges the case toward "high" via the
#                       numeric score so a reviewer still triages same-day
#                       rather than being told to drop everything.
COMBO_EMERGENCY = [
    ({"chest_pain", "breathlessness"}, "Chest pain combined with breathlessness (possible cardiac/respiratory emergency pattern)"),
    ({"high_fever", "infant_poor_feeding"}, "High fever with poor feeding in an infant (danger sign)"),
    ({"severe_abdominal_pain", "pregnancy_bleeding"}, "Severe abdominal pain with pregnancy-related bleeding"),
]
COMBO_HIGH_BOOST = [
    ({"persistent_vomiting", "dehydration_signs"}, "Persistent vomiting with signs of dehydration - urgent same-day review recommended", 20),
]


def compute_risk(symptoms, vitals, age_text=None):
    """
    Combine symptom flags + vital sign thresholds into a risk tier.
    Returns: dict(tier, score, reasons: [str])
    """
    reasons = []
    score = 0
    canonical_set = {s["canonical"] for s in symptoms}

    hard_red_flag = False
    for s in symptoms:
        score += s["weight"] + (s["severity_level"] * 3)
        if s["red_flag"]:
            hard_red_flag = True
            reasons.append(f"Urgency keyword detected: '{s['matched_phrase']}' ({s['category']})")

    for combo, label in COMBO_EMERGENCY:
        if combo.issubset(canonical_set):
            hard_red_flag = True
            reasons.append(label)

    for combo, label, boost in COMBO_HIGH_BOOST:
        if combo.issubset(canonical_set):
            score += boost
            reasons.append(label)

    vitals_reasons = evaluate_vitals(vitals)
    for vr in vitals_reasons:
        score += vr["weight"]
        reasons.append(vr["reason"])
        if vr["weight"] >= 30:
            hard_red_flag = True

    # Simple age-based sensitivity: infants and elderly get a bump since
    # deterioration can be faster / less obvious in these groups.
    #
    # Real, confirmed bug fixed here: the old logic only ever recognised "months" (via a bare
    # "month" in str.lower()) vs. everything else treated as "years", and had NO handling for
    # "days" at all - even though the intake form's own validation (app.py) explicitly accepts
    # years/yrs, months/mos, AND days as valid age units. That meant:
    #   - ANY age given in months was unconditionally treated as "<1 year" regardless of the
    #     number - "18 months" (a toddler) was mislabeled "Infant patient (<1 year)".
    #   - An age given in days (e.g. "75 days", a genuine newborn) fell through to the "years"
    #     branch and had its raw number treated as YEARS - "75 days" was mislabeled "Elderly
    #     patient (70+)", while a real newborn described in days got no infant safety bump at all.
    # Ages are now normalized to a fractional number of years before either threshold is applied.
    age_flag = None
    if age_text:
        digits = re.findall(r"\d+", str(age_text))
        if digits:
            age_val = int(digits[0])
            unit_match = re.search(r"\b(day|days|month|months|mo|mos|year|years|yr|yrs)\b", str(age_text).lower())
            unit = unit_match.group(1) if unit_match else "years"
            if unit in ("day", "days"):
                age_in_years = age_val / 365.0
            elif unit in ("month", "months", "mo", "mos"):
                age_in_years = age_val / 12.0
            else:
                age_in_years = age_val
            if age_in_years < 1:
                score += 10
                age_flag = "Infant patient (<1 year) - lower threshold for urgent review"
            elif age_in_years >= 70:
                score += 8
                age_flag = "Elderly patient (70+) - lower threshold for urgent review"
    if age_flag:
        reasons.append(age_flag)

    if hard_red_flag or score >= 45:
        tier = "emergency"
    elif score >= 25:
        tier = "high"
    elif score >= 10:
        tier = "medium"
    else:
        tier = "low"

    if not reasons:
        reasons.append("No urgency signals detected from provided information; routine review recommended.")

    return {"tier": tier, "score": score, "reasons": reasons}


RISK_TIER_LABELS = {
    "emergency": {"label": "Emergency - See Immediately", "color": "#c62828"},
    "high": {"label": "High Priority", "color": "#ef6c00"},
    "medium": {"label": "Medium Priority", "color": "#f9a825"},
    "low": {"label": "Low Priority / Routine", "color": "#2e7d32"},
    "unclassified": {"label": "Unclassified", "color": "#616161"},
}
