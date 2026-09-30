"""
Rules-based information extraction: turns free-text (typed or voice-
transcribed) patient-reported symptoms into structured fields, identifies
missing information, and drafts follow-up questions for the health worker
or reviewer to ask.

No diagnosis is ever produced here - only organisation of what the patient
said and what is still unknown.
"""

from . import risk_rules

CATEGORY_FOLLOWUPS = {
    "cardiac": [
        "Does the pain radiate to the arm, jaw, or back?",
        "Is there associated sweating, nausea, or breathlessness?",
    ],
    "respiratory": [
        "Is there wheezing or noisy breathing?",
        "Does breathlessness occur at rest or only on exertion?",
    ],
    "neuro": [
        "When exactly did the symptoms start, and did they come on suddenly?",
        "Is there any weakness, numbness, or trouble speaking?",
    ],
    "gi": [
        "How many episodes in the last 24 hours?",
        "Is there blood in vomit or stool?",
    ],
    "obstetric": [
        "How many weeks pregnant is the patient?",
        "Any history of complications in this or a previous pregnancy?",
    ],
    "pediatric": [
        "Is the child alert and responsive between episodes?",
        "How many wet diapers / urinations in the last 24 hours?",
    ],
    "trauma": [
        "How did the injury occur?",
        "Can the affected area be moved normally?",
    ],
    "infection": [
        "Has the temperature been measured with a thermometer? What was the reading?",
        "Any associated chills, rash, or localized pain?",
    ],
    "mental_health": [
        "Is the patient safe right now and not alone?",
        "Escalate immediately to a qualified counselor / medical officer - do not rely on this tool for mental health crises.",
    ],
    "occupational": [
        "What substance or material was involved in the exposure?",
        "Was protective equipment being used at the time?",
    ],
    "general": [
        "Any other symptoms not yet mentioned?",
        "Any known chronic conditions or ongoing medications?",
    ],
    "derm": ["Is the rash spreading, and is there any associated fever?"],
    "ent": ["Any fever, ear pain, or difficulty breathing along with the throat symptoms?"],
    "urinary": ["Any fever, back pain, or blood in urine?"],
    "musculoskeletal": ["Is there swelling, redness, or inability to bear weight?"],
}

UNIT_TO_DAYS = {
    "hour": 1 / 24, "hours": 1 / 24, "hr": 1 / 24, "hrs": 1 / 24,
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
}


def build_chief_complaint(symptoms):
    if not symptoms:
        return "No specific symptom keywords detected in provided text - reviewer should read raw notes directly."
    # Chief complaint = highest-weight symptom(s), human-readable
    top = sorted(symptoms, key=lambda s: s["weight"], reverse=True)[:2]
    names = [s["canonical"].replace("_", " ") for s in top]
    return " and ".join(names).capitalize()


def build_timeline(raw_text, symptoms):
    """Very lightweight timeline: pairs nearby duration mentions with the
    closest preceding symptom phrase in the raw text. This is a heuristic,
    not a guaranteed-correct parse, and is labelled as such for reviewers.
    """
    text_lower = (raw_text or "").lower()
    events = []
    for s in symptoms:
        idx = text_lower.find(s["matched_phrase"].lower())
        if idx == -1:
            continue
        window = text_lower[idx: idx + 60]
        durations = risk_rules.extract_durations(window)
        if durations:
            value, unit = durations[0]
            events.append({
                "symptom": s["canonical"].replace("_", " "),
                "onset_text": f"{value} {unit} ago",
                "sort_days": value * UNIT_TO_DAYS.get(unit, 1),
            })
        else:
            events.append({
                "symptom": s["canonical"].replace("_", " "),
                "onset_text": "duration not specified",
                "sort_days": None,
            })
    # Sort oldest-onset first when duration known, unknowns last
    events.sort(key=lambda e: (e["sort_days"] is None, -(e["sort_days"] or 0)))
    return events


def detect_missing_info(patient, symptoms, vitals, timeline):
    missing = []
    if not patient.get("age"):
        missing.append("Patient age not recorded")
    if not patient.get("sex"):
        missing.append("Patient sex not recorded")
    if not symptoms:
        missing.append("No structured symptoms could be identified from free text - manual review of raw notes required")
    if symptoms and not any(e.get("sort_days") is not None for e in timeline):
        missing.append("Duration/onset of symptoms not clearly stated")
    if not vitals:
        missing.append("No vital signs recorded (temperature, pulse, BP, SpO2, respiratory rate)")
    else:
        for key in ("temperature_f", "temperature_c"):
            if key in vitals:
                break
        else:
            missing.append("Temperature not recorded")
    severity_terms_present = any(s["severity_level"] > 0 for s in symptoms)
    if symptoms and not severity_terms_present:
        missing.append("Severity of symptoms not clearly described (mild/moderate/severe)")
    return missing


def generate_follow_up_questions(symptoms, missing_info):
    questions = []
    seen_categories = set()
    for s in symptoms:
        cat = s["category"]
        if cat in seen_categories:
            continue
        seen_categories.add(cat)
        for q in CATEGORY_FOLLOWUPS.get(cat, [])[:2]:
            questions.append(q)
    for m in missing_info:
        if "age" in m.lower():
            questions.append("What is the patient's exact age (or date of birth)?")
        if "duration" in m.lower():
            questions.append("Exactly when did the symptoms start?")
        if "vital signs" in m.lower():
            questions.append("Please record temperature, pulse, blood pressure, and SpO2 if a device is available.")
    # De-duplicate while preserving order
    out = []
    for q in questions:
        if q not in out:
            out.append(q)
    return out[:8]


def extract_structured_note(raw_text, patient, ocr_text=None):
    """
    Main entry point: given raw patient-reported text (+ optional OCR text
    from an uploaded lab report) and patient demographic dict, returns a
    fully structured, reviewer-facing note.
    """
    # Real, confirmed bug fixed here: this variable's own name says "combined", but it was never
    # actually combined with ocr_text - only raw_text was ever scanned for symptoms/red-flags.
    # OCR'd document text (e.g. a scanned referral note or discharge slip) was only ever mined
    # for numeric VITALS below, never for symptom keywords - meaning a document that plainly says
    # "unconscious" or "severe bleeding" produced NO red-flag detection at all if the patient's
    # own typed/spoken text was empty or didn't repeat those words. Now both sources are scanned.
    combined_text_for_symptoms = f"{raw_text or ''} {ocr_text or ''}".strip()
    symptoms = risk_rules.detect_symptoms(combined_text_for_symptoms)
    negated_symptoms = risk_rules.detect_negated_symptoms(combined_text_for_symptoms)

    vitals = risk_rules.extract_vitals(raw_text)
    if ocr_text:
        ocr_vitals = risk_rules.extract_vitals(ocr_text)
        # OCR values take precedence where present (assumed more reliable
        # than a free-text patient description of a lab slip)
        vitals.update(ocr_vitals)

    # Also paired against the combined text (not raw_text alone) so a duration mentioned only in
    # an OCR'd document ("fever x 3 days" on a referral slip) can still be matched to the symptom
    # it was found alongside above.
    timeline = build_timeline(combined_text_for_symptoms, symptoms)
    missing_info = detect_missing_info(patient, symptoms, vitals, timeline)
    follow_up_questions = generate_follow_up_questions(symptoms, missing_info)
    chief_complaint = build_chief_complaint(symptoms)

    risk = risk_rules.compute_risk(symptoms, vitals, age_text=patient.get("age"))

    return {
        "chief_complaint": chief_complaint,
        "symptoms": symptoms,
        # Kept as canonical symptom keys (not humanized here) - real, confirmed bug fixed: this
        # used to pre-humanize with .replace("_", " ") before storage, which meant
        # translator.translate_symptom_label() (keyed by the underscored canonical form) could
        # never find a match and always fell back to English display, regardless of the selected
        # language. Humanizing/translating now happens only at display time (app.py, templates,
        # and the two internal AI-prompt/plain-text-summary call sites that need a readable
        # string), exactly like the (correctly-handled) present-symptoms list already did.
        "negated_symptoms": negated_symptoms,
        "timeline": timeline,
        "vitals": vitals,
        "missing_info": missing_info,
        "follow_up_questions": follow_up_questions,
        "risk_tier": risk["tier"],
        "risk_score": risk["score"],
        "risk_reasons": risk["reasons"],
    }
