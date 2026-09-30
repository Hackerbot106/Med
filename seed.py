"""
Seeds the database with demo users and a handful of SYNTHETIC triage cases
spanning different risk tiers, facility types, and scenarios, so the
reviewer dashboard is populated immediately for a demo/evaluation.

No real patient data is used anywhere in this repository.

Usage: python seed.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triage_engine import database, ocr as ocr_module, llm, pipeline, security
from sample_data import make_sample_reports

DEMO_USERS = [
    ("worker1", "demo123", "intake_worker", "Priya Nayak (Health Worker)", "PHC Balianta"),
    ("worker2", "demo123", "intake_worker", "Suresh Behera (Health Worker)", "Industrial Estate Clinic, Rasulgarh"),
    ("reviewer1", "demo123", "reviewer", "Dr. Anil Mohanty (Medical Officer)", "CHC Khordha"),
    ("reviewer2", "demo123", "reviewer", "Dr. Sneha Patra (Nurse Practitioner)", "Government Hospital, Bhubaneswar"),
    ("auditor1", "demo123", "auditor", "Facility Admin - Data Officer", "District Health Office"),
]

# Each: (facility_type, facility_name, scenario, language, display_name, age, sex, raw_text, input_mode,
#        status, reviewer_notes, ocr_image_filename, next_followup_date)
DEMO_CASES = [
    (
        "Government Hospital", "Govt. Hospital, Bhubaneswar", "outpatient_queue", "en",
        "Demo Patient A", "58 years", "Male",
        "Severe chest pain since 2 hours, radiating to left arm, breathlessness and sweating a lot.",
        "text", "new", None, None, None,
    ),
    (
        "Primary Health Centre (PHC)", "PHC Balianta", "public_health_camp", "hi",
        "Demo Infant B", "4 months", "Female",
        "Baby has high fever since yesterday and is refusing feed, not waking up easily to feed.",
        "voice", "new", None, None, None,
    ),
    (
        "Company Clinic", "Clinic - TCS Campus", "outpatient_queue", "en",
        "Demo Patient C", "29 years", "Female",
        "Severe abdominal pain since this morning, moderate severity, no vomiting.",
        "text", "new", None, None, None,
    ),
    (
        "Campus Health Centre", "University Health Centre", "campus_fever", "en",
        "Demo Student D", "6 years", "Male",
        "Vomiting repeatedly since last night, about 6 times, child looks very thirsty and passed very little urine today.",
        "text", "in_review", None, None, None,
    ),
    (
        "Primary Health Centre (PHC)", "PHC Balianta", "outpatient_queue", "hi",
        "Demo Patient E", "34 years", "Male",
        "Fever for 3 days, dry cough, mild headache. Attaching vitals slip from earlier visit.",
        "text", "new", None, "sample_lab_report_fever.png", None,
    ),
    (
        "Industrial Estate Health Unit", "IE Health Unit, Rasulgarh", "occupational_health", "en",
        "Demo Worker F", "40 years", "Male",
        "Mild chemical fumes exposure at work today, some throat irritation and cough, no breathlessness.",
        "text", "reviewed", "Observed for 30 min, vitals stable, advised rest and follow-up if symptoms worsen. Reported to safety officer.", None, None,
    ),
    (
        "Campus Health Centre", "University Health Centre", "campus_fever", "en",
        "Demo Student G", "22 years", "Other",
        "Mild headache and general fatigue for 1 day, no fever. Routine screening.",
        "text", "closed", "Advised rest and hydration, no red flags. Case closed after routine review.", "sample_lab_report_routine.png", None,
    ),
    (
        "Primary Health Centre (PHC)", "PHC Balianta", "chronic_checkin", "en",
        "Demo Patient H", "70 years", "Female",
        "Ongoing body ache and mild fatigue, known chronic condition, monthly check-in call.",
        "text", "new", None, None, "2026-09-05",  # deliberately in the past -> shows as OVERDUE on dashboard
    ),
    (
        "Primary Health Centre (PHC)", "PHC Balianta", "maternal_followup", "en",
        "Demo Patient I", "27 years", "Female",
        "28 weeks pregnant, routine follow-up call, feels fine, baby moving normally, no bleeding.",
        "text", "new", None, None, "2026-09-20",  # upcoming
    ),
]


def run(reset=True):
    database.init_db(reset=reset)
    make_sample_reports.generate_all()

    conn = database.get_connection()
    for username, password, role, display_name, facility_name in DEMO_USERS:
        # Passwords are hashed (werkzeug/pbkdf2) even for these synthetic demo
        # accounts - never stored in plaintext, matching how a real deployment
        # must handle credentials (see triage_engine/security.py). Existing
        # rows are left untouched (INSERT OR IGNORE); app.py transparently
        # upgrades any pre-existing plaintext row on next successful login.
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password, role, display_name, facility_name) VALUES (?,?,?,?,?)",
            (username, security.hash_password(password), role, display_name, facility_name),
        )
    conn.commit()

    for (facility_type, facility_name, scenario, language, display_name, age, sex,
         raw_text, input_mode, status, reviewer_notes, ocr_image, next_followup_date) in DEMO_CASES:
        triage_id = database.new_triage_id()
        patient = {
            "triage_id": triage_id, "display_name": display_name, "age": age, "sex": sex,
            "facility_type": facility_type, "facility_name": facility_name,
            "preferred_language": language,
        }

        ocr_text = None
        ocr_filename = None
        if ocr_image:
            image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", ocr_image)
            if os.path.exists(image_path):
                ocr_text = ocr_module.run_ocr(image_path)
                ocr_filename = ocr_image

        structured = pipeline.run_triage_pipeline(raw_text, patient, ocr_text=ocr_text)
        ai_fields = llm.build_ai_fields(structured, patient, raw_text)
        now = database.now_iso()

        conn.execute(
            "INSERT INTO patients (triage_id, display_name, age, sex, facility_type, facility_name, "
            "preferred_language, consent_given, consent_timestamp, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (triage_id, display_name, age, sex, facility_type, facility_name, language, 1, now, now),
        )
        patient_row_id = conn.execute("SELECT id FROM patients WHERE triage_id=?", (triage_id,)).fetchone()["id"]

        reviewer_id = None
        reviewed_at = None
        if status in ("reviewed", "closed"):
            reviewer_row = conn.execute("SELECT id FROM users WHERE username='reviewer1'").fetchone()
            reviewer_id = reviewer_row["id"] if reviewer_row else None
            reviewed_at = now

        conn.execute(
            """INSERT INTO triage_notes
               (patient_id, scenario, input_mode, raw_symptom_text, chief_complaint, symptoms_json,
                negated_symptoms_json, timeline_json, missing_info_json, follow_up_questions_json, vitals_json,
                ocr_extracted_text, ocr_source_filename, risk_tier, risk_score, risk_reasons_json,
                ai_summary, ai_summary_model, ai_summary_generated_at, ai_followups_json,
                extraction_source, ai_powered, rules_only_symptoms_json, rules_only_risk_tier,
                next_followup_date,
                status, reviewer_id, reviewer_notes, reviewed_at, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                patient_row_id, scenario, input_mode, raw_text, structured["chief_complaint"],
                json.dumps(structured["symptoms"]), json.dumps(structured["negated_symptoms"]),
                json.dumps(structured["timeline"]),
                json.dumps(structured["missing_info"]), json.dumps(structured["follow_up_questions"]),
                json.dumps(structured["vitals"]), ocr_text, ocr_filename,
                structured["risk_tier"], structured["risk_score"], json.dumps(structured["risk_reasons"]),
                ai_fields["ai_summary"], ai_fields["ai_summary_model"], ai_fields["ai_summary_generated_at"],
                ai_fields["ai_followups_json"],
                structured.get("extraction_source"), 1 if structured.get("ai_powered") else 0,
                json.dumps(structured.get("rules_only_symptoms", [])), structured.get("rules_only_risk_tier"),
                next_followup_date,
                status, reviewer_id, reviewer_notes, reviewed_at, now, now,
            ),
        )
        print(f"Seeded {triage_id}: {structured['chief_complaint']} -> {structured['risk_tier']} ({structured['risk_score']})")

    conn.commit()
    conn.close()

    ai_available, ai_message = llm.availability_status()
    print(f"\nAI summarization: {'ENABLED' if ai_available else 'not active'} - {ai_message}")
    print("Seed complete. Demo accounts: worker1/demo123, reviewer1/demo123, auditor1/demo123 (password: demo123)")


if __name__ == "__main__":
    run(reset=True)
