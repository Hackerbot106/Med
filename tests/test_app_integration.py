"""
Flask test-client integration tests covering auth, role-based access
control, and the intake -> dashboard flow end to end against a real
(but fully isolated, temporary, throwaway) SQLite database - never the
developer's actual data/triage.db.

Isolation strategy: triage_engine.database.DB_PATH is monkeypatched to a
fresh temp file BEFORE `app` is imported, since app.py calls
database.init_db() at import time. This means running `pytest` never
touches, seeds into, or purges real demo/local data.

CSRF is left enabled in test_csrf_enforcement.py-style coverage lives in
test_security.py (pure unit tests) plus one focused check here
(test_post_without_csrf_token_is_rejected); every other test below runs
under APP_ENV=testing, which disables CSRF (config.TestingConfig) - the
standard, common approach for exercising application logic without
re-deriving a fresh token before every single POST in every test.
"""

import os
import sys
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["APP_ENV"] = "testing"

_TEST_DIR = tempfile.mkdtemp(prefix="triage_assistant_test_")

from triage_engine import database  # noqa: E402
database.DB_PATH = os.path.join(_TEST_DIR, "test_triage.db")

import app as app_module  # noqa: E402  (import AFTER DB_PATH patch - triggers init_db() against the temp path)
from triage_engine import security  # noqa: E402
from werkzeug.security import generate_password_hash  # noqa: E402


def _seed_minimal_users():
    conn = database.get_connection()
    users = [
        ("t_worker", "pw123", "intake_worker", "Test Worker", "Test Facility"),
        ("t_worker2", "pw123", "intake_worker", "Test Worker Two", "Other Facility"),
        ("t_reviewer", "pw123", "reviewer", "Test Reviewer", "Test Facility"),
        ("t_auditor", "pw123", "auditor", "Test Auditor", "Test Facility"),
    ]
    for username, password, role, display_name, facility in users:
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password, role, display_name, facility_name) VALUES (?,?,?,?,?)",
            (username, generate_password_hash(password), role, display_name, facility),
        )
    conn.commit()
    conn.close()


def _get_csrf_token(html):
    m = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return m.group(1) if m else None


class TriageAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _seed_minimal_users()
        cls.app = app_module.app
        cls.app.config.update(TESTING=True)

    def setUp(self):
        self.client = self.app.test_client()

    def _login(self, username, password):
        r = self.client.get("/login")
        token = _get_csrf_token(r.get_data(as_text=True)) or ""
        return self.client.post(
            "/login", data={"username": username, "password": password, "csrf_token": token},
            follow_redirects=True,
        )


def tearDownModule():
    # Runs once after every TestCase class in this module has finished -
    # NOT per-class, since all classes below share the same temp DB file.
    shutil.rmtree(_TEST_DIR, ignore_errors=True)


class TestAuth(TriageAppTestCase):
    def test_login_page_loads(self):
        r = self.client.get("/login")
        self.assertEqual(r.status_code, 200)

    def test_valid_login_succeeds(self):
        r = self._login("t_reviewer", "pw123")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Signed in as", r.data)

    def test_invalid_password_rejected(self):
        r = self._login("t_reviewer", "wrong-password")
        self.assertIn(b"Invalid demo credentials", r.data)

    def test_unknown_username_rejected(self):
        r = self._login("nobody_at_all", "whatever")
        self.assertIn(b"Invalid demo credentials", r.data)

    def test_logout_clears_session(self):
        self._login("t_reviewer", "pw123")
        self.client.get("/logout")
        r = self.client.get("/dashboard", follow_redirects=True)
        self.assertIn(b"sign in", r.data.lower())

    def test_post_without_csrf_token_is_rejected(self):
        # Explicitly re-enable CSRF enforcement for this one request to prove
        # the before_request hook actually rejects a forged/missing token,
        # independent of the CSRF-disabled default used elsewhere in this
        # test module for convenience.
        self.app.config["WTF_CSRF_ENABLED"] = True
        try:
            r = self.client.post("/login", data={"username": "t_reviewer", "password": "pw123"})
            self.assertEqual(r.status_code, 400)
        finally:
            self.app.config["WTF_CSRF_ENABLED"] = False

    def test_login_next_parameter_rejects_off_site_redirect(self):
        # Regression test for a real, confirmed open-redirect bug: /login's `next` parameter was
        # passed straight into redirect() with no validation, so a crafted link with an
        # off-site `next` would send a victim to an attacker-controlled site right after a
        # legitimate sign-in - a classic phishing-enabling open redirect.
        token = _get_csrf_token(self.client.get("/login").get_data(as_text=True)) or ""
        r = self.client.post(
            "/login?next=https://evil.example/phish",
            data={"username": "t_reviewer", "password": "pw123", "csrf_token": token},
        )
        self.assertEqual(r.status_code, 302)
        self.assertNotIn("evil.example", r.headers.get("Location", ""))

    def test_login_next_parameter_allows_relative_same_site_path(self):
        token = _get_csrf_token(self.client.get("/login").get_data(as_text=True)) or ""
        r = self.client.post(
            "/login?next=/dashboard",
            data={"username": "t_reviewer", "password": "pw123", "csrf_token": token},
        )
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r.headers.get("Location"), "/dashboard")

    def test_login_next_parameter_rejects_protocol_relative_redirect(self):
        # "//evil.example" has no scheme but browsers treat it as protocol-relative and follow
        # it off-site - must be rejected exactly like a full https:// URL.
        token = _get_csrf_token(self.client.get("/login").get_data(as_text=True)) or ""
        r = self.client.post(
            "/login?next=//evil.example",
            data={"username": "t_reviewer", "password": "pw123", "csrf_token": token},
        )
        self.assertEqual(r.status_code, 302)
        self.assertNotIn("evil.example", r.headers.get("Location", ""))


class TestRoleBasedAccess(TriageAppTestCase):
    def test_worker_cannot_access_dashboard(self):
        self._login("t_worker", "pw123")
        r = self.client.get("/dashboard")
        self.assertEqual(r.status_code, 403)

    def test_reviewer_can_access_dashboard(self):
        self._login("t_reviewer", "pw123")
        r = self.client.get("/dashboard")
        self.assertEqual(r.status_code, 200)

    def test_worker_can_access_intake(self):
        self._login("t_worker", "pw123")
        r = self.client.get("/consent")
        self.assertEqual(r.status_code, 200)

    def test_auditor_cannot_access_intake(self):
        self._login("t_auditor", "pw123")
        r = self.client.get("/consent")
        self.assertEqual(r.status_code, 403)

    def test_unauthenticated_redirects_to_login(self):
        r = self.client.get("/dashboard", follow_redirects=True)
        self.assertIn(b"sign in", r.data.lower() + b"" if b"sign in" in r.data.lower() else r.data.lower())
        # More robust: unauthenticated dashboard access should not itself
        # succeed with a 200 dashboard render.
        r2 = self.client.get("/dashboard")
        self.assertIn(r2.status_code, (302, 403))


class TestIntakeWorkerOwnershipAccessControl(TriageAppTestCase):
    """Regression tests for a real, confirmed access-control gap: /triage/<note_id> and
    /intake/confirmation/<note_id> grant the intake_worker role access at all ONLY so "Open in
    Reviewer View" works right after a worker submits their own case - but neither route checked
    that note_id actually belonged to the worker viewing it, so any intake worker could view any
    OTHER patient's full reviewer detail page (chief complaint, AI summary, risk reasons,
    uploaded-photo AI findings) across every facility just by editing the note_id in the URL.
    Fixed via a new created_by_user_id column set at intake time and checked on both routes.
    """

    def _consent_and_get_intake_token(self):
        r = self.client.get("/consent")
        token = _get_csrf_token(r.get_data(as_text=True)) or ""
        self.client.post("/consent", data={"consent_ack": "on", "csrf_token": token}, follow_redirects=True)
        r = self.client.get("/intake")
        return _get_csrf_token(r.get_data(as_text=True)) or ""

    def _create_note_as(self, username, password):
        self._login(username, password)
        token = self._consent_and_get_intake_token()
        self.client.post("/intake", data={
            "csrf_token": token, "display_name": "Ownership Test Patient", "age": "35 years",
            "sex": "Male", "facility_type": "Government Hospital", "facility_name": "Test Hospital",
            "preferred_language": "en", "scenario": "outpatient_queue", "input_mode": "text",
            "symptom_text": "Fever and cough for two days.",
        }, follow_redirects=True)
        self.client.get("/logout")
        conn = database.get_connection()
        note_id = conn.execute("SELECT id FROM triage_notes ORDER BY id DESC LIMIT 1").fetchone()["id"]
        conn.close()
        return note_id

    def test_worker_can_view_their_own_note_detail(self):
        note_id = self._create_note_as("t_worker", "pw123")
        self._login("t_worker", "pw123")
        r = self.client.get(f"/triage/{note_id}")
        self.assertEqual(r.status_code, 200)

    def test_worker_cannot_view_another_workers_note_detail(self):
        note_id = self._create_note_as("t_worker", "pw123")
        self._login("t_worker2", "pw123")
        r = self.client.get(f"/triage/{note_id}")
        self.assertEqual(r.status_code, 403)

    def test_worker_cannot_view_another_workers_intake_confirmation(self):
        note_id = self._create_note_as("t_worker", "pw123")
        self._login("t_worker2", "pw123")
        r = self.client.get(f"/intake/confirmation/{note_id}")
        self.assertEqual(r.status_code, 403)

    def test_reviewer_can_still_view_any_note_detail(self):
        note_id = self._create_note_as("t_worker", "pw123")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}")
        self.assertEqual(r.status_code, 200)


class TestIntakeFlow(TriageAppTestCase):
    def _consent_and_get_intake_token(self):
        r = self.client.get("/consent")
        token = _get_csrf_token(r.get_data(as_text=True)) or ""
        self.client.post("/consent", data={"consent_ack": "on", "csrf_token": token}, follow_redirects=True)
        r = self.client.get("/intake")
        return _get_csrf_token(r.get_data(as_text=True)) or ""

    def test_full_intake_submission_creates_a_triage_note(self):
        self._login("t_worker", "pw123")
        token = self._consent_and_get_intake_token()
        r = self.client.post("/intake", data={
            "csrf_token": token, "display_name": "Integration Test Patient", "age": "34 years", "sex": "Male",
            "facility_type": "Government Hospital", "facility_name": "Test Hospital",
            "preferred_language": "en", "scenario": "outpatient_queue", "input_mode": "text",
            "symptom_text": "Severe chest pain and breathlessness since this morning.",
        }, follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"created and queued for review", r.data)

    def test_intake_without_symptom_text_is_rejected_with_validation_message(self):
        self._login("t_worker", "pw123")
        token = self._consent_and_get_intake_token()
        r = self.client.post("/intake", data={
            "csrf_token": token, "facility_type": "Government Hospital",
            "scenario": "outpatient_queue", "input_mode": "text", "symptom_text": "",
        }, follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"describe the patient", r.data.lower())

    def test_invalid_facility_type_is_rejected(self):
        self._login("t_worker", "pw123")
        token = self._consent_and_get_intake_token()
        r = self.client.post("/intake", data={
            "csrf_token": token, "facility_type": "Not A Real Facility",
            "scenario": "outpatient_queue", "input_mode": "text", "symptom_text": "Fever.",
        }, follow_redirects=True)
        self.assertIn(b"valid facility type", r.data.lower())

    def test_form_values_are_preserved_after_a_validation_error(self):
        # Regression test for a real usability bug: on a validation error, the form used to
        # re-render completely blank, silently discarding everything the worker had already
        # typed (including a long symptom description) and forcing a full retype.
        self._login("t_worker", "pw123")
        token = self._consent_and_get_intake_token()
        r = self.client.post("/intake", data={
            "csrf_token": token, "facility_type": "Not A Real Facility", "display_name": "Jane Doe",
            "age": "45 years", "scenario": "outpatient_queue", "input_mode": "text",
            "symptom_text": "Severe chest pain radiating to the left arm since this morning.",
        }, follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("Severe chest pain radiating to the left arm since this morning.", html)
        self.assertIn('value="Jane Doe"', html)
        self.assertIn('value="45 years"', html)

    def test_emergency_case_appears_on_dashboard_sorted_first(self):
        self._login("t_worker", "pw123")
        token = self._consent_and_get_intake_token()
        self.client.post("/intake", data={
            "csrf_token": token, "facility_type": "Government Hospital",
            "scenario": "outpatient_queue", "input_mode": "text",
            "symptom_text": "Patient had a seizure just now and is unconscious.",
        }, follow_redirects=True)
        self.client.get("/logout")
        self._login("t_reviewer", "pw123")
        r = self.client.get("/dashboard")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"emergency", r.data.lower())


class TestExtraFieldTranslation(TriageAppTestCase):
    """Regression tests for a real, confirmed gap: switching the language selector to Hindi/Odia
    only translated the AI narrative summary - chief complaint, missing info, extra AI follow-up
    questions, and the vision model's photo description/findings were shown in English
    regardless of the selected language. app.py's _translate_field()/_translate_list_field() (used
    from the /triage/<id> view) fix this; these tests confirm the wiring end-to-end through the
    real Flask routes, with the (slow, CPU-bound) translation call itself mocked out.
    """

    def _consent_and_get_intake_token(self):
        r = self.client.get("/consent")
        token = _get_csrf_token(r.get_data(as_text=True)) or ""
        self.client.post("/consent", data={"consent_ack": "on", "csrf_token": token}, follow_redirects=True)
        r = self.client.get("/intake")
        return _get_csrf_token(r.get_data(as_text=True)) or ""

    def _create_note_with_extra_ai_fields(self):
        self._login("t_worker", "pw123")
        token = self._consent_and_get_intake_token()
        self.client.post("/intake", data={
            "csrf_token": token, "display_name": "Translation Test Patient", "age": "30 years",
            "sex": "Female", "facility_type": "Government Hospital", "facility_name": "Test Hospital",
            "preferred_language": "hi", "scenario": "outpatient_queue", "input_mode": "text",
            "symptom_text": "Fever and cough for three days.",
        }, follow_redirects=True)
        self.client.get("/logout")

        conn = database.get_connection()
        row = conn.execute("SELECT id FROM triage_notes ORDER BY id DESC LIMIT 1").fetchone()
        note_id = row["id"]
        conn.execute(
            "UPDATE triage_notes SET image_ai_description=?, image_ai_findings_json=?, "
            "ai_followups_json=?, missing_info_json=? WHERE id=?",
            (
                "A close-up photo showing reddened skin with mild swelling.",
                '["possibly consistent with contact dermatitis (redness, no discharge)"]',
                '["Has the patient had a fever in the last 24 hours?"]',
                '["Duration of symptoms not specified"]',
                note_id,
            ),
        )
        conn.commit()
        conn.close()
        return note_id

    def test_extra_fields_are_shown_in_english_when_translation_unavailable(self):
        # With no IndicTrans2/Qwen model available in this test environment, translation isn't
        # possible - the page must still render the ORIGINAL English text (never blank), which is
        # exactly the safe-fallback behavior _translate_field() promises.
        note_id = self._create_note_with_extra_ai_fields()
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=hi")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"reddened skin with mild swelling", r.data)
        self.assertIn(b"contact dermatitis", r.data)

    def test_extra_fields_use_translated_text_when_available(self):
        note_id = self._create_note_with_extra_ai_fields()

        def fake_translate(text, lang_code):
            return f"[HI]{text}"

        with patch("app.indic_translate.translate_summary", side_effect=fake_translate):
            self._login("t_reviewer", "pw123")
            r = self.client.get(f"/triage/{note_id}?lang=hi")

        self.assertEqual(r.status_code, 200)
        self.assertIn(b"[HI]A close-up photo showing reddened skin with mild swelling.", r.data)
        self.assertIn(b"[HI]possibly consistent with contact dermatitis", r.data)
        self.assertIn(b"[HI]Duration of symptoms not specified", r.data)
        self.assertIn(b"[HI]Has the patient had a fever in the last 24 hours?", r.data)
        # Original untranslated English must NOT still be showing for these fields.
        self.assertNotIn(b">A close-up photo showing reddened skin", r.data)

    def test_translation_is_cached_and_not_recomputed_on_second_view(self):
        note_id = self._create_note_with_extra_ai_fields()
        call_count = {"n": 0}

        def fake_translate(text, lang_code):
            call_count["n"] += 1
            return f"[HI]{text}"

        self._login("t_reviewer", "pw123")
        with patch("app.indic_translate.translate_summary", side_effect=fake_translate):
            self.client.get(f"/triage/{note_id}?lang=hi")
            first_call_count = call_count["n"]
            self.client.get(f"/triage/{note_id}?lang=hi")
            second_call_count = call_count["n"]

        self.assertGreater(first_call_count, 0)
        # The second view must not have triggered any NEW translation calls - the cache in
        # extra_translations_json should have served every field from the first view already.
        self.assertEqual(first_call_count, second_call_count)

    def test_english_language_shows_original_untranslated_text(self):
        note_id = self._create_note_with_extra_ai_fields()
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=en")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"A close-up photo showing reddened skin with mild swelling.", r.data)

    def test_regenerating_ai_clears_stale_extra_translation_cache(self):
        # Regression test for a real, confirmed bug: "Regenerate AI Summary" reset
        # ai_summary_translations_json but NOT extra_translations_json, so a reviewer who had
        # already viewed a Hindi/Odia translation of e.g. missing_info would keep seeing that
        # STALE cached translation next to freshly regenerated English text after regenerating -
        # a silent mismatch with no visible warning.
        note_id = self._create_note_with_extra_ai_fields()

        def fake_translate(text, lang_code):
            return f"[HI]{text}"

        self._login("t_reviewer", "pw123")
        with patch("app.indic_translate.translate_summary", side_effect=fake_translate):
            self.client.get(f"/triage/{note_id}?lang=hi")  # populates extra_translations_json

        conn = database.get_connection()
        cached_before = conn.execute(
            "SELECT extra_translations_json FROM triage_notes WHERE id=?", (note_id,)
        ).fetchone()["extra_translations_json"]
        conn.close()
        self.assertIsNotNone(cached_before)
        self.assertIn("[HI]", cached_before)

        token = _get_csrf_token(self.client.get(f"/triage/{note_id}").get_data(as_text=True)) or ""
        # Deliberately NOT following the redirect - triage_detail() would otherwise immediately
        # re-populate a (negative-result) cache on the very next GET, which would obscure
        # whether the regenerate route itself actually cleared the stale cache.
        self.client.post(f"/triage/{note_id}/regenerate-ai", data={"csrf_token": token})

        conn = database.get_connection()
        cached_after = conn.execute(
            "SELECT extra_translations_json FROM triage_notes WHERE id=?", (note_id,)
        ).fetchone()["extra_translations_json"]
        conn.close()
        self.assertIsNone(cached_after)


class TestSymptomCategoryAndRiskReasonTranslation(TriageAppTestCase):
    """Regression tests for a second, related gap found while fixing the one above: the
    "Explicitly Denied by Patient" list showed raw, untranslated (and un-humanized, e.g.
    'chest_pain' instead of 'Chest pain') canonical symptom names; the rules-engine cross-check
    panel had the same problem; the symptom category next to each reported symptom
    ('category: cardiac') was never translated; and the plain-language "why this priority" risk
    reasons under Risk Priority - the most prominent AI/rules-generated text on the whole page -
    were shown in English regardless of the selected language. Fixed via translator.py's now-
    complete SYMPTOM_TRANSLATIONS/CATEGORY_TRANSLATIONS phrase banks (negated symptoms, the
    cross-check list, and category are deterministic dictionary lookups - no model needed) and,
    for risk reasons (freeform generated sentences), the same on-demand model-translation cache
    used for chief complaint/missing info/etc.
    """

    def _consent_and_get_intake_token(self):
        r = self.client.get("/consent")
        token = _get_csrf_token(r.get_data(as_text=True)) or ""
        self.client.post("/consent", data={"consent_ack": "on", "csrf_token": token}, follow_redirects=True)
        r = self.client.get("/intake")
        return _get_csrf_token(r.get_data(as_text=True)) or ""

    def _create_note(self, symptom_text, lang="hi"):
        self._login("t_worker", "pw123")
        token = self._consent_and_get_intake_token()
        self.client.post("/intake", data={
            "csrf_token": token, "display_name": "Translation Test Patient 2", "age": "40 years",
            "sex": "Male", "facility_type": "Government Hospital", "facility_name": "Test Hospital",
            "preferred_language": lang, "scenario": "outpatient_queue", "input_mode": "voice",
            "symptom_text": symptom_text,
        }, follow_redirects=True)
        self.client.get("/logout")
        conn = database.get_connection()
        note_id = conn.execute("SELECT id FROM triage_notes ORDER BY id DESC LIMIT 1").fetchone()["id"]
        conn.close()
        return note_id

    def test_reported_symptom_and_category_are_translated_in_hindi(self):
        note_id = self._create_note("Patient reports a seizure and chest pain since yesterday.")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=hi")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("दौरा", html)      # seizure - previously untranslated (outside the old 7-symptom list)
        self.assertIn("सीने में दर्द", html)  # chest pain

    def test_reported_symptom_is_translated_in_odia(self):
        note_id = self._create_note("Patient reports a seizure.", lang="or")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=or")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("ଝଡ଼କା", html)

    def test_negated_symptom_is_translated_not_raw_canonical(self):
        note_id = self._create_note("Patient has chest pain but denies breathlessness.")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=hi")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("सांस लेने में तकलीफ", html)  # translated "breathlessness" in the denied-symptoms line
        self.assertNotIn("breathlessness</p>", html)  # raw canonical must not leak into the denied list

    def test_negated_symptom_is_humanized_in_english_too(self):
        # Bug existed even without translation involved: the denied-symptoms list rendered the
        # raw snake_case canonical ('chest_pain') instead of a readable label ('Chest pain').
        note_id = self._create_note("Patient has fever but denies chest pain.", lang="en")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=en")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Chest pain", html)
        self.assertNotIn("chest_pain", html)

    def test_risk_reasons_are_translated_when_available(self):
        note_id = self._create_note("Patient became unconscious suddenly.")

        def fake_translate(text, lang_code):
            return f"[HI]{text}"

        with patch("app.indic_translate.translate_summary", side_effect=fake_translate):
            self._login("t_reviewer", "pw123")
            r = self.client.get(f"/triage/{note_id}?lang=hi")

        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("[HI]", html)
        self.assertIn("Urgency keyword detected", html)  # inside the [HI]-prefixed mocked translation

    def test_risk_reasons_stay_english_when_language_is_english(self):
        note_id = self._create_note("Patient became unconscious suddenly.", lang="en")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=en")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Urgency keyword detected", html)
        self.assertNotIn("[HI]", html)


class TestStaticUiStringsTranslatedOnTriageDetail(TriageAppTestCase):
    """Regression coverage for a real, confirmed bug: the triage detail page's language switcher
    only ever translated 4 headings (chief complaint, risk priority, missing info, follow-up
    questions) - every OTHER section heading, button, static label, and the risk-tier/status
    badges themselves were hardcoded English directly in triage_detail.html and stayed in
    English regardless of the selected language. See README §11.13.
    """

    def _consent_and_get_intake_token(self):
        r = self.client.get("/consent")
        token = _get_csrf_token(r.get_data(as_text=True)) or ""
        self.client.post("/consent", data={"consent_ack": "on", "csrf_token": token}, follow_redirects=True)
        r = self.client.get("/intake")
        return _get_csrf_token(r.get_data(as_text=True)) or ""

    def _create_note(self, lang="or"):
        self._login("t_worker", "pw123")
        token = self._consent_and_get_intake_token()
        self.client.post("/intake", data={
            "csrf_token": token, "display_name": "Static UI Test Patient", "age": "45 years",
            "sex": "Female", "facility_type": "Government Hospital", "facility_name": "Test Hospital",
            "preferred_language": lang, "scenario": "outpatient_queue", "input_mode": "text",
            "symptom_text": "Fever and cough for three days.",
        }, follow_redirects=True)
        self.client.get("/logout")
        conn = database.get_connection()
        note_id = conn.execute("SELECT id FROM triage_notes ORDER BY id DESC LIMIT 1").fetchone()["id"]
        conn.close()
        return note_id

    def test_odia_page_shows_translated_static_labels_not_english(self):
        note_id = self._create_note(lang="or")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=or")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        # Section headings / labels that were previously always English
        self.assertIn("ରୋଗୀ (ପରିଚୟ ଗୋପନ ଦୃଶ୍ୟ)", html)   # "Patient (anonymised view)"
        self.assertIn("ସମୀକ୍ଷକ କାର୍ଯ୍ୟ", html)              # "Reviewer Actions"
        self.assertIn("ସମୀକ୍ଷା ସେଭ୍ କରନ୍ତୁ", html)          # "Save Review"
        self.assertIn("ରେଫରାଲ PDF ଡାଉନଲୋଡ୍ କରନ୍ତୁ", html)  # "Download Referral PDF"
        self.assertIn("ରେକର୍ଡ", html)                       # "Record"
        # None of the old hardcoded English strings should appear anymore
        self.assertNotIn("Patient (anonymised view)", html)
        self.assertNotIn("Reviewer Actions", html)
        self.assertNotIn(">Save Review<", html)
        self.assertNotIn(">Reviewed Symptoms<", html)

    def test_odia_page_translates_the_risk_tier_badge(self):
        note_id = self._create_note(lang="or")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=or")
        html = r.get_data(as_text=True)
        # Whatever tier this note landed in, the English label must not appear, and the
        # corresponding Odia label must.
        conn = database.get_connection()
        tier = conn.execute("SELECT risk_tier FROM triage_notes ORDER BY id DESC LIMIT 1").fetchone()["risk_tier"]
        conn.close()
        from triage_engine import translator
        self.assertIn(translator.translate_tier_label(tier, "or"), html)
        self.assertNotIn(translator.translate_tier_label(tier, "en"), html)

    def test_odia_page_translates_the_status_dropdown_and_label(self):
        note_id = self._create_note(lang="or")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=or")
        html = r.get_data(as_text=True)
        self.assertIn("ନୂଆ", html)  # "New" status option, translated
        self.assertNotIn(">New<", html)

    def test_english_page_still_shows_english_static_labels(self):
        note_id = self._create_note(lang="en")
        self._login("t_reviewer", "pw123")
        r = self.client.get(f"/triage/{note_id}?lang=en")
        html = r.get_data(as_text=True)
        self.assertIn("Patient (anonymised view)", html)
        self.assertIn("Reviewer Actions", html)
        self.assertIn("Save Review", html)


class TestHealthEndpoint(TriageAppTestCase):
    def test_health_endpoint_reports_structure(self):
        r = self.client.get("/health")
        self.assertIn(r.status_code, (200, 503))
        body = r.get_json()
        self.assertIn("database", body)
        self.assertIn("ai_text_model", body)
        self.assertIn("stats", body["ai_text_model"])


if __name__ == "__main__":
    unittest.main()
