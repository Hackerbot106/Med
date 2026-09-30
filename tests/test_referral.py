"""
Tests for triage_engine/referral.py (referral PDF generation).

Regression coverage for a real, confirmed bug: free text (reviewer notes, AI-generated summary/
chief complaint/risk reasons) was inserted directly into reportlab Paragraph() objects, which
parse their input as a small XML/HTML-like markup dialect rather than plain text. An unescaped
'<' not immediately followed by a matching closing tag (very plausible in ordinary reviewer
shorthand, e.g. "BP normal, improving<discharge planned") raised an uncaught ValueError from
reportlab's parser - a hard crash with no test coverage at all before this file existed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import referral

PATIENT = {
    "triage_id": "TRG-TEST-000001", "display_name": "Test Patient", "age": "30 years", "sex": "Female",
    "facility_type": "Government Hospital", "facility_name": "Test Hospital",
}

STRUCTURED = {
    "risk_tier": "high",
    "chief_complaint": "Fever and cough",
    "risk_reasons": ["Urgency keyword detected: 'fever' (infection)"],
    "symptoms": [{"canonical": "high_fever", "category": "infection"}],
    "vitals": {"temperature_f": "103"},
    "missing_info": ["Duration of symptoms not specified"],
}


class TestReferralPdfDoesNotCrashOnUnescapedText(unittest.TestCase):
    def test_reviewer_notes_with_unmatched_angle_bracket_does_not_crash(self):
        # The exact real-world reproduction: natural reviewer shorthand with a stray '<'.
        buf = referral.build_referral_pdf(
            PATIENT, STRUCTURED, reviewer_name="Dr. Test",
            reviewer_notes="BP normal, improving<discharge planned", referral_facility="General Hospital",
        )
        self.assertGreater(len(buf.getvalue()), 0)

    def test_ampersand_in_free_text_does_not_crash(self):
        structured = dict(STRUCTURED, chief_complaint="Pain & swelling in left leg")
        buf = referral.build_referral_pdf(PATIENT, structured, reviewer_notes="Salt & pepper appearance noted")
        self.assertGreater(len(buf.getvalue()), 0)

    def test_ai_summary_with_angle_brackets_does_not_crash(self):
        buf = referral.build_referral_pdf(
            PATIENT, STRUCTURED, ai_summary="Patient reports pain <5/10 in severity, improving",
            ai_summary_model="test-model",
        )
        self.assertGreater(len(buf.getvalue()), 0)

    def test_risk_reason_with_angle_bracket_does_not_crash(self):
        structured = dict(STRUCTURED, risk_reasons=["Temperature <90F noted as abnormally low"])
        buf = referral.build_referral_pdf(PATIENT, structured)
        self.assertGreater(len(buf.getvalue()), 0)

    def test_reviewer_notes_newlines_still_render_as_line_breaks(self):
        # Escaping must happen BEFORE the deliberate \n -> <br/> substitution, not after -
        # otherwise our own <br/> markup would itself get escaped into visible text.
        buf = referral.build_referral_pdf(
            PATIENT, STRUCTURED, reviewer_notes="Line one\nLine two",
        )
        self.assertGreater(len(buf.getvalue()), 0)

    def test_normal_case_still_produces_a_valid_pdf(self):
        buf = referral.build_referral_pdf(
            PATIENT, STRUCTURED, reviewer_name="Dr. Test", reviewer_notes="Stable, referred for follow-up.",
            referral_facility="District Hospital",
        )
        content = buf.getvalue()
        self.assertTrue(content.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
