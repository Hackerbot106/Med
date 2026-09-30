"""
Unit tests for the deterministic rules engine (triage_engine/risk_rules.py) -
the ONE component that decides risk tier, per the safety-first design (see
triage_engine/pipeline.py's module docstring). These tests exist precisely
because that safety guarantee is only as good as this module's correctness.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import risk_rules


class TestSymptomDetection(unittest.TestCase):
    def test_detects_known_symptom(self):
        found = risk_rules.detect_symptoms("Patient has severe chest pain since 2 hours.")
        canonicals = {s["canonical"] for s in found}
        self.assertIn("chest_pain", canonicals)

    def test_compound_or_negation_excludes_both_symptoms(self):
        # Regression test for a real, confirmed bug: "does not have X or Y" only excluded X
        # (the item directly adjacent to the negation cue) - Y was wrongly reported as PRESENT
        # because the fixed-width lookback window couldn't reach back past the first item to see
        # the negation cue. _is_negated() now widens the search specifically for a symptom
        # directly preceded by a bare "or"/"and"/"nor" connector.
        found = risk_rules.detect_symptoms("Patient does not have breathlessness or chest pain today.")
        canonicals = {s["canonical"] for s in found}
        self.assertNotIn("chest_pain", canonicals)
        self.assertNotIn("breathlessness", canonicals)

    def test_compound_negation_does_not_cross_a_contrastive_but(self):
        # The compound-list widening must NOT cross "but"/"however" - a genuinely present
        # symptom mentioned before a contrastive conjunction must not be swept into a LATER
        # negation ("has chest pain BUT denies breathlessness" must leave chest pain present).
        found = risk_rules.detect_symptoms("Patient has chest pain but denies breathlessness.")
        canonicals = {s["canonical"] for s in found}
        self.assertIn("chest_pain", canonicals)
        self.assertNotIn("breathlessness", canonicals)

    def test_compound_negation_does_not_cross_a_comma_clause_boundary(self):
        # Comma is deliberately NOT treated as a list connector (see risk_rules.py's comment) -
        # it's too ambiguous between "another denied item" and "a new clause entirely". A
        # genuinely present symptom mentioned after a comma must not be wrongly negated.
        found = risk_rules.detect_symptoms("No history of fever, chest pain now present for two days.")
        canonicals = {s["canonical"] for s in found}
        self.assertIn("chest_pain", canonicals)

    def test_negation_excludes_symptom_from_present_list(self):
        found = risk_rules.detect_symptoms("No chest pain, no breathlessness, just a mild headache.")
        canonicals = {s["canonical"] for s in found}
        self.assertNotIn("chest_pain", canonicals)
        self.assertNotIn("breathlessness", canonicals)

    def test_negated_symptom_is_reported_separately(self):
        negated = risk_rules.detect_negated_symptoms("No chest pain reported.")
        self.assertIn("chest_pain", negated)

    def test_empty_text_returns_no_symptoms(self):
        self.assertEqual(risk_rules.detect_symptoms(""), [])
        self.assertEqual(risk_rules.detect_symptoms(None), [])


class TestVitalsExtraction(unittest.TestCase):
    def test_extracts_spo2(self):
        vitals = risk_rules.extract_vitals("SpO2: 88%, temp 101F")
        self.assertIn("spo2", vitals)
        self.assertEqual(vitals["spo2"], "88")

    def test_low_spo2_flagged_as_dangerous(self):
        reasons = risk_rules.evaluate_vitals({"spo2": "85"})
        self.assertTrue(any("oxygen" in r["reason"].lower() for r in reasons))

    def test_normal_vitals_not_flagged(self):
        reasons = risk_rules.evaluate_vitals({"spo2": "98", "heart_rate": "72", "temperature_f": "98.6"})
        self.assertEqual(reasons, [])

    def test_malformed_vitals_do_not_raise(self):
        # evaluate_vitals must degrade gracefully on garbage input - it feeds
        # off best-effort regex extraction from free text/OCR, which can
        # occasionally produce a non-numeric capture.
        reasons = risk_rules.evaluate_vitals({"spo2": "not-a-number", "blood_pressure": "oops"})
        self.assertEqual(reasons, [])

    def test_pain_scale_is_not_misread_as_blood_pressure(self):
        # Regression test for a real, confirmed bug: the blood_pressure regex matched ANY
        # "N/N" fragment with no plausibility check, so a 0-10 pain scale ("pain is 12/10") was
        # silently accepted as a blood pressure reading and then flagged "abnormal".
        vitals = risk_rules.extract_vitals("pain is 12/10 unbearable")
        self.assertNotIn("blood_pressure", vitals)

    def test_date_like_fraction_is_not_misread_as_blood_pressure(self):
        vitals = risk_rules.extract_vitals("please follow up on 15/11 for review")
        self.assertNotIn("blood_pressure", vitals)

    def test_plausible_blood_pressure_is_still_extracted(self):
        vitals = risk_rules.extract_vitals("BP 140/90 mmHg recorded")
        self.assertEqual(vitals.get("blood_pressure"), "140/90")

    def test_plausible_low_blood_pressure_is_still_extracted(self):
        vitals = risk_rules.extract_vitals("BP 70/40, patient looks unwell")
        self.assertEqual(vitals.get("blood_pressure"), "70/40")


class TestComputeRisk(unittest.TestCase):
    def test_red_flag_symptom_forces_emergency(self):
        symptoms = risk_rules.detect_symptoms("Patient had a seizure just now.")
        risk = risk_rules.compute_risk(symptoms, {})
        self.assertEqual(risk["tier"], "emergency")

    def test_low_oxygen_forces_high_urgency(self):
        symptoms = risk_rules.detect_symptoms("Cough for two days.")
        risk = risk_rules.compute_risk(symptoms, {"spo2": "85"})
        self.assertIn(risk["tier"], ("emergency", "high"))

    def test_mild_symptom_alone_is_not_emergency(self):
        symptoms = risk_rules.detect_symptoms("Mild headache and slight fatigue for one day.")
        risk = risk_rules.compute_risk(symptoms, {})
        self.assertNotEqual(risk["tier"], "emergency")

    def test_no_symptoms_and_no_vitals_is_low_or_unclassified(self):
        risk = risk_rules.compute_risk([], {})
        self.assertIn(risk["tier"], ("low", "unclassified"))

    def test_combo_rule_chest_pain_and_breathlessness_is_emergency(self):
        symptoms = risk_rules.detect_symptoms("Chest pain and breathlessness since this morning.")
        risk = risk_rules.compute_risk(symptoms, {})
        self.assertEqual(risk["tier"], "emergency")
        self.assertTrue(any("cardiac" in r.lower() or "respiratory" in r.lower() for r in risk["reasons"]))

    def test_every_tier_has_a_reason(self):
        symptoms = risk_rules.detect_symptoms("Fever for 3 days.")
        risk = risk_rules.compute_risk(symptoms, {})
        self.assertTrue(len(risk["reasons"]) >= 1)

    def test_toddler_in_months_is_not_flagged_as_infant(self):
        # Regression test for a real, confirmed bug: ANY age given in months was unconditionally
        # treated as "<1 year", regardless of the actual number - "18 months" (a toddler) was
        # mislabeled "Infant patient (<1 year)".
        risk = risk_rules.compute_risk([], {}, age_text="18 months")
        self.assertFalse(any("infant" in r.lower() for r in risk["reasons"]))

    def test_newborn_in_months_is_flagged_as_infant(self):
        risk = risk_rules.compute_risk([], {}, age_text="6 months")
        self.assertTrue(any("infant" in r.lower() for r in risk["reasons"]))

    def test_newborn_in_days_is_flagged_as_infant_not_elderly(self):
        # Regression test for a real, confirmed bug: "days" was never handled as a unit at all,
        # so the raw number fell through to the "years" branch - "75 days" (a newborn) was
        # mislabeled "Elderly patient (70+)" instead of getting the infant safety bump.
        risk = risk_rules.compute_risk([], {}, age_text="75 days")
        reasons_lower = [r.lower() for r in risk["reasons"]]
        self.assertTrue(any("infant" in r for r in reasons_lower))
        self.assertFalse(any("elderly" in r for r in reasons_lower))

    def test_elderly_in_years_is_still_flagged(self):
        risk = risk_rules.compute_risk([], {}, age_text="75 years")
        self.assertTrue(any("elderly" in r.lower() for r in risk["reasons"]))


if __name__ == "__main__":
    unittest.main()
