"""
Unit tests for the phrase-bank translator (triage_engine/translator.py).

Regression coverage for a real, confirmed bug: SYMPTOM_TRANSLATIONS originally covered only 7 of
the 27 canonical symptoms defined in triage_engine/risk_rules.py's SYMPTOM_ONTOLOGY. Any symptom
outside that first 7 (seizure, rash, sore throat, etc.) silently fell back to an English label via
translate_symptom_label()'s fallback, even when a reviewer had explicitly selected Hindi or Odia -
meaning a patient's Hindi/Odia voice input could be reflected back to the reviewer partly in
English regardless of language choice. The completeness tests below make sure every symptom (and
every symptom category) the rules engine knows about has a Hindi/Odia entry, so this can't quietly
regress if a new symptom is ever added to the ontology without a matching translation.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from triage_engine import translator
from triage_engine.risk_rules import SYMPTOM_ONTOLOGY
from triage_engine.extractor import CATEGORY_FOLLOWUPS


class TestSymptomTranslationCompleteness(unittest.TestCase):
    def test_every_canonical_symptom_has_hindi_and_odia_translations(self):
        for canonical in SYMPTOM_ONTOLOGY:
            entry = translator.SYMPTOM_TRANSLATIONS.get(canonical)
            self.assertIsNotNone(entry, f"'{canonical}' is missing from SYMPTOM_TRANSLATIONS entirely")
            self.assertTrue(entry.get("hi"), f"'{canonical}' has no Hindi translation")
            self.assertTrue(entry.get("or"), f"'{canonical}' has no Odia translation")

    def test_every_symptom_category_has_hindi_and_odia_translations(self):
        categories = {meta["category"] for meta in SYMPTOM_ONTOLOGY.values()}
        for category in categories:
            entry = translator.CATEGORY_TRANSLATIONS.get(category)
            self.assertIsNotNone(entry, f"'{category}' is missing from CATEGORY_TRANSLATIONS entirely")
            self.assertTrue(entry.get("hi"), f"'{category}' has no Hindi translation")
            self.assertTrue(entry.get("or"), f"'{category}' has no Odia translation")

    def test_translate_symptom_label_returns_known_translation(self):
        self.assertEqual(translator.translate_symptom_label("seizure", "hi"), "दौरा (मिर्गी)")
        self.assertEqual(translator.translate_symptom_label("seizure", "or"), "ଝଡ଼କା (ମୃଗୀ)")

    def test_translate_symptom_label_falls_back_gracefully_for_unknown_canonical(self):
        # An unrecognised canonical (e.g. a future symptom added without a translation yet) must
        # never raise - it should degrade to a readable English label, not crash the page.
        result = translator.translate_symptom_label("some_future_symptom", "hi")
        self.assertEqual(result, "Some future symptom")

    def test_translate_category_label_returns_known_translation(self):
        self.assertEqual(translator.translate_category_label("neuro", "hi"), "तंत्रिका तंत्र संबंधी")
        self.assertEqual(translator.translate_category_label("neuro", "or"), "ସ୍ନାୟୁ ସମ୍ବନ୍ଧୀୟ")

    def test_translate_category_label_falls_back_gracefully_for_unknown_category(self):
        result = translator.translate_category_label("some_future_category", "hi")
        self.assertEqual(result, "Some future category")

    def test_english_language_returns_english_label(self):
        self.assertEqual(translator.translate_symptom_label("seizure", "en"), "Seizure")
        self.assertEqual(translator.translate_category_label("neuro", "en"), "Neurological")


class TestFollowUpQuestionTranslationCompleteness(unittest.TestCase):
    """Regression coverage for a real, confirmed bug: FOLLOWUP_TRANSLATIONS_HI/OR originally
    covered only 6 of the ~29 distinct follow-up-question strings extractor.py's
    CATEGORY_FOLLOWUPS/generate_follow_up_questions() can actually produce (only cardiac,
    infection, and the 3 generic age/duration/vitals questions were covered) - every other
    category (respiratory, neuro, gi, obstetric, pediatric, trauma, mental_health, occupational,
    general, derm, ent, urinary, musculoskeletal) fell straight through to raw English on the
    triage detail page even when Hindi or Odia was selected.
    """

    GENERIC_FOLLOWUPS = [
        "What is the patient's exact age (or date of birth)?",
        "Exactly when did the symptoms start?",
        "Please record temperature, pulse, blood pressure, and SpO2 if a device is available.",
    ]

    def _all_fixed_followup_questions(self):
        questions = set(self.GENERIC_FOLLOWUPS)
        for category_questions in CATEGORY_FOLLOWUPS.values():
            questions.update(category_questions)
        return questions

    def test_every_category_follow_up_question_has_hindi_translation(self):
        for question in self._all_fixed_followup_questions():
            self.assertIn(question, translator.FOLLOWUP_TRANSLATIONS_HI,
                           f"Missing Hindi translation for follow-up question: {question!r}")

    def test_every_category_follow_up_question_has_odia_translation(self):
        for question in self._all_fixed_followup_questions():
            self.assertIn(question, translator.FOLLOWUP_TRANSLATIONS_OR,
                           f"Missing Odia translation for follow-up question: {question!r}")

    def test_translate_followup_question_no_longer_falls_back_to_english_for_every_category(self):
        # Spot-check one question from a category that was previously completely uncovered.
        q = "Is there wheezing or noisy breathing?"  # respiratory
        self.assertNotEqual(translator.translate_followup_question(q, "hi"), q)
        self.assertNotEqual(translator.translate_followup_question(q, "or"), q)


class TestTierAndStatusTranslation(unittest.TestCase):
    """Regression coverage for a real, confirmed bug: the risk-tier badge at the very top of the
    triage detail page (e.g. 'Emergency - See Immediately') and the note status ('New', 'In
    review', ...) were read directly off RISK_TIER_LABELS/STATUS_FLOW - which only ever hold
    English text - and rendered with no translation step at all, so they stayed in English
    regardless of the language switcher. See README §11.13.
    """

    def test_translate_tier_label_covers_every_real_tier(self):
        for tier in ("emergency", "high", "medium", "low", "unclassified"):
            for lang in ("hi", "or"):
                label = translator.translate_tier_label(tier, lang)
                self.assertNotEqual(label, translator.translate_tier_label(tier, "en"))

    def test_translate_tier_label_falls_back_gracefully_for_unknown_tier(self):
        result = translator.translate_tier_label("some_future_tier", "hi")
        self.assertEqual(result, "Some future tier")

    def test_translate_status_label_covers_every_status_flow_value(self):
        for status in ("new", "in_review", "reviewed", "referred", "closed"):
            for lang in ("hi", "or"):
                label = translator.translate_status_label(status, lang)
                self.assertNotEqual(label, translator.translate_status_label(status, "en"))

    def test_translate_status_label_falls_back_gracefully_for_unknown_status(self):
        result = translator.translate_status_label("some_future_status", "hi")
        self.assertEqual(result, "Some future status")


class TestUiStringCompleteness(unittest.TestCase):
    """Every UI_STRINGS entry must have both a Hindi and an Odia value - a key present only in
    English would silently render as English text on any non-English page view via
    get_ui_string()'s fallback, which is exactly the bug this whole file's test additions target.
    """

    def test_every_ui_string_has_hindi_and_odia(self):
        for key, entry in translator.UI_STRINGS.items():
            self.assertTrue(entry.get("hi"), f"UI_STRINGS['{key}'] has no Hindi translation")
            self.assertTrue(entry.get("or"), f"UI_STRINGS['{key}'] has no Odia translation")

    def test_get_ui_string_duration_not_specified(self):
        self.assertNotEqual(translator.get_ui_string("duration_not_specified", "hi"),
                             translator.get_ui_string("duration_not_specified", "en"))
        self.assertNotEqual(translator.get_ui_string("duration_not_specified", "or"),
                             translator.get_ui_string("duration_not_specified", "en"))


if __name__ == "__main__":
    unittest.main()
