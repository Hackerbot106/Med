"""
Unit tests for the AI safety filter, diagnosis-attribution logic, and
translation-quality checks in triage_engine/llm.py. These run WITHOUT the
actual model loaded/downloaded - they exercise the pure-Python guardrails
directly, which is exactly what a hackathon judge (or a future
contributor) should be able to verify without a 1GB download.

Regression coverage note: test_required_disclaimer_line_is_never_rejected
below is a direct regression test for a real bug found and fixed earlier
in this project's history - the safety filter used to reject EVERY
compliant AI summary because they were required to end with the word
"diagnosis" in a disclaimer, and the filter didn't yet know to treat a
negated/disclaiming mention as safe. Keep this test if that logic is ever
touched again.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import llm


class TestSafetyFilter(unittest.TestCase):
    def test_required_disclaimer_line_is_never_rejected(self):
        text = (
            "Patient reports fever and cough for three days. "
            "For reviewer confirmation - not a diagnosis."
        )
        self.assertIsNotNone(llm._safety_filter(text))

    def test_assertive_new_diagnosis_is_rejected(self):
        self.assertIsNone(llm._safety_filter("The diagnosis is pneumonia."))
        self.assertIsNone(llm._safety_filter("This confirms a diagnosis of malaria."))

    def test_direct_second_person_diagnosis_is_rejected(self):
        self.assertIsNone(llm._safety_filter("You have pneumonia and should rest."))

    def test_prescription_language_is_rejected(self):
        self.assertIsNone(llm._safety_filter("We recommend treatment with antibiotics."))
        self.assertIsNone(llm._safety_filter("Take 500 mg paracetamol every 6 hours."))

    def test_attributed_existing_condition_is_allowed(self):
        text = "Patient reports known diagnosis of asthma and is currently stable."
        self.assertIsNotNone(llm._safety_filter(text))
        text2 = "Patient has a history of diabetes, diagnosed 5 years ago per patient report."
        self.assertIsNotNone(llm._safety_filter(text2))

    def test_empty_or_blank_text_is_rejected(self):
        self.assertIsNone(llm._safety_filter(""))
        self.assertIsNone(llm._safety_filter("   "))
        self.assertIsNone(llm._safety_filter(None))

    def test_ordinary_neutral_summary_passes(self):
        text = (
            "Patient reports fever and mild cough for two days, no breathlessness. "
            "Vitals within normal range. For reviewer confirmation - not a diagnosis."
        )
        self.assertEqual(llm._safety_filter(text), text)


class TestExtractionValidation(unittest.TestCase):
    def test_valid_extraction_passes(self):
        data = {
            "chief_complaint": "fever and cough",
            "symptoms": [{"name": "fever", "severity": "mild", "category": "infection"}],
        }
        result = llm._validate_extraction_json(data)
        self.assertIsNotNone(result)
        self.assertEqual(result["chief_complaint"], "fever and cough")
        self.assertEqual(result["symptoms"][0]["severity"], "mild")

    def test_empty_extraction_is_rejected(self):
        self.assertIsNone(llm._validate_extraction_json({"chief_complaint": "", "symptoms": []}))

    def test_invalid_severity_falls_back_to_unspecified(self):
        data = {"chief_complaint": "x", "symptoms": [{"name": "pain", "severity": "extreme"}]}
        result = llm._validate_extraction_json(data)
        self.assertEqual(result["symptoms"][0]["severity"], "unspecified")

    def test_invalid_category_falls_back_to_general(self):
        data = {"chief_complaint": "x", "symptoms": [{"name": "pain", "category": "not_a_real_category"}]}
        result = llm._validate_extraction_json(data)
        self.assertEqual(result["symptoms"][0]["category"], "general")

    def test_non_dict_input_is_rejected(self):
        self.assertIsNone(llm._validate_extraction_json(["not", "a", "dict"]))
        self.assertIsNone(llm._validate_extraction_json(None))

    def test_json_schema_is_serializable_and_shaped_correctly(self):
        import json
        schema_str = json.dumps(llm.EXTRACTION_JSON_SCHEMA)
        reloaded = json.loads(schema_str)
        self.assertEqual(reloaded["type"], "object")
        self.assertIn("chief_complaint", reloaded["properties"])
        self.assertIn("symptoms", reloaded["properties"])


class TestJsonParsing(unittest.TestCase):
    def test_parses_clean_json(self):
        self.assertEqual(llm._parse_json_object('{"a": 1}'), {"a": 1})

    def test_strips_markdown_fences(self):
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(llm._parse_json_object(text), {"a": 1})

    def test_extracts_json_from_surrounding_prose(self):
        text = 'Sure, here is the JSON: {"a": 1} - hope that helps!'
        self.assertEqual(llm._parse_json_object(text), {"a": 1})

    def test_unparseable_text_returns_none(self):
        self.assertIsNone(llm._parse_json_object("not json at all"))
        self.assertIsNone(llm._parse_json_object(""))
        self.assertIsNone(llm._parse_json_object(None))


class TestTranslationQualityChecks(unittest.TestCase):
    def test_script_match_ratio_full_match(self):
        odia_text = "ମୁଁ ଭଲ ଅଛି"  # sample Odia text
        self.assertGreater(llm._script_match_ratio(odia_text, "or"), 0.9)

    def test_script_match_ratio_wrong_script_is_low(self):
        # Bengali text is a different (though visually similar-ish) Unicode
        # block from Odia - this is the exact real failure mode observed:
        # a small model drifting into Bengali while believing it wrote Odia.
        bengali_text = "আমি ভালো আছি"
        self.assertLess(llm._script_match_ratio(bengali_text, "or"), 0.6)

    def test_script_match_ratio_no_range_defined_passes_through(self):
        self.assertEqual(llm._script_match_ratio("anything at all", "en"), 1.0)

    def test_degenerate_repetition_is_detected(self):
        looping_text = "the patient has a fever " * 10
        self.assertTrue(llm._looks_degenerate(looping_text))

    def test_normal_length_varied_text_is_not_degenerate(self):
        normal_text = (
            "Patient reports fever and mild cough for two days, no breathlessness, "
            "denies chest pain, vitals stable, advised rest and follow-up if worsening."
        )
        self.assertFalse(llm._looks_degenerate(normal_text))

    def test_short_text_is_never_flagged_degenerate(self):
        self.assertFalse(llm._looks_degenerate("short text"))


class TestExtractionRetryWidensTokenBudget(unittest.TestCase):
    """Regression test for a real bug found on first real-hardware run: with
    grammar-constrained decoding active, a verbose case (e.g. a pediatric
    vomiting case with a long missing_info list) could exhaust max_tokens
    before the JSON closed, producing a syntactically incomplete document
    that fails to parse - identically on every retry, since a retry at the
    SAME token budget and low temperature reproduces the same truncation
    point. The fix: each retry uses a wider max_tokens (see
    EXTRACTION_MAX_TOKENS / EXTRACTION_MAX_TOKENS_RETRY). This test locks
    that behavior in without needing the actual model loaded.
    """

    @patch.object(llm, "_get_extraction_grammar", return_value=None)
    def test_retry_uses_a_wider_token_budget_than_the_first_attempt(self, _mock_grammar):
        seen_max_tokens = []

        def fake_chat(system_prompt, user_prompt, max_tokens=280, temperature=0.2, grammar=None):
            seen_max_tokens.append(max_tokens)
            return None  # simulate every attempt failing, so both attempts run

        with patch.object(llm, "_chat", side_effect=fake_chat):
            result = llm.generate_ai_extraction("Some symptom text.", {"age": "5 years", "sex": "Female"})

        self.assertIsNone(result)  # every attempt failed -> caller falls back, as designed
        self.assertEqual(len(seen_max_tokens), llm.MAX_EXTRACTION_RETRIES + 1)
        self.assertEqual(seen_max_tokens[0], llm.EXTRACTION_MAX_TOKENS)
        for later_attempt_tokens in seen_max_tokens[1:]:
            self.assertEqual(later_attempt_tokens, llm.EXTRACTION_MAX_TOKENS_RETRY)
            self.assertGreater(later_attempt_tokens, seen_max_tokens[0])

    def test_schema_size_caps_are_internally_consistent(self):
        # The grammar's array/string caps and the post-hoc Python-side
        # validation caps must agree (single source of truth) - if someone
        # changes one without the other, this catches the drift.
        schema = llm.EXTRACTION_JSON_SCHEMA
        self.assertEqual(schema["properties"]["symptoms"]["maxItems"], llm.MAX_SYMPTOMS)
        self.assertEqual(schema["properties"]["chief_complaint"]["maxLength"], llm.MAX_CHIEF_COMPLAINT_LEN)
        self.assertEqual(schema["properties"]["follow_up_questions"]["maxItems"], llm.MAX_FOLLOWUP_QUESTIONS)


if __name__ == "__main__":
    unittest.main()
