"""
Tests for triage_engine/pipeline.py's merge logic - the core safety
guarantee of this whole project: "the AI can add urgency signals the
rules engine missed, but can never make it drop one it already caught."

The local AI model is mocked out entirely here (via unittest.mock.patch)
so these tests run instantly and deterministically, without needing the
~1GB model file downloaded - they test the MERGE LOGIC, not the model's
output quality (that's what a judge running the live demo sees).
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import pipeline, llm, ocr, extractor


PATIENT = {
    "triage_id": "TRG-TEST-000001", "display_name": "Test", "age": "40 years", "sex": "Male",
    "facility_type": "Government Hospital", "facility_name": "Test Hospital",
    "preferred_language": "en",
}


class TestPipelineFallback(unittest.TestCase):
    @patch.object(llm, "model_file_present", return_value=False)
    def test_falls_back_to_rules_when_model_not_downloaded(self, _mock):
        result = pipeline.run_triage_pipeline("Fever and cough for two days.", PATIENT)
        self.assertFalse(result["ai_powered"])
        self.assertIn("not downloaded", result["extraction_source"])

    @patch.object(llm, "generate_ai_extraction", return_value=None)
    @patch.object(llm, "model_file_present", return_value=True)
    def test_falls_back_to_rules_when_ai_returns_none(self, _mock1, _mock2):
        result = pipeline.run_triage_pipeline("Fever and cough for two days.", PATIENT)
        self.assertFalse(result["ai_powered"])
        self.assertIn("did not return a usable result", result["extraction_source"])


class TestPipelineAiUnionSafety(unittest.TestCase):
    """The critical safety property: the FINAL risk tier is computed on the
    UNION of what rules + AI each independently found, so the AI reporting
    fewer/milder symptoms than the rules engine can never silently lower
    urgency below what the deterministic engine alone would have assigned.
    """

    @patch.object(llm, "model_file_present", return_value=True)
    def test_ai_cannot_downgrade_a_rules_detected_red_flag(self, _mock):
        raw_text = "Patient had a seizure just now."  # rules engine: red-flag -> emergency

        # Simulate a (badly) underperforming AI that only reports a mild,
        # unrelated symptom and misses the seizure entirely.
        with patch.object(llm, "generate_ai_extraction", return_value={
            "chief_complaint": "mild fatigue",
            "symptoms": [{"name": "fatigue", "severity": "mild", "category": "general"}],
            "negated_symptoms": [], "timeline": [], "missing_info": [], "follow_up_questions": [],
        }):
            result = pipeline.run_triage_pipeline(raw_text, PATIENT)

        self.assertTrue(result["ai_powered"])
        # The rules-only cross-check should still show emergency...
        self.assertEqual(result["rules_only_risk_tier"], "emergency")
        # ...AND the actual displayed/acted-on tier must ALSO be emergency,
        # because compute_risk() runs on the union, not on the AI's list alone.
        self.assertEqual(result["risk_tier"], "emergency")

    @patch.object(llm, "model_file_present", return_value=True)
    def test_ai_can_add_a_symptom_rules_engine_missed(self, _mock):
        # Free text the rules engine's keyword list doesn't recognize at all.
        raw_text = "Patient describes their vision going dark suddenly."

        with patch.object(llm, "generate_ai_extraction", return_value={
            "chief_complaint": "sudden vision loss",
            "symptoms": [{"name": "sudden vision loss", "severity": "severe", "category": "neuro"}],
            "negated_symptoms": [], "timeline": [], "missing_info": [], "follow_up_questions": [],
        }):
            result = pipeline.run_triage_pipeline(raw_text, PATIENT)

        self.assertTrue(result["ai_powered"])
        # The rules-only pass found nothing actionable in this text...
        self.assertIn(result["rules_only_risk_tier"], ("low", "unclassified"))
        # ...but the AI's genuinely new finding should be reflected in the
        # final structured note (this is "the AI can only ADD signals").
        ai_reported_names = [s.get("ai_reported_name", "") for s in result["symptoms"]]
        self.assertTrue(any("vision" in n.lower() for n in ai_reported_names))

    @patch.object(llm, "model_file_present", return_value=True)
    def test_extraction_source_always_labeled_for_transparency(self, _mock):
        with patch.object(llm, "generate_ai_extraction", return_value={
            "chief_complaint": "fever", "symptoms": [{"name": "fever", "severity": "mild", "category": "infection"}],
            "negated_symptoms": [], "timeline": [], "missing_info": [], "follow_up_questions": [],
        }):
            result = pipeline.run_triage_pipeline("Fever for two days.", PATIENT)
        self.assertIn("extraction_source", result)
        self.assertTrue(result["extraction_source"])


class TestOcrPlaceholderTextIsFiltered(unittest.TestCase):
    """Regression test for a real, confirmed bug: uploading a non-text photo
    (an X-ray, a wound photo) still runs OCR alongside the vision model
    (see app.py's intake handler), and run_ocr() correctly returns a
    human-readable placeholder like "[No text could be extracted...]" when
    there's no text to find - but that placeholder string was being fed
    straight into both the rules-based vitals scan and the AI extraction
    prompt as if it were real report content, producing confusing
    extraction output that had nothing to do with the AI vision
    description the user actually expected. run_triage_pipeline() must
    filter it out via ocr.is_usable_ocr_text() before it reaches either
    extraction path - real OCR text must still go through untouched.
    """

    NO_TEXT_PLACEHOLDER = (
        "[No text could be extracted from this file - image may be low quality; please attach "
        "for manual reviewer inspection.]"
    )

    @patch.object(llm, "model_file_present", return_value=True)
    def test_ocr_placeholder_is_not_passed_to_ai_extraction(self, _mock):
        with patch.object(llm, "generate_ai_extraction", return_value={
            "chief_complaint": "fever", "symptoms": [{"name": "fever", "severity": "mild", "category": "infection"}],
            "negated_symptoms": [], "timeline": [], "missing_info": [], "follow_up_questions": [],
        }) as mock_ai_extract:
            pipeline.run_triage_pipeline(
                "Fever for two days.", PATIENT, ocr_text=self.NO_TEXT_PLACEHOLDER,
            )
        # generate_ai_extraction must have been called with ocr_text=None, not the placeholder -
        # confirms the placeholder never reaches the AI prompt as if it were report content.
        _args, kwargs = mock_ai_extract.call_args
        self.assertIsNone(kwargs.get("ocr_text"))

    @patch.object(llm, "model_file_present", return_value=False)
    def test_ocr_placeholder_is_not_passed_to_rules_based_extraction(self, _mock):
        # With no AI model, run_triage_pipeline() falls back to extractor.extract_structured_note()
        # directly - the placeholder must be filtered before reaching that path too.
        with patch("triage_engine.pipeline.extractor.extract_structured_note") as mock_extract:
            mock_extract.return_value = {"symptoms": [], "extraction_source": "", "ai_powered": False}
            pipeline.run_triage_pipeline(
                "Fever for two days.", PATIENT, ocr_text=self.NO_TEXT_PLACEHOLDER,
            )
        _args, kwargs = mock_extract.call_args
        self.assertIsNone(kwargs.get("ocr_text"))

    @patch.object(llm, "model_file_present", return_value=True)
    def test_real_ocr_text_still_reaches_ai_extraction_unfiltered(self, _mock):
        real_ocr_text = "BP 140/90, Temp 101F"
        with patch.object(llm, "generate_ai_extraction", return_value={
            "chief_complaint": "fever", "symptoms": [{"name": "fever", "severity": "mild", "category": "infection"}],
            "negated_symptoms": [], "timeline": [], "missing_info": [], "follow_up_questions": [],
        }) as mock_ai_extract:
            pipeline.run_triage_pipeline("Fever for two days.", PATIENT, ocr_text=real_ocr_text)
        _args, kwargs = mock_ai_extract.call_args
        self.assertEqual(kwargs.get("ocr_text"), real_ocr_text)

    def test_filter_helper_agrees_with_ocr_module(self):
        # Sanity check that pipeline.py is using the same shared helper, not a reimplementation.
        self.assertFalse(ocr.is_usable_ocr_text(self.NO_TEXT_PLACEHOLDER))


class TestOcrTextIsScannedForSymptoms(unittest.TestCase):
    """Regression test for a real, confirmed bug: extractor.extract_structured_note()'s
    'combined_text_for_symptoms' variable was never actually combined with ocr_text - only
    numeric vitals regexes read OCR'd document text; symptom/red-flag keyword detection only
    ever looked at the patient's own typed/spoken text. A scanned referral note that plainly
    said "unconscious" or "severe bleeding" produced NO red-flag detection at all if the
    intake worker left the typed-symptom box empty (a very real scenario: an image-only upload).
    """

    def test_red_flag_in_ocr_text_alone_is_detected(self):
        structured = extractor.extract_structured_note(
            raw_text="", patient={"age": "40 years", "sex": "Male"},
            ocr_text="Referral note: patient unconscious after fall, severe bleeding from head wound.",
        )
        canonicals = {s["canonical"] for s in structured["symptoms"]}
        self.assertIn("unconsciousness", canonicals)
        self.assertIn("severe_bleeding", canonicals)
        self.assertEqual(structured["risk_tier"], "emergency")

    @patch.object(llm, "model_file_present", return_value=False)
    def test_red_flag_in_ocr_text_alone_is_detected_via_full_pipeline(self, _mock):
        result = pipeline.run_triage_pipeline(
            "", PATIENT, ocr_text="Patient reports severe bleeding and is unconscious.",
        )
        self.assertEqual(result["risk_tier"], "emergency")


class TestMatchOntologyIsNotOverPermissive(unittest.TestCase):
    """Regression test for a real, confirmed bug: _match_ontology() matched bidirectionally
    (alias-in-name OR name-in-alias), so a short, generic AI-reported symptom name that happened
    to be a substring of a longer, more specific red-flag alias was wrongly matched to it -
    "weakness"/"weak" matched stroke_signs via "sudden weakness one side", "speech" matched it
    via "cannot speak properly", and "bleeding" matched severe_bleeding via "severe bleeding".
    All three are red_flag=True, so this silently forced a false "emergency" tier for routine
    complaints. Matching is now one-directional (alias must appear IN the AI's name), which still
    resolves genuine paraphrases/elaborations correctly.
    """

    def test_generic_short_names_no_longer_false_match_red_flags(self):
        from triage_engine import risk_rules
        for name in ("weakness", "weak", "speech", "bleeding", "pain"):
            canonical = pipeline._match_ontology(name)
            if canonical is not None:
                self.assertFalse(
                    risk_rules.SYMPTOM_ONTOLOGY[canonical]["red_flag"],
                    f"{name!r} should not match a red-flag alias (matched {canonical!r})",
                )

    def test_elaborate_ai_description_still_matches_correctly(self):
        self.assertEqual(pipeline._match_ontology("chest tightness"), "chest_pain")
        self.assertEqual(pipeline._match_ontology("severe chest pain radiating to the arm"), "chest_pain")
        self.assertEqual(pipeline._match_ontology("convulsions in child"), "seizure")

    @patch.object(llm, "model_file_present", return_value=True)
    def test_generic_ai_symptom_name_no_longer_forces_false_emergency(self, _mock):
        # "patient feels weak and tired" is a routine, non-emergency complaint - the AI reporting
        # it with a terse name like "weak" must not force an emergency tier via a false match to
        # a red-flag ontology entry (this was the real, reproducible bug).
        with patch.object(llm, "generate_ai_extraction", return_value={
            "chief_complaint": "feeling weak", "symptoms": [{"name": "weak", "severity": "mild", "category": "general"}],
            "negated_symptoms": [], "timeline": [], "missing_info": [], "follow_up_questions": [],
        }):
            result = pipeline.run_triage_pipeline("Patient feels weak and tired.", PATIENT)
        self.assertNotEqual(result["risk_tier"], "emergency")


class TestNegatedSymptomCanonicalMapping(unittest.TestCase):
    """Regression test for a real, confirmed bug: extractor.py pre-humanized negated symptoms
    with .replace('_',' ') before storing them (so translator.translate_symptom_label(), which
    is keyed by the underscored canonical form, could never find a match and always fell back to
    English) - and the AI-powered path never mapped its free-text negated-symptom strings back to
    the ontology at all, unlike present symptoms. Both are fixed: negated symptoms are now stored
    as canonical keys, matched through the same ontology lookup as present symptoms.
    """

    def test_rules_path_keeps_negated_symptoms_as_canonical_keys(self):
        structured = extractor.extract_structured_note(
            "Patient has breathlessness but denies chest pain.", {"age": "40 years", "sex": "Male"},
        )
        self.assertIn("chest_pain", structured["negated_symptoms"])

    @patch.object(llm, "model_file_present", return_value=True)
    def test_ai_path_maps_negated_symptoms_to_canonical_keys(self, _mock):
        with patch.object(llm, "generate_ai_extraction", return_value={
            "chief_complaint": "fever", "symptoms": [{"name": "fever", "severity": "mild", "category": "infection"}],
            "negated_symptoms": ["chest pain"], "timeline": [], "missing_info": [], "follow_up_questions": [],
        }):
            result = pipeline.run_triage_pipeline("Fever, no chest pain.", PATIENT)
        self.assertIn("chest_pain", result["negated_symptoms"])

    @patch.object(llm, "model_file_present", return_value=True)
    def test_ai_path_falls_back_to_raw_text_for_unmatched_negated_symptom(self, _mock):
        with patch.object(llm, "generate_ai_extraction", return_value={
            "chief_complaint": "fever", "symptoms": [{"name": "fever", "severity": "mild", "category": "infection"}],
            "negated_symptoms": ["some unusual complaint"], "timeline": [], "missing_info": [], "follow_up_questions": [],
        }):
            result = pipeline.run_triage_pipeline("Fever only.", PATIENT)
        self.assertIn("some unusual complaint", result["negated_symptoms"])


if __name__ == "__main__":
    unittest.main()
