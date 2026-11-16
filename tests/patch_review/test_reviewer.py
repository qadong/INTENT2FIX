from INTENT2FIX.intent_dialogue.developer_agent import DeveloperSession
from INTENT2FIX.patch_review.reviewer import (
    PatchSurfaceReviewer,
    SubmitPatchReview,
    SubmitPatchReviewArgs,
)
from INTENT2FIX.patch_review.schema import PatchReviewResult


def test_patch_review_is_binary_and_rejection_requires_report():
    session = DeveloperSession()
    evidence = session.record(
        "grep -RInF -- 'document_reference' include src",
        "include/document_reference.h:42",
        0,
    )
    submit = SubmitPatchReview(session=session)

    missing_report = submit.execute(
        SubmitPatchReviewArgs(
            thoughts="",
            approved=False,
            evidence_ids=[evidence.id],
            surface_checks=[],
        )
    )
    assert missing_report.expect_correction

    rejected = submit.execute(
        SubmitPatchReviewArgs(
            thoughts="",
            approved=False,
            report="The parallel document_reference wrapper is unchanged.",
            evidence_ids=[evidence.id],
            surface_checks=[
                {
                    "surface": "document_reference",
                    "disposition": "required",
                    "rationale": "The existing sibling operation is forwarded here.",
                    "counterexample": (
                        "Calling the requested operation through "
                        "document_reference remains unavailable."
                    ),
                    "evidence_ids": [evidence.id],
                }
            ],
        )
    )
    result = PatchReviewResult.model_validate(
        rejected.properties["patch_review"]
    )
    assert rejected.terminal
    assert result.approved is False
    assert "document_reference" in result.report


def test_patch_review_requires_repository_evidence_for_approval():
    submit = SubmitPatchReview(session=DeveloperSession())

    observation = submit.execute(
        SubmitPatchReviewArgs(
            thoughts="",
            approved=True,
            report="No omissions found.",
            surface_checks=[],
        )
    )

    assert observation.expect_correction
    assert "Inspect at least one" in observation.message


def test_patch_review_accepts_evidence_cited_by_surface_checks():
    session = DeveloperSession()
    evidence = session.record(
        "grep -RInF -- 'wide_join' include",
        "include/xchar.h:42 wide_join",
        0,
    )

    observation = SubmitPatchReview(session=session).execute(
        SubmitPatchReviewArgs(
            thoughts="",
            approved=False,
            report="The wide entry point still exposes the old behavior.",
            surface_checks=[
                {
                    "surface": "wide join entry point",
                    "disposition": "required",
                    "rationale": "It forwards through the same public family.",
                    "counterexample": "Calling wide join still fails.",
                    "evidence_ids": [evidence.id],
                }
            ],
        )
    )

    result = PatchReviewResult.model_validate(
        observation.properties["patch_review"]
    )
    assert observation.terminal
    assert result.evidence_ids == [evidence.id]


def test_patch_review_cannot_approve_a_required_surface():
    session = DeveloperSession()
    evidence = session.record(
        "grep -RInF -- 'count_elements' include",
        "include/document.h:42 count_elements()",
        0,
    )
    observation = SubmitPatchReview(session=session).execute(
        SubmitPatchReviewArgs(
            thoughts="",
            approved=True,
            evidence_ids=[evidence.id],
            surface_checks=[
                {
                    "surface": "document forwarding wrapper",
                    "disposition": "required",
                    "rationale": "The sibling operation is forwarded by document.",
                    "counterexample": (
                        "The document forwarding entry point still exposes the "
                        "old behavior."
                    ),
                    "evidence_ids": [evidence.id],
                }
            ],
        )
    )

    assert observation.expect_correction
    assert "conflicts with required" in observation.message


def test_patch_review_can_approve_with_advisory_surface():
    session = DeveloperSession()
    evidence = session.record(
        "grep -RInF -- 'document_reference' include src",
        "include/document_reference.h:42 sibling operation",
        0,
    )
    observation = SubmitPatchReview(session=session).execute(
        SubmitPatchReviewArgs(
            thoughts="",
            approved=True,
            report="Adjacent wrapper inspected; no blocking omission established.",
            evidence_ids=[evidence.id],
            surface_checks=[
                {
                    "surface": "document_reference sibling operation",
                    "disposition": "advisory",
                    "rationale": (
                        "The wrapper has a similar operation, but evidence does "
                        "not show that the requested behavior is exposed here."
                    ),
                    "evidence_ids": [evidence.id],
                }
            ],
        )
    )

    result = PatchReviewResult.model_validate(
        observation.properties["patch_review"]
    )
    assert result.approved is True
    assert result.surface_checks[0].disposition == "advisory"


def test_patch_review_required_surface_needs_counterexample():
    try:
        PatchReviewResult.model_validate({
            "approved": False,
            "report": "Missing wrapper.",
            "evidence_ids": ["SEARCH-1"],
            "surface_checks": [{
                "surface": "wrapper",
                "disposition": "required",
                "rationale": "The wrapper forwards the sibling operation.",
                "evidence_ids": ["SEARCH-1"],
            }],
        })
    except ValueError as exc:
        assert "counterexample" in str(exc)
    else:
        raise AssertionError("required surface without counterexample was accepted")


def test_patch_review_validates_surface_evidence_ids():
    session = DeveloperSession()
    evidence = session.record("grep -RInF -- 'wrapper' include", "", 1)
    observation = SubmitPatchReview(session=session).execute(
        SubmitPatchReviewArgs(
            thoughts="",
            approved=True,
            evidence_ids=[evidence.id],
            surface_checks=[
                {
                    "surface": "wrapper",
                    "disposition": "not_required",
                    "rationale": "A direct API distinction was inspected.",
                    "evidence_ids": ["SEARCH-999"],
                }
            ],
        )
    )

    assert observation.expect_correction
    assert "SEARCH-999" in observation.message


def test_patch_review_prompt_rejects_lower_level_extraction_as_facade_coverage():
    source = __import__("inspect").getsource(PatchSurfaceReviewer)

    assert "same public abstraction and invocation level" in source
    assert "lower-level receiver" in source
