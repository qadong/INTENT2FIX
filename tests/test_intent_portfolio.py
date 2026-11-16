import pytest

from INTENT2FIX.intent_dialogue import IntentDialogue, RepairIntentContract
from INTENT2FIX.intent_portfolio import (
    BranchComplianceDecision,
    CandidateAssessment,
    CandidateAuditDecision,
    CrossExecutionResult,
    DimensionOutcome,
    HypothesisMergeAdjudication,
    IntentHypothesis,
    IntentHypothesisPool,
    IntentObligation,
    IntentPortfolio,
    PortfolioCandidate,
    PortfolioCoverageAudit,
    ProbeExecutionResult,
    RepairProbeEvidence,
    MergeBoundaryAlternative,
    MergeBoundaryDecision,
    UnresolvedDimension,
    anonymize_baseline_execution,
    anonymize_candidates,
    build_branch_compliance_prompt,
    build_candidate_audit_prompt,
    build_candidate_repair_contract,
    build_hypothesis_merge_adjudication_prompt,
    build_hypothesis_prompt,
    build_hypothesis_context,
    build_portfolio_coverage_audit_prompt,
    build_portfolio_revision_prompt,
    deduplicate_candidates,
    render_hypothesis_for_repair,
    resolve_audit_selection,
    _scout_policy_evidence,
)
from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame


def _candidate(hypothesis_id: str, patch: str) -> PortfolioCandidate:
    return PortfolioCandidate.create(
        hypothesis_id=hypothesis_id,
        branch_assumption=f"behavior for {hypothesis_id}",
        patch=patch,
        kept_paths=["src/value.cc"],
        excluded_paths={"tests/repro.cc": "test_or_fixture"},
        finish_succeeded=True,
        basic_tests_passed=False,
        cross_execution=CrossExecutionResult(
            status="failed", exit_code=1, patch_applied=True
        ),
    )


def _hard_fix() -> IntentObligation:
    return IntentObligation(
        id="FIX-1",
        kind="fix",
        statement="return the requested value",
        authority="issue_explicit",
        strength="hard",
    )


def _hard_coverage(*ids: str) -> dict[str, str]:
    return {item: f"Concrete repair action satisfying {item}" for item in ids}


def test_merge_adjudication_prompt_is_bounded_and_recall_preserving():
    prompt = build_hypothesis_merge_adjudication_prompt(
        issue="join should use ADL",
        issue_evidence={"claims": ["ADL"]},
        repair_intent_bundle={
            "repository_facts": ["member and ADL helpers both exist"],
            "unrelated_large_trace": "must not be included",
        },
        repair_intent_contract={
            "target_behavior": ["ADL participates"],
            "open_decisions": ["member-first versus ADL-first"],
            "candidate_boundaries": ["both providers coexist"],
        },
    )

    assert "member-first versus ADL-first" in prompt
    assert "both providers coexist" in prompt
    assert "inconclusive_keep" in prompt
    assert "fallback coverage separate from coexistence precedence" in prompt
    assert "ordinary unqualified" in prompt
    assert "member-presence pre-check" in prompt
    assert "collapses to the fallback policy" in prompt
    assert "must not be included" not in prompt


def test_hypothesis_prompt_requires_separate_observable_alternatives():
    adjudication = HypothesisMergeAdjudication(
        decisions=[
            MergeBoundaryDecision(
                id="MERGE-1",
                question="Which provider wins?",
                verdict="keep_separate",
                alternatives=[
                    MergeBoundaryAlternative(
                        label="member-first",
                        selection_rule="member wins",
                        contract_viability="plausible",
                        viability_reason="It satisfies the common target.",
                    ),
                    MergeBoundaryAlternative(
                        label="ADL-first",
                        selection_rule="ADL wins",
                        contract_viability="plausible",
                        viability_reason="It satisfies the common target.",
                    ),
                ],
                boundary_scenarios=["member and ADL coexist"],
                reason="The public iterator differs.",
            )
        ]
    )

    prompt = build_hypothesis_prompt(
        issue="join should use ADL",
        issue_evidence={},
        repair_intent_bundle={},
        repair_intent_contract={},
        scout_summary="",
        merge_adjudication=adjudication.model_dump(),
    )

    assert '"verdict": "keep_separate"' in prompt
    assert '"label": "ADL-first"' in prompt
    assert "retain separate hypotheses" in prompt


def test_policy_pool_prompt_crosses_supported_mechanism_and_scope_options():
    prompt = build_hypothesis_prompt(
        issue="Make the generic predicate return a direct bool.",
        issue_evidence={},
        repair_intent_bundle={
            "candidate_boundaries": [
                "The generic and completely enumerated named predicates share forwarding."
            ]
        },
        repair_intent_contract={},
        scout_summary="",
        expanded_pool=True,
        policy_prompt=True,
    )

    normalized = " ".join(prompt.split())
    assert "pair it with each compatible causal mechanism" in normalized
    assert "family-wide or alternate-entry scope only to one mechanism" in normalized
    assert "completely enumerates that scope" in normalized
    assert 'do not equate "does not call the fault site"' in normalized
    assert "retain a complete broader repair-scope hypothesis" in normalized
    assert "an absent direct call is not enough" in normalized.lower()


def test_policy_handoff_separates_cited_causal_facts_from_open_graph_leads():
    bundle = {
        "cited_repository_evidence": [{"id": "SEARCH-4", "observation": "return consume(state);"}],
        "intent_reports": [{
            "focus_intent_ids": ["INTENT-1"], "status": "partial",
            "answer": "The public entry forwards state to its consumer (SEARCH-4).",
        }],
        "repository_facts": [
            "The entry forwards state (SEARCH-4).",
            "UNCITED_SCOUT_GUESS about a sibling.",
        ],
        "investigation_paths": [
            {"question": "Which consumer reads state?", "finding": "Confirmed consumer",
             "status": "supported", "source_evidence_id": "SEARCH-4", "source_quote": "return consume(state);"},
            {"question": "Does a sibling share state?", "finding": "Generic graph placeholder",
             "status": "unresolved", "missing_evidence": "Read the sibling source."},
        ],
        "repair_surface_leads": [
            {"id": "SURFACE-1", "statement": "Forwarding entry", "evidence_ids": ["SEARCH-4"]},
            {"id": "SURFACE-2", "statement": "Uncited sibling", "evidence_ids": []},
        ],
        "open_decisions": ["Whether the sibling shares the same rule remains unknown."],
    }
    evidence = _scout_policy_evidence(bundle)
    assert [lead["id"] for lead in evidence["repair_surface_leads"]] == ["SURFACE-1"]
    assert len(evidence["investigation_paths"]) == 1
    assert evidence["open_investigation_questions"][0]["question"] == "Does a sibling share state?"

    generation = build_hypothesis_prompt(
        issue="Preserve state.", issue_evidence={}, repair_intent_bundle=bundle,
        repair_intent_contract={}, scout_summary="UNCITED_SCOUT_GUESS",
        expanded_pool=True, policy_prompt=True,
    )
    pool = IntentHypothesisPool(
        obligations=[_hard_fix()],
        hypotheses=[IntentHypothesis(
            id="H0", title="forward state", branch_assumption="none",
            covered_obligation_ids=["FIX-1"], hard_obligation_coverage=_hard_coverage("FIX-1"),
        )],
    )
    audit = build_portfolio_coverage_audit_prompt(
        issue="Preserve state.", issue_evidence={}, repair_intent_bundle=bundle,
        repair_intent_contract={}, scout_summary="UNCITED_SCOUT_GUESS",
        merge_adjudication={}, portfolio=pool,
    )
    for prompt in (generation, audit):
        assert "The entry forwards state (SEARCH-4)." in prompt
        assert "The public entry forwards state to its consumer" in prompt
        assert "Does a sibling share state?" in prompt
        assert "Generic graph placeholder" not in prompt
        assert "UNCITED_SCOUT_GUESS" not in prompt
        assert '"id": "SURFACE-1"' in prompt
        assert '"id": "SURFACE-2"' not in prompt


def test_portfolio_coverage_audit_checks_recall_without_selecting_intent():
    portfolio = IntentPortfolio(
        obligations=[_hard_fix()],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="bounded repair",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
            )
        ],
    )

    prompt = build_portfolio_coverage_audit_prompt(
        issue="Return the requested value.",
        issue_evidence={"claims": ["requested value"]},
        repair_intent_bundle={},
        repair_intent_contract={},
        scout_summary="SURFACE-1 is the public entry point.",
        merge_adjudication={"decisions": []},
        portfolio=portfolio,
    )

    assert "pre-Repair coverage and consistency Audit" in prompt
    assert "Do not select which plausible hypothesis" in prompt
    assert "Every supplied SURFACE-N lead" in prompt
    assert "upstream selection points before downstream symptoms" in prompt
    assert "exception behavior" in prompt
    assert "Issue-author implementation suggestion" in prompt
    assert "golden patches" in prompt


def test_policy_pool_keeps_grounded_uncertainty_as_conditional_branch():
    bundle = {
        "candidate_boundaries": [{
            "anchor_operation": "second public entry",
            "candidate_public_forms": ["first entry", "second entry"],
            "evidence_ids": ["SEARCH-1"],
            "rationale": "Both handle the same representation; the second body is not fully checked.",
        }],
        "repair_surface_leads": [{
            "id": "SURFACE-1",
            "statement": "Second entry handles the same representation",
            "evidence_ids": ["SEARCH-1"],
        }],
        "cited_repository_evidence": [{
            "id": "SEARCH-1",
            "observation": "Both entries receive the Issue-relevant representation.",
        }],
    }
    generation = build_hypothesis_prompt(
        issue="Repair the first public entry.", issue_evidence={},
        repair_intent_bundle=bundle, repair_intent_contract={}, scout_summary="",
        expanded_pool=True, policy_prompt=True,
    )
    assert "Uncertainty about which Issue-related cause or scope is correct" in generation
    assert "branch, not a reason to erase every alternative" in generation
    assert "keep a local branch and a conditional shared-scope branch" in generation
    assert "Defer a material lead out of every hypothesis only when" in generation
    assert "premise is unverified." in generation
    assert "matching the exact failing expression is not the causal test" in generation
    assert "conditional shared-scope lead, not proof" in generation
    assert "second public entry" in generation

    pool = IntentHypothesisPool(
        obligations=[_hard_fix()],
        hypotheses=[IntentHypothesis(
            id="H0", title="first entry repair", branch_assumption="first entry only",
            covered_obligation_ids=["FIX-1"], hard_obligation_coverage=_hard_coverage("FIX-1"),
        )],
    )
    audit = build_portfolio_coverage_audit_prompt(
        issue="Repair the first public entry.", issue_evidence={},
        repair_intent_bundle=bundle, repair_intent_contract={}, scout_summary="",
        merge_adjudication={}, portfolio=pool,
    )
    assert "Do not demand proof that its premise is true before ranking it for Repair" in audit
    assert "signature-to-output type flow" in audit
    assert "`deferred` is not a substitute" in audit


def test_portfolio_revision_is_bounded_to_audited_evidence():
    portfolio = IntentPortfolio(
        obligations=[_hard_fix()],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="bounded repair",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
            )
        ],
    )
    audit = PortfolioCoverageAudit(
        decision="REVISE",
        causal_coverage_gaps=["H0 omits evidenced SURFACE-2."],
        reason="A required public entry remains unchanged.",
    )

    prompt = build_portfolio_revision_prompt(
        issue="Return the requested value.",
        issue_evidence={},
        repair_intent_bundle={},
        repair_intent_contract={},
        scout_summary="SURFACE-2 is on the same public call path.",
        merge_adjudication={"decisions": []},
        portfolio=portfolio,
        audit=audit,
    )

    assert "one bounded revision" in prompt
    assert "review feedback rather than new" in prompt
    assert "product evidence" in prompt
    assert "H0 omits evidenced SURFACE-2" in prompt
    assert "Return one complete corrected IntentPortfolio" in prompt


def test_portfolio_coverage_audit_cannot_accept_reported_gaps():
    audit = PortfolioCoverageAudit(
        decision="ACCEPT",
        repair_surface_gaps=["H0 omits SURFACE-2."],
        reason="The surface is missing.",
    )

    assert audit.decision == "REVISE"


def test_candidate_deduplication_uses_submitted_patch_not_hypothesis():
    patch = "diff --git a/src/value.cc b/src/value.cc\n"
    candidates = [_candidate("H0", patch), _candidate("H1", patch)]

    assert deduplicate_candidates(candidates) == [candidates[0]]


def test_candidate_anonymization_is_independent_of_hypothesis_order():
    first = _candidate("H0", "patch-b")
    second = _candidate("H1", "patch-a")

    payload_a, mapping_a = anonymize_candidates([first, second])
    payload_b, mapping_b = anonymize_candidates([second, first])

    assert payload_a == payload_b
    assert [item.patch_sha256 for item in mapping_a.values()] == [
        item.patch_sha256 for item in mapping_b.values()
    ]
    assert all("hypothesis_id" not in item for item in payload_a)


def test_candidate_anonymization_exposes_repair_evidence_without_branch_ids():
    candidate = PortfolioCandidate.create(
        hypothesis_id="H1",
        branch_assumption="free lookup wins",
        discriminator="public_call(overlap)",
        predicted_outcome="returns free result",
        patch="patch-h1",
        kept_paths=["src/value.cc"],
        excluded_paths={},
        finish_succeeded=True,
        basic_tests_passed=False,
        repair_probes=[
            RepairProbeEvidence(
                probe_id="REPAIR-H1-1",
                source_hypothesis_id="H1",
                kind="discriminator",
                command="./probe",
                observed_status="passed",
                observed_exit_code=0,
                production_patch_sha256="abc",
            )
        ],
        cross_execution=CrossExecutionResult(
            status="unavailable",
            patch_applied=True,
            probe_results=[
                ProbeExecutionResult(
                    check_id="REPAIR-H1-1",
                    authority="hypothesis",
                    origin="repair",
                    source_hypothesis_id="H1",
                    status="passed",
                )
            ],
        ),
    )

    payload, _ = anonymize_candidates([candidate])
    serialized = str(payload)

    assert payload[0]["assigned_discriminator"] == "public_call(overlap)"
    assert payload[0]["repair_verification"][0][
        "source_hypothesis_id"
    ] == "CANDIDATE-1"
    assert "H1" not in serialized


def test_candidate_anonymization_keeps_same_hypothesis_variants_distinct():
    candidates = []
    for variant_id, patch in (
        ("initial", "patch-initial"),
        ("continuation_1", "patch-corrected"),
    ):
        source_id = f"H1@{variant_id}"
        candidates.append(
            PortfolioCandidate.create(
                hypothesis_id="H1",
                variant_id=variant_id,
                branch_assumption="free lookup wins",
                patch=patch,
                kept_paths=["src/value.cc"],
                excluded_paths={},
                finish_succeeded=True,
                basic_tests_passed=False,
                repair_probes=[
                    RepairProbeEvidence(
                        probe_id=f"REPAIR-H1-1@{variant_id}",
                        source_hypothesis_id=source_id,
                        command="./probe",
                        observed_status="passed",
                        production_patch_sha256="abc",
                    )
                ],
                cross_execution=CrossExecutionResult(
                    status="passed",
                    patch_applied=True,
                    probe_results=[
                        ProbeExecutionResult(
                            check_id=f"REPAIR-H1-1@{variant_id}",
                            authority="hypothesis",
                            origin="repair",
                            source_hypothesis_id=source_id,
                            status="passed",
                        )
                    ],
                ),
            )
        )

    anonymous, _ = anonymize_candidates(candidates)

    assert len(anonymous) == 2
    for item in anonymous:
        assert item["repair_verification"][0]["source_hypothesis_id"] == (
            item["candidate_id"]
        )
        assert item["cross_execution"]["probe_results"][0][
            "source_hypothesis_id"
        ] == item["candidate_id"]
    assert "H1" not in str(anonymous)


def test_baseline_and_candidate_probe_ids_share_one_anonymous_mapping():
    raw_probe = ProbeExecutionResult(
        check_id="REPAIR-H1-1@initial",
        authority="hypothesis",
        origin="repair",
        source_hypothesis_id="H1@initial",
        command=(
            "./probe && echo INTENT2FIX_DISCRIMINATOR_H1_PASS"
        ),
        status="passed",
        output="INTENT2FIX_DISCRIMINATOR_H1_PASS",
    )
    candidate = PortfolioCandidate.create(
        hypothesis_id="H1",
        variant_id="initial",
        branch_assumption="free lookup wins",
        patch="patch-h1",
        kept_paths=["src/value.cc"],
        excluded_paths={},
        finish_succeeded=True,
        basic_tests_passed=False,
        cross_execution=CrossExecutionResult(
            status="passed",
            patch_applied=True,
            probe_results=[raw_probe],
        ),
    )
    baseline = CrossExecutionResult(
        status="failed",
        probe_results=[raw_probe.model_copy(update={"status": "failed"})],
    )

    anonymous_candidates, _ = anonymize_candidates([candidate])
    anonymous_baseline = anonymize_baseline_execution(baseline, [candidate])
    candidate_probe = anonymous_candidates[0]["cross_execution"][
        "probe_results"
    ][0]
    baseline_probe = anonymous_baseline.probe_results[0].model_dump()

    assert baseline_probe["check_id"] == candidate_probe["check_id"]
    assert baseline_probe["source_hypothesis_id"] == (
        candidate_probe["source_hypothesis_id"]
    )
    assert "H1" not in str(anonymous_baseline.model_dump())
    assert "H1" not in str(anonymous_candidates)


def test_candidate_anonymization_exposes_complete_contract_without_branch_id():
    portfolio = IntentPortfolio(
        obligations=[_hard_fix()],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="repair the selection point",
                branch_assumption="the upstream route must win",
                root_cause="The upstream selector chooses the wrong route.",
                repair_scope=["selector"],
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
            )
        ],
    )
    candidate = PortfolioCandidate.create(
        hypothesis_id="H0",
        branch_assumption="the upstream route must win",
        repair_contract=build_candidate_repair_contract(
            portfolio, portfolio.hypotheses[0]
        ),
        patch="patch-a",
        kept_paths=["src/value.cc"],
        excluded_paths={},
        finish_succeeded=True,
        basic_tests_passed=True,
        cross_execution=CrossExecutionResult(
            status="passed", patch_applied=True
        ),
    )

    payload, _ = anonymize_candidates([candidate])

    assert payload[0]["repair_contract"]["branch"]["root_cause"] == (
        "The upstream selector chooses the wrong route."
    )
    assert payload[0]["repair_contract"]["branch"]["repair_scope"] == [
        "selector"
    ]
    assert "H0" not in str(payload)


def test_candidate_contract_retains_non_branch_scope_constraints_without_ids():
    portfolio = IntentPortfolio(
        obligations=[_hard_fix()],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="bounded repair",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
            )
        ],
        unresolved_dimensions=[
            UnresolvedDimension(
                id="DIM-2",
                statement="A neighboring storage rewrite is unrelated.",
                impact="low",
                disposition="soft_constraint",
                reason="No issue or repository evidence connects it.",
            )
        ],
    )

    context = build_hypothesis_context(portfolio, portfolio.hypotheses[0])
    contract = build_candidate_repair_contract(
        portfolio, portfolio.hypotheses[0]
    )

    assert context["non_goals_or_soft_constraints"] == (
        contract["non_goals_or_soft_constraints"]
    )
    assert "neighboring storage rewrite" in str(contract)
    assert "DIM-2" not in str(contract)
    assert "H0" not in str(contract)


def test_portfolio_accepts_single_unambiguous_hypothesis():
    portfolio = IntentPortfolio(
        obligations=[_hard_fix()],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="literal issue",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
            )
        ],
    )

    assert len(portfolio.hypotheses) == 1


def test_portfolio_allows_shared_discriminator_without_redundant_local_ones():
    portfolio = IntentPortfolio(
        obligations=[_hard_fix()],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="first",
                branch_assumption="first behavior",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
            ),
            IntentHypothesis(
                id="H1",
                title="second",
                branch_assumption="second behavior",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
            ),
        ],
        unresolved_dimensions=[
            UnresolvedDimension(
                id="DIM-1",
                statement="which behavior is required",
                impact="high",
                disposition="branched",
                hypothesis_ids=["H0", "H1"],
                shared_discriminator="one shared public scenario",
                outcomes=[
                    DimensionOutcome(
                        hypothesis_id="H0", predicted_outcome="returns A"
                    ),
                    DimensionOutcome(
                        hypothesis_id="H1", predicted_outcome="returns B"
                    ),
                ],
                reason="The observable result differs.",
            )
        ],
    )

    assert all(not item.discriminator for item in portfolio.hypotheses)


def test_portfolio_allows_branch_specific_discriminators():
    portfolio = IntentPortfolio(
        obligations=[_hard_fix()],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="first",
                branch_assumption="first behavior",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
                discriminator="public_call(overlap_input)",
                predicted_outcome="returns A",
            ),
            IntentHypothesis(
                id="H1",
                title="second",
                branch_assumption="second behavior",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
                discriminator="compile(sibling_overload)",
                predicted_outcome="compiles",
            ),
        ],
        unresolved_dimensions=[
            UnresolvedDimension(
                id="DIM-1",
                statement="which behavior is required",
                impact="high",
                disposition="branched",
                hypothesis_ids=["H0", "H1"],
                shared_discriminator="one shared public scenario",
                outcomes=[
                    DimensionOutcome(
                        hypothesis_id="H0", predicted_outcome="returns A"
                    ),
                    DimensionOutcome(
                        hypothesis_id="H1", predicted_outcome="returns B"
                    ),
                ],
                reason="The observable result differs.",
            )
        ],
    )

    assert portfolio.hypotheses[1].discriminator == "compile(sibling_overload)"


def test_invalid_branched_dimension_is_salvaged_as_deferred():
    portfolio = IntentPortfolio(
            obligations=[_hard_fix()],
            hypotheses=[
                IntentHypothesis(
                    id="H0",
                    title="old wins",
                    branch_assumption="old path wins",
                    covered_obligation_ids=["FIX-1"],
                    hard_obligation_coverage=_hard_coverage("FIX-1"),
                    discriminator="old-only fixture",
                    predicted_outcome="returns old",
                ),
                IntentHypothesis(
                    id="H1",
                    title="new wins",
                    branch_assumption="new path wins",
                    covered_obligation_ids=["FIX-1"],
                    hard_obligation_coverage=_hard_coverage("FIX-1"),
                    discriminator="new-only fixture",
                    predicted_outcome="returns new",
                ),
            ],
            unresolved_dimensions=[
                UnresolvedDimension(
                    id="DIM-1",
                    statement="which applicable provider wins",
                    impact="high",
                    disposition="branched",
                    hypothesis_ids=["H0", "H1"],
                    outcomes=[
                        DimensionOutcome(
                            hypothesis_id="H0",
                            predicted_outcome="returns old",
                        ),
                        DimensionOutcome(
                            hypothesis_id="H1",
                            predicted_outcome="returns new",
                        ),
                    ],
                    reason="The outcome changes public behavior.",
                )
            ],
        )

    assert portfolio.unresolved_dimensions[0].disposition == "deferred"


def test_invalid_branched_outcome_mapping_is_salvaged_as_deferred():
    common = {
        "obligations": [_hard_fix()],
        "hypotheses": [
            IntentHypothesis(
                id="H0",
                title="first",
                branch_assumption="first path wins",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
                discriminator="shared conflict fixture",
                predicted_outcome="returns first",
            ),
            IntentHypothesis(
                id="H1",
                title="second",
                branch_assumption="second path wins",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
                discriminator="shared conflict fixture",
                predicted_outcome="returns second",
            ),
        ],
    }

    missing_outcome = IntentPortfolio(
            **common,
            unresolved_dimensions=[
                UnresolvedDimension(
                    id="DIM-1",
                    statement="winner",
                    impact="high",
                    disposition="branched",
                    hypothesis_ids=["H0", "H1"],
                    shared_discriminator="both providers return distinct values",
                    outcomes=[
                        DimensionOutcome(
                            hypothesis_id="H0",
                            predicted_outcome="returns first",
                        )
                    ],
                    reason="Observable winner is unresolved.",
                )
            ],
        )

    equivalent_outcomes = IntentPortfolio(
            **common,
            unresolved_dimensions=[
                UnresolvedDimension(
                    id="DIM-1",
                    statement="winner",
                    impact="high",
                    disposition="branched",
                    hypothesis_ids=["H0", "H1"],
                    shared_discriminator="both providers return distinct values",
                    outcomes=[
                        DimensionOutcome(
                            hypothesis_id="H0",
                            predicted_outcome="same result",
                        ),
                        DimensionOutcome(
                            hypothesis_id="H1",
                            predicted_outcome=" SAME   RESULT ",
                        ),
                    ],
                    reason="Observable winner is unresolved.",
                )
            ],
        )

    assert missing_outcome.unresolved_dimensions[0].disposition == "deferred"
    assert equivalent_outcomes.unresolved_dimensions[0].disposition == "deferred"


def test_portfolio_salvages_missing_hard_closure_mapping():
    portfolio = IntentPortfolio(
            obligations=[
                _hard_fix(),
                IntentObligation(
                    id="CLOSURE-1",
                    kind="closure",
                    statement="update the public overload family",
                    authority="repository_structure",
                    strength="hard",
                ),
            ],
            hypotheses=[
                IntentHypothesis(
                    id="H0",
                    title="incomplete",
                    branch_assumption="none",
                    covered_obligation_ids=["FIX-1"],
                    hard_obligation_coverage=_hard_coverage("FIX-1"),
                )
            ],
        )

    hypothesis = portfolio.hypotheses[0]
    assert "CLOSURE-1" in hypothesis.covered_obligation_ids
    assert "CLOSURE-1" in hypothesis.hard_obligation_coverage


def test_portfolio_salvages_missing_hard_obligation_claim():
    portfolio = IntentPortfolio(
            obligations=[_hard_fix()],
            hypotheses=[
                IntentHypothesis(
                    id="H0",
                    title="unsupported claim",
                    branch_assumption="none",
                    covered_obligation_ids=["FIX-1"],
                )
            ],
        )

    assert portfolio.hypotheses[0].hard_obligation_coverage["FIX-1"]


def test_portfolio_normalizes_recoverable_obligation_id_drift():
    portfolio = IntentPortfolio.model_validate(
        {
            "obligations": [
                {
                    "id": "PRES-1",
                    "kind": "preserve",
                    "statement": "Keep the public fallback working.",
                    "authority": "public_contract",
                    "strength": "hard",
                }
            ],
            "hypotheses": [
                {
                    "id": "h0",
                    "title": "repair candidate",
                    "branch_assumption": "none",
                    "covered_obligation_ids": ["PRES-1"],
                    "hard_obligation_coverage": {
                        "PRES-1": "Exercise the fallback after the change."
                    },
                }
            ],
        }
    )

    assert portfolio.obligations[0].id == "PRESERVE-1"
    assert portfolio.hypotheses[0].covered_obligation_ids == ["PRESERVE-1"]
    assert set(portfolio.hypotheses[0].hard_obligation_coverage) == {
        "PRESERVE-1"
    }


def test_legacy_behavior_is_deterministically_demoted_from_hard_preserve():
    obligation = IntentObligation(
        id="PRESERVE-1",
        kind="preserve",
        statement="keep sibling error text byte-for-byte",
        authority="legacy_behavior",
        strength="hard",
    )

    assert obligation.strength == "soft"


def test_hypothesis_prompts_cover_shared_diagnostic_argument_roles():
    prompt = build_hypothesis_prompt(
        issue="Explain which public argument is invalid.",
        issue_evidence={},
        repair_intent_bundle={},
        repair_intent_contract={},
        scout_summary="",
    )
    assert "Diagnostic argument-role closure" in prompt
    assert "every distinct argument or operand role" in prompt
    assert "receiver-specific special case" in prompt


def test_portfolio_scout_can_skip_property_materialization():
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("property extractor should not run")

    dialogue = IntentDialogue(
        extractor=fail_if_called,
        investigator=object(),
        enable_property_materialization=False,
    )

    assert dialogue._extract_intent_properties(
        "issue",
        IssueEvidenceFrame(),
        RepairIntentContract(),
        [],
        [],
    ) is None


def test_branch_prompt_exposes_competing_prediction_and_requires_discriminator():
    portfolio = IntentPortfolio(
        obligations=[_hard_fix()],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="old wins",
                branch_assumption="old path wins",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
                discriminator="public_call(overlap)",
                predicted_outcome="returns old",
            ),
            IntentHypothesis(
                id="H1",
                title="new wins",
                branch_assumption="new path wins",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
                discriminator="compile(new_path)",
                predicted_outcome="returns new",
            ),
        ],
        unresolved_dimensions=[
            UnresolvedDimension(
                id="DIM-1",
                statement="which simultaneously applicable path wins",
                impact="high",
                disposition="branched",
                hypothesis_ids=["H0", "H1"],
                shared_discriminator=(
                    "Run public_call(overlap) with old and new paths both "
                    "available and returning distinct values."
                ),
                outcomes=[
                    DimensionOutcome(
                        hypothesis_id="H0",
                        predicted_outcome="returns old",
                    ),
                    DimensionOutcome(
                        hypothesis_id="H1",
                        predicted_outcome="returns new",
                    ),
                ],
                reason="The public result changes.",
            )
        ],
    )

    context = build_hypothesis_context(portfolio, portfolio.hypotheses[1])
    prompt = render_hypothesis_for_repair(portfolio, portfolio.hypotheses[1])

    assert context["competing_predictions"][0]["predicted_outcome"] == (
        "returns old"
    )
    assert context["assigned_dimensions"][0]["assigned_outcome"] == (
        "returns new"
    )
    assert "old and new paths both available" in (
        context["assigned_dimensions"][0]["shared_discriminator"]
    )
    assert "complete bounded repair contract" in prompt
    assert "assigned_dimensions outcome is a hard proposal constraint" in prompt
    assert "shared_discriminator is diagnostic evidence" in prompt
    assert "Run it at most once" in prompt
    assert "INTENT2FIX_DISCRIMINATOR_H1_PASS" in prompt
    assert "does not excuse changing the assigned outcome" in prompt
    assert "call Finish so Branch Compliance and the Selector can assess" in prompt
    assert "Tests remain soft evidence" in prompt
    assert "generated, amalgamated, vendored, or mirrored" in prompt
    assert "search hints, not admission conditions" not in prompt


def test_branch_compliance_prompt_checks_only_assigned_hypothesis():
    probe = RepairProbeEvidence(
        probe_id="REPAIR-H1-1",
        source_hypothesis_id="H1",
        authority="hypothesis",
        kind="discriminator",
        description="Both providers are present.",
        expected_outcome="new provider wins",
        command="./probe && echo INTENT2FIX_DISCRIMINATOR_H1_PASS",
        observed_status="passed",
        observed_exit_code=0,
        observed_output="INTENT2FIX_DISCRIMINATOR_H1_PASS",
        production_patch_sha256="a" * 64,
    )

    prompt = build_branch_compliance_prompt(
        hypothesis_context={
            "branch": {"id": "H1"},
            "assigned_dimensions": [
                {"assigned_outcome": "new provider wins"}
            ],
        },
        patch="diff --git a/src/a.cc b/src/a.cc\n",
        repair_probes=[probe],
        regression_evidence={"passed": False},
    )

    assert "normalized hypothesis context is the complete contract" in prompt
    assert "true intent of the original" in prompt
    assert "later candidate Selector" in prompt
    assert "competing branch" in prompt
    assert "noncompliant" in prompt
    assert "Legacy/basic tests are supporting evidence" in prompt
    assert "actionable_correction=true" in prompt
    assert "INTENT2FIX_DISCRIMINATOR_H1_PASS" in prompt


def test_branch_compliance_actionable_correction_requires_concrete_clause():
    with pytest.raises(ValueError, match="violated clause IDs"):
        BranchComplianceDecision(
            status="noncompliant",
            observed_behavior="The competing path wins.",
            reason="The patch differs from the assigned hypothesis.",
            actionable_correction=True,
        )


def test_portfolio_prompts_use_generic_evidence_gated_closure_patterns():
    prompt_args = {
        "issue": "The public operation returns the requested value.",
        "issue_evidence": {},
        "repair_intent_bundle": {},
        "repair_intent_contract": {},
        "scout_summary": "",
    }
    generation_prompt = build_hypothesis_prompt(**prompt_args)
    combined = generation_prompt

    for pattern in (
        "Shared-cause equivalence",
        "Dependency propagation",
        "Synchronized representations",
        "Compatibility boundary",
        "Conjunctive causality",
        "Competing resolution semantics",
        "Selection-chain closure",
        "Artifact-route closure",
        "Equivalent access-form closure",
        "Baseline-value fidelity",
        "Existing-contract promotion",
        "Compositional behavior",
        "first library-owned ingress or seed operation",
        "Issue-authorized operation family",
        "Mandatory family-ingress intersection audit",
        "Mandatory artifact-route audit",
        "Mandatory equivalent-access audit",
        "Do not turn a failure trace into post-fix intent",
        "Absence of an Issue-specified selection winner is not positive authority",
        "exactly one `surface_dispositions` item",
        "does not need to appear under an Expected",
        "Separate implementation shape from stated behavioral breadth",
        "two independent facts coincide",
        'Do not treat "and siblings"',
        "Reconcile conflicting evidence explicitly",
        "counterfactual necessity",
    ):
        assert pattern in combined

    for case_shaped_rule in (
        "Result<T>",
        "range-for",
        "prefixes, signs",
        "decoding/parsing setup",
        "distinctly named internal helper",
    ):
        assert case_shaped_rule not in combined


def test_candidate_checks_require_evidenced_artifact_route_provenance():
    portfolio = IntentPortfolio(
        obligations=[_hard_fix()],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="bounded repair",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage=_hard_coverage("FIX-1"),
            )
        ],
    )
    candidate = _candidate("H0", "patch-a")
    anonymous, _ = anonymize_candidates([candidate])
    audit_prompt = build_candidate_audit_prompt(
        issue="Fix the public operation.",
        portfolio=portfolio,
        anonymous_candidates=anonymous,
        baseline_execution=CrossExecutionResult(status="unavailable"),
    )

    assert "build/import route exercised" in audit_prompt
    assert "similar filename or common project convention" in audit_prompt
    assert "unary provenance label" in audit_prompt
    assert "prefer a compliant candidate merely for fidelity" in audit_prompt
    normalized_prompt = " ".join(audit_prompt.split())
    assert "never reject a noncompliant or inconclusive candidate" in (
        normalized_prompt
    )


def test_audit_cannot_select_candidate_with_unrelated_production_scope():
    candidate = _candidate("H0", "patch-a")
    _, mapping = anonymize_candidates([candidate])
    decision = CandidateAuditDecision(
        selection="SELECTED",
        selected_candidate_id="CANDIDATE-1",
        assessments=[
            CandidateAssessment(
                candidate_id="CANDIDATE-1",
                target_coverage="complete",
                preservation="supported",
                production_scope="unrelated",
                legacy_test_interpretation="inconclusive",
            )
        ],
        decision_reason="The diff is unrelated.",
    )

    assert resolve_audit_selection(decision, mapping) is None


def test_audit_cannot_select_execution_contradicted_candidate():
    candidate = _candidate("H0", "patch-a")
    _, mapping = anonymize_candidates([candidate])
    decision = CandidateAuditDecision(
        selection="SELECTED",
        selected_candidate_id="CANDIDATE-1",
        assessments=[
            CandidateAssessment(
                candidate_id="CANDIDATE-1",
                target_coverage="complete",
                preservation="supported",
                production_scope="focused",
                legacy_test_interpretation="inconclusive",
                execution_support="contradicted",
            )
        ],
        decision_reason="The full-context target still fails.",
    )

    assert resolve_audit_selection(decision, mapping) is None


def test_final_audit_does_not_reapply_branch_compliance_gate():
    candidate = _candidate("H0", "patch-a").model_copy(
        update={"branch_compliance": "noncompliant"}
    )
    _, mapping = anonymize_candidates([candidate])
    decision = CandidateAuditDecision(
        selection="SELECTED",
        selected_candidate_id="CANDIDATE-1",
        assessments=[
            CandidateAssessment(
                candidate_id="CANDIDATE-1",
                target_coverage="complete",
                preservation="supported",
                production_scope="focused",
                legacy_test_interpretation="supports_candidate",
                execution_support="supported",
            )
        ],
        decision_reason="The target probe passes.",
    )

    assert resolve_audit_selection(decision, mapping) is candidate


def _correction_fixture():
    import hashlib
    from INTENT2FIX.intent_portfolio import BranchComplianceDecision
    patch = "a frozen production patch\n"
    context = {
        "hard_obligations": [{"id": "FIX-1", "statement": "Reject marker N.", "strength": "hard"}],
        "soft_evidence": [{"id": "PRESERVE-3", "statement": "Legacy N silently returns.", "strength": "soft"}],
        "assigned_dimensions": [{"dimension_id": "DIM-1", "assigned_outcome": "Reject marker N."}],
        "branch": {"expected_behavior": ["Reject marker N."]},
    }
    decision = BranchComplianceDecision(
        status="noncompliant", observed_behavior="N silently returns.", reason="An explicit case fails.",
        actionable_correction=True, violated_clause_ids=["FIX-1"], suggested_correction="Reject N.",
        production_patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
        violations=[dict(clause_id="FIX-1", clause_quote="Reject marker N.", scenario="Input N",
                         expected_behavior="parse_error", observed_behavior="silent return", evidence_ids=["PATCH"])],
    )
    return context, patch, decision


@pytest.mark.parametrize("clause_id", ["FIX-1", "DIM-1", "BRANCH-EXPECTED-1"])
def test_current_explicit_contract_violation_can_authorize_correction(clause_id):
    from INTENT2FIX.intent_portfolio import validate_branch_correction
    context, patch, decision = _correction_fixture()
    decision.violated_clause_ids = [clause_id]
    decision.violations[0].clause_id = clause_id
    assert validate_branch_correction(decision, hypothesis_context=context, patch=patch,
                                      repair_probes=[], regression_evidence={}).actionable_correction


@pytest.mark.parametrize("fault", ["soft", "unknown", "quote", "patch", "unexecuted", "stale", "missing"])
def test_audit_cannot_authorize_unfounded_or_stale_correction(fault):
    from INTENT2FIX.intent_portfolio import validate_branch_correction, RepairProbeEvidence
    context, patch, decision = _correction_fixture()
    probe = RepairProbeEvidence(probe_id="PROBE-1", source_hypothesis_id="H0", command="probe",
                                observed_status="passed", production_patch_sha256=decision.production_patch_sha256)
    if fault in {"soft", "unknown"}:
        cid = "PRESERVE-3" if fault == "soft" else "FIX-404"
        decision.violated_clause_ids = [cid]
        decision.violations[0].clause_id = cid
    elif fault == "quote":
        decision.violations[0].clause_quote = "Preserve silent returns."
    elif fault == "patch":
        decision.production_patch_sha256 = "old-patch"
    elif fault in {"unexecuted", "stale"}:
        decision.violations[0].evidence_ids = [probe.probe_id]
        if fault == "stale":
            probe.observed_exit_code = 0
            probe.production_patch_sha256 = "old-patch"
    else:
        decision.violations = []
    checked = validate_branch_correction(decision, hypothesis_context=context, patch=patch,
                                        repair_probes=[probe], regression_evidence={})
    assert checked.status == "inconclusive"
    assert not checked.actionable_correction
    assert checked.suggested_correction == ""
