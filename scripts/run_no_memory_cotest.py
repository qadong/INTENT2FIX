import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import copy
import time
import traceback
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Type

from dotenv import load_dotenv
from litellm import RateLimitError
from pydantic import ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from INTENT2FIX.benchmark.swebench import create_repository
from INTENT2FIX.artifact_paths import results_path, submission_path
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.agent.code_agent import CodingAgent
from INTENT2FIX.agent.test_agent import TestAgent, TestEnhancementResult, RepairResult
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.loop import AgenticLoop
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.actions.verification_sync import (
    PRODUCTION_ISOLATION,
    current_probe_change_seq,
    sync_verification_files,
)
from INTENT2FIX.actions.run_intent_check import RunIntentCheck
from INTENT2FIX.actions.command_policy import verification_exit_status_rejection
from INTENT2FIX.actions.run_regression import RunRegression, RunRegressionArgs
from INTENT2FIX.actions.test_parser import LogType
from INTENT2FIX.actions.finish import Finish
from INTENT2FIX.actions.verified_finish import VerifiedFinish
from INTENT2FIX.intent_dialogue import (
    ContainerRepositoryInvestigator,
    IntentDialogue,
    IntentDialogueResult,
    RepairIntentContract,
    IntentEmbeddingUnavailableError,
    compact_reproduction_context,
)
from INTENT2FIX.intent_dialogue.developer_agent import (
    frozen_probe_command_rejection,
)
from INTENT2FIX.intent_dialogue.schema import (
    ContrastiveIntentDecision,
    InvestigationPath,
)
from INTENT2FIX.intent_guard import IntentGuard
from INTENT2FIX.completion.base import LLMResponseFormat
from INTENT2FIX.schema import MessageHistoryType
from INTENT2FIX.repo_structure import CodeParser, DirectoryAnalyzer, show_code_structure, show_cpp_repo_structure
from INTENT2FIX.memory.memory import Memory
from INTENT2FIX.scout_probe_handoff import (
    build_probe_selection_prompt,
    probe_selection_schema,
)
from INTENT2FIX.hypothesis_evidence import HypothesisEvidenceTools
from INTENT2FIX.patch_review import (
    CandidatePatch,
    PatchSurfaceReviewer,
    sanitize_candidate_patch,
)
from INTENT2FIX.patch_intent_audit import PatchIntentAuditor
from INTENT2FIX.patch_intent_audit.trajectory_leads import (
    AuditLead,
    TrajectoryLeadResult,
    extract_trajectory_leads,
    load_trajectory,
    slice_trajectory,
)
from INTENT2FIX.reproduction_loop import ReproductionLoop
from INTENT2FIX.reproduce.reproduction_agent import ReproductionAgent
from INTENT2FIX.reproduce.intent_pipeline import (
    build_intent_adjudication_prompt,
    build_issue_evidence_prompt,
    merge_issue_evidence_with_fallback,
    normalize_repair_intent_bundle,
    parse_structured_response,
    unresolved_bundle,
)
from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame, RepairIntentBundle
from INTENT2FIX.reproduce.state import is_dedicated_reproduction_path
from INTENT2FIX.completion.lite_llm import LiteLLM
from INTENT2FIX.completion.structured_compactor import (
    StructuredCompactionError,
    compact_structured_payload_once,
    is_compactable_validation_error,
    parse_json_object,
)
from INTENT2FIX.model_config import get_model_config
from INTENT2FIX.intent_portfolio import (
    BranchCorrectionRefinement,
    branch_correction_clauses,
    BranchComplianceDecision,
    CandidateAuditDecision,
    CrossExecutionResult,
    HypothesisMergeAdjudication,
    HypothesisPoolAudit,
    IntentHypothesis,
    IntentHypothesisPool,
    IntentObligation,
    IntentPortfolio,
    PortfolioCoverageAudit,
    PortfolioCandidate,
    ProbeExecutionResult,
    RepairProbeEvidence,
    anonymize_candidates,
    anonymize_baseline_execution,
    build_branch_compliance_prompt,
    validate_branch_correction,
    build_candidate_repair_contract,
    build_candidate_audit_prompt,
    build_hypothesis_context,
    build_hypothesis_merge_adjudication_prompt,
    build_hypothesis_prompt,
    build_portfolio_coverage_audit_prompt,
    build_portfolio_revision_prompt,
    deduplicate_candidates,
    render_hypothesis_for_repair,
    render_assigned_discriminator,
    render_assigned_outcome,
    resolve_audit_selection,
    select_hypothesis_pool,
)
from INTENT2FIX.utils.patch import production_patch_sha256

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOTENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_env() -> None:
    load_dotenv(DOTENV_PATH, override=False)


def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return default
    return str(v).strip()


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


def _repair_agent_system_prompt(explicit_prompt: str | None) -> str | None:
    """Resolve an opt-in Repair prompt without changing other coding agents."""
    if explicit_prompt is not None:
        return explicit_prompt
    prompt_path = _env_str("COTEST_REPAIR_SYSTEM_PROMPT_FILE", "")
    if not prompt_path:
        return None
    path = Path(prompt_path)
    if not path.is_absolute():
        raise ValueError("COTEST_REPAIR_SYSTEM_PROMPT_FILE must be absolute")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("COTEST_REPAIR_SYSTEM_PROMPT_FILE is empty")
    return prompt


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _graph_ablation_enabled() -> bool:
    return _env_bool("COTEST_ABLATE_DUAL_LAYER_GRAPH", False)


def _repair_graph_enabled() -> bool:
    return not _graph_ablation_enabled() and _env_bool("COTEST_REPAIR_GRAPH_SEARCH", True)


def _intent_grounding_enabled() -> bool:
    return _env_bool(
        "COTEST_ENABLE_INTENT_GROUNDING",
        _env_bool(
            "COTEST_ENABLE_CAUSAL_INVESTIGATION",
            _env_bool("COTEST_ENABLE_INTENT_DIALOGUE", False),
        ),
    )


def _test_derived_intent_enabled() -> bool:
    return (
        _intent_grounding_enabled()
        and _env_str("COTEST_INTENT_DIALOGUE_MODE", "contrastive")
        in {"contrastive", "poc"}
    )


def _reproducer_fallback_enabled() -> bool:
    """Keep the legacy fallback opt-in for compatibility experiments only."""
    return _env_bool("COTEST_ENABLE_REPRODUCER_FALLBACK", False)


def _legacy_reproducer_enabled() -> bool:
    """Preserve legacy Stage 1 behavior unless an experiment disables it."""
    return _env_bool("COTEST_ENABLE_LEGACY_REPRODUCER", True)


def _intent_evidence_steps() -> int:
    """Shared action budget for the test-derived evidence trajectory."""
    return _env_int(
        "COTEST_INTENT_EVIDENCE_STEPS",
        _env_int(
            "COTEST_INTENT_DIALOGUE_TOTAL_STEPS",
            _env_int(
                "COTEST_INTENT_INVESTIGATOR_STEPS",
                _env_int("COTEST_INTENT_DIALOGUE_DEVELOPER_STEPS", 60),
            ),
        ),
    )


def _patch_reviewer_enabled() -> bool:
    configured = os.getenv("COTEST_ENABLE_PATCH_REVIEWER")
    if configured is not None and configured.strip():
        return _env_bool("COTEST_ENABLE_PATCH_REVIEWER")
    return _test_derived_intent_enabled()


def _patch_intent_audit_enabled() -> bool:
    return _env_bool("COTEST_ENABLE_PATCH_INTENT_AUDIT", False)


def _trajectory_audit_leads_enabled() -> bool:
    return _env_bool("COTEST_ENABLE_TRAJECTORY_AUDIT_LEADS", False)


def _patch_intent_audit_unresolved(status: str) -> bool:
    return status in {
        "inconclusive_admission_unavailable",
        "inconclusive_infrastructure",
        "underfit_refinement_started",
        "underfit_refine_empty",
        "underfit_refine_no_persisted_change",
        "underfit_persists_after_refine",
        "inconclusive_recheck",
    }


def _submission_patch_after_test_enhancement(
    repair_patch: str, diagnostic_patch: str
) -> str:
    """TestAgent validates the repair; its generated tests are not submission code."""
    return repair_patch


def _has_valid_contrastive_witness(memory: Memory | None) -> bool:
    return bool(memory and memory.reproduction_result.strip())


def _dialogue_accepted_contrastive_witness(result) -> bool:
    decision = result.contrastive_decision if result is not None else None
    return bool(
        decision
        and decision.witness_valid
        and result.pair_status == "validated"
        and result.stop_reason
        in {"contrastive_boundary_resolved", "contrastive_witness_only"}
    )


def _selected_dialogue_contrastive_witness(result):
    if not _dialogue_accepted_contrastive_witness(result):
        return None
    attempts = getattr(result, "contrastive_attempts", [])
    if attempts:
        return next(
            (
                attempt.witness
                for attempt in reversed(attempts)
                if attempt.witness is not None
                and attempt.decision.witness_valid
            ),
            None,
        )
    investigation = getattr(result, "investigation", None)
    if investigation is None:
        return None
    return investigation.report.contrastive_witness


def _merge_intent_and_reproduction_memory(
    intent_memory: Memory | None,
    reproduction_memory: Memory | None,
) -> Memory | None:
    if reproduction_memory is None:
        return intent_memory
    if intent_memory is None:
        return reproduction_memory
    return reproduction_memory.model_copy(update={
        "repair_summary": intent_memory.repair_summary,
        "issue_evidence": (
            intent_memory.issue_evidence
            or reproduction_memory.issue_evidence
        ),
        "repair_intent_bundle": intent_memory.repair_intent_bundle,
        "repair_intent_contract": intent_memory.repair_intent_contract,
    })


def _generate_submission_patch(file_context: FileContext) -> str:
    return sanitize_candidate_patch(file_context).patch


def _run_portfolio_finish_tail(
    loop: AgenticLoop,
    final_trajectory,
    *,
    max_finish_attempts: int,
):
    """Give a non-empty proposal an explicit Finish-only tail."""
    if (
        max_finish_attempts <= 0
        or final_trajectory.is_finished()
        or not _generate_submission_patch(final_trajectory.file_context).strip()
        or sanitize_candidate_patch(final_trajectory.file_context).artifact_errors
    ):
        return final_trajectory

    finish_actions = [
        action
        for action in loop.agent.actions
        if isinstance(action, (VerifiedFinish, Finish))
    ]
    if not finish_actions:
        logger.warning("Portfolio Finish tail unavailable: no terminal Finish action.")
        return final_trajectory

    loop.append_user_feedback(
        "The portfolio proposal phase is over. Preserve the current production "
        "diff exactly; do not run another test, search, or edit. This candidate "
        "will be compared with independent proposals and is not being declared "
        "globally correct. Call Finish now and report the assigned branch behavior, "
        "the verification that actually ran, and every unresolved or failing "
        "check honestly. Missing or failed legacy, discriminator, or acceptance "
        "evidence is selector evidence, not a reason to withhold this non-empty "
        "proposal."
    )
    finish_action_types = {action.args_schema for action in finish_actions}
    loop.agent.set_execution_action_allowlist(finish_action_types)
    try:
        loop.max_iterations = len(loop.root.get_all_nodes()) + max_finish_attempts
        return loop.run()
    finally:
        loop.agent.set_execution_action_allowlist(None)


def _patches_equivalent(first: str, second: str) -> bool:
    """Compare persisted submission patches while ignoring edge whitespace."""
    return (first or "").strip() == (second or "").strip()


def _register_intent_verification_contract(
    file_context: FileContext,
    memory: Memory,
) -> None:
    raw_contract = memory.repair_intent_contract or {}
    if not raw_contract:
        return
    contract = RepairIntentContract.model_validate(raw_contract)
    spec = contract.verification_spec()
    if (memory.repair_intent_bundle or {}).get("source") == "scout_and_hypothesis_observations":
        spec = _portfolio_scout_verification_spec(contract)
    if not spec["checks"]:
        return
    file_context.verification["intent_contract"] = spec
    logger.info(
        "Registered intent verification contract: targets=%d preserves=%d "
        "acceptance=%d diagnostics=%d",
        len(spec["target_ids"]),
        len(spec["preserve_ids"]),
        len(spec["acceptance_ids"]),
        len(spec["diagnostic_ids"]),
    )


def _save_patch_sanitization_audit(
    file_context: FileContext,
    output_dir: str,
    label: str,
) -> CandidatePatch:
    candidate = sanitize_candidate_patch(file_context)
    raw_path = os.path.join(output_dir, f"solution_raw_{label}.patch")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(candidate.raw_patch)
    audit_path = os.path.join(output_dir, f"patch_sanitization_{label}.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kept_paths": candidate.kept_paths,
                "excluded_paths": candidate.excluded_paths,
                "artifact_errors": list(candidate.artifact_errors),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(
        "Candidate patch sanitation (%s): kept=%d excluded=%d",
        label,
        len(candidate.kept_paths),
        len(candidate.excluded_paths),
    )
    return candidate


def _save_repair_verification_status(
    output_dir: str,
    *,
    verified_finish: bool,
    basic_tests_passed: bool,
    patch_review_status: str,
    patch_intent_audit_status: str | None = None,
) -> None:
    unresolved_review = patch_review_status in {
        "rejected_refinement_started",
        "rejected_refine_empty",
        "rejected_refine_no_persisted_change",
    }
    unresolved_audit = _patch_intent_audit_unresolved(
        patch_intent_audit_status or ""
    )
    path = os.path.join(output_dir, "repair_verification_status.json")
    payload = {
        "verified_finish": verified_finish,
        "basic_tests_passed": basic_tests_passed,
        "patch_review_status": patch_review_status,
        "submission_status": (
            "verified"
            if verified_finish
            and basic_tests_passed
            and not unresolved_review
            and not unresolved_audit
            else "unverified"
        ),
    }
    if patch_intent_audit_status is not None:
        payload["patch_intent_audit_status"] = patch_intent_audit_status
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _portfolio_candidate_variant_payload(
    *,
    variant_id: str,
    finish_succeeded: bool,
    basic_tests_passed: bool,
    candidate: CandidatePatch | None,
    branch_compliance: BranchComplianceDecision | None = None,
    discriminator_verified: bool = False,
    discriminator_required: bool = False,
    blocking_compile_regression: bool = False,
    unresolved_focused_failure: bool = False,
    regression_evidence: Dict[str, Any] | None = None,
    repair_probes: List[RepairProbeEvidence] | None = None,
) -> Dict[str, Any]:
    sanitized_patch = candidate.patch if candidate is not None else ""
    return {
        "variant_id": variant_id,
        "patch": sanitized_patch,
        "finish_succeeded": finish_succeeded,
        "sanitized_patch_nonempty": bool(sanitized_patch.strip()),
        "admitted": bool(
            finish_succeeded
            and sanitized_patch.strip()
            and not getattr(candidate, "artifact_errors", ())
        ),
        "basic_tests_passed": basic_tests_passed,
        "branch_compliance": (
            branch_compliance.status
            if branch_compliance is not None
            else "not_checked"
        ),
        "branch_compliance_decision": (
            branch_compliance.model_dump()
            if branch_compliance is not None
            else None
        ),
        "discriminator_verified": discriminator_verified,
        "discriminator_required": discriminator_required,
        "blocking_compile_regression": blocking_compile_regression,
        "unresolved_focused_failure": unresolved_focused_failure,
        "regression_evidence": regression_evidence,
        "kept_paths": candidate.kept_paths if candidate is not None else [],
        "excluded_paths": (
            candidate.excluded_paths if candidate is not None else {}
        ),
        "artifact_errors": list(getattr(candidate, "artifact_errors", ())),
        "repair_probes": [
            item.model_dump() for item in (repair_probes or [])
        ],
    }


def _save_portfolio_candidate_status(
    output_dir: str,
    *,
    finish_succeeded: bool,
    basic_tests_passed: bool,
    candidate: CandidatePatch | None,
    branch_compliance: str = "not_checked",
    discriminator_verified: bool = False,
    discriminator_required: bool = False,
    blocking_compile_regression: bool = False,
    unresolved_focused_failure: bool = False,
    regression_evidence: Dict[str, Any] | None = None,
    repair_probes: List[RepairProbeEvidence] | None = None,
    branch_compliance_decision: BranchComplianceDecision | None = None,
    candidate_variants: List[Dict[str, Any]] | None = None,
) -> None:
    sanitized_patch = candidate.patch if candidate is not None else ""
    current_variant = _portfolio_candidate_variant_payload(
        variant_id="final",
        finish_succeeded=finish_succeeded,
        basic_tests_passed=basic_tests_passed,
        candidate=candidate,
        branch_compliance=branch_compliance_decision,
        discriminator_verified=discriminator_verified,
        discriminator_required=discriminator_required,
        blocking_compile_regression=blocking_compile_regression,
        unresolved_focused_failure=unresolved_focused_failure,
        regression_evidence=regression_evidence,
        repair_probes=repair_probes,
    )
    if branch_compliance_decision is None:
        current_variant["branch_compliance"] = branch_compliance
    variants = candidate_variants or [current_variant]
    payload = {
        "finish_succeeded": finish_succeeded,
        "sanitized_patch_nonempty": bool(sanitized_patch.strip()),
        "admitted": any(bool(item.get("admitted")) for item in variants),
        "admission_policy": "finished_nonempty_production_patch",
        "basic_tests_passed": basic_tests_passed,
        "regression_passed": basic_tests_passed,
        "regression_role": "soft_evidence",
        "basic_tests_role": "soft_evidence",
        "branch_compliance": branch_compliance,
        "branch_compliance_decision": (
            branch_compliance_decision.model_dump()
            if branch_compliance_decision is not None
            else None
        ),
        "branch_compliance_role": "selector_evidence",
        "discriminator_verified": discriminator_verified,
        "discriminator_required": discriminator_required,
        "discriminator_role": "selector_evidence",
        "blocking_compile_regression": blocking_compile_regression,
        "unresolved_focused_failure": unresolved_focused_failure,
        "focused_failure_role": "selector_evidence",
        "focused_oracle_satisfied": any(
            item.observed_status == "passed"
            for item in (repair_probes or [])
        ),
        "regression_evidence": regression_evidence,
        "kept_paths": candidate.kept_paths if candidate is not None else [],
        "excluded_paths": (
            candidate.excluded_paths if candidate is not None else {}
        ),
        "repair_probes": [
            item.model_dump() for item in (repair_probes or [])
        ],
        "candidate_variants": variants,
    }
    path = os.path.join(output_dir, "portfolio_candidate_status.json")
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def _portfolio_patch_sha256(patch: str) -> str:
    return production_patch_sha256(patch)


def _retain_portfolio_candidate_variant(
    variants_by_hash: Dict[str, Dict[str, Any]],
    variant: Dict[str, Any],
) -> None:
    patch_sha256 = _portfolio_patch_sha256(str(variant.get("patch") or ""))
    previous = variants_by_hash.get(patch_sha256)
    if previous is None or variant.get("admitted") or not previous.get("admitted"):
        variants_by_hash[patch_sha256] = variant


def _portfolio_discriminator_marker(hypothesis_id: str) -> str:
    return f"INTENT2FIX_DISCRIMINATOR_{hypothesis_id}_PASS"


def _repair_discriminator_verified(
    probes: List[RepairProbeEvidence], hypothesis_id: str
) -> bool:
    marker = _portfolio_discriminator_marker(hypothesis_id)
    return any(
        probe.kind == "discriminator"
        and probe.observed_status == "passed"
        and marker in probe.observed_output
        for probe in probes
    )


def _verification_observation_status(step) -> tuple[str, int | None, str]:
    observation = step.observation
    properties = observation.properties if observation is not None else {}
    exit_code = properties.get("exit_code")
    output = ""
    test_summary = properties.get("test_summary") or {}
    if isinstance(test_summary, dict):
        output = str(test_summary.get("raw_output") or "")
    if not output:
        output = str(properties.get("output") or "")
    if not output and observation is not None:
        output = str(observation.message or "")
    oracle_satisfied = properties.get("oracle_satisfied")
    if isinstance(oracle_satisfied, bool) and exit_code is not None:
        return (
            "passed" if oracle_satisfied else "failed",
            int(exit_code),
            output,
        )
    compilation_failure = bool(
        properties.get("compilation_failure_signature")
    )
    masked_nonzero = bool(
        re.search(
            r"(?m)\b(?:EXIT|R[0-9]+|RC|STATUS)=([1-9][0-9]*)\b",
            output,
            flags=re.IGNORECASE,
        )
    )
    compiler_error = bool(
        exit_code == 0
        and re.search(
            r"(?m)(?:^|\n)(?:[^\n]*:)?(?:fatal )?error:|"
            r"undefined reference|collect2: error|Segmentation fault|"
            r"AddressSanitizer:(?:DEADLYSIGNAL|[^\n]*ERROR)|"
            r"UndefinedBehaviorSanitizer:[^\n]*(?:ERROR|runtime error)",
            output,
        )
    )
    if exit_code is None:
        return "inconclusive", None, output
    if exit_code != 0 or compilation_failure or masked_nonzero or compiler_error:
        return "failed", int(exit_code), output
    return "passed", int(exit_code), output


def _portfolio_probe_setup_failure(
    command: str,
    output: str,
    *,
    file_context: FileContext | None = None,
    production_patch: str = "",
) -> bool:
    """Return whether a probe failed before reaching its behavioral target."""
    for match in re.finditer(
        r"(?:fatal )?error:\s*([^:\n]+): No such file or directory",
        output,
        flags=re.IGNORECASE,
    ):
        missing_input = match.group(1).strip().strip("'\"")
        if missing_input and missing_input in command:
            return True

    # A one-off, registered probe that cannot resolve a header has not reached
    # its behavioral oracle. Do not let include-path corrections consume the
    # semantic diagnostic allowance. A dependency newly introduced by the
    # production diff remains a real candidate compilation failure.
    ephemeral_paths = set(
        getattr(file_context, "verification", {}).get("ephemeral_paths", [])
        if file_context is not None
        else []
    )
    probe_invocation = any(
        path and (str(path) in command or str(path) in output)
        for path in ephemeral_paths
    )
    if probe_invocation:
        added_production_lines = "\n".join(
            line[1:]
            for line in production_patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        for match in re.finditer(
            r"fatal error:\s*([^:\n]+): No such file or directory",
            output,
            flags=re.IGNORECASE,
        ):
            missing_header = match.group(1).strip().strip("'\"")
            if missing_header and missing_header not in added_production_lines:
                return True

    semantic_diagnostic = re.search(
        r"ambiguous|candidate:|no matching function|no match for|"
        r"static assertion failed|undefined reference",
        output,
        flags=re.IGNORECASE,
    )
    missing_command = re.search(
        r"(?m)(?:^|\n)sh: [0-9]+: [^\n]+: not found|command not found",
        output,
        flags=re.IGNORECASE,
    )
    probe_wiring_failure = re.search(
        r"syntax error:.*unexpected|"
        r"(?:undefined reference to|multiple definition of) [`'\"]?main\b",
        output,
        flags=re.IGNORECASE,
    )
    return bool(
        probe_wiring_failure
        or (missing_command and not semantic_diagnostic)
    )


def _portfolio_branch_discriminator_probe(
    file_context: FileContext,
    command: str,
) -> bool:
    """Return whether command executes a candidate-authored branch oracle."""
    marker = re.compile(r"INTENT2FIX_DISCRIMINATOR_H[0-9]+_PASS")
    if marker.search(command):
        return True
    for raw_path in file_context.verification.get("ephemeral_paths", []):
        path = str(raw_path)
        if not path or path not in command:
            continue
        context_file = file_context.get_file(path)
        if context_file is not None and marker.search(context_file.content):
            return True
    return False


def _portfolio_regression_derived_failure(
    file_context: FileContext,
    command: str,
    step,
) -> bool:
    """Keep a targeted rerun of a known regression failure at soft authority."""
    normalize = lambda value: re.sub(r"\s+", " ", str(value)).strip()
    normalized_command = normalize(command)
    intent_contract = file_context.verification.get("intent_contract") or {}
    for probe in (intent_contract.get("probes") or {}).values():
        if normalize((probe or {}).get("command", "")) == normalized_command:
            return False

    current_change_seq = getattr(file_context, "change_seq", 0)
    current_records = [
        record
        for record in (
            file_context.verification.get("regression") or {},
            file_context.verification.get("production_regression") or {},
        )
        if record.get("change_seq") == current_change_seq
    ]
    # A focused failure from a candidate-edited test view is still soft
    # evidence.  It must not become a new intent authority merely because the
    # unchanged production suite was reused instead of rebuilding that view.
    if any(record.get("test_view_stale") for record in current_records):
        return True
    current_failures = [
        record for record in current_records if record.get("passed") is False
    ]
    failed_names = {
        str(name).strip()
        for record in current_failures
        for name in record.get("failed_test_names", [])
        if str(name).strip()
    }
    if not failed_names:
        return False
    failed_aliases = set(failed_names)
    sorted_names = sorted(failed_names)
    for index, first in enumerate(sorted_names):
        for second in sorted_names[index + 1 :]:
            prefix = os.path.commonprefix((first, second))
            if not prefix or prefix[-1:] not in {"-", "_", ".", ":", "/"}:
                continue
            alias = prefix.rstrip("-_.:/")
            # Avoid collapsing unrelated test names to generic prefixes such
            # as `test`; configuration variants normally share a substantial
            # executable/target stem.
            if len(alias) >= 6 and len(alias) * 2 >= min(
                len(first), len(second)
            ):
                failed_aliases.add(alias)
    properties = step.observation.properties if step.observation else {}
    summary = properties.get("test_summary") or {}
    observed_names = {
        str(item.get("name") or "").strip()
        for item in summary.get("results", [])
        if str(item.get("status") or "").lower() == "failed"
    }
    return bool(
        failed_names.intersection(observed_names)
        or any(name in command for name in failed_aliases)
    )


def _collect_final_patch_repair_probes(
    *,
    final_trajectory,
    candidate: CandidatePatch,
    hypothesis_context: Dict[str, Any] | None,
) -> List[RepairProbeEvidence]:
    if final_trajectory is None or not candidate.patch.strip():
        return []
    final_patch_sha256 = _portfolio_patch_sha256(candidate.patch)
    branch = (hypothesis_context or {}).get("branch") or {}
    hypothesis_id = str(branch.get("id") or "H0")
    discriminator = str(branch.get("discriminator") or "")
    predicted_outcome = str(branch.get("predicted_outcome") or "")
    discriminator_marker = _portfolio_discriminator_marker(hypothesis_id)
    all_nodes = final_trajectory.get_all_nodes()
    records: List[RepairProbeEvidence] = []
    seen_executions: set[tuple] = set()
    for node in reversed(all_nodes):
        for step in reversed(node.action_steps):
            if step.action.name not in {"RunSingleTest", "Bash"}:
                continue
            if step.action.name == "Bash" and not getattr(step.action, "verification", False):
                continue
            properties = step.observation.properties if step.observation else {}
            execution = properties.get("execution_snapshot") or {}
            # Old node-level snapshots cannot establish which revision a test
            # executed when a batch also edited code. Require action evidence.
            if (
                execution.get("production_patch_sha256") != final_patch_sha256
                or properties.get("fail_reason")
                or execution.get("artifact_errors")
                or execution.get("command") != getattr(step.action, "command", "")
                or execution.get("expected_exit_code") != getattr(step.action, "expected_exit_code", 0)
                or execution.get("required_output_regex") != getattr(step.action, "required_output_regex", "")
            ):
                continue
            command = str(getattr(step.action, "command", "")).strip()
            if not command:
                continue
            files = dict(execution.get("probe_files") or {})
            identity = (
                command, execution.get("expected_exit_code"),
                execution.get("required_output_regex"),
                json.dumps(files, sort_keys=True, ensure_ascii=False),
            )
            if identity in seen_executions:
                continue
            seen_executions.add(identity)
            status, exit_code, output = _verification_observation_status(step)
            if status == "failed" and _portfolio_probe_setup_failure(
                command,
                output,
                file_context=node.file_context,
                production_patch=candidate.patch,
            ):
                continue
            regression_derived = (
                status == "failed"
                and _portfolio_regression_derived_failure(
                    node.file_context,
                    command,
                    step,
                )
            )
            kind = (
                "discriminator"
                if discriminator
                and (
                    discriminator_marker in command
                    or discriminator_marker in output
                )
                else (
                    "soft_regression"
                    if regression_derived
                    else "focused"
                )
            )
            description = (
                f"Repair final-patch verification for {hypothesis_id}."
            )
            if not files:
                description += " No temporary fixture source was declared for this execution."
            if kind == "discriminator":
                description += f" Assigned discriminator: {discriminator}"
            records.append(
                RepairProbeEvidence(
                    probe_id="",
                    source_hypothesis_id=hypothesis_id,
                    kind=kind,
                    description=description,
                    expected_outcome=predicted_outcome,
                    command=command,
                    files=files,
                    expected_exit_code=int(
                        getattr(step.action, "expected_exit_code", 0) or 0
                    ),
                    required_output_regex=str(
                        getattr(step.action, "required_output_regex", "")
                        or ""
                    ),
                    observed_status=status,
                    observed_exit_code=exit_code,
                    observed_output=_bounded_command_output(output),
                    production_patch_sha256=final_patch_sha256,
                )
            )

    # A corrected link command for the exact same frozen probe source proves
    # that an earlier linker-only failure never reached the behavior oracle.
    # Drop only that superseded setup evidence; semantic failures remain.
    successful_probe_files = [
        item.files
        for item in records
        if item.observed_status == "passed" and item.files
    ]
    if successful_probe_files:
        records = [
            item
            for item in records
            if not (
                item.observed_status == "failed"
                and item.files in successful_probe_files
                and re.search(
                    r"undefined reference|collect2: error|"
                    r"ld returned [1-9][0-9]* exit status",
                    item.observed_output,
                    flags=re.IGNORECASE,
                )
            )
        ]

    # Preserve all distinct current-revision checks with their exact fixtures.
    # Ordering or a two-record cap cannot decide which test establishes intent.
    return [
        item.model_copy(
            update={"probe_id": f"REPAIR-{hypothesis_id}-{index}"}
        )
        for index, item in enumerate(reversed(records), start=1)
    ]


def _portfolio_has_unresolved_focused_failure(
    probes: List[RepairProbeEvidence], *, discriminator_required: bool
) -> bool:
    """Return whether final-patch target/preservation evidence still fails.

    A required discriminator is adjudicated separately from its branch marker
    and compliance audit. It must not suppress an independent focused failure
    from the same final production revision.
    """
    return any(
        probe.observed_status == "failed"
        and probe.kind != "soft_regression"
        and (not discriminator_required or probe.kind != "discriminator")
        for probe in probes
    )


def _portfolio_focused_failure_diagnostic(
    probes: List[RepairProbeEvidence],
) -> str:
    failures = [
        probe
        for probe in probes
        if probe.observed_status == "failed"
    ]
    if not failures:
        return ""
    sections = []
    for probe in failures:
        sections.append(
            "Command:\n"
            f"{probe.command}\n"
            "Observed failure:\n"
            f"{probe.observed_output}"
        )
    return _bounded_command_output("\n\n".join(sections))


def _patch_intent_audit_contracts(memory: Memory) -> tuple[str, str]:
    raw_contract = memory.repair_intent_contract or {}
    review_contract = memory.repair_summary or ""
    normative_contract = ""
    if not raw_contract:
        return review_contract, normative_contract
    try:
        contract = RepairIntentContract.model_validate(raw_contract)
    except ValidationError:
        logger.warning(
            "Could not render structured Intent context for Patch Intent "
            "Audit; falling back to the non-authoritative Repair handoff.",
            exc_info=True,
        )
        return review_contract, normative_contract

    review_contract = contract.render_for_review()
    normative_lines = [
        f"[{item.id}] [{item.role.upper()}] {item.text}"
        for item in contract.intent_ledger
    ]
    if not contract.intent_ledger:
        normative_lines.extend(
            f"[TARGET] {item}" for item in contract.target_behavior
        )
        normative_lines.extend(
            f"[PRESERVE] {item}" for item in contract.preserve_behavior
        )
    normative_lines.extend(
        f"[{item.id}] [{item.role.upper()}] {item.predicate}"
        for item in contract.properties
    )
    normative_lines.extend(
        f"[ACCEPT-{index}] {check.description}"
        for index, check in enumerate(contract.acceptance_checks, 1)
        if check.authority == "normative"
    )
    normative_contract = "\n".join(normative_lines)
    if normative_contract:
        review_contract += (
            "\n\n<blocking_oracle_authority>\n"
            "Only exact spans from this section may support an "
            "accepted-contract blocking oracle. Open decisions, candidate "
            "boundaries, validation gaps, and compatibility notes are excluded.\n"
            + normative_contract
            + "\n</blocking_oracle_authority>"
        )
    return review_contract, normative_contract


def _render_trajectory_audit_leads(leads: list[AuditLead]) -> str:
    if not leads:
        return ""
    return json.dumps(
        [lead.model_dump() for lead in leads],
        ensure_ascii=False,
        indent=2,
    )


def _extract_repair_trajectory_audit_leads(
    *,
    persist_path: str,
    candidate_patch: str,
    output_dir: str,
    model_name: str,
) -> TrajectoryLeadResult | None:
    if not _trajectory_audit_leads_enabled():
        return None
    artifact_path = os.path.join(
        output_dir,
        "repair_trajectory_audit_leads.json",
    )
    try:
        trajectory = load_trajectory(persist_path)
        trajectory_slice = slice_trajectory(
            trajectory,
            candidate_patch,
            max_events=_env_int("COTEST_TRAJECTORY_AUDIT_LEAD_MAX_EVENTS", 16),
        )
        lead_model = _env_str(
            "COTEST_TRAJECTORY_AUDIT_LEAD_MODEL",
            _env_str("COTEST_PATCH_INTENT_AUDIT_MODEL", model_name),
        )
        result = extract_trajectory_leads(trajectory_slice, lead_model)
    except Exception:
        logger.exception(
            "Repair trajectory AuditLead extraction failed; continuing without leads."
        )
        return None
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info(
        "Repair trajectory AuditLead extraction: status=%s finished=%s "
        "events=%d/%d leads=%d rejected=%d artifact=%s",
        result.extraction_status,
        result.trajectory_slice.repair_finished,
        result.trajectory_slice.stats.selected_events,
        result.trajectory_slice.stats.events_created,
        len(result.leads),
        len(result.validation_errors),
        artifact_path,
    )
    return result


def _run_patch_intent_audit(
    *,
    instance_name: str,
    model_name: str,
    repository,
    docker_env: DockerEnv,
    code_index: CodeIndex,
    memory: Memory,
    candidate: CandidatePatch,
    file_context: FileContext,
    output_dir: str,
    trajectory_audit_leads: str = "",
):
    sync_error = sync_verification_files(
        file_context,
        docker_env,
        included_paths=set(candidate.kept_paths),
    )
    if sync_error:
        raise RuntimeError(
            "Could not prepare the production-only Patch Intent Audit view: "
            + sync_error
        )
    audit_model = _env_str("COTEST_PATCH_INTENT_AUDIT_MODEL", model_name)
    admission_model = _env_str(
        "COTEST_PATCH_INTENT_ADMISSION_MODEL",
        audit_model,
    )

    def _semantic_adjudicator(prompt: str, schema: Type):
        return _run_structured_extraction(
            prompt,
            schema,
            model_override=admission_model,
            max_tokens=_env_int(
                "COTEST_PATCH_INTENT_ADMISSION_MAX_TOKENS",
                4096,
            ),
            glm_reasoning_effort=_env_str(
                "COTEST_PATCH_INTENT_ADMISSION_REASONING_EFFORT",
                _env_str("COTEST_GLM_REASONING_EFFORT", "low"),
            ),
        )

    review_contract, normative_contract = _patch_intent_audit_contracts(memory)
    auditor = PatchIntentAuditor.create(
        repository=repository,
        docker_env=docker_env,
        code_index=code_index,
        issue=memory.problem_statement,
        contract=normative_contract,
        included_paths=set(candidate.kept_paths),
        model=audit_model,
        semantic_adjudicator=_semantic_adjudicator,
        max_steps=_env_int("COTEST_PATCH_INTENT_AUDIT_STEPS", 40),
        **_completion_protocol_kwargs(audit_model),
        **_model_kwargs(audit_model, "COTEST_PATCH_INTENT_AUDIT_MAX_TOKENS"),
    )
    result = auditor.audit(
        issue=memory.problem_statement,
        repair_intent_contract=review_contract,
        patch=candidate.patch,
        file_context=file_context,
        excluded_paths=candidate.excluded_paths,
        trajectory_audit_leads=trajectory_audit_leads,
    )
    payload = {
        "result": result.model_dump(),
        "metrics": auditor.metrics,
        "trajectory_audit_leads_supplied": bool(trajectory_audit_leads),
    }
    if auditor.trajectory is not None:
        payload["trajectory"] = auditor.trajectory.dump_as_list(
            exclude_none=True
        )
    artifact_path = os.path.join(
        output_dir,
        f"patch_intent_audit_{instance_name}.json",
    )
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    report_path = os.path.join(output_dir, "patch_intent_audit_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"verdict={result.verdict}\n")
        f.write(f"blocking={str(result.blocking).lower()}\n\n")
        f.write(result.report or "No Issue-supported counterexample found.")
        f.write("\n")
    logger.info(
        "Patch Intent Audit: verdict=%s blocking=%s steps=%d/%d "
        "tool_calls=%d rejected_tools=%d no_action_retries=%d "
        "termination=%s report=%s",
        result.verdict,
        result.blocking,
        auditor.steps_used,
        auditor.max_steps,
        auditor.tool_calls_attempted,
        auditor.rejected_tool_calls,
        auditor.no_action_retries,
        auditor.termination_reason,
        result.report[:500],
    )
    return auditor, result


def _render_patch_intent_probe_feedback(
    audit_result,
    probe,
    *,
    element_name: str,
) -> str:
    if probe is None:
        return ""
    semantic_admission = getattr(audit_result, "semantic_admission", None)
    mechanism_confirmed = (
        semantic_admission is None
        or semantic_admission.candidate_prediction_matches_observation
    )
    if mechanism_confirmed:
        mechanism_feedback = (
            f"Candidate prediction: {probe.candidate_prediction}\n"
            f"Alternative prediction: {probe.alternative_prediction}\n"
        )
    else:
        mechanism_feedback = (
            "Mechanism attribution: unconfirmed. The pre-run mechanism "
            "predictions are omitted because execution contradicted the "
            "candidate prediction; infer root cause from the source and "
            "observed behavior.\n"
        )
    return (
        f"\n\n<{element_name}>\n"
        f"Issue authority: {probe.oracle_reference}\n"
        f"Oracle: {probe.oracle}\n"
        f"Oracle mode: {probe.oracle_mode}\n"
        f"Purpose: {probe.purpose or '(not stated)'}\n"
        f"Distinguishing input: {probe.distinguishing_input}\n"
        + mechanism_feedback
        + "Observed exits: "
        f"control_compile={probe.control_compile_exit_code}, "
        f"control_run={probe.control_run_exit_code}, "
        f"target_compile={probe.target_compile_exit_code}, "
        f"target_run={probe.target_run_exit_code}\n"
        "Shared fixture:\n"
        + _compact_reproduction_output(probe.fixture_content, 8000)
        + "\n\nPassing control:\n"
        + _compact_reproduction_output(probe.control_content, 8000)
        + f"\nCommands: {probe.control_compile_command} ; "
        f"{probe.control_run_command}\n"
        "Failing target:\n"
        + _compact_reproduction_output(probe.target_content, 8000)
        + f"\nCommands: {probe.target_compile_command} ; "
        f"{probe.target_run_command}\n"
        "Observed pair output:\n"
        + _compact_reproduction_output(probe.output, 8000)
        + f"\n</{element_name}>"
    )


def _render_patch_intent_counterexample_feedback(audit_result) -> str:
    return _render_patch_intent_probe_feedback(
        audit_result,
        audit_result.counterexample,
        element_name="frozen_contrastive_counterexample",
    )


def _render_patch_intent_unresolved_discriminator_feedback(audit_result) -> str:
    return _render_patch_intent_probe_feedback(
        audit_result,
        audit_result.unresolved_discriminator,
        element_name="unresolved_mechanism_discriminator",
    )


def _recheck_patch_intent_counterexample(
    *,
    auditor,
    audit_result,
    file_context: FileContext,
    candidate: CandidatePatch,
    output_dir: str,
    label: str,
) -> str:
    if auditor is None or audit_result is None:
        return "not_available"
    if audit_result.counterexample is None:
        return "stale_after_later_patch_change"
    try:
        recheck = auditor.recheck(
            audit_result.counterexample,
            file_context=file_context,
            included_paths=set(candidate.kept_paths),
        )
    except Exception as exc:
        logger.exception("Frozen Patch Intent Audit recheck failed.")
        payload = {
            "conclusive": False,
            "passed": False,
            "error": str(exc),
        }
        status = "inconclusive_recheck"
    else:
        payload = recheck.model_dump()
        if not recheck.conclusive:
            status = "inconclusive_recheck"
        else:
            status = (
                "underfit_refined"
                if recheck.passed
                else "underfit_persists_after_refine"
            )
    path = os.path.join(
        output_dir,
        f"patch_intent_counterexample_recheck_{label}.json",
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return status


def _run_patch_surface_review(
    *,
    instance_name: str,
    model_name: str,
    repository,
    docker_env: DockerEnv,
    code_index: CodeIndex,
    memory: Memory,
    candidate: CandidatePatch,
    file_context: FileContext,
    output_dir: str,
    pass_number: int,
):
    sync_error = sync_verification_files(
        file_context,
        docker_env,
        included_paths=set(candidate.kept_paths),
    )
    if sync_error:
        raise RuntimeError(
            "Could not prepare the production-only Patch Reviewer view: "
            + sync_error
        )
    reviewer_model = _env_str("COTEST_PATCH_REVIEW_MODEL", model_name)
    reviewer = PatchSurfaceReviewer.create(
        repository=repository,
        docker_env=docker_env,
        code_index=code_index,
        model=reviewer_model,
        max_steps=_env_int("COTEST_PATCH_REVIEW_STEPS", 50),
        **_completion_protocol_kwargs(reviewer_model),
        **_model_kwargs(reviewer_model, "COTEST_PATCH_REVIEW_MAX_TOKENS"),
    )
    structural_routes = [
        route
        for route in (memory.repair_intent_contract or {}).get(
            "validation_routes", []
        )
        if route.get("mode") == "structural_review"
    ]
    structural_section = ""
    if structural_routes:
        property_by_id = {
            item.get("id"): item
            for item in (memory.repair_intent_contract or {}).get(
                "properties", []
            )
        }
        route_lines = [
            f"- [{route.get('id', '')}; {route.get('property_id', '')}] "
            f"{property_by_id.get(route.get('property_id'), {}).get('predicate', '')} "
            f"Review lead: {route.get('generation_plan', '')}"
            for route in structural_routes
        ]
        structural_section = (
            "\n\n<intent_validation_reviewer_leads>\n"
            "These are non-binding structural investigation leads selected "
            "by the validation router. Require an omission only with direct "
            "repository evidence.\n"
            + "\n".join(route_lines)
            + "\n</intent_validation_reviewer_leads>"
        )
    review_contract = memory.repair_summary or ""
    raw_contract = memory.repair_intent_contract or {}
    if raw_contract:
        try:
            review_contract = RepairIntentContract.model_validate(
                raw_contract
            ).render_for_review()
        except ValidationError:
            logger.warning(
                "Could not render structured Intent context for Patch Reviewer; "
                "falling back to the Repair handoff.",
                exc_info=True,
            )
    result = reviewer.review(
        issue=memory.problem_statement,
        repair_intent_contract=review_contract + structural_section,
        patch=candidate.patch,
        file_context=file_context,
        excluded_paths=candidate.excluded_paths,
    )
    path = os.path.join(
        output_dir,
        f"patch_surface_review_{instance_name}_pass{pass_number}.json",
    )
    payload = {"result": result.model_dump()}
    if reviewer.trajectory is not None:
        payload["trajectory"] = reviewer.trajectory.dump_as_list(
            exclude_none=True
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    report_path = os.path.join(
        output_dir,
        f"patch_surface_review_report_pass{pass_number}.txt",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"approved={str(result.approved).lower()}\n\n")
        f.write(result.report or "Patch surface review passed.")
        f.write("\n")
    logger.info(
        "Patch surface review pass %d: approved=%s report=%s",
        pass_number,
        result.approved,
        result.report[:500],
    )
    return result


def _remove_host_reproduction_artifacts(repo_path: str, paths: List[str]) -> None:
    for path in set(paths):
        if not is_dedicated_reproduction_path(path):
            continue
        full_path = os.path.abspath(os.path.join(repo_path, path))
        if os.path.commonpath([os.path.abspath(repo_path), full_path]) != os.path.abspath(repo_path):
            continue
        if os.path.isfile(full_path):
            os.remove(full_path)
            logger.info("Removed staged reproduction artifact: %s", full_path)


def _run_intent_grounding(
    *,
    instance: Dict[str, Any],
    repository,
    docker_env: DockerEnv,
    code_index: CodeIndex | None,
    issue_evidence: IssueEvidenceFrame,
    reproduction_context: Dict[str, Any],
    output_dir: str,
    fallback_model: str,
):
    investigator_model = _env_str(
        "COTEST_INTENT_DIALOGUE_MODEL",
        _env_str("COTEST_EXTRACTION_MODEL", fallback_model),
    )
    dialogue_mode = _env_str("COTEST_INTENT_DIALOGUE_MODE", "contrastive")
    investigator = ContainerRepositoryInvestigator.create(
        repository=repository,
        docker_env=docker_env,
        code_index=code_index,
        model=investigator_model,
        mode=dialogue_mode,
        enable_graph_search=(
            not _graph_ablation_enabled()
            and _env_bool("COTEST_SCOUT_GRAPH_SEARCH", True)
        ),
        max_steps=_intent_evidence_steps(),
        **_completion_protocol_kwargs(investigator_model),
        **_model_kwargs(
            investigator_model,
            "COTEST_INTENT_DIALOGUE_GENERATION_MAX_TOKENS",
            legacy_max_tokens_key="COTEST_INTENT_DIALOGUE_MAX_TOKENS",
            glm_reasoning_effort=_intent_scout_reasoning_effort(),
        ),
    )
    dialogue = IntentDialogue(
        extractor=_run_intent_dialogue_extraction,
        evidence_sufficiency_reviewer=_run_intent_evidence_review,
        investigator=investigator,
        mode=dialogue_mode,
        enable_test_anchor_retrieval=(
            dialogue_mode == "contrastive"
            and not _graph_ablation_enabled()
            and _env_bool("COTEST_INTENT_TEST_ANCHOR_RETRIEVAL", True)
        ),
        enable_property_materialization=(
            _env_bool(
                "COTEST_PORTFOLIO_SCOUT_MATERIALIZE_BOUNDARIES",
                False,
            )
            if _env_bool("COTEST_ENABLE_INTENT_PORTFOLIO", False)
            else True
        ),
        max_tasks=_env_int(
            "COTEST_INTENT_MAX_TASKS",
            _env_int(
                "COTEST_INTENT_EXPLORATION_MAX_OBJECTIVES",
                _env_int("COTEST_INTENT_DIALOGUE_MAX_ROUNDS", 10),
            ),
        ),
    )
    result = dialogue.run(
        issue=instance["problem_statement"],
        issue_evidence=issue_evidence,
        reproduction=reproduction_context,
    )
    path = os.path.join(output_dir, "intent_grounding_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info(
        "Intent grounding completed: mode=%s tasks=%d report=%s "
        "stop_reason=%s pair_status=%s resolution=%s adjudication=%s "
        "contrastive=%s evidence_ready=%s semantic_audit=%s "
        "version_space=%s repair_intent_contract=%s",
        result.exploration_plan.mode,
        len(result.exploration_plan.tasks),
        "present" if result.investigation else "none",
        result.stop_reason,
        result.pair_status or "not_applicable",
        (
            ",".join(
                "resolved" if item.resolved else "unresolved"
                for item in result.resolution.resolutions
            )
            if result.resolution
            else "none"
        ),
        "present" if result.adjudication else "none",
        (
            "boundary_resolved"
            if (
                result.contrastive_decision
                and result.contrastive_decision.boundary_resolved
            )
            else "witness_only"
            if (
                result.contrastive_decision
                and result.contrastive_decision.witness_valid
            )
            else "repository_only"
            if (
                result.contrastive_decision
                and result.contrastive_decision.repository_effects
            )
            else "no_effect"
            if result.contrastive_decision
            else "none"
        ),
        (
            "ready"
            if getattr(result, "hypothesis_ready", None) is True
            else "not_ready"
            if getattr(result, "hypothesis_ready", None) is False
            else "not_reviewed"
        ),
        (
            "consistent"
            if result.contract_semantic_audit
            and result.contract_semantic_audit.consistent
            else "inconsistent"
            if result.contract_semantic_audit
            else "none"
        ),
        ",".join(item.status for item in result.version_space) or "none",
        "present" if result.render_for_repair() else "empty",
    )
    return result


def run_test_derived_intent_step(
    instance_name: str,
    model_name: str,
    docker_env: DockerEnv,
    output_dir: str,
) -> Memory:
    """Build repair memory from repository contract and optional test evidence."""
    logger.info("--- Starting Intent Evidence Step for %s ---", instance_name)
    instance = get_benchmark_instance(instance_name)

    cache_dir = (
        "" if _graph_ablation_enabled()
        else os.getenv("COTEST_INTENT_SCOUT_CACHE_DIR", "").strip()
    )
    if cache_dir:
        cache_path = os.path.join(
            _resolve_jsonl_path(cache_dir),
            instance_name,
            "intent_grounding_result.json",
        )
        if os.path.isfile(cache_path):
            cached = IntentDialogueResult.model_validate(
                _read_json_file(cache_path)
            )
            manifest_path = os.path.join(
                os.path.dirname(cache_path), "intent_scout_checkpoint.json"
            )
            expected_manifest = _intent_scout_checkpoint_manifest(
                instance_name, instance
            )
            manifest_matches = (
                os.path.isfile(manifest_path)
                and _read_json_file(manifest_path) == expected_manifest
            )
            trust_legacy = _env_bool(
                "COTEST_INTENT_SCOUT_CACHE_TRUST_LEGACY", False
            )
            probe_rejection = _cached_intent_scout_probe_rejection(cached)
            if (manifest_matches or trust_legacy) and not probe_rejection:
                issue_evidence = cached.issue_evidence
                os.makedirs(output_dir, exist_ok=True)
                with open(
                    os.path.join(output_dir, "issue_evidence.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        issue_evidence.model_dump(),
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                with open(
                    os.path.join(output_dir, "intent_grounding_result.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        cached.model_dump(), f, indent=2, ensure_ascii=False
                    )
                with open(
                    os.path.join(output_dir, "intent_scout_checkpoint.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(expected_manifest, f, indent=2, ensure_ascii=False)
                if trust_legacy and not manifest_matches:
                    logger.warning(
                        "Reused legacy Intent Scout checkpoint without a "
                        "manifest because explicit trust was enabled: %s",
                        cache_path,
                    )
                else:
                    logger.info(
                        "Reused validated Intent Scout checkpoint: %s",
                        cache_path,
                    )
                return _memory_from_intent_dialogue_result(
                    instance_name=instance_name,
                    problem_statement=instance["problem_statement"],
                    issue_evidence=issue_evidence,
                    result=cached,
                    output_dir=output_dir,
                )
            if probe_rejection:
                logger.warning(
                    "Ignoring Intent Scout checkpoint with a non-replayable "
                    "frozen probe (%s): %s",
                    probe_rejection,
                    cache_path,
                )

            else:
                logger.warning(
                    "Ignoring Intent Scout checkpoint with no matching "
                    "manifest: %s",
                    cache_path,
                )

    issue_evidence = _extract_issue_evidence(instance["problem_statement"])

    repository = create_repository(instance)

    code_index = None
    if not _graph_ablation_enabled():
        index_store_dir = os.getenv("INDEX_STORE_DIR")
        if not index_store_dir:
            raise ValueError("INDEX_STORE_DIR environment variable is not set.")
        os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
        code_index = CodeIndex.from_index_name(
            index_name=instance_name,
            file_repo=repository,
            index_store_dir=index_store_dir,
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "issue_evidence.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(issue_evidence.model_dump(), f, indent=2, ensure_ascii=False)

    result = None
    with docker_env:
        logger.info(
            "Docker container '%s' is running for Intent evidence induction.",
            docker_env.container_name,
        )
        try:
            result = _run_intent_grounding(
                instance=instance,
                repository=repository,
                docker_env=docker_env,
                code_index=code_index,
                issue_evidence=issue_evidence,
                reproduction_context={
                    "reproduction_status": "not_reproduced",
                    "evidence_validated": False,
                    "source": (
                        "Separate Reproducer has not run. Scout may run an "
                        "Issue-faithful diagnostic POC or return repository "
                        "evidence only."
                        if _env_str("COTEST_INTENT_DIALOGUE_MODE", "contrastive")
                        == "poc"
                        else "Separate Reproducer has not run. The Investigator "
                        "may establish an Issue-guided TARGET-1 or return "
                        "repository contract evidence only."
                    ),
                    "execution_records": [],
                },
                output_dir=output_dir,
                fallback_model=model_name,
            )
        except IntentEmbeddingUnavailableError:
            raise
        except Exception:
            logger.exception(
                "Intent evidence induction failed; continuing with the "
                "original Issue contract."
            )

    if result is not None:
        with open(
            os.path.join(output_dir, "intent_scout_checkpoint.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                _intent_scout_checkpoint_manifest(instance_name, instance),
                f,
                indent=2,
                ensure_ascii=False,
            )

    return _memory_from_intent_dialogue_result(
        instance_name=instance_name,
        problem_statement=instance["problem_statement"],
        issue_evidence=issue_evidence,
        result=result,
        output_dir=output_dir,
    )


_INTENT_SCOUT_POLICY_VERSION = "2026-09-15-branch-closure-review-v19"


def _intent_scout_reasoning_effort() -> str:
    return _env_str(
        "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT",
        "low",
    )


def _intent_evidence_review_reasoning_effort() -> str:
    return _env_str(
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
        "low",
    )


def _intent_scout_checkpoint_manifest(
    instance_name: str,
    instance: Dict[str, Any],
) -> Dict[str, str]:
    problem_statement = str(instance["problem_statement"])
    manifest = {
        "instance_id": instance_name,
        "base_commit": str(instance.get("base_commit") or ""),
        "problem_sha256": hashlib.sha256(
            problem_statement.encode("utf-8")
        ).hexdigest(),
        "policy_version": _INTENT_SCOUT_POLICY_VERSION,
        "investigation_mode": _env_str(
            "COTEST_INTENT_DIALOGUE_MODE", "contrastive"
        ),
        "scout_reasoning_effort": _intent_scout_reasoning_effort(),
        "evidence_review_reasoning_effort": (
            _intent_evidence_review_reasoning_effort()
        ),
    }
    if _graph_ablation_enabled():
        manifest["dual_layer_graph"] = "disabled"
    return manifest


def _atomic_repair_surface_statements(facts: List[str]) -> List[str]:
    statements: List[str] = []
    splitter = re.compile(
        r";\s+|,\s+(?=(?:and|but|so)\s+"
        r"(?:(?:the|a|an|this|that)\s+)?[A-Za-z_])"
    )
    for fact in facts:
        for part in splitter.split(fact):
            statement = re.sub(r"^(?:and|but|so)\s+", "", part.strip())
            if statement:
                statements.append(statement)
    return statements


_HANDOFF_EVIDENCE_ID = re.compile(
    r"(?<![A-Z0-9_-])(?:SEARCH|RUN|GRAPH)-\d+(?![A-Z0-9_-])"
)


def _referenced_handoff_evidence_ids(value: Any) -> List[str]:
    """Return evidence IDs in the order the final report refers to them."""
    ordered: List[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            ordered.extend(match.group(0) for match in _HANDOFF_EVIDENCE_ID.finditer(item))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "model_dump"):
            visit(item.model_dump())

    visit(value)
    return list(dict.fromkeys(ordered))


def _cached_intent_scout_probe_rejection(
    result: IntentDialogueResult,
) -> str:
    for index, check in enumerate(
        result.intent_contract.acceptance_checks, start=1
    ):
        rejection = frozen_probe_command_rejection(check.command, check.files)
        if rejection:
            return f"ACCEPT-{index}: {rejection}"
    return ""


def _handoff_observation(item) -> str:
    """Budget graph records as whole JSON, sharing the observation budget."""
    if not item.id.startswith("GRAPH-"):
        return item.observation[:6000]
    try:
        graph = json.loads(item.observation)
    except json.JSONDecodeError:
        return item.observation[:6000]
    if len(item.observation) <= 6000:
        return item.observation
    # Long C++ USRs recur in every edge. Preserve their exact identity once,
    # leaving the source excerpts and relation directions intact.
    ids = {}

    def compact(value, key=""):
        if isinstance(value, dict):
            return {name: compact(child, name) for name, child in value.items()}
        if isinstance(value, list):
            return [compact(child, key) for child in value]
        if isinstance(value, str) and key in {"id", "func_id", "seed_id", "source", "target"}:
            if value not in ids:
                ids[value] = f"E{len(ids) + 1}"
            return ids[value]
        return value

    entities = compact(graph)
    observation = json.dumps({
        "entities": entities,
        "entity_ids": {alias: original for original, alias in ids.items()},
    }, ensure_ascii=False, separators=(",", ":"))
    if len(observation) > 60000:
        return json.dumps({
            "evidence_id": item.id,
            "observation_omitted": "Graph exceeds the shared 60000-character budget; request a narrower query before claiming source coverage.",
            "original_chars": len(item.observation),
        })
    return observation


_GENERIC_GRAPH_SYMBOL_WORDS = frozenset({
    "array", "basic_string_view", "demand", "document", "format", "include",
    "iterate", "json", "object", "parse", "parser", "pointer", "source",
    "string_view", "test", "value",
})


def _distinctive_issue_graph_symbols(issue_evidence: IssueEvidenceFrame) -> List[str]:
    """Return explicit Issue symbols precise enough to admit extra graph evidence."""
    symbols: List[str] = []
    for raw_symbol in issue_evidence.symbols:
        if "/" in raw_symbol or raw_symbol.lower().endswith((".h", ".hh", ".hpp", ".md")):
            continue
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw_symbol):
            if token.lower() in _GENERIC_GRAPH_SYMBOL_WORDS or len(token) < 5:
                continue
            has_case_boundary = any(char.islower() for char in token) and any(
                char.isupper() for char in token
            )
            if "_" in token or has_case_boundary:
                symbols.append(token.lower())
    return list(dict.fromkeys(symbols))


def _issue_relevant_graph_evidence_ids(
    graph_evidence_ids: List[str],
    observations: Dict[str, str],
    issue_evidence: IssueEvidenceFrame,
) -> List[str]:
    """Select graphs with a verified entity named by an explicit Issue symbol."""
    symbols = _distinctive_issue_graph_symbols(issue_evidence)
    if not symbols:
        return []

    def graph_matches(raw_observation: str) -> bool:
        try:
            payload = json.loads(raw_observation)
        except json.JSONDecodeError:
            return False
        roots = payload.get("entities", []) if isinstance(payload, dict) else payload
        pending = list(roots) if isinstance(roots, list) else []
        while pending:
            entity = pending.pop()
            if not isinstance(entity, dict):
                continue
            members = entity.get("members")
            if isinstance(members, list):
                pending.extend(members)
            identity = " ".join(
                str(entity.get(field) or "")
                for field in (
                    "fq_name", "display_name", "name", "short_name",
                    "feature_name",
                )
            ).lower()
            if (entity.get("source_status") == "verified"
                    and any(symbol in identity for symbol in symbols)):
                return True
        return False

    return [
        evidence_id for evidence_id in graph_evidence_ids
        if graph_matches(observations[evidence_id])
    ]


def _memory_from_intent_dialogue_result(
    *,
    instance_name: str,
    problem_statement: str,
    issue_evidence: IssueEvidenceFrame,
    result: IntentDialogueResult | None,
    output_dir: str | None = None,
) -> Memory:
    repair_summary = result.render_for_repair() if result is not None else ""
    contract = result.intent_contract.model_dump() if result is not None else {}
    poc_mode = (
        getattr(getattr(result, "exploration_plan", None), "mode", None) == "poc"
    )
    evidence_bundle: Dict[str, Any] = {}
    if result is not None and result.investigation is not None:
        investigation = result.investigation
        report = investigation.report
        report_records = list(getattr(investigation, "report_history", []))
        reports = [item.report for item in report_records] or [report]
        repository_facts = list(dict.fromkeys(
            fact
            for item in reports
            for fact in item.repository_facts
        ))
        open_decisions = list(dict.fromkeys(
            decision
            for item in reports
            for decision in item.open_decisions
        ))
        input_path_checks = [
            item.input_path_check
            for item in reports
            if item.input_path_check is not None
        ]
        candidate_boundaries = []
        seen_boundaries = set()
        for boundary in [
            *(boundary for item in reports for boundary in item.candidate_boundaries),
            *result.intent_contract.candidate_boundaries,
        ]:
            boundary_key = (
                boundary.anchor_operation,
                tuple(boundary.issue_intent_ids),
                tuple(boundary.candidate_public_forms),
            )
            if boundary_key in seen_boundaries:
                continue
            seen_boundaries.add(boundary_key)
            candidate_boundaries.append(boundary)
        command_actions = {
            action["evidence_id"]: action
            for action in getattr(investigation, "actions", [])
            if action.get("action") == "RunInvestigationCommand"
            and action.get("evidence_id")
        }
        failed_ids = {key for key, action in command_actions.items()
                      if action.get("exit_code") not in (None, 0)}
        repository_evidence = list(investigation.repository_evidence)
        evidence_by_id = {item.id: item for item in repository_evidence}
        observations = {item.id: _handoff_observation(item)
                        for item in repository_evidence}
        fact_evidence_ids = _referenced_handoff_evidence_ids(repository_facts)
        premise_ids = {
            item["evidence_id"] for item in getattr(investigation, "actions", [])
            if item.get("action") == "RunInvestigationCommand"
            and item.get("evidence_role") in {"premise", "control"}
        }
        path_evidence_ids = set()
        path_evidence_order = []
        path_history = []
        for action in getattr(investigation, "actions", []):
            if action.get("live_path"):
                path_history.append({
                    key: value for key, value in action["live_path"].items()
                    if key != "checkpoint_decision"
                })
            if action.get("action") == "SubmitInvestigationReport":
                path_history.extend(action.get("report", {}).get("investigation_paths", []))
        path_history.extend(
            path.model_dump()
            for item in reports
            for path in getattr(item, "investigation_paths", [])
        )
        merged_paths = {}
        for raw_path in path_history:
            path = InvestigationPath.model_validate(raw_path)
            key = path.path_id or "question:" + " ".join(path.question.lower().split())
            prior = merged_paths.get(key)
            if prior is not None:
                path = path.model_copy(update={
                    "evidence_ids": list(dict.fromkeys(
                        prior.evidence_ids + path.evidence_ids
                    ))[:6],
                })
            merged_paths[key] = path
        investigation_paths = list(merged_paths.values())[:8]
        for path in investigation_paths:
            path_evidence_order.extend(path.evidence_ids)
            if path.source_evidence_id:
                path_evidence_order.append(path.source_evidence_id)
        input_path_evidence_ids = []
        for input_path_check in input_path_checks:
            input_path_evidence_ids.append(
                input_path_check.condition_evidence_id
            )
            if input_path_check.verification_run_id:
                input_path_evidence_ids.append(
                    input_path_check.verification_run_id
                )
        path_evidence_order = list(dict.fromkeys(
            input_path_evidence_ids + path_evidence_order
        ))
        path_evidence_ids.update(path_evidence_order)
        boundary_evidence_ids = []
        for boundary in candidate_boundaries:
            boundary_evidence_ids.extend(boundary.evidence_ids)
        boundary_evidence_ids = list(dict.fromkeys(
            [
                evidence_id
                for item in reports
                for evidence_id in item.preserve_evidence_ids
            ]
            + boundary_evidence_ids
        ))
        evidence_order = [item.id for item in repository_evidence]
        ordered_failed_ids = [eid for eid in evidence_order if eid in failed_ids]
        ordered_premise_ids = [eid for eid in evidence_order if eid in premise_ids]
        graph_evidence_ids = [
            eid for eid in evidence_order if eid.startswith("GRAPH-")
        ]
        graph_exit_codes = {
            action["evidence_id"]: action.get("exit_code")
            for action in getattr(investigation, "actions", [])
            if action.get("action") == "RepositoryGraphQuery"
            and action.get("evidence_id")
        }
        successful_graph_ids = [
            eid for eid in graph_evidence_ids
            if graph_exit_codes.get(eid, 0) == 0
        ]
        issue_graph_ids = _issue_relevant_graph_evidence_ids(
            successful_graph_ids,
            observations,
            issue_evidence,
        )
        # Prefer an Issue-linked graph lead. Otherwise retain one recent
        # successful navigation result; failed and broad uncited graph records
        # stay in the Scout audit trail instead of crowding Hypothesis.
        navigation_graph_ids = (
            [] if issue_graph_ids else successful_graph_ids[-1:]
        )
        priority_ids = list(dict.fromkeys(
            fact_evidence_ids
            + path_evidence_order
            + boundary_evidence_ids
            + ordered_failed_ids
            + ordered_premise_ids
            + navigation_graph_ids
            + [
                evidence_id
                for item in reports
                for evidence_id in item.repository_evidence_ids
            ]
        ))
        ranked_evidence = [
            evidence_by_id[eid] for eid in priority_ids if eid in evidence_by_id
        ]
        selected_evidence = ranked_evidence
        # Keep source definitions alongside diagnostics: a run-heavy report
        # must not crowd every upstream predicate out of the factual handoff.
        fact_source_ids = [
            eid for eid in fact_evidence_ids if eid.startswith("SEARCH-")
        ]
        source_reserve_ids = list(dict.fromkeys(
            fact_source_ids
            + [item.id for item in ranked_evidence
               if item.id.startswith("SEARCH-")]
        ))[:5]
        source_reserve = [
            evidence_by_id[eid] for eid in source_reserve_ids
            if eid in evidence_by_id
        ]
        protected_ids = (
            path_evidence_ids
            | set(boundary_evidence_ids)
            | set(fact_evidence_ids)
            | set(source_reserve_ids)
        )
        for source in source_reserve:
            if any(item.id == source.id for item in selected_evidence):
                continue
            for index in range(len(selected_evidence) - 1, -1, -1):
                if selected_evidence[index].id not in protected_ids:
                    selected_evidence[index] = source
                    break
        # Admit evidence in report-semantic priority order. A graph can use more
        # than one legacy 6000-character slot, but it no longer jumps ahead of
        # source explicitly cited by the Scout's final facts.
        remaining_chars = 60000
        budgeted_evidence = []
        for item in selected_evidence:
            size = len(observations[item.id])
            if size <= remaining_chars:
                budgeted_evidence.append(item)
                remaining_chars -= size
        selected_evidence = budgeted_evidence
        # A graph record can fall just beyond the legacy ten-record cutoff even
        # while most of the shared character budget is unused. Recover only
        # successful records that contain a distinctive symbol explicitly named
        # by the Issue. This keeps a relevant bootstrap analogue without admitting
        # every broad semantic result or displacing final-fact citations.
        supplemental_graph_ids = []
        selected_ids = {item.id for item in selected_evidence}
        for evidence_id in _issue_relevant_graph_evidence_ids(
            graph_evidence_ids, observations, issue_evidence,
        ):
            if (evidence_id in selected_ids
                    or graph_exit_codes.get(evidence_id, 0) != 0):
                continue
            size = len(observations[evidence_id])
            if size > remaining_chars:
                continue
            selected_evidence.append(evidence_by_id[evidence_id])
            selected_ids.add(evidence_id)
            supplemental_graph_ids.append(evidence_id)
            remaining_chars -= size
        if _env_bool("COTEST_PORTFOLIO_RERANK", False):
            # A Scout's final citations are a priority order, not a recall gate.
            # Use spare space in the existing 10 * 6000 observation budget for
            # raw records it read but omitted from its concluding report.
            selected_ids = {item.id for item in selected_evidence}
            for item in investigation.repository_evidence:
                size = len(observations[item.id])
                if item.id not in selected_ids and size <= remaining_chars:
                    selected_evidence.append(item)
                    selected_ids.add(item.id)
                    remaining_chars -= size
        selected_ids = {item.id for item in selected_evidence}
        surface_leads = []
        for index, statement in enumerate(
            _atomic_repair_surface_statements(repository_facts),
            start=1,
        ):
            lead = {
                "id": f"SURFACE-{index}",
                "kind": "repository_fact",
                "statement": statement,
            }
            evidence_ids = _referenced_handoff_evidence_ids(statement)
            if evidence_ids:
                lead["evidence_ids"] = evidence_ids
                omitted_ids = [
                    eid for eid in evidence_ids if eid not in selected_ids
                ]
                if omitted_ids:
                    lead["omitted_evidence_ids"] = omitted_ids
            surface_leads.append(lead)
        next_surface_index = len(surface_leads) + 1
        surface_leads.extend(
            {
                "id": f"SURFACE-{next_surface_index + index}",
                "kind": "candidate_boundary",
                "statement": (
                    f"{boundary.anchor_operation}: "
                    + "; ".join(boundary.candidate_public_forms)
                ),
                "evidence_ids": list(boundary.evidence_ids),
            }
            for index, boundary in enumerate(candidate_boundaries)
        )
        evidence_bundle = {
            "source": "intent_dialogue_investigation",
            "hypothesis_ready": (
                getattr(result, "hypothesis_ready", None) is True
            ),
            "evidence_sufficiency_review": (
                getattr(result, "evidence_sufficiency_review").model_dump()
                if getattr(result, "evidence_sufficiency_review", None)
                is not None
                else None
            ),
            "evidence_sufficiency_followup_performed": (
                getattr(
                    result,
                    "evidence_sufficiency_followup_performed",
                    False,
                )
            ),
            "repository_facts": repository_facts,
            "input_path_check": report.input_path_check.model_dump() if report.input_path_check else None,
            "input_path_checks": [
                item.model_dump() for item in input_path_checks
            ],
            "investigation_paths": [
                {**path.model_dump(), "omitted_evidence_ids": [
                    eid for eid in dict.fromkeys([*path.evidence_ids, *([path.source_evidence_id] if path.source_evidence_id else [])])
                    if eid not in {item.id for item in selected_evidence}
                ]}
                for path in investigation_paths
            ],
            "candidate_boundaries": [
                item.model_dump() for item in candidate_boundaries
            ],
            "open_decisions": open_decisions,
            "intent_reports": [
                {
                    "focus_intent_ids": list(record.focus_intent_ids),
                    "status": record.report.status,
                    "answer": record.report.answer,
                    "repository_facts": list(record.report.repository_facts),
                    "repository_evidence_ids": list(
                        record.report.repository_evidence_ids
                    ),
                    "input_path_check": (
                        record.report.input_path_check.model_dump()
                        if record.report.input_path_check is not None
                        else None
                    ),
                    "candidate_boundaries": [
                        item.model_dump()
                        for item in record.report.candidate_boundaries
                    ],
                    "open_decisions": list(record.report.open_decisions),
                }
                for record in report_records
            ],
            "intent_evidence_index": [
                {
                    "focus_intent_ids": list(record.focus_intent_ids),
                    "repository_evidence_ids": list(
                        record.report.repository_evidence_ids
                    ),
                }
                for record in report_records
            ],
            "repair_surface_leads": surface_leads[:30],
            "supplemental_graph_evidence_ids": supplemental_graph_ids,
            "omitted_fact_evidence_ids": [
                eid for eid in fact_evidence_ids if eid not in selected_ids
            ],
            "cited_repository_evidence": [
                {
                    "id": item.id,
                    "path": item.path,
                    "locator": item.locator,
                    "observation": observations[item.id],
                    **({"exit_code": command_actions[item.id]["exit_code"]}
                       if item.id in command_actions and "exit_code" in command_actions[item.id]
                       else {}),
                    **({"observation_truncated": True, "original_chars": len(item.observation)}
                       if not item.id.startswith("GRAPH-") and len(item.observation) > 6000
                       else {}),
                }
                for item in selected_evidence
            ],
            "authority": (
                "Scout-authored POCs are diagnostic observations, never "
                "acceptance gates. Repository facts and candidate boundaries "
                "are structural repair leads."
                if poc_mode
                else "Repository facts and candidate boundaries are structural "
                "repair leads. Executed witness variants remain diagnostic "
                "unless the Issue or a public contract makes them normative."
            ),
        }
        snapshots = [action for action in getattr(investigation, "actions", [])
                     if action.get("frozen_probe_files") and action.get("evidence_id")]
        if snapshots:
            if poc_mode:
                snapshot = _select_scout_probe_snapshot(
                    problem_statement, snapshots,
                    {item.id: item.observation for item in repository_evidence},
                    output_dir=output_dir,
                )
            else:
                # Preserve the legacy contrastive contract's explicit witness.
                verification_ids = {
                    item.verification_run_id for item in input_path_checks
                    if item.verification_run_id
                }
                snapshot = next((
                    action
                    for action in reversed(snapshots)
                    if action["evidence_id"] in verification_ids
                ), snapshots[-1])
        else:
            snapshot = None
        if snapshot is not None:
            evidence_bundle["scout_probe_snapshot"] = {
                "evidence_id": snapshot["evidence_id"],
                "command": snapshot.get("command", ""),
                "files": dict(snapshot["frozen_probe_files"]),
            }
            for field in ("exit_code", "evidence_role"):
                if field in snapshot:
                    evidence_bundle["scout_probe_snapshot"][field] = snapshot[field]
            if poc_mode:
                evidence_bundle["scout_probe_snapshot"]["baseline_observation"] = (
                    evidence_by_id[snapshot["evidence_id"]].observation
                    if snapshot["evidence_id"] in evidence_by_id else ""
                )
            status_warning = verification_exit_status_rejection(snapshot.get("command", ""))
            if status_warning:
                evidence_bundle["scout_probe_snapshot"]["exit_status_warning"] = status_warning
    selected_witness = _selected_dialogue_contrastive_witness(result)
    trace_delta = (
        selected_witness.observed_trace_delta
        if selected_witness is not None
        else ""
    )
    witness_valid = selected_witness is not None
    boundary_resolved = bool(
        result is not None
        and result.contrastive_decision is not None
        and result.contrastive_decision.boundary_resolved
    )
    required_target = bool(
        boundary_resolved
        and result is not None
        and result.contrastive_decision is not None
        and result.contrastive_decision.acceptance_authority == "normative"
    )
    return Memory(
        instance_id=instance_name,
        problem_statement=problem_statement,
        reproduction_summary=(
            "Scout collected repository evidence and an optional diagnostic "
            "POC. Any Scout-authored POC is not an acceptance gate."
            if poc_mode
            else
            "Intent Evidence validated a baseline-passing repository anchor "
            "and a baseline-failing Issue-guided target, and resolved the public "
            "behavior boundary represented by the target."
            if boundary_resolved
            else "Intent Evidence validated a local baseline/target witness, "
            "but it did not establish a complete repair boundary. Use it only as "
            "diagnostic evidence and retain the original Issue authority."
            if witness_valid
            else "No validated contrastive target was established; cited "
            "repository contract evidence remains available below."
        ),
        repair_summary=repair_summary,
        reproduction_result=trace_delta if witness_valid else "",
        expected_reproduction_result=(
            "The frozen Issue-guided TARGET-1 check exits with code 0 after repair."
            if required_target
            else ""
        ),
        issue_evidence=issue_evidence.model_dump(),
        repair_intent_bundle=evidence_bundle,
        repair_intent_contract=contract,
    )


def _select_scout_probe_snapshot(
    problem_statement: str, snapshots: list[dict], observations: dict[str, str],
    *, output_dir: str | None = None,
) -> dict | None:
    prompt = build_probe_selection_prompt(problem_statement, snapshots, observations)
    schema = probe_selection_schema([item["evidence_id"] for item in snapshots])
    trace = {"selection_method": "lightweight_llm", "input_prompt": prompt}
    selected = None
    try:
        decision = _run_structured_extraction(
            prompt + "\nSchema:\n" + json.dumps(schema.model_json_schema()),
            schema, max_tokens=1024, glm_reasoning_effort="low",
        )
        if decision is None:
            raise ValueError("No validated Scout POC selection returned")
        trace["decision"] = decision.model_dump()
        if decision.selected_evidence_id is not None:
            selected = next(item for item in snapshots
                            if item["evidence_id"] == decision.selected_evidence_id)
        trace["status"] = "selected" if selected is not None else "no_useful_poc"
    except Exception as error:
        trace.update(status="unavailable", error=type(error).__name__)
        logger.warning(
            "Scout POC selection unavailable (%s); preserving other evidence "
            "and continuing without a guessed snapshot", type(error).__name__,
        )
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "scout_probe_selection.json"), "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
    return selected


def _parse_skip_instances() -> set[str]:
    raw = os.getenv("COTEST_SKIP_INSTANCE_IDS", "")
    if not raw.strip():
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _resolve_jsonl_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def load_instances_from_jsonl(file_path: str) -> Dict[str, Dict[str, Any]]:
    instances: Dict[str, Dict[str, Any]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                instance = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping line due to JSON decode error: {line.strip()[:100]}...")
                continue

            instance_id = instance.get("instance_id")
            if instance_id:
                instances[instance_id] = instance
    return instances


def get_benchmark_instance(instance_id: str) -> Dict[str, Any]:
    jsonl_path = _resolve_jsonl_path(_env_str("COTEST_JSONL", "INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl"))
    instances = load_instances_from_jsonl(jsonl_path)
    instance = instances.get(instance_id)
    if not instance:
        raise ValueError(f"Instance {instance_id} not found in benchmark JSONL: {jsonl_path}")
    return dict(instance)


def parse_instance_string(repo_str: str):
    pattern = r'^([a-zA-Z0-9_.-]+)__([a-zA-Z0-9_.-]+)[-_](\d+)$'
    match = re.match(pattern, repo_str)
    if not match:
        return None
    return {"org": match.group(1), "repo": match.group(2), "number": match.group(3)}
    
def get_repo_structure(repo_path: str) -> str:
    parser = CodeParser()
    analyzer = DirectoryAnalyzer(parser)
    repo_structure = analyzer.analyze(repo_path)
    return show_code_structure(repo_structure)


def _repo_structure_for_model(repo_path: str, model: str) -> str:
    if (
        "deepseek-v4" in model
        or "glm-5.3-flash" in model
        or "qwen3.8-flash" in model
    ):
        return show_cpp_repo_structure(repo_path)
    return get_repo_structure(repo_path)


def _deepseek_v4_extra_body(model: str) -> Dict[str, Any] | None:
    if "deepseek-v4" in model:
        thinking_type = _env_str("COTEST_DEEPSEEK_V4_THINKING", "enabled")
        return {"thinking": {"type": thinking_type}}
    return None


def _glm_53_flash_extra_body(model: str) -> Dict[str, Any] | None:
    if "glm-5.3-flash" not in model:
        return None
    thinking_type = _env_str("COTEST_GLM_THINKING", "enabled")
    reasoning_effort = _env_str("COTEST_GLM_REASONING_EFFORT", "low")
    return {
        "thinking": {
            "type": thinking_type,
            "clear_thinking": False,
        },
        "reasoning_effort": reasoning_effort,
    }


def _qwen_38_flash_extra_body(model: str) -> Dict[str, Any] | None:
    if "qwen3.8-flash" not in model:
        return None
    thinking_enabled = _env_bool("COTEST_QWEN_THINKING", True)
    extra_body: Dict[str, Any] = {
        "enable_thinking": thinking_enabled,
        "preserve_thinking": thinking_enabled,
    }
    if thinking_enabled:
        extra_body["reasoning_effort"] = _env_str(
            "COTEST_QWEN_REASONING_EFFORT",
            "xhigh",
        )
    return extra_body


def _model_extra_body(model: str) -> Dict[str, Any] | None:
    return (
        _deepseek_v4_extra_body(model)
        or _glm_53_flash_extra_body(model)
        or _qwen_38_flash_extra_body(model)
    )


def _model_temperature(model: str, default: float = 0.2) -> float | None:
    if "deepseek-v4" in model:
        return None
    if "glm-5.3-flash" in model:
        return 1.0
    if "qwen3.8-flash" in model:
        return 1.0
    return default


def _model_max_tokens(
    model: str,
    env_key: str,
    default: int = 2000,
    *,
    legacy_env_key: str | None = None,
) -> int:
    if "glm-5.3-flash" in model:
        # GLM uses a model-specific cap so the repository's 128K DeepSeek
        # stage settings do not accidentally enable very long reasoning turns.
        return _env_int("COTEST_GLM_MAX_TOKENS", 32768)
    if "deepseek-v4" in model:
        default = 128000
    if "qwen3.8-flash" in model:
        default = 131072
    selected_key = env_key
    if (
        legacy_env_key
        and os.getenv(env_key) is None
        and os.getenv(legacy_env_key) is not None
    ):
        selected_key = legacy_env_key
    return _env_int(selected_key, default)


def _model_kwargs(
    model: str,
    max_tokens_key: str,
    *,
    legacy_max_tokens_key: str | None = None,
    glm_reasoning_effort: str | None = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "temperature": _model_temperature(model),
        "max_tokens": _model_max_tokens(
            model,
            max_tokens_key,
            legacy_env_key=legacy_max_tokens_key,
        ),
    }
    extra_body = _model_extra_body(model)
    if extra_body and glm_reasoning_effort and "glm-5.3-flash" in model:
        extra_body["reasoning_effort"] = glm_reasoning_effort
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _completion_protocol_kwargs(model: str) -> Dict[str, Any]:
    if (
        "deepseek-v4" in model
        or "glm-5.3-flash" in model
        or "qwen3.8-flash" in model
    ):
        return {
            "response_format": LLMResponseFormat.TOOLS,
            "message_history_type": MessageHistoryType.MESSAGES,
        }
    return {
        "response_format": LLMResponseFormat.REACT,
        "message_history_type": MessageHistoryType.REACT,
    }


def _compact_reproduction_output(output: str, limit: int = 5000) -> str:
    if len(output) <= limit:
        return output
    head = limit // 2
    tail = limit - head
    return (
        output[:head]
        + f"\n... [compiler output compacted; {len(output) - limit} characters omitted] ...\n"
        + output[-tail:]
    )


def _refine_audit_selection_once(raw: str, pool: IntentHypothesisPool, complete) -> HypothesisPoolAudit | None:
    """Restore a selection list only when existing reviews already fix its membership."""
    field = "selected_hypothesis_ids"
    try:
        source = parse_json_object(raw)
        if not isinstance(source, dict) or source.get("decision") != "ACCEPT":
            return None
        reasons = source.get("selection_reasons")
        if not isinstance(reasons, dict):
            return None
        expected = list(reasons)
        previous = source.get(field) or []
        if (not isinstance(previous, list) or not all(isinstance(i, str) for i in previous)
                or len(set(previous)) != len(previous) or set(previous) - set(expected)):
            return None
        if set(previous) == set(expected):
            return None
        # Missing judgments, genuine budget changes and semantic conflicts must
        # go through the original Audit path, never this field-only recovery.
        checked = HypothesisPoolAudit.model_validate({**source, field: expected})
        checked.validate_pool(pool)
        if checked.decision != "ACCEPT":
            return None
    except Exception:
        return None

    prompt = (
        "Restore a missing or incomplete selected_hypothesis_ids field in an existing "
        "Audit JSON object. The selection decision is already made: include every key "
        "of selection_reasons exactly once, and no other ID. Preserve the relative order "
        "of previously listed IDs. Do not rerank policies, add judgments, or change any "
        "other field. Return only JSON with one key: selected_hypothesis_ids.\n"
        + json.dumps({field: previous, "selection_reasons": reasons}, ensure_ascii=False)
    )
    patch = parse_json_object(complete(prompt))
    if not isinstance(patch, dict) or set(patch) != {field}:
        raise ValueError("Audit refine must return only selected_hypothesis_ids")
    selected = patch[field]
    if (not isinstance(selected, list) or not all(isinstance(i, str) for i in selected)
            or len(selected) != len(expected) or set(selected) != set(expected)
            or [i for i in selected if i in previous] != previous):
        raise ValueError("Audit refine changed the existing selection decision or order")
    refined = HypothesisPoolAudit.model_validate({**source, field: selected})
    refined.validate_pool(pool)
    if refined.decision != "ACCEPT":
        raise ValueError("Audit refine cannot bypass a required upstream revision")
    return refined


def _refine_audit_shape_once(raw: str, pool: IntentHypothesisPool, complete) -> HypothesisPoolAudit | None:
    """Repair delimiters or redundant review references without editing judgments."""
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if not text.startswith("{") or len(text) > 24000:
        return None
    try:
        source = json.loads(text)
    except json.JSONDecodeError:
        # Offer only closing delimiters at existing root-field boundaries that
        # already pass the complete schema/pool checks. This keeps the small
        # model away from raw offsets, JSON string contents and new judgments.
        options = []
        for match in re.finditer(r',\s*"([^"\\]+)"\s*:', text):
            before = match[0]
            if match[1] not in HypothesisPoolAudit.model_fields or text.count(before) != 1:
                continue
            for insertion in ("}", "]", "}}", "]}", "}]", "]]"):
                try:
                    candidate = json.loads(text[:match.start()] + insertion + text[match.start():])
                    checked = HypothesisPoolAudit.model_validate(candidate)
                    checked.validate_pool(pool)
                except (ValueError, TypeError):
                    continue
                options.append({"before": before, "insert": insertion})
        if not options:
            return None
        prompt = (
            'Fix a missing closing JSON delimiter in an existing Audit. The following '
            'exact insertion options have passed all local schema and pool checks. '
            'Choose one option verbatim; return only its before and insert fields. '
            'Preserve every character of both values, including leading commas and quotes. '
            'Do not change judgments or add other fields.\n' + json.dumps(options)
        )
        patch = parse_json_object(complete(prompt))
        if not isinstance(patch, dict) or patch not in options:
            raise ValueError("Audit syntax refine must return one delimiter insertion")
        before, insertion = patch["before"], patch["insert"]
        if (not isinstance(before, str) or not before or text.count(before) != 1
                or not isinstance(insertion, str) or not 1 <= len(insertion) <= 2
                or any(c not in "}]" for c in insertion)):
            raise ValueError("Audit syntax refine cannot rewrite existing content")
        quoted = escaped = False
        for char in text[:text.index(before)]:
            if escaped:
                escaped = False
            elif char == "\\" and quoted:
                escaped = True
            elif char == '"':
                quoted = not quoted
        if quoted:
            raise ValueError("Audit syntax refine cannot insert into a judgment string")
        source = json.loads(text.replace(before, insertion + before, 1))
    else:
        if not isinstance(source, dict) or not isinstance(source.get("hypothesis_reviews"), dict):
            return None
        reviews = source["hypothesis_reviews"]
        ids = {h.id for h in pool.hypotheses}
        notes = {k: v for k, v in reviews.items() if k not in ids}
        # Only redundant references are safe to move without a new semantic
        # review. Arbitrary annotations may describe a real blocker.
        if not notes or any(k != k.removesuffix("-note") + "-note"
                            or k.removesuffix("-note") not in ids
                            or v != f"See {k.removesuffix('-note')} entry above."
                            for k, v in notes.items()):
            return None
        source = {**source, "hypothesis_reviews": {k: v for k, v in reviews.items() if k in ids},
                  "nonblocking_notes": [*source.get("nonblocking_notes", []),
                                        *(f"{k}: {v}" for k, v in notes.items())]}
        checked = HypothesisPoolAudit.model_validate(source)
        checked.validate_pool(pool)
        patch = parse_json_object(complete(
            'Relocate these redundant references from hypothesis_reviews to '
            'nonblocking_notes. Do not classify or alter judgments. Return only '
            '{"move_review_notes":[all the provided keys]}.\n' + json.dumps(notes)))
        if (not isinstance(patch, dict) or set(patch) != {"move_review_notes"}
                or not isinstance(patch["move_review_notes"], list)
                or sorted(patch["move_review_notes"]) != sorted(notes)):
            raise ValueError("Audit note refine must preserve every reference")
    refined = HypothesisPoolAudit.model_validate(source)
    refined.validate_pool(pool)
    return refined


_INVALID_WITNESS_FIELDS = {
    "selected_anchor_id": "",
    "selected_target_id": "",
    "boundary_resolved": False,
    "witnessed_delta": "",
    "supported_delta": "",
    "covered_issue_claim_ids": [],
    "covered_issue_intent_ids": [],
    "unresolved_issue_claim_ids": [],
    "unresolved_issue_intent_ids": [],
    "scope_evidence": [],
    "uncertainty": [],
}


def _contrastive_protocol_refinement_plan(
    source: Any,
    error: ValidationError,
) -> Dict[str, Any] | None:
    """Identify protocol-only repairs that can be applied conservatively."""
    if not isinstance(source, dict):
        return None
    drop_effects: set[int] = set()
    allowed_next_actions: list[str] = []
    for item in error.errors(include_url=False):
        location = tuple(item.get("loc") or ())
        message = str(item.get("msg") or "")
        if (
            item.get("type") == "value_error"
            and not location
            and "test-pair refinement requires a valid local witness" in message
            and source.get("witness_valid") is False
            and source.get("next_action") == "refine_test_pair"
        ):
            allowed_next_actions = ["reject_witness", "leave_open"]
            continue
        if (
            item.get("type") == "string_pattern_mismatch"
            and len(location) == 5
            and location[0] == "repository_effects"
            and isinstance(location[1], int)
            and location[2] == "evidence"
            and isinstance(location[3], int)
            and location[4] == "evidence_id"
        ):
            drop_effects.add(location[1])
            continue
        if (
            item.get("type") in {"missing", "string_too_short"}
            and len(location) == 5
            and location[0] == "repository_effects"
            and isinstance(location[1], int)
            and location[2] == "evidence"
            and isinstance(location[3], int)
            and location[4] == "quote"
        ):
            drop_effects.add(location[1])
            continue
        return None
    if not allowed_next_actions and not drop_effects:
        return None
    return {
        "allowed_next_actions": allowed_next_actions,
        "drop_repository_effect_indexes": sorted(drop_effects),
        "clear_invalid_witness_fields": source.get("witness_valid") is False,
    }


def _refine_contrastive_protocol_once(
    raw: str,
    error: ValidationError,
    complete,
    *,
    plan: Dict[str, Any] | None = None,
) -> ContrastiveIntentDecision | None:
    """Repair protocol fields without regenerating the evidence judgment."""
    source = parse_json_object(raw)
    plan = plan or _contrastive_protocol_refinement_plan(source, error)
    if plan is None:
        return None
    allowed_actions = plan["allowed_next_actions"]
    required_drops = plan["drop_repository_effect_indexes"]
    prompt = (
        "Repair only the protocol consistency of an existing contrastive intent decision. "
        "Do not add evidence, replace evidence IDs, rewrite behavioral claims, or change "
        "witness_valid. Repository-contract effects with missing citations, empty quotes, or "
        "non-SEARCH-N evidence IDs will be removed mechanically; never fabricate or relabel "
        "citation evidence. Return exactly one compact JSON "
        "object with the key next_action. Choose one of the allowed actions when that list is "
        "non-empty; otherwise return null so the existing action remains unchanged.\n"
        + json.dumps(
            {
                "allowed_next_actions": allowed_actions,
                "mechanical_drop_repository_effect_indexes": required_drops,
                "validation_errors": [
                    {
                        "type": item.get("type"),
                        "location": list(item.get("loc") or ()),
                        "message": item.get("msg"),
                    }
                    for item in error.errors(include_url=False)
                ],
                "decision": source,
            },
            ensure_ascii=False,
        )
    )
    try:
        patch = parse_json_object(complete(prompt))
    except Exception:
        patch = {}
    if not isinstance(patch, dict):
        patch = {}
    requested_action = patch.get("next_action")
    if allowed_actions:
        # `leave_open` is the conservative protocol closure when the lightweight
        # model does not return one of the two authorized semantic choices.
        next_action = (
            requested_action
            if requested_action in allowed_actions
            else "leave_open"
        )
        action_source = (
            "lightweight_llm"
            if requested_action in allowed_actions
            else "conservative_leave_open"
        )
    else:
        # Evidence-ID cleanup is already determined by validation locations.
        # Do not let the model change an otherwise valid action.
        next_action = None
        action_source = "preserved"
    logger.info(
        "Contrastive protocol patch selected: action_source=%s "
        "mechanical_effect_drops=%d",
        action_source,
        len(required_drops),
    )

    refined = json.loads(json.dumps(source, ensure_ascii=False))
    if next_action is not None:
        refined["next_action"] = next_action
        refined["followup_request"] = None
    for index in reversed(required_drops):
        effects = refined.get("repository_effects")
        if not isinstance(effects, list) or index >= len(effects):
            raise ValueError("Contrastive protocol refine referenced a missing effect")
        del effects[index]
    if plan["clear_invalid_witness_fields"]:
        for field, empty_value in _INVALID_WITNESS_FIELDS.items():
            refined[field] = empty_value
    try:
        return ContrastiveIntentDecision.model_validate(refined)
    except ValidationError as post_error:
        # A nested repository citation error can prevent Pydantic's model-level
        # validator from exposing an invalid-witness transition on the first
        # pass. Close that newly visible protocol defect locally, without a
        # second model call.
        followup_plan = _contrastive_protocol_refinement_plan(refined, post_error)
        if followup_plan is None:
            raise
        if followup_plan["allowed_next_actions"]:
            refined["next_action"] = "leave_open"
            refined["followup_request"] = None
        for index in reversed(followup_plan["drop_repository_effect_indexes"]):
            effects = refined.get("repository_effects")
            if not isinstance(effects, list) or index >= len(effects):
                raise ValueError("Contrastive protocol refine referenced a missing effect")
            del effects[index]
        if followup_plan["clear_invalid_witness_fields"]:
            for field, empty_value in _INVALID_WITNESS_FIELDS.items():
                refined[field] = empty_value
        return ContrastiveIntentDecision.model_validate(refined)


def _run_structured_extraction(
    prompt: str,
    schema: Type,
    *,
    model_override: str | None = None,
    max_tokens: int | None = None,
    max_tokens_key: str = "COTEST_EXTRACTION_MAX_TOKENS",
    legacy_max_tokens_key: str | None = None,
    glm_reasoning_effort: str | None = None,
    audit_pool: IntentHypothesisPool | None = None,
    audit_tools: HypothesisEvidenceTools | None = None,
    max_attempts: int = 2,
) -> Any:
    _load_env()
    extraction_model = model_override or _env_str(
        "COTEST_EXTRACTION_MODEL",
        "openrouter/deepseek/deepseek-v3.2",
    )
    provider_kwargs: Dict[str, Any] = {}
    configured_extra_body: Dict[str, Any] | None = None
    compact_structured_output = False
    try:
        extraction_config = get_model_config(extraction_model).copy()
        extraction_model = extraction_config["model"]
        if extraction_config.get("model_base_url"):
            provider_kwargs["api_base"] = extraction_config["model_base_url"]
        api_key = extraction_config.get("model_api_key")
        api_key_env = extraction_config.get("model_api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(api_key_env)
        if not api_key and extraction_config.get("model_base_url"):
            api_key = os.getenv("CUSTOM_LLM_API_KEY")
        if api_key:
            provider_kwargs["api_key"] = api_key
        configured_extra_body = extraction_config.get("extra_body")
        compact_structured_output = bool(
            extraction_config.get("compact_structured_output", False)
        )
    except ValueError:
        # Provider-qualified model names do not need a project alias.
        pass
    base_max_tokens = (
        max_tokens
        if max_tokens is not None
        else _model_max_tokens(
            extraction_model,
            max_tokens_key,
            4000,
            legacy_env_key=legacy_max_tokens_key,
        )
    )
    extraction_kwargs: Dict[str, Any] = {
        "model": extraction_model,
        "response_format": {"type": "json_object"},
        "max_tokens": base_max_tokens,
        **provider_kwargs,
    }
    if "glm-5.3-flash" in extraction_model:
        # This endpoint's JSON mode removes literal "json" inside identifiers
        # (e.g. from_json -> from_). Text mode preserves bytes; the same local
        # JSON/schema parser and bounded retries still validate the response.
        extraction_kwargs.pop("response_format")
    # Environment switches override the alias default for known thinking
    # models; other providers retain their configured request body.
    extra_body = _model_extra_body(extraction_model) or configured_extra_body
    if glm_reasoning_effort and "glm-5.3-flash" in extraction_model:
        extra_body = dict(extra_body or {})
        extra_body["thinking"] = {
            "type": "enabled",
            "clear_thinking": False,
        }
        extra_body["reasoning_effort"] = glm_reasoning_effort
    if extra_body:
        extraction_kwargs["extra_body"] = extra_body
    if schema is BranchCorrectionRefinement:
        extraction_kwargs.update(timeout=180.0, max_retries=0)
    last_error = ""
    previous_audit_payload = ""
    audit_refine_used = False
    for attempt in range(max_attempts):
        try:
            attempt_kwargs = dict(extraction_kwargs)
            if attempt:
                attempt_kwargs["max_tokens"] = min(
                    base_max_tokens * 2,
                    128000,
                )
            llm = LiteLLM(**attempt_kwargs)
            attempt_prompt = prompt
            if compact_structured_output:
                attempt_prompt += (
                    "\nUse internal reasoning for analysis, but keep the returned "
                    "JSON as a compact transport artifact: output only schema "
                    "fields, use exact evidence/object IDs instead of restating "
                    "observations, keep bounded strings well below their limits, "
                    "and rank list items before respecting their declared maxima."
                )
            if attempt:
                attempt_prompt += (
                    "\nYour previous response did not match the required JSON schema. "
                    "Correct the specific validation failure below and return one "
                    "compact JSON object with all required fields. Escape quotes, "
                    "backslashes, and "
                    "newlines. Do not use markdown fences or copy multiline code into "
                    f"string fields.\nValidation failure: {last_error[:2000]}"
                )
                if previous_audit_payload:
                    attempt_prompt += (
                        "\nPrevious Audit submission (unvalidated data):\n"
                        + previous_audit_payload
                        + "\nCorrect the identified fields and resubmit the complete audit. "
                        "Reuse unaffected reviews and evidence; do not replace missing "
                        "judgments with assumed support. Preserve concrete contradictions "
                        "unless new cited evidence corrects them. The pool and its original "
                        "scenarios remain authoritative, not this failed submission."
                    )
            if schema is HypothesisPoolAudit and (
                "glm-5.3-flash" in extraction_model or audit_tools is not None
            ):
                # Native tool arguments avoid malformed free-text JSON in long
                # audits without enabling the provider's corrupting JSON mode.
                import litellm

                tool_schema = schema.model_json_schema()
                if audit_pool is not None:
                    for field, ids in {
                        "hypothesis_reviews": [h.id for h in audit_pool.hypotheses],
                        "discriminator_reviews": [d.id for d in audit_pool.unresolved_dimensions
                                                  if d.disposition == "branched"],
                    }.items():
                        mapping = tool_schema["properties"][field]
                        value_schema = mapping["additionalProperties"]
                        mapping.update(properties={i: value_schema for i in ids},
                                       required=ids, additionalProperties=False)
                        if ids and field not in tool_schema.setdefault("required", []):
                            tool_schema["required"].append(field)
                    size = len(audit_pool.hypotheses)
                    tool_schema.setdefault("allOf", []).append({
                        "if": {"properties": {"decision": {"const": "ACCEPT"}}},
                        "then": {
                            "required": [
                                "selected_hypothesis_ids",
                                "selection_reasons",
                                "unselected_reasons",
                                "joint_policy_losses",
                            ],
                            "properties": {
                                "selected_hypothesis_ids": {
                                    "minItems": 1, "maxItems": min(3, size),
                                    "uniqueItems": True,
                                    "items": {"enum": [h.id for h in audit_pool.hypotheses]},
                                },
                            },
                        },
                    })
                if audit_tools is not None:
                    if _env_bool("COTEST_PORTFOLIO_AUDIT_STREAM", False):
                        attempt_kwargs["stream"] = True
                    raw_response_message = audit_tools.complete(
                        attempt_kwargs, attempt_prompt, tool_schema
                    )
                else:
                    audit_response = litellm.completion(
                        **attempt_kwargs,
                        messages=[{"role": "user", "content": attempt_prompt +
                                   "\nReturn the entire audit by calling SubmitHypothesisAudit exactly once."}],
                        tools=[{"type": "function", "function": {
                            "name": "SubmitHypothesisAudit",
                            "description": "Submit the complete hypothesis pool review and budget assignment.",
                            "parameters": tool_schema,
                        }}],
                        tool_choice="auto",
                        timeout=600.0,
                        max_retries=0,
                    )
                    message = audit_response.choices[0].message
                    calls = message.tool_calls or []
                    if not calls:
                        # Text fallback must pass the identical schema/pool checks.
                        raw_response_message = message.content or ""
                    elif len(calls) != 1 or calls[0].function.name != "SubmitHypothesisAudit":
                        raise ValueError("Expected one SubmitHypothesisAudit tool call")
                    else:
                        raw_response_message = calls[0].function.arguments
            else:
                if schema is BranchCorrectionRefinement:
                    # Retry rejected requests without consuming a JSON/Repair correction turn.
                    response = Retrying(
                        retry=retry_if_exception_type(RateLimitError),
                        wait=wait_exponential_jitter(initial=5, max=60, jitter=1),
                        stop=stop_after_attempt(5),
                        reraise=True,
                        before_sleep=lambda state: logger.warning(
                            "Branch correction refine HTTP 429; retry %d/4 in %.1f seconds",
                            state.attempt_number, state.next_action.sleep,
                        ),
                    )(llm.generate, attempt_prompt)
                else:
                    response = llm.generate(attempt_prompt)
                raw_response_message = response.message if response else ""
            if not raw_response_message or not raw_response_message.strip():
                raise ValueError("LLM returned an empty structured response")
            if schema is HypothesisPoolAudit:
                previous_audit_payload = raw_response_message
            try:
                parsed = parse_structured_response(raw_response_message, schema)
                if audit_pool is not None:
                    parsed.validate_pool(audit_pool)
                return parsed
            except (ValidationError, ValueError) as validation_error:
                if (
                    isinstance(validation_error, ValidationError)
                    and schema is ContrastiveIntentDecision
                ):
                    source = parse_json_object(raw_response_message)
                    protocol_plan = _contrastive_protocol_refinement_plan(
                        source,
                        validation_error,
                    )
                    if protocol_plan is not None:
                        refine_kwargs = dict(extraction_kwargs)
                        refine_kwargs.update(
                            max_tokens=2048,
                            max_retries=0,
                            timeout=60,
                        )
                        if "glm-5.3-flash" in extraction_model:
                            refine_kwargs["extra_body"] = {
                                "thinking": {
                                    "type": "enabled",
                                    "clear_thinking": False,
                                },
                                "reasoning_effort": "low",
                            }

                        def _complete_contrastive_protocol_refine(
                            refine_prompt: str,
                        ) -> str:
                            response = LiteLLM(**refine_kwargs).generate(
                                refine_prompt
                            )
                            return response.message if response else ""

                        try:
                            refined = _refine_contrastive_protocol_once(
                                raw_response_message,
                                validation_error,
                                _complete_contrastive_protocol_refine,
                                plan=protocol_plan,
                            )
                            if refined is not None:
                                logger.info(
                                    "ContrastiveIntentDecision protocol restored "
                                    "in one lightweight LLM turn"
                                )
                                return refined
                        except Exception as refine_error:
                            logger.error(
                                "Lightweight contrastive protocol refine failed: %s",
                                type(refine_error).__name__,
                            )
                        # A recognized protocol defect must not trigger another
                        # full adjudicator generation. An unavailable patch stays
                        # explicitly unresolved.
                        return None
                if schema is HypothesisPoolAudit and audit_pool is not None and not audit_refine_used:
                    def _complete_audit_refine(refine_prompt: str) -> str:
                        nonlocal audit_refine_used
                        audit_refine_used = True
                        refine_kwargs = dict(extraction_kwargs)
                        refine_kwargs.update(max_tokens=2048, max_retries=0, timeout=60)
                        if "glm-5.3-flash" in extraction_model:
                            # This endpoint requires thinking even for a field patch.
                            refine_kwargs["extra_body"] = {
                                "thinking": {"type": "enabled", "clear_thinking": False},
                                "reasoning_effort": "low",
                            }
                        response = LiteLLM(**refine_kwargs).generate(refine_prompt)
                        return response.message if response else ""

                    try:
                        refined = _refine_audit_selection_once(
                            raw_response_message, audit_pool, _complete_audit_refine)
                        if refined is None and not audit_refine_used:
                            refined = _refine_audit_shape_once(
                                raw_response_message, audit_pool, _complete_audit_refine)
                        if refined is not None:
                            logger.info(
                                "HypothesisPoolAudit structure restored in one lightweight LLM turn"
                            )
                            return refined
                    except Exception as refine_error:
                        logger.warning("Lightweight Audit structure refine failed: %s", type(refine_error).__name__)
                if not (
                    isinstance(validation_error, ValidationError)
                    and schema is not BranchCorrectionRefinement
                    and compact_structured_output
                    and attempt == 0
                    and is_compactable_validation_error(validation_error)
                ):
                    raise
                source_validation_error = validation_error

                compactor_kwargs = dict(extraction_kwargs)
                compactor_kwargs["max_tokens"] = min(
                    max(base_max_tokens, 2048),
                    8192,
                )
                compactor_kwargs["max_retries"] = 0
                if "glm-5.3-flash" in extraction_model:
                    compactor_kwargs["timeout"] = 600.0
                    compactor_kwargs["extra_body"] = {
                        "thinking": {
                            "type": "enabled",
                            "clear_thinking": False,
                        },
                        "reasoning_effort": _env_str(
                            "COTEST_GLM_REASONING_EFFORT", "low"
                        ),
                    }

                def _complete_compaction(compaction_prompt: str) -> str:
                    compactor = LiteLLM(**compactor_kwargs)
                    compacted_response = compactor.generate(compaction_prompt)
                    return compacted_response.message if compacted_response else ""

                try:
                    validated, audit = compact_structured_payload_once(
                        raw_response_message,
                        schema,
                        _complete_compaction,
                    )
                except (StructuredCompactionError, ValidationError, ValueError) as compact_error:
                    logger.warning(
                        "Single-turn semantic compaction failed for %s: %s",
                        schema.__name__,
                        compact_error,
                    )
                    raise source_validation_error
                logger.info(
                    "Semantically compacted %s in one isolated LLM turn: "
                    "paths=%s preserved_ids=%d",
                    schema.__name__,
                    ",".join(audit.mutable_paths),
                    len(audit.reference_ids),
                )
                if audit_pool is not None:
                    validated.validate_pool(audit_pool)
                return validated
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            log = logger.warning if attempt == 0 else logger.error
            log(f"Structured extraction attempt {attempt + 1} failed for {schema.__name__}: {e}")
    return None


def _run_intent_dialogue_extraction(prompt: str, schema: Type) -> Any:
    return _run_structured_extraction(
        prompt,
        schema,
        max_tokens_key="COTEST_INTENT_STRUCTURED_GENERATION_MAX_TOKENS",
        legacy_max_tokens_key="COTEST_INTENT_DIALOGUE_MAX_TOKENS",
    )


def _run_intent_evidence_review(prompt: str, schema: Type) -> Any:
    return _run_structured_extraction(
        prompt,
        schema,
        max_tokens_key="COTEST_INTENT_STRUCTURED_GENERATION_MAX_TOKENS",
        legacy_max_tokens_key="COTEST_INTENT_DIALOGUE_MAX_TOKENS",
        glm_reasoning_effort=_intent_evidence_review_reasoning_effort(),
    )


def _extract_issue_evidence(problem_statement: str) -> IssueEvidenceFrame:
    frame = _run_structured_extraction(
        build_issue_evidence_prompt(problem_statement),
        IssueEvidenceFrame,
    )
    merged_frame = merge_issue_evidence_with_fallback(problem_statement, frame)
    if not any(
        claim.kind == "expected_behavior"
        for claim in merged_frame.claims
    ):
        logger.warning(
            "Issue evidence has no grounded expected behavior; retrying "
            "target extraction from exact issue spans."
        )
        target_retry_prompt = build_issue_evidence_prompt(problem_statement) + (
            "\nThe previous extraction produced no grounded expected_behavior. "
            "Retry the complete extraction, paying special attention to the issue "
            "title, requested-feature bullet lists, and expected-result sections. "
            "A noun-phrase bullet under wording such as 'lacking', 'wanted', "
            "'should support', or 'would be great' can state a requested behavior. "
            "For every expected_behavior, copy issue_span character-for-character "
            "from one physical line of the Issue. Do not infer a target from ordinary "
            "observations or implementation suggestions."
        )
        recovered_frame = _run_structured_extraction(
            target_retry_prompt,
            IssueEvidenceFrame,
        )
        recovered_merged = merge_issue_evidence_with_fallback(
            problem_statement,
            recovered_frame,
        )
        if any(
            claim.kind == "expected_behavior"
            for claim in recovered_merged.claims
        ):
            merged_frame = recovered_merged
            logger.info("Recovered grounded expected behavior on extraction retry.")
        else:
            logger.warning(
                "Issue target extraction retry still produced no grounded "
                "expected behavior."
            )
    if frame is not None and merged_frame.extraction_status == "fallback":
        logger.warning("Issue evidence failed ID/span validation; using the original issue fallback.")
    logger.info(
        "Issue intent extraction completed: claims=%d atomic_intents=%d targets=%d preserves=%d status=%s",
        len(merged_frame.claims),
        len(merged_frame.intent_units),
        sum(unit.role == "target" for unit in merged_frame.intent_units),
        sum(unit.role == "preserve" for unit in merged_frame.intent_units),
        merged_frame.extraction_status,
    )
    return merged_frame


def _adjudicate_repair_intent(problem_statement: str, issue_evidence: IssueEvidenceFrame, repro_result) -> RepairIntentBundle:
    prompt = build_intent_adjudication_prompt(
        issue=problem_statement,
        issue_evidence=issue_evidence,
        repository_evidence=repro_result.repository_evidence,
        execution_records=repro_result.execution_records,
        tests=repro_result.test_artifacts,
        test_files=repro_result.test_files,
        graph_bindings=[item.model_dump() for item in repro_result.graph_bindings],
    )
    bundle = _run_structured_extraction(prompt, RepairIntentBundle)
    if bundle is None:
        bundle = unresolved_bundle(
            "Grounded intent adjudication failed.",
            execution_records=repro_result.execution_records,
            tests=repro_result.test_artifacts,
        )
    bundle = normalize_repair_intent_bundle(bundle)
    bundle.execution_records = repro_result.execution_records
    bundle.tests = repro_result.test_artifacts
    reported_bindings = {
        (item.entity, item.file, item.relation)
        for item in repro_result.graph_bindings
    }
    bundle.graph_bindings = [
        item for item in bundle.graph_bindings
        if (item.entity, item.file, item.relation) in reported_bindings
    ]
    return bundle

def run_reproduction_step(
    instance_name: str,
    model_name: str,
    docker_env: DockerEnv,
    output_dir: str,
    *,
    run_intent_analysis: bool = True,
) -> Memory | None:
    """
    Runs reproduction and the optional dialogue, returning available evidence.
    """
    logger.info(f"--- Starting Reproduction Step for {instance_name} ---")
    
    instance = get_benchmark_instance(instance_name)
    repository = create_repository(instance)
    dialogue_enabled = (
        _intent_grounding_enabled() if run_intent_analysis else False
    )
    intent_enabled = (
        _env_bool("COTEST_ENABLE_INTENT", True)
        and not dialogue_enabled
        and run_intent_analysis
    )
    issue_evidence = (
        _extract_issue_evidence(instance["problem_statement"])
        if intent_enabled or dialogue_enabled
        else None
    )
    if dialogue_enabled:
        if _test_derived_intent_enabled():
            logger.info(
                "Intent pipeline: test-derived contrastive evidence enabled "
                "(shared evidence-action budget=%d, reproducer fallback=%s)",
                _intent_evidence_steps(),
                _reproducer_fallback_enabled(),
            )
        else:
            logger.info(
                "Intent pipeline: legacy repository dialogue enabled "
                "(max_tasks=%d, shared action budget=%d)",
                _env_int(
                    "COTEST_INTENT_MAX_TASKS",
                    _env_int(
                        "COTEST_INTENT_EXPLORATION_MAX_OBJECTIVES",
                        _env_int("COTEST_INTENT_DIALOGUE_MAX_ROUNDS", 10),
                    ),
                ),
                _intent_evidence_steps(),
            )
    else:
        logger.info(
            "Intent pipeline: %s",
            "legacy enabled" if intent_enabled else "disabled",
        )

    index_store_dir = os.getenv("INDEX_STORE_DIR")
    if not index_store_dir:
        raise ValueError("INDEX_STORE_DIR environment variable is not set.")
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    code_index = CodeIndex.from_index_name(
        index_name=instance_name,
        file_repo=repository,
        index_store_dir=index_store_dir,
    )

    os.makedirs(output_dir, exist_ok=True)
    if issue_evidence is not None:
        issue_evidence_path = os.path.join(output_dir, "issue_evidence.json")
        with open(issue_evidence_path, "w", encoding="utf-8") as f:
            json.dump(issue_evidence.model_dump(), f, indent=2, ensure_ascii=False)

    dialogue_prompt = ""
    dialogue_contract: Dict[str, Any] = {}
    with docker_env:
        logger.info(f"Docker container '{docker_env.container_name}' is running for reproduction.")
        max_turns = _env_int("COTEST_REPRO_STEPS", 50)
        logger.info(f"Reproduction max turns: {max_turns}")
        
        agent = ReproductionAgent.create(
            repository=repository,
            code_index=code_index,
            model=model_name,
            docker_env=docker_env,
            message_cache=True,
            intent_validating=intent_enabled,
            **_model_kwargs(model_name, "COTEST_REPRO_MAX_TOKENS"),
        )

        loop = ReproductionLoop(
            agent=agent,
            docker_env=docker_env,
            repository=repository,
            repo_path=repository.repo_path,
            max_turns=max_turns,
            require_intent_evidence=intent_enabled,
        )

        repo_structure = (
            _repo_structure_for_model(repository.repo_path, model_name)
            if include_repository_outline
            else ""
        )
        issue_message = (
            f"<problem_statement>\n{instance['problem_statement']}\n</problem_statement>\n"
        )
        if issue_evidence is not None:
            issue_message += (
                f"<issue_evidence>\n{issue_evidence.model_dump_json(indent=2)}\n"
                "</issue_evidence>\n"
            )
        issue_message += f"<repository_outline>\n{repo_structure}\n</repository_outline>"
        try:
            repro_result = loop.run(issue=issue_message)
        except Exception as e:
            # Catch any exception from loop.run() to prevent process crash
            logger.error(f"Reproduction loop failed with exception: {e}", exc_info=True)
            from INTENT2FIX.exceptions import CompletionRuntimeError
            if isinstance(e, CompletionRuntimeError):
                logger.error(f"CompletionRuntimeError detected during reproduction. This may be due to incorrect model name or API issues.")
            return None

        if dialogue_enabled:
            logger.info(
                "Starting evidence-bounded intent grounding in the live repository "
                "container (reproduction_status=%s).",
                "reproduced" if repro_result.is_reproduced else "not_reproduced",
            )
            try:
                dialogue_result = _run_intent_grounding(
                    instance=instance,
                    repository=repository,
                    docker_env=docker_env,
                    code_index=code_index,
                    issue_evidence=issue_evidence,
                    reproduction_context=compact_reproduction_context(
                        repro_result
                    ),
                    output_dir=output_dir,
                    fallback_model=model_name,
                )
                dialogue_prompt = dialogue_result.render_for_repair()
                dialogue_contract = dialogue_result.intent_contract.model_dump()
            except Exception:
                logger.exception(
                    "Intent grounding failed; continuing without repository evidence."
                )

    created_paths = [
        step.action.path
        for node in repro_result.full_trajectory.get_all_nodes()
        for step in node.action_steps
        if step.action.name == "CreateFile" and hasattr(step.action, "path")
    ] if repro_result.full_trajectory else []
    _remove_host_reproduction_artifacts(repository.repo_path, created_paths)

    # --- Process and Save results ---
    if repro_result.full_trajectory:
        # Save trajectory to the specified output directory
        persist_path = os.path.join(output_dir, f"repro_trajectory_{instance_name}.json")
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        logger.info(f"Saving full reproduction trajectory to {persist_path}")
        with open(persist_path, "w") as f:
            json.dump(repro_result.full_trajectory.model_dump(exclude_none=True), f, indent=2)
    else:
        logger.error("Reproduction task failed to produce a trajectory.")
        return None

    if issue_evidence is not None:
        valid_claim_ids = {claim.id for claim in issue_evidence.claims}
        unknown_claim_ids = sorted(set(repro_result.matched_claim_ids) - valid_claim_ids)
        if repro_result.is_reproduced and unknown_claim_ids:
            repro_result.is_reproduced = False
            repro_result.unresolved.append(
                f"Finish referenced unknown issue claim IDs: {', '.join(unknown_claim_ids)}"
            )

    if repro_result.is_reproduced:
        logger.info("Reproduction successful.")
        bundle = None
        if intent_enabled:
            logger.info("Adjudicating grounded repair intent...")
            proposed_bundle = _adjudicate_repair_intent(
                instance["problem_statement"], issue_evidence, repro_result
            )
            bundle = proposed_bundle
        if intent_enabled and _env_bool("COTEST_INTENT_GUARD"):
            logger.info("IntentGuard enabled; challenging the proposed repair intent.")
            guard_result = IntentGuard(_run_structured_extraction).validate(
                proposed_intent=proposed_bundle,
                issue=instance["problem_statement"],
                issue_evidence=issue_evidence,
                reproduction=repro_result,
                code_index=code_index,
                repository=repository,
            )
            guard_path = os.path.join(output_dir, "intent_guard_result.json")
            with open(guard_path, "w", encoding="utf-8") as f:
                json.dump(guard_result.model_dump(), f, indent=2, ensure_ascii=False)
            bundle = guard_result.effective_bundle
            logger.info(
                "IntentGuard decision: %s (%s)",
                guard_result.decision,
                guard_result.execution_status,
            )
        if bundle is not None:
            bundle_path = os.path.join(output_dir, "repair_intent_bundle.json")
            with open(bundle_path, "w", encoding="utf-8") as f:
                json.dump(bundle.model_dump(), f, indent=2, ensure_ascii=False)

        if bundle is not None and bundle.reproduction_status == "MISMATCHED":
            logger.warning("Adjudication rejected the reproduction as inconsistent with the issue.")
            return None

        summary = (bundle.summary if bundle is not None else "") or (
            f"Reproduced {repro_result.failure_class} with {repro_result.test_command}."
        )

        new_memory = Memory(
            instance_id=instance_name,
            problem_statement=instance["problem_statement"],
            reproduction_summary=summary,
            repair_summary=dialogue_prompt,
            golden_patch="",
            test_command=repro_result.test_command or "",
            test_files=repro_result.test_files,
            reproduction_result=repro_result.reproduction_output or "",
            expected_reproduction_result=repro_result.expected_output or "",
            issue_evidence=issue_evidence.model_dump() if issue_evidence is not None else {},
            repair_intent_bundle=bundle.compact_dump() if bundle is not None else {},
            repair_intent_contract=dialogue_contract,
            reproduction_execution_records=[item.model_dump() for item in repro_result.execution_records],
        )
        logger.info(f"Successfully created memory for instance {instance_name}.")
        return new_memory
    else:
        if dialogue_enabled:
            logger.warning(
                "Reproduction was unsuccessful. Continuing with issue- and "
                "repository-grounded intent analysis only."
            )
            return Memory(
                instance_id=instance_name,
                problem_statement=instance["problem_statement"],
                reproduction_summary=(
                    "The reproduction attempt did not establish a validated "
                    "failure."
                ),
                repair_summary=dialogue_prompt,
                golden_patch="",
                issue_evidence=(
                    issue_evidence.model_dump()
                    if issue_evidence is not None
                    else {}
                ),
                repair_intent_contract=dialogue_contract,
            )
        logger.warning("Reproduction was unsuccessful. No memory will be created.")
        return None

def run_repair_step(
    instance_name: str,
    model_name: str,
    docker_env: DockerEnv,
    memory: Memory,
    output_dir: str,
    *,
    candidate_mode: bool = False,
    candidate_hypothesis_context: Dict[str, Any] | None = None,
    agent_system_prompt: str | None = None,
    include_repository_outline: bool = True,
) -> str | None:
    """
    Runs the bug repair step using the provided memory from the reproduction step.
    Note: This version does NOT use MemoryModule for searching past experiences.
    """

    logger.info(f"--- Starting Repair Step for {instance_name} ---")
    _load_env()
    
    instance = get_benchmark_instance(instance_name)
    repository = create_repository(instance)
    
    code_index = None
    if _repair_graph_enabled():
        index_store_dir = os.getenv("INDEX_STORE_DIR")
        if not index_store_dir:
            raise ValueError("INDEX_STORE_DIR environment variable is not set. Please set it in .env file.")
        os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
        logger.info(f"Loading index for instance: {instance_name}")
        logger.info(f"INDEX_STORE_DIR: {index_store_dir}")
        code_index = CodeIndex.from_index_name(
            index_name=instance_name,
            file_repo=repository,
            index_store_dir=index_store_dir,
        )

    context_archive = None
    evidence_reader_kwargs = {}
    context_mode = _env_str("COTEST_BRANCH_CONTEXT", "off")
    if not _repair_graph_enabled() and context_mode == "graph":
        context_mode = "text"
    if candidate_mode and context_mode != "off":
        import shlex

        from INTENT2FIX.branch_context import prepare_branch_context
        from INTENT2FIX.diagnostic_evidence import record_digest

        diagnostic_kwargs = {}
        if _env_bool("COTEST_BRANCH_DIAGNOSTIC_SUMMARIES", False):
            case_dir = next((p for p in (Path(output_dir), *Path(output_dir).parents)
                             if p.name == instance_name), None)
            if case_dir is None:
                raise ValueError("Diagnostic preparation requires a case output directory")
            diagnostic_kwargs = dict(
                summarize=lambda prompt, schema: _run_structured_extraction(
                    prompt, schema, model_override=model_name, max_tokens=2048,
                    glm_reasoning_effort="low", max_attempts=1,
                ),
                summary_model_key=f"{model_name}:low:2048",
                diagnostic_source_dir=case_dir,
                diagnostic_cache_dir=case_dir / "diagnostic_summaries",
            )

        memory, archive_path, archive = prepare_branch_context(
            memory, candidate_hypothesis_context or {}, repository,
            instance["base_commit"], output_dir, context_mode,
            select=lambda prompt, schema: _run_structured_extraction(
                prompt, schema, model_override=model_name, max_tokens=4096,
                glm_reasoning_effort="low", max_attempts=1,
            ),
            code_index=code_index,
            **diagnostic_kwargs,
        )
        context_archive = (archive_path, archive)
        if diagnostic_kwargs:
            evidence_reader_kwargs = dict(
                evidence_archive_path=str(Path(output_dir).resolve() / "branch_evidence_archive.json"),
                evidence_archive_sha256=record_digest(archive),
            )

    with docker_env:
        logger.info(f"Docker container '{docker_env.container_name}' is running for repair.")
        if context_archive:
            archive_path, archive = context_archive
            docker_env.write_file(archive_path, json.dumps(archive, ensure_ascii=False))
            archive_check, archive_exit = docker_env.run_bash(
                "python3 -c " + shlex.quote(
                    "import json; d=json.load(open(" + repr(archive_path) + ")); "
                    "assert d['contract_sha256']==" + repr(archive["contract_sha256"])
                ), timeout=30,
            )
            if archive_exit:
                raise ValueError("Prepared branch evidence archive is not readable in Repair")
        instance_str = parse_instance_string(instance_name)

        file_context = FileContext(repo=repository, runtime=docker_env)
        _register_intent_verification_contract(file_context, memory)
        if candidate_mode:
            file_context.verification["finish_policy"] = (
                "portfolio_soft_evidence"
            )

        if memory.test_files:
            immutable_reproduction_paths = []
            for test_file in memory.test_files:
                # Ensure the path from memory is treated as a relative path
                relative_path = test_file['path'].lstrip('/')
                immutable_reproduction_paths.append(relative_path)
                
                # Construct the full path inside the container's workspace
                full_container_path = os.path.join(docker_env.work_dir, relative_path)
                
                # Write the file to the correct workspace location
                docker_env.write_file(full_container_path, test_file['content'])

                # RunSingleTest copies from the host repository into the container.
                full_host_path = repository.get_full_path(relative_path)
                os.makedirs(os.path.dirname(full_host_path), exist_ok=True)
                with open(full_host_path, "w") as f:
                    f.write(test_file['content'])

                # Add the file to the context using its relative path
                file_context.add_file(relative_path, content=test_file['content'])
                logger.info(f"Created test file in container workspace at {relative_path} and added to context.")
            file_context.verification["immutable_reproduction_paths"] = (
                immutable_reproduction_paths
            )
        else:
            logger.info("No test files from reproduction step. Proceeding with repair.")

        repair_tool_profile = _env_str("COTEST_REPAIR_TOOL_PROFILE", "") or None
        agent_system_prompt = _repair_agent_system_prompt(agent_system_prompt)
        agent = CodingAgent.create(
            repository=repository,
            code_index=code_index,
            enable_indexed_tools=_repair_graph_enabled(),
            model=model_name,
            runtime=docker_env,
            message_cache=True,
            soft_verification_finish=candidate_mode,
            allow_regression=True,
            system_prompt_override=agent_system_prompt,
            tool_profile=repair_tool_profile,
            graph_issue_text=(
                memory.problem_statement
                if repair_tool_profile == "unified_four" else ""
            ),
            **evidence_reader_kwargs,
            **_completion_protocol_kwargs(model_name),
            **_model_kwargs(model_name, "COTEST_REPAIR_MAX_TOKENS"),
        )
        finish_tool_name = (
            "Finish"
            if any(isinstance(action, Finish) for action in agent.actions)
            else "VerifiedFinish"
        )

        if _env_bool("COTEST_REPAIR_OMIT_REPOSITORY_OUTLINE", False):
            include_repository_outline = False

        repo_structure = _repo_structure_for_model(repository.repo_path, model_name)

        message = (
            f"<task>\n<problem_statement>\n{memory.problem_statement}\n</problem_statement>\n"
        )

        if memory.test_files:
            poc_test_paths = [f"  - {test_file['path']}" for test_file in memory.test_files]
            poc_block = "<Poc_test>\n" + "\n".join(poc_test_paths) + "\n</Poc_test>"
            message += (
                f"The following Proof-of-Concept (PoC) test file(s) have been created to demonstrate the bug. "
                f"You should run these tests when appropriate to understand the failure and verify your fix.\n"
                f"{poc_block}\n\n"
            )
        if memory.test_command:
            command_block = f"<Poc_command>\n{memory.test_command}\n</Poc_command>"
            logger.info(f'Poc_command: f{command_block}')
            message += (
                f"Use the following command to run the PoC test:\n"
                f"{command_block}\n\n"
            )
        if memory.reproduction_summary:
            message += f"Reproduction Summary:\n{memory.reproduction_summary}\n\n"
        if memory.reproduction_result:
            message += (
                "Observed Failure Evidence:\n"
                f"{_compact_reproduction_output(memory.reproduction_result)}\n\n"
            )
        if memory.repair_summary:
            message += memory.repair_summary + "\n\n"
        if memory.issue_evidence:
            message += (
                "<issue_evidence>\n"
                + json.dumps(memory.issue_evidence, ensure_ascii=False)
                + "\n</issue_evidence>\n"
                "These claims have exact issue spans. The focused verification must cover "
                "every issue-explicit operand form and affected type category; do not reduce "
                "the contract to only the minimal reproducer. Reproduction PoC files are "
                "immutable evidence: do not edit or append to them, and do not add assertions "
                "for behavior absent from issue-explicit claims. Fix production code and run "
                "the supplied PoC unchanged; never substitute /tmp.\n\n"
            )
        if memory.repair_intent_bundle:
            if candidate_mode:
                bundle_guidance = (
                    "Treat hard obligations as required behavior. The active "
                    "<intent_hypothesis> assigned above defines the mandatory "
                    "observable outcome for this proposal; do not replace it with "
                    "another portfolio branch. Other soft hypotheses and mechanism "
                    "suggestions are search guidance only. Preserve unresolved scope "
                    "instead of guessing.\n\n"
                )
            else:
                bundle_guidance = (
                    "Treat hard obligations as required behavior. Use soft "
                    "hypotheses only as search guidance, do not revive rejected "
                    "interpretations, and preserve unresolved scope instead of "
                    "guessing.\n\n"
                )
            message += (
                "<repair_intent_bundle>\n"
                + json.dumps(memory.repair_intent_bundle, ensure_ascii=False)
                + "\n</repair_intent_bundle>\n"
                + bundle_guidance
            )
        
        if include_repository_outline:
            message += (
                f"<repository_outline>\n{repo_structure}\n</repository_outline>\n"
            )
        message += "Fix the issue and verify the final code state.\n</task>"

        persist_path = os.path.join(output_dir, f"repair_trajectory_{instance_name}.json")
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        repair_steps = (
            _env_int("COTEST_PORTFOLIO_REPAIR_STEPS", 50)
            if candidate_mode
            else _env_int("COTEST_REPAIR_STEPS", 120)
        )
        finish_reserve_steps = (
            min(
                max(
                    0,
                    _env_int("COTEST_PORTFOLIO_FINISH_RESERVE_STEPS", 2),
                ),
                max(0, repair_steps - 1),
            )
            if candidate_mode
            else 0
        )
        proposal_steps = max(1, repair_steps - finish_reserve_steps)
        logger.info(
            "Repair action budget: %d (%d proposal, %d Finish-only reserve)",
            repair_steps,
            proposal_steps,
            finish_reserve_steps,
        )
        metadata = {
            "instance_id": instance_name,
            "problem_statement": memory.problem_statement,
            "reproduction_summary": memory.reproduction_summary,
            "reproduction_result": memory.reproduction_result,
            "expected_reproduction_result": memory.expected_reproduction_result,
            "causal_investigation_evidence": memory.repair_summary,
            "repair_intent_bundle": memory.repair_intent_bundle,
            "repair_intent_contract": memory.repair_intent_contract,
            "result_persist_path": output_dir, # Use the dynamic output directory
        }

        # Create loop WITHOUT memory_module by manually creating and disabling it
        loop = AgenticLoop.create(
            agent=agent,
            file_context=file_context,
            message=message,
            repository=repository,
            runtime=docker_env,
            code_index=code_index,
            metadata=metadata,
            persist_path=persist_path,
            max_iterations=proposal_steps
        )
        
        # Disable memory module - set it to None after creation
        loop.memory_module = None

        # Run the loop WITHOUT search_weights (since we don't have memory_module)
        logger.info("Running repair loop without memory module...")
        try:
            final_trajectory = loop.run()
            if candidate_mode:
                final_trajectory = _run_portfolio_finish_tail(
                    loop,
                    final_trajectory,
                    max_finish_attempts=finish_reserve_steps,
                )
        except Exception as e:
            # Catch any exception from loop.run() to prevent process crash
            logger.error(f"Repair loop failed with exception: {e}", exc_info=True)
            # Check if it's a CompletionRuntimeError due to model name issues
            from INTENT2FIX.exceptions import CompletionRuntimeError
            if isinstance(e, CompletionRuntimeError):
                logger.error(f"CompletionRuntimeError detected. This may be due to incorrect model name or API issues.")
            return None

        try:
            solution_patch = _generate_submission_patch(final_trajectory.file_context)
            # Check if there are any patches (non-empty string after stripping)
            # Empty string from generate_git_patch() means no patches, which is falsy
            has_patches = solution_patch and solution_patch.strip()
            artifact_errors = (
                sanitize_candidate_patch(final_trajectory.file_context).artifact_errors
                if candidate_mode else ()
            )
            if final_trajectory and (has_patches or artifact_errors):
                verified_finish = final_trajectory.is_finished()
                blocking_compile_regression = False
                if candidate_mode and verified_finish:
                    _ensure_portfolio_regression(
                        final_trajectory.file_context, docker_env
                    )
                    blocking_compile_regression = (
                        _portfolio_compile_regression_failure(
                            final_trajectory.file_context
                        )
                    )
                basic_tests_passed = (
                    _portfolio_regression_passed(
                        final_trajectory.file_context
                    )
                    if candidate_mode
                    else _basic_tests_passed(final_trajectory.file_context)
                )
                patch_review_status = (
                    "disabled" if not _patch_reviewer_enabled() else "pending"
                )
                patch_intent_audit_status = (
                    "disabled" if not _patch_intent_audit_enabled() else "pending"
                )
                auditor = None
                audit_result = None
                patch_review_changed = False
                if basic_tests_passed:
                    logger.info(
                        "Repair patch generated with current basic verification."
                    )
                else:
                    logger.warning(
                        "Unverified repair patch generated; current basic "
                        "verification did not pass."
                    )
                # Save solution patch to the dynamic output directory
                patch_path = os.path.join(output_dir, "solution.patch")
                with open(patch_path, "w") as f:
                    f.write(solution_patch)
                logger.info(f"Solution patch saved to {patch_path}")
                if candidate_mode:
                    frozen_candidate = _save_patch_sanitization_audit(
                        final_trajectory.file_context,
                        output_dir,
                        "portfolio_candidate",
                    )
                    solution_patch = frozen_candidate.patch
                    if not verified_finish and not frozen_candidate.artifact_errors:
                        _save_portfolio_candidate_status(
                            output_dir,
                            finish_succeeded=False,
                            basic_tests_passed=basic_tests_passed,
                            candidate=frozen_candidate,
                        )
                        logger.warning(
                            "Portfolio branch ended without a successfully "
                            "executed terminal Finish; excluding its patch."
                        )
                        return None
                    if not solution_patch.strip() and not frozen_candidate.artifact_errors:
                        _save_portfolio_candidate_status(
                            output_dir,
                            finish_succeeded=True,
                            basic_tests_passed=basic_tests_passed,
                            candidate=frozen_candidate,
                        )
                        logger.warning(
                            "Portfolio branch Finish produced no sanitized "
                            "production patch; excluding it."
                        )
                        return None
                    branch = (candidate_hypothesis_context or {}).get(
                        "branch"
                    ) or {}
                    hypothesis_id = str(branch.get("id") or "H0")
                    discriminator_required = bool(
                        branch.get("discriminator")
                        or (candidate_hypothesis_context or {}).get(
                            "assigned_dimensions"
                        )
                    )
                    repair_probes = _collect_final_patch_repair_probes(
                        final_trajectory=final_trajectory,
                        candidate=frozen_candidate,
                        hypothesis_context=candidate_hypothesis_context,
                    )
                    discriminator_verified = (
                        _repair_discriminator_verified(
                            repair_probes, hypothesis_id
                        )
                    )
                    unresolved_focused_failure = (
                        _portfolio_has_unresolved_focused_failure(
                            repair_probes,
                            discriminator_required=discriminator_required,
                        )
                    )
                    compliance = _audit_portfolio_branch_compliance(
                        hypothesis_context=(
                            candidate_hypothesis_context or {}
                        ),
                        candidate=frozen_candidate,
                        repair_probes=repair_probes,
                        file_context=final_trajectory.file_context,
                        model_name=model_name,
                        output_dir=output_dir,
                        label="initial",
                    )
                    engineering_diagnostic = _portfolio_engineering_diagnostic(
                        final_trajectory.file_context, frozen_candidate, repair_probes
                    )
                    candidate_variants_by_hash: Dict[str, Dict[str, Any]] = {}
                    initial_variant = _portfolio_candidate_variant_payload(
                        variant_id="initial",
                        finish_succeeded=verified_finish,
                        basic_tests_passed=basic_tests_passed,
                        candidate=frozen_candidate,
                        branch_compliance=compliance,
                        discriminator_verified=discriminator_verified,
                        discriminator_required=discriminator_required,
                        blocking_compile_regression=blocking_compile_regression,
                        unresolved_focused_failure=(
                            unresolved_focused_failure
                        ),
                        regression_evidence=_fresh_portfolio_regression(
                            final_trajectory.file_context
                        ),
                        repair_probes=repair_probes,
                    )
                    _retain_portfolio_candidate_variant(
                        candidate_variants_by_hash, initial_variant
                    )
                    _save_portfolio_candidate_status(
                        output_dir,
                        finish_succeeded=verified_finish,
                        basic_tests_passed=basic_tests_passed,
                        candidate=frozen_candidate,
                        branch_compliance=compliance.status,
                        discriminator_verified=discriminator_verified,
                        discriminator_required=discriminator_required,
                        blocking_compile_regression=(
                            blocking_compile_regression
                        ),
                        unresolved_focused_failure=(
                            unresolved_focused_failure
                        ),
                        regression_evidence=_fresh_portfolio_regression(
                            final_trajectory.file_context
                        ),
                        repair_probes=repair_probes,
                        branch_compliance_decision=compliance,
                        candidate_variants=list(
                            candidate_variants_by_hash.values()
                        ),
                    )
                    if _portfolio_branch_requires_correction(
                        compliance=compliance,
                        engineering_diagnostic=engineering_diagnostic,
                    ):
                        continuation_steps = _env_int(
                            "COTEST_PORTFOLIO_CONTINUATION_STEPS", 10
                        )
                        continuation_proposal_steps = continuation_steps
                        continuation_finish_steps = 2
                        continuation_rounds = _env_int(
                            "COTEST_PORTFOLIO_CONTINUATION_ROUNDS", 2
                        )
                        for continuation_round in range(
                            1, max(1, continuation_rounds) + 1
                        ):
                            feedback = _render_portfolio_branch_correction_feedback(
                                continuation_round=continuation_round,
                                continuation_rounds=max(
                                    1, continuation_rounds
                                ),
                                hypothesis_context=(
                                    candidate_hypothesis_context or {}
                                ),
                                compliance=compliance,
                                engineering_diagnostic=engineering_diagnostic,
                            )
                            loop.append_user_feedback(feedback)
                            loop.max_iterations = len(
                                loop.root.get_all_nodes()
                            ) + max(1, continuation_proposal_steps)
                            logger.info(
                                "Continuing the same portfolio branch for round "
                                "%d/%d with up to %d proposal steps and 2 "
                                "reserved Finish slots.",
                                continuation_round,
                                max(1, continuation_rounds),
                                continuation_proposal_steps,
                            )
                            final_trajectory = loop.run()
                            final_trajectory = _run_portfolio_finish_tail(
                                loop,
                                final_trajectory,
                                max_finish_attempts=(
                                    continuation_finish_steps
                                ),
                            )
                            verified_finish = final_trajectory.is_finished()
                            if (
                                verified_finish
                                and _generate_submission_patch(
                                    final_trajectory.file_context
                                ).strip()
                            ):
                                _ensure_portfolio_regression(
                                    final_trajectory.file_context, docker_env
                                )
                            blocking_compile_regression = (
                                _portfolio_compile_regression_failure(
                                    final_trajectory.file_context
                                )
                            )
                            basic_tests_passed = (
                                _portfolio_regression_passed(
                                    final_trajectory.file_context
                                )
                            )
                            label = f"after_continuation_{continuation_round}"
                            frozen_candidate = _save_patch_sanitization_audit(
                                final_trajectory.file_context,
                                output_dir,
                                f"portfolio_candidate_{label}",
                            )
                            solution_patch = frozen_candidate.patch
                            repair_probes = (
                                _collect_final_patch_repair_probes(
                                    final_trajectory=final_trajectory,
                                    candidate=frozen_candidate,
                                    hypothesis_context=(
                                        candidate_hypothesis_context
                                    ),
                                )
                            )
                            discriminator_verified = (
                                _repair_discriminator_verified(
                                    repair_probes, hypothesis_id
                                )
                            )
                            unresolved_focused_failure = (
                                _portfolio_has_unresolved_focused_failure(
                                    repair_probes,
                                    discriminator_required=(
                                        discriminator_required
                                    ),
                                )
                            )
                            compliance = _audit_portfolio_branch_compliance(
                                hypothesis_context=(
                                    candidate_hypothesis_context or {}
                                ),
                                candidate=frozen_candidate,
                                repair_probes=repair_probes,
                                file_context=final_trajectory.file_context,
                                model_name=model_name,
                                output_dir=output_dir,
                                label=(
                                    "after_continuation_"
                                    f"{continuation_round}"
                                ),
                            )
                            engineering_diagnostic = _portfolio_engineering_diagnostic(
                                final_trajectory.file_context, frozen_candidate, repair_probes
                            )
                            continued_variant = (
                                _portfolio_candidate_variant_payload(
                                    variant_id=(
                                        "continuation_"
                                        f"{continuation_round}"
                                    ),
                                    finish_succeeded=verified_finish,
                                    basic_tests_passed=basic_tests_passed,
                                    candidate=frozen_candidate,
                                    branch_compliance=compliance,
                                    discriminator_verified=(
                                        discriminator_verified
                                    ),
                                    discriminator_required=(
                                        discriminator_required
                                    ),
                                    blocking_compile_regression=(
                                        blocking_compile_regression
                                    ),
                                    unresolved_focused_failure=(
                                        unresolved_focused_failure
                                    ),
                                    regression_evidence=(
                                        _fresh_portfolio_regression(
                                            final_trajectory.file_context
                                        )
                                    ),
                                    repair_probes=repair_probes,
                                )
                            )
                            _retain_portfolio_candidate_variant(
                                candidate_variants_by_hash, continued_variant
                            )
                            if not _portfolio_branch_requires_correction(
                                compliance=compliance,
                                engineering_diagnostic=engineering_diagnostic,
                            ):
                                break
                    with open(patch_path, "w") as f:
                        f.write(solution_patch)
                    _save_portfolio_candidate_status(
                        output_dir,
                        finish_succeeded=verified_finish,
                        basic_tests_passed=basic_tests_passed,
                        candidate=frozen_candidate,
                        branch_compliance=compliance.status,
                        discriminator_verified=discriminator_verified,
                        discriminator_required=discriminator_required,
                        blocking_compile_regression=(
                            blocking_compile_regression
                        ),
                        unresolved_focused_failure=(
                            unresolved_focused_failure
                        ),
                        regression_evidence=_fresh_portfolio_regression(
                            final_trajectory.file_context
                        ),
                        repair_probes=repair_probes,
                        branch_compliance_decision=compliance,
                        candidate_variants=list(
                            candidate_variants_by_hash.values()
                        ),
                    )
                    admitted_variants = [
                        item
                        for item in candidate_variants_by_hash.values()
                        if item.get("admitted")
                    ]
                    if not admitted_variants:
                        logger.warning(
                            "Portfolio branch produced no finished, nonempty "
                            "candidate variant."
                        )
                        return None
                    selected_variant = admitted_variants[-1]
                    solution_patch = str(selected_variant["patch"])
                    with open(patch_path, "w") as f:
                        f.write(solution_patch)
                    logger.info(
                        "Portfolio branch retained %d candidate variant(s) for "
                        "cross-execution; compliance and discriminator results "
                        "remain Selector evidence.",
                        len(admitted_variants),
                    )
                    return solution_patch
                if not verified_finish:
                    unverified_path = os.path.join(
                        output_dir,
                        "solution_unverified.patch",
                    )
                    with open(unverified_path, "w") as f:
                        f.write(solution_patch)
                    logger.warning(
                        "Patch is retained for analysis but marked unverified: %s",
                        unverified_path,
                    )

                # Check if basic tests passed, then trigger TestAgent for test enhancement
                if basic_tests_passed:
                    if _patch_intent_audit_enabled():
                        if not final_trajectory.is_finished():
                            patch_intent_audit_status = (
                                "skipped_without_verified_finish"
                            )
                            logger.warning(
                                "Repair trajectory ended without VerifiedFinish. "
                                "Skipping Patch Intent Audit."
                            )
                        else:
                            pre_audit_candidate = _save_patch_sanitization_audit(
                                final_trajectory.file_context,
                                output_dir,
                                "before_patch_intent_audit",
                            )
                            pre_audit_patch = pre_audit_candidate.patch
                            solution_patch = pre_audit_patch
                            pre_audit_path = os.path.join(
                                output_dir,
                                "solution_before_intent_audit.patch",
                            )
                            with open(pre_audit_path, "w") as f:
                                f.write(pre_audit_patch)
                            pre_audit_trajectory = final_trajectory
                            trajectory_lead_result = (
                                _extract_repair_trajectory_audit_leads(
                                    persist_path=persist_path,
                                    candidate_patch=pre_audit_patch,
                                    output_dir=output_dir,
                                    model_name=model_name,
                                )
                            )
                            trajectory_audit_leads = (
                                _render_trajectory_audit_leads(
                                    trajectory_lead_result.leads
                                )
                                if trajectory_lead_result is not None
                                else ""
                            )
                            try:
                                auditor, audit_result = _run_patch_intent_audit(
                                    instance_name=instance_name,
                                    model_name=model_name,
                                    repository=repository,
                                    docker_env=docker_env,
                                    code_index=code_index,
                                    memory=memory,
                                    candidate=pre_audit_candidate,
                                    file_context=final_trajectory.file_context,
                                    output_dir=output_dir,
                                    trajectory_audit_leads=trajectory_audit_leads,
                                )
                            except Exception:
                                patch_intent_audit_status = (
                                    "inconclusive_infrastructure"
                                )
                                logger.exception(
                                    "Patch Intent Audit failed. Keeping the "
                                    "already verified candidate and continuing "
                                    "with later stages."
                                )
                            if audit_result is not None:
                                if (
                                    audit_result.semantic_admission is not None
                                    and not audit_result.semantic_admission.available
                                ):
                                    patch_intent_audit_status = (
                                        "inconclusive_admission_unavailable"
                                    )
                                else:
                                    patch_intent_audit_status = audit_result.verdict
                            if audit_result is not None and audit_result.blocking:
                                patch_intent_audit_status = (
                                    "underfit_refinement_started"
                                )
                                refine_steps = _env_int(
                                    "COTEST_PATCH_INTENT_AUDIT_REFINE_STEPS",
                                    40,
                                )
                                feedback = (
                                    "<patch_intent_audit>\n"
                                    "The candidate patch fails one frozen public "
                                    "counterexample backed by the original Issue or "
                                    "accepted repair contract. Refine the smallest "
                                    "production behavior needed to satisfy this exact "
                                    "counterexample. Do not broaden the scope from graph "
                                    "proximity, modify existing tests, or redesign "
                                    "unresolved product policy. Re-run current verification "
                                    f"and call {finish_tool_name} again.\n\n"
                                    f"{audit_result.report}"
                                    f"{_render_patch_intent_counterexample_feedback(audit_result)}\n"
                                    "</patch_intent_audit>"
                                )
                                loop.append_user_feedback(feedback)
                                loop.max_iterations = len(
                                    loop.root.get_all_nodes()
                                ) + max(1, refine_steps)
                                logger.info(
                                    "Continuing the same Repair trajectory for up to %d "
                                    "Patch Intent Audit refinement steps.",
                                    refine_steps,
                                )
                                final_trajectory = loop.run()
                                refined_candidate = _save_patch_sanitization_audit(
                                    final_trajectory.file_context,
                                    output_dir,
                                    "after_patch_intent_audit",
                                )
                                refined_patch = refined_candidate.patch
                                after_audit_path = os.path.join(
                                    output_dir,
                                    "solution_after_intent_refine.patch",
                                )
                                with open(after_audit_path, "w") as f:
                                    f.write(refined_patch or "")
                                if not refined_patch:
                                    patch_intent_audit_status = (
                                        "underfit_refine_empty"
                                    )
                                    solution_patch = pre_audit_patch
                                    final_trajectory = pre_audit_trajectory
                                elif _patches_equivalent(
                                    pre_audit_patch,
                                    refined_patch,
                                ):
                                    patch_intent_audit_status = (
                                        "underfit_refine_no_persisted_change"
                                    )
                                    solution_patch = refined_patch
                                else:
                                    solution_patch = refined_patch
                                    with open(patch_path, "w") as f:
                                        f.write(solution_patch)
                                    patch_intent_audit_status = (
                                        _recheck_patch_intent_counterexample(
                                            auditor=auditor,
                                            audit_result=audit_result,
                                            file_context=(
                                                final_trajectory.file_context
                                            ),
                                            candidate=refined_candidate,
                                            output_dir=output_dir,
                                            label="after_intent_refine",
                                        )
                                    )
                                verified_finish = final_trajectory.is_finished()
                                basic_tests_passed = _basic_tests_passed(
                                    final_trajectory.file_context
                                )
                            elif (
                                audit_result is not None
                                and audit_result.unresolved_discriminator
                                is not None
                            ):
                                patch_intent_audit_status = (
                                    "unresolved_discriminator_review_started"
                                )
                                used_steps = len(loop.root.get_all_nodes())
                                remaining_steps = max(0, repair_steps - used_steps)
                                feedback = (
                                    "<patch_intent_audit>\n"
                                    "The audit executed a valid public discriminator "
                                    "for a semantic choice made by the candidate. The "
                                    "Issue makes this dimension relevant but does not "
                                    "authorize either probe outcome, so this is not a "
                                    "failing test or a new requirement. Reconsider only "
                                    "the selected mechanism: compare it with the literal "
                                    "Issue wording and independently inspect repository "
                                    "documentation or public tests if they normatively "
                                    "resolve the choice. Otherwise keep the verified "
                                    "candidate. Do not edit existing tests or broaden "
                                    "the repair. Re-run current verification and call "
                                    f"{finish_tool_name}.\n\n"
                                    f"{audit_result.report}"
                                    f"{_render_patch_intent_unresolved_discriminator_feedback(audit_result)}\n"
                                    "</patch_intent_audit>"
                                )
                                if remaining_steps > 0:
                                    loop.append_user_feedback(feedback)
                                    loop.max_iterations = repair_steps
                                    logger.info(
                                        "Continuing the same Repair trajectory within "
                                        "the original budget (%d steps remain) to "
                                        "reconsider one unresolved mechanism choice.",
                                        remaining_steps,
                                    )
                                    final_trajectory = loop.run()
                                    reconsidered_candidate = (
                                        _save_patch_sanitization_audit(
                                            final_trajectory.file_context,
                                            output_dir,
                                            "after_unresolved_discriminator",
                                        )
                                    )
                                    reconsidered_patch = (
                                        reconsidered_candidate.patch
                                    )
                                    reconsidered_path = os.path.join(
                                        output_dir,
                                        "solution_after_unresolved_discriminator.patch",
                                    )
                                    with open(reconsidered_path, "w") as f:
                                        f.write(reconsidered_patch or "")
                                    reconsidered_verified = (
                                        final_trajectory.is_finished()
                                        and _basic_tests_passed(
                                            final_trajectory.file_context
                                        )
                                    )
                                    if not reconsidered_verified:
                                        patch_intent_audit_status = (
                                            "unresolved_discriminator_reverted_unverified"
                                        )
                                        final_trajectory = pre_audit_trajectory
                                        solution_patch = pre_audit_patch
                                    elif not reconsidered_patch or _patches_equivalent(
                                        pre_audit_patch,
                                        reconsidered_patch,
                                    ):
                                        patch_intent_audit_status = (
                                            "unresolved_discriminator_retained"
                                        )
                                        solution_patch = (
                                            reconsidered_patch or pre_audit_patch
                                        )
                                    else:
                                        patch_intent_audit_status = (
                                            "unresolved_discriminator_revised"
                                        )
                                        solution_patch = reconsidered_patch
                                        with open(patch_path, "w") as f:
                                            f.write(solution_patch)
                                else:
                                    patch_intent_audit_status = (
                                        "unresolved_discriminator_no_budget"
                                    )
                                verified_finish = final_trajectory.is_finished()
                                basic_tests_passed = _basic_tests_passed(
                                    final_trajectory.file_context
                                )

                    if (
                        basic_tests_passed
                        and not _patch_intent_audit_unresolved(
                            patch_intent_audit_status
                        )
                        and _patch_reviewer_enabled()
                    ):
                        if not final_trajectory.is_finished():
                            patch_review_status = "skipped_without_verified_finish"
                            logger.warning(
                                "Repair trajectory ended without VerifiedFinish. "
                                "Skipping Patch Reviewer and keeping the generated "
                                "submission patch."
                            )
                        else:
                            pre_review_candidate = _save_patch_sanitization_audit(
                                final_trajectory.file_context,
                                output_dir,
                                "before_patch_review",
                            )
                            pre_review_patch = pre_review_candidate.patch
                            solution_patch = pre_review_patch
                            pre_review_patch_path = os.path.join(
                                output_dir,
                                "solution_before_patch_review.patch",
                            )
                            with open(pre_review_patch_path, "w") as f:
                                f.write(pre_review_patch)
                            logger.info(
                                "Patch before surface review saved to %s",
                                pre_review_patch_path,
                            )
                            pre_review_trajectory = final_trajectory
                            review_result = _run_patch_surface_review(
                                instance_name=instance_name,
                                model_name=model_name,
                                repository=repository,
                                docker_env=docker_env,
                                code_index=code_index,
                                memory=memory,
                                candidate=pre_review_candidate,
                                file_context=final_trajectory.file_context,
                                output_dir=output_dir,
                                pass_number=1,
                            )
                            if review_result.approved:
                                patch_review_status = "approved"
                                post_review_patch_path = os.path.join(
                                    output_dir,
                                    "solution_after_patch_review.patch",
                                )
                                with open(post_review_patch_path, "w") as f:
                                    f.write(solution_patch)
                            if not review_result.approved:
                                patch_review_status = "rejected_refinement_started"
                                refine_steps = _env_int(
                                    "COTEST_PATCH_REVIEW_REFINE_STEPS",
                                    40,
                                )
                                feedback = (
                                    "<patch_surface_review>\n"
                                    "The read-only Patch Reviewer rejected the candidate. "
                                    "Address only the evidence-supported omissions below; "
                                    "do not redesign the requested behavior or modify existing "
                                    "tests, fixtures, expected outputs, or test registration to "
                                    "evade the defect. Re-run available verification after the "
                                    "final production-code edit, report unresolved failures "
                                    f"honestly, then call {finish_tool_name} again.\n\n"
                                    f"{review_result.report}\n"
                                    "</patch_surface_review>"
                                )
                                loop.append_user_feedback(feedback)
                                loop.max_iterations = len(loop.root.get_all_nodes()) + max(
                                    1, refine_steps
                                )
                                logger.info(
                                    "Continuing the same Repair trajectory for up to %d "
                                    "review-driven refinement steps.",
                                    refine_steps,
                                )
                                final_trajectory = loop.run()
                                refined_candidate = _save_patch_sanitization_audit(
                                    final_trajectory.file_context,
                                    output_dir,
                                    "after_patch_review",
                                )
                                refined_patch = refined_candidate.patch
                                post_review_patch_path = os.path.join(
                                    output_dir,
                                    "solution_after_patch_review.patch",
                                )
                                with open(post_review_patch_path, "w") as f:
                                    f.write(refined_patch or "")
                                logger.info(
                                    "Patch after surface-review refinement saved to %s",
                                    post_review_patch_path,
                                )
                                if not refined_patch:
                                    patch_review_status = "rejected_refine_empty"
                                    logger.warning(
                                        "Patch review refinement produced an empty patch. "
                                        "Keeping the pre-review patch for submission."
                                    )
                                    solution_patch = pre_review_patch
                                    final_trajectory = pre_review_trajectory
                                elif _patches_equivalent(
                                    pre_review_patch,
                                    refined_patch,
                                ):
                                    patch_review_status = (
                                        "rejected_refine_no_persisted_change"
                                    )
                                    logger.warning(
                                        "Patch Reviewer rejected the candidate, but "
                                        "refinement produced no persisted submission "
                                        "change. Keeping both audit artifacts and the "
                                        "candidate patch; review remains unresolved."
                                    )
                                    solution_patch = refined_patch
                                    with open(patch_path, "w") as f:
                                        f.write(solution_patch)
                                else:
                                    patch_review_status = "rejected_then_refined"
                                    patch_review_changed = True
                                    solution_patch = refined_patch
                                    with open(patch_path, "w") as f:
                                        f.write(solution_patch)

                            verified_finish = final_trajectory.is_finished()
                            basic_tests_passed = _basic_tests_passed(
                                final_trajectory.file_context
                            )
                            if (
                                patch_review_changed
                                and audit_result is not None
                            ):
                                latest_candidate = sanitize_candidate_patch(
                                    final_trajectory.file_context
                                )
                                patch_intent_audit_status = (
                                    _recheck_patch_intent_counterexample(
                                        auditor=auditor,
                                        audit_result=audit_result,
                                        file_context=(
                                            final_trajectory.file_context
                                        ),
                                        candidate=latest_candidate,
                                        output_dir=output_dir,
                                        label="after_patch_review_refine",
                                    )
                                )

                    if _patch_intent_audit_unresolved(
                        patch_intent_audit_status
                    ):
                        logger.warning(
                            "The frozen Patch Intent Audit counterexample remains "
                            "unresolved. Retaining the candidate for analysis and "
                            "skipping later reviewers and TestAgent enhancement."
                        )
                        _save_repair_verification_status(
                            output_dir,
                            verified_finish=verified_finish,
                            basic_tests_passed=basic_tests_passed,
                            patch_review_status=patch_review_status,
                            patch_intent_audit_status=patch_intent_audit_status,
                        )
                        return solution_patch

                    if not basic_tests_passed:
                        logger.warning(
                            "The post-review candidate no longer has current "
                            "basic verification. Retaining it for analysis and "
                            "skipping TestAgent enhancement."
                        )
                        _save_repair_verification_status(
                            output_dir,
                            verified_finish=verified_finish,
                            basic_tests_passed=False,
                            patch_review_status=patch_review_status,
                            patch_intent_audit_status=(
                                patch_intent_audit_status
                                if _patch_intent_audit_enabled()
                                else None
                            ),
                        )
                        return solution_patch

                    # Save the patch before iteration (iter0)
                    solution_before_iter_path = os.path.join(output_dir, "solution_before_iteration.patch")
                    with open(solution_before_iter_path, "w") as f:
                        f.write(solution_patch)
                    logger.info(f"Solution patch before iteration saved to {solution_before_iter_path}")
                    
                    # Test enhancement and follow-up repair rounds.
                    max_iterations = _env_int("COTEST_TEST_ENHANCEMENT_ROUNDS", 1)
                    if max_iterations > 0:
                        logger.info(
                            "Basic tests passed. Starting TestAgent enhancement "
                            "(%d round(s)).",
                            max_iterations,
                        )
                    else:
                        logger.info(
                            "Basic tests passed. TestAgent enhancement is disabled "
                            "(rounds=0)."
                        )
                    current_patch = solution_patch
                    current_file_context = final_trajectory.file_context
                    test_model_name = model_name  # Can use different model for test enhancement
                    
                    for iteration in range(max_iterations):
                        logger.info(f"Test Enhancement iteration {iteration + 1}/{max_iterations}")
                        
                        # Reuse the existing repair container for test enhancement
                        docker_env.reuse_existing = True
                        enhancement_result = run_test_enhancement_step(
                            instance_name=instance_name,
                            model_name=test_model_name,
                            docker_env=docker_env,
                            file_context=current_file_context,
                            memory=memory,
                            output_dir=output_dir,
                            iteration=iteration + 1
                        )

                        if enhancement_result is None:
                            logger.warning(
                                "Test enhancement returned no result. Keeping the last verified repair patch."
                            )
                            break

                        if enhancement_result.success:
                            logger.info("Test enhancement successful! All tests passed.")
                            diagnostic_patch = enhancement_result.enhanced_patch
                            enhanced_patch_path = os.path.join(output_dir, f"enhanced_solution_iter{iteration + 1}.patch")
                            with open(enhanced_patch_path, "w") as f:
                                f.write(diagnostic_patch)
                            logger.info(f"Test enhancement diagnostic patch saved to {enhanced_patch_path}")
                            solution_patch = _submission_patch_after_test_enhancement(
                                current_patch, diagnostic_patch
                            )
                            logger.info(
                                "Keeping the verified repair patch for submission; "
                                "TestAgent-generated tests remain diagnostic only."
                            )
                            break
                        else:
                            logger.warning(f"Test enhancement iteration {iteration + 1} failed. Attempting repair...")
                            
                            if not enhancement_result.failed_tests and not enhancement_result.failure_analysis:
                                logger.warning("No failure information available. Stopping iteration.")
                                break
                            
                            # Trigger CodeAgent for iterative repair
                            # Reuse the existing repair container for iterative repair
                            docker_env.reuse_existing = True
                            repair_result = run_iterative_repair_step(
                                instance_name=instance_name,
                                model_name=model_name,
                                docker_env=docker_env,
                                file_context=current_file_context,
                                memory=memory,
                                test_failures=enhancement_result.failed_tests,
                                failure_analysis=enhancement_result.failure_analysis or "",
                                output_dir=output_dir,
                                iteration=iteration + 1
                            )
                            
                            if repair_result.success:
                                current_patch = repair_result.enhanced_patch
                                current_file_context = repair_result.file_context
                                solution_patch = current_patch  # Update solution_patch for return value
                                # Save repaired patch
                                repaired_patch_path = os.path.join(output_dir, f"repaired_solution_iter{iteration + 1}.patch")
                                with open(repaired_patch_path, "w") as f:
                                    f.write(current_patch)
                                logger.info(f"Repaired solution patch saved to {repaired_patch_path}")
                                # Update solution.patch with the repaired version
                                with open(patch_path, "w") as f:
                                    f.write(current_patch)
                                logger.info(f"Updated solution.patch with repaired version")
                                if audit_result is not None:
                                    latest_candidate = sanitize_candidate_patch(
                                        current_file_context
                                    )
                                    patch_intent_audit_status = (
                                        _recheck_patch_intent_counterexample(
                                            auditor=auditor,
                                            audit_result=audit_result,
                                            file_context=current_file_context,
                                            candidate=latest_candidate,
                                            output_dir=output_dir,
                                            label=(
                                                "after_testagent_refine_"
                                                f"{iteration + 1}"
                                            ),
                                        )
                                    )
                                    if _patch_intent_audit_unresolved(
                                        patch_intent_audit_status
                                    ):
                                        logger.warning(
                                            "Stopping TestAgent refinement because "
                                            "the frozen Patch Intent Audit check is "
                                            "unresolved."
                                        )
                                        break
                            else:
                                logger.error("Iterative repair failed. Stopping.")
                                break
                else:
                    if patch_review_status == "pending":
                        patch_review_status = "skipped_basic_failure"
                    logger.warning("Basic tests did not pass. Skipping test enhancement.")

                _save_repair_verification_status(
                    output_dir,
                    verified_finish=verified_finish,
                    basic_tests_passed=basic_tests_passed,
                    patch_review_status=patch_review_status,
                    patch_intent_audit_status=(
                        patch_intent_audit_status
                        if _patch_intent_audit_enabled()
                        else None
                    ),
                )
                return solution_patch
            else:
                # Check if there were any file modifications at all
                has_any_patches = final_trajectory.file_context.has_patch(ignore_tests=False)
                if has_any_patches:
                    logger.warning(f"Repair completed but generate_git_patch() returned empty string. "
                                 f"FileContext has patches but they may have been reverted or only test files were modified.")
                else:
                    logger.warning("Repair was unsuccessful. No solution patch found and no file modifications detected.")
                if candidate_mode:
                    _save_portfolio_candidate_status(
                        output_dir,
                        finish_succeeded=final_trajectory.is_finished(),
                        basic_tests_passed=_portfolio_regression_passed(
                            final_trajectory.file_context
                        ),
                        candidate=None,
                    )
                return None
        except Exception as e:
            logger.error(f"Error generating or saving solution patch: {e}", exc_info=True)
            return None

def _basic_tests_passed(file_context: FileContext) -> bool:
    """
    检查基本测试是否通过（RunRegression 和 RunSingleTest）
    """
    try:
        verification = getattr(file_context, "verification", {}) or {}
        change_seq = getattr(file_context, "change_seq", 0)
        
        reg = verification.get("regression") or {}
        single = verification.get("single_test") or {}
        probe_change_seq = current_probe_change_seq(file_context)

        reg_ok = bool(reg.get("passed")) and reg.get("change_seq") == change_seq
        single_ok = (
            bool(single.get("passed"))
            and single.get("change_seq") == change_seq
            and single.get("probe_change_seq", probe_change_seq)
            == probe_change_seq
        )
        
        return reg_ok and single_ok
    except Exception as e:
        logger.warning(f"Error checking basic tests passed: {e}")
        return False


def _fresh_portfolio_regression(file_context: FileContext) -> Dict[str, Any] | None:
    """Return current production-view regression evidence when available."""
    verification = getattr(file_context, "verification", {}) or {}
    change_seq = getattr(file_context, "change_seq", 0)
    digest = production_patch_sha256(sanitize_candidate_patch(file_context).patch)
    for key in ("production_regression", "regression"):
        record = verification.get(key) or {}
        if (
            record.get("change_seq") == change_seq
            and record.get("verification_view") == "production"
            and record.get("production_patch_sha256") == digest
            and record.get("verification_isolation") == PRODUCTION_ISOLATION
        ):
            return record
    return None


def _portfolio_regression_passed(file_context: FileContext) -> bool:
    """Return only the current production regression result for a candidate."""
    record = _fresh_portfolio_regression(file_context)
    return bool(record and record.get("passed"))


def _portfolio_compliance_regression_evidence(
    file_context: FileContext,
) -> Dict[str, Dict[str, Any]]:
    """Return bounded current and production-only regression views for Audit."""
    verification = getattr(file_context, "verification", {}) or {}
    change_seq = getattr(file_context, "change_seq", 0)
    digest = production_patch_sha256(sanitize_candidate_patch(file_context).patch)
    evidence: Dict[str, Dict[str, Any]] = {}
    for key, label in (
        ("regression", "working_tree_with_verification_changes"),
        ("production_regression", "production_only_with_original_tests"),
    ):
        record = verification.get(key) or {}
        if record.get("change_seq") != change_seq:
            continue
        evidence[label] = {
            field: record.get(field)
            for field in (
                "passed",
                "exit_code",
                "log_type",
                "summary",
                "failed_test_names",
                "test_view",
                "test_view_stale",
                "verification_view",
                "production_patch_sha256",
                "verification_isolation",
            )
            if record.get(field) is not None
        }
        if record.get("output"):
            evidence[label]["output"] = _bounded_command_output(str(record["output"]))
        if (
            record.get("verification_isolation") != PRODUCTION_ISOLATION
            or record.get("production_patch_sha256") != digest
        ):
            evidence[label]["verification_view"] = "unproven"
    return evidence


def _audit_portfolio_branch_compliance(
    *,
    hypothesis_context: Dict[str, Any],
    candidate: CandidatePatch,
    repair_probes: List[RepairProbeEvidence],
    file_context: FileContext,
    model_name: str,
    output_dir: str,
    label: str,
) -> BranchComplianceDecision:
    """Check the frozen candidate against its assigned hypothesis branch."""
    prompt = build_branch_compliance_prompt(
        hypothesis_context=hypothesis_context,
        patch=candidate.patch,
        repair_probes=repair_probes,
        regression_evidence=_portfolio_compliance_regression_evidence(
            file_context
        ),
        verification_test_patch=getattr(candidate, "excluded_test_patch", ""),
    )
    try:
        if getattr(candidate, "artifact_errors", ()):
            decision = BranchComplianceDecision(
                status="inconclusive", observed_behavior="Submission artifact classification failed.",
                reason="Engineering correction is required before semantic verification.",
            )
        else:
            decision = _run_structured_extraction(
                prompt,
                BranchComplianceDecision,
                model_override=model_name,
                max_tokens=_env_int(
                    "COTEST_PORTFOLIO_BRANCH_AUDIT_MAX_TOKENS", 3000
                ),
                glm_reasoning_effort=_env_str(
                    "COTEST_GLM_REASONING_EFFORT", "low"
                ),
            )
        if not isinstance(decision, BranchComplianceDecision):
            raise ValueError(
                "Branch compliance Audit returned no validated payload"
            )
        raw_path = os.path.join(output_dir, f"branch_compliance_{label}_raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(decision.model_dump(), f, indent=2, ensure_ascii=False)
        raw_decision = decision
        decision = validate_branch_correction(
            decision, hypothesis_context=hypothesis_context, patch=candidate.patch,
            repair_probes=repair_probes,
            regression_evidence=_portfolio_compliance_regression_evidence(file_context),
            verification_test_patch=getattr(candidate, "excluded_test_patch", ""),
        )
        # Only fill evidence fields for the SAME current-patch correction. A
        # soft/unknown clause or stale patch cannot be promoted by refinement.
        if (
            raw_decision.actionable_correction and not decision.actionable_correction
            and raw_decision.production_patch_sha256 == production_patch_sha256(candidate.patch)
            and set(raw_decision.violated_clause_ids) <= set(branch_correction_clauses(hypothesis_context))
        ):
            try:
                refinement = _run_structured_extraction(
                    "Complete only the violations evidence structure for the existing correction below. "
                    "Keep exactly its violated_clause_ids; do not change the judgment, scope, hypothesis, "
                    "patch, suggested correction, test expectations, or observed facts. Copy exact clause "
                    "quotes from correction_authority_clauses and use only current PATCH/probe/production "
                    "regression or VERIFICATION_TEST_PATCH evidence IDs. If the evidence cannot support the original claim, return "
                    "an empty violations list. Return JSON with only violations.\n"
                    "verification_test_patch is evidence of tests already written but excluded from the "
                    "production submission. If it contradicts a claim that these tests were not written, "
                    "return an empty violations list. Never ask to ship test files as production code.\n"
                    + json.dumps({"original_decision": raw_decision.model_dump(),
                                  "validation_failure": decision.reason}, ensure_ascii=False)
                    + "\nFrozen audit evidence and full clause definitions:\n"
                    + prompt.split("Assigned-hypothesis evidence:\n", 1)[1]
                    + "\nReturn only JSON matching this schema:\n"
                    + json.dumps(BranchCorrectionRefinement.model_json_schema()),
                    BranchCorrectionRefinement, model_override=model_name,
                    max_tokens=4096, glm_reasoning_effort="low", max_attempts=1,
                )
                if not isinstance(refinement, BranchCorrectionRefinement):
                    raise ValueError("Correction evidence refine returned no validated payload")
                with open(os.path.join(output_dir, f"branch_compliance_{label}_refine.json"), "w", encoding="utf-8") as f:
                    json.dump(refinement.model_dump(), f, indent=2, ensure_ascii=False)
                decision = validate_branch_correction(
                    raw_decision.model_copy(update={"violations": refinement.violations}),
                    hypothesis_context=hypothesis_context, patch=candidate.patch, repair_probes=repair_probes,
                    regression_evidence=_portfolio_compliance_regression_evidence(file_context),
                    verification_test_patch=getattr(candidate, "excluded_test_patch", ""),
                )
            except Exception as exc:
                logger.warning("Branch correction evidence refine unavailable: %s", type(exc).__name__)
    except Exception as exc:
        logger.exception("Portfolio branch compliance Audit failed.")
        decision = BranchComplianceDecision(
            status="inconclusive",
            observed_behavior="The branch compliance Audit did not complete.",
            reason=str(exc),
        )

    path = os.path.join(output_dir, f"branch_compliance_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            decision.model_dump(), f, indent=2, ensure_ascii=False
        )
    return decision


def _portfolio_branch_requires_correction(
    *,
    compliance: BranchComplianceDecision,
    engineering_diagnostic: str = "",
) -> bool:
    return bool(
        engineering_diagnostic or (
            compliance.status == "noncompliant" and compliance.actionable_correction
        )
    )


def _render_portfolio_branch_correction_feedback(
    *,
    continuation_round: int,
    continuation_rounds: int,
    hypothesis_context: Dict[str, Any],
    compliance: BranchComplianceDecision,
    engineering_diagnostic: str = "",
) -> str:
    reasons: List[str] = []
    if compliance.status == "noncompliant" and compliance.actionable_correction:
        reasons.append(
            "The proposal does not implement its assigned hypothesis contract. "
            "Correct only the concrete hypothesis mismatch identified below; "
            "do not use this feedback to decide whether the hypothesis itself "
            "is the true Issue intent."
        )
        reasons.append(
            "Audit status: "
            f"{compliance.status}\n"
            f"Audit observation: {compliance.observed_behavior}\n"
            f"Reason: {compliance.reason}\n"
            f"Violated clauses: {compliance.violated_clause_ids}\n"
            f"Suggested correction: {compliance.suggested_correction}\n"
            "Hypothesis context: "
            + json.dumps(hypothesis_context, ensure_ascii=False)
        )
    if engineering_diagnostic:
        reasons.append(
            "The orchestrator found a concrete submission/build problem. Correct "
            "the file classification or implementation within the SAME assigned "
            "hypothesis; do not change its required behavior or test expectations.\n"
            + engineering_diagnostic
        )
        if not compliance.actionable_correction:
            reasons.append("No semantic contract correction is authorized by the audit.")
        reasons.append("Hypothesis context: " + json.dumps(hypothesis_context, ensure_ascii=False))
    return (
        "<portfolio_branch_compliance>\n"
        f"Continuation {continuation_round} of {continuation_rounds}. "
        + "\n\n".join(reasons)
        + "\n\nKeep required production files, exclude only disposable probes, "
        "verify the submitted production view, and call Finish again when the "
        "corrected production candidate is ready.\n"
        "</portfolio_branch_compliance>"
    )


def _ensure_portfolio_regression(
    file_context: FileContext, docker_env: DockerEnv
) -> Dict[str, Any] | None:
    """Mechanically run one current regression after a portfolio Finish."""
    record = _fresh_portfolio_regression(file_context)
    if record is not None:
        return record
    try:
        RunRegression(runtime=docker_env).execute(
            RunRegressionArgs(
                thoughts=(
                    "Check the finished portfolio candidate against the "
                    "repository build and regression suite."
                )
            ),
            file_context,
        )
    except Exception:
        logger.exception(
            "Automatic portfolio regression was unavailable; retaining it as "
            "missing soft evidence."
        )
        return None
    return _fresh_portfolio_regression(file_context)


def _portfolio_compile_regression_failure(file_context: FileContext) -> bool:
    record = _fresh_portfolio_regression(file_context) or {}
    diagnostic = str(record.get("summary", "")) + "\n" + str(record.get("output", ""))
    environment_failure = re.search(
        r"command not found|No CMAKE_CXX_COMPILER|Could NOT find|Cannot connect to the Docker daemon",
        diagnostic, re.I,
    )
    source_error = re.search(r":\d+(?::\d+)?:\s*(?:fatal\s+)?error:", diagnostic, re.I)
    return (
        not bool(record.get("passed"))
        and record.get("log_type") == LogType.COMPILATION_ERROR.value
        and (not environment_failure or bool(source_error))
    )


def _portfolio_engineering_diagnostic(
    file_context: FileContext, candidate: CandidatePatch,
    repair_probes: List[RepairProbeEvidence] | None = None,
) -> str:
    if candidate.artifact_errors:
        return "\n".join(candidate.artifact_errors)
    if _portfolio_compile_regression_failure(file_context):
        return "The submitted production view does not compile:\n" + _portfolio_regression_diagnostic(file_context)
    for probe in repair_probes or []:
        if (
            probe.production_patch_sha256 != production_patch_sha256(candidate.patch)
            or probe.expected_exit_code != 0 or probe.observed_status != "failed"
            or probe.observed_exit_code in (None, 0, 124, 125, 126, 127)
            or probe.kind == "soft_regression"
        ):
            continue
        # Require a compiler diagnostic in a delivered production file, not an
        # old test assertion, malformed probe, missing tool, or expected rejection.
        paths = re.findall(r"(?:^|\n)([^\n:]+):\d+(?::\d+)?:\s*(?:fatal\s+)?error:", probe.observed_output)
        if any(path.strip() == kept or path.strip().endswith("/" + kept)
               for path in paths for kept in candidate.kept_paths):
            return "A positive probe cannot compile the submitted production code:\n" + _bounded_command_output(
                probe.command + "\n" + probe.observed_output
            )
    return ""


def _portfolio_regression_diagnostic(file_context: FileContext) -> str:
    record = _fresh_portfolio_regression(file_context) or {}
    parts = [
        str(record.get("summary") or "Repository compilation failed."),
        str(record.get("output") or ""),
    ]
    return _bounded_command_output("\n".join(item for item in parts if item))

def _format_test_failures(failed_tests: List[Dict[str, str]]) -> str:
    """
    格式化失败的测试信息，供 CodeAgent 使用
    """
    if not failed_tests:
        return ""
    
    formatted = "The following tests failed:\n\n"
    for i, test in enumerate(failed_tests, 1):
        formatted += f"Test {i}:\n"
        formatted += f"  File: {test.get('file', 'Unknown')}\n"
        formatted += f"  Test Name: {test.get('test_name', 'Unknown')}\n"
        formatted += f"  Error: {test.get('error', 'Unknown error')}\n"
        if test.get('expected'):
            formatted += f"  Expected: {test.get('expected')}\n"
        if test.get('actual'):
            formatted += f"  Actual: {test.get('actual')}\n"
        if test.get('location'):
            formatted += f"  Location: {test.get('location')}\n"
        formatted += "\n"
    
    return formatted

def run_test_enhancement_step(
    instance_name: str,
    model_name: str,
    docker_env: DockerEnv,
    file_context: FileContext,
    memory: Memory,
    output_dir: str,
    iteration: int = 1
) -> TestEnhancementResult:
    """
    运行测试增强步骤，生成对抗测试和边界值测试用例
    
    返回 TestEnhancementResult，包含：
    - success: 是否所有测试都通过
    - enhanced_patch: 增强后的 patch
    - failed_tests: 失败的测试列表（如果有）
    - failure_analysis: 失败原因分析
    """
    logger.info(f"--- Starting Test Enhancement Step (iteration {iteration}) for {instance_name} ---")
    _load_env()
    
    instance = get_benchmark_instance(instance_name)
    repository = create_repository(instance)
    
    # Set SWEBENCH_INSTANCE_NAME for entity_tools
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    if not index_store_dir:
        raise ValueError("INDEX_STORE_DIR environment variable is not set. Please set it in .env file.")
    
    # Load the index
    code_index = CodeIndex.from_index_name(
        index_name=instance_name,
        file_repo=repository,
        index_store_dir=index_store_dir
    )
    
    # Note: docker_env is already in a 'with' context from run_repair_step, so we don't need another one
    logger.info(f"Docker container '{docker_env.container_name}' is running for test enhancement.")
    
    # Create a new file context from the existing one (copy state)
    test_file_context = FileContext(repo=repository, runtime=docker_env)
    # Reproduction files are evidence, not test-enhancement workspaces.  Carry
    # this invariant into the new context so action-level guards still apply.
    source_verification = getattr(file_context, "verification", {}) or {}
    immutable_reproduction_paths = list(
        source_verification.get("immutable_reproduction_paths", [])
    )
    if immutable_reproduction_paths:
        test_file_context.verification["immutable_reproduction_paths"] = (
            immutable_reproduction_paths
        )
    
    # Copy all files from the original file_context (not just patched files)
    # This ensures reproduction-created files are also available
    # 
    # IMPORTANT: In repair phase, file changes are stored as patches in FileContext,
    # not written to disk. We need to:
    # 1. Get the patched content using context_file.content (applies patch)
    # 2. Write it to the container so tests run against patched code
    # NOTE: Avoid writing patched content back to host repo to prevent base-content drift.
    for file_path, context_file in file_context._files.items():
        try:
            # Get content with patch applied (this is what TestAgent should see)
            file_content = context_file.content
            
            # IMPORTANT: Preserve original base content from the existing context
            # Avoid re-reading from repo if it may already contain patched content.
            base_content = None
            if context_file.patch:
                base_content = getattr(context_file, "_cached_base_content", None)
                if base_content is None:
                    # Get base content from context_file (before applying its patch)
                    # This handles initial_patch if present
                    try:
                        base_content = context_file.get_base_content()
                    except Exception as e:
                        logger.warning(f"Failed to get base content from context_file for {file_path}: {e}")
                        # Fallback: try to get from repository (may already be patched)
                        if repository.file_exists(file_path):
                            base_content = repository.get_file_content(file_path)
            
            # Add to test_file_context with content
            test_context_file = test_file_context.add_file(file_path, content=file_content)
            
            # IMPORTANT: Preserve the patch from original context_file if it exists
            # This ensures generate_git_patch() will include project code changes
            if context_file.patch:
                # Set the patch so generate_git_patch() can collect it
                test_context_file.patch = context_file.patch
                # Set base_content to the original repository content (before repair)
                if base_content is not None:
                    test_context_file._cached_base_content = base_content
                    test_context_file._initial_patch = None  # No initial patch needed
            
            # Write to container to ensure TestAgent sees patched code
            full_container_path = os.path.join(docker_env.work_dir, file_path)
            docker_env.write_file(full_container_path, file_content)
            
            # Do NOT write patched content back to host repo here; it can corrupt
            # base content for subsequent patch applications across stages.
            logger.debug(f"Copied patched file {file_path} to test_file_context (with patch preserved) and container")
        except Exception as e:
            logger.warning(f"Failed to get patched content for {file_path}: {e}. Trying fallback...")
            # Fallback: Try to get content from repository (may be unpatched)
            try:
                if repository and repository.file_exists(file_path):
                    file_content = repository.get_file_content(file_path)
                    if file_content is not None:
                        test_file_context.add_file(file_path, content=file_content)
                        logger.warning(f"Used unpatched content from repository for {file_path}")
                        continue
            except Exception as e2:
                logger.warning(f"Failed to read {file_path} from repository: {e2}")
            
            # Last resort: Use base content (before patch)
            try:
                file_content = context_file.get_base_content()
                test_file_context.add_file(file_path, content=file_content)
                logger.warning(f"Used base content (before patch) for {file_path}")
            except Exception as e3:
                logger.error(f"Failed to copy {file_path} to test_file_context: {e3}. Skipping this file.")
    
    # Also copy test files from memory (ensure they're in the context even if not in file_context)
    if memory.test_files:
        for test_file in memory.test_files:
            relative_path = test_file['path'].lstrip('/')
            full_container_path = os.path.join(docker_env.work_dir, relative_path)
            docker_env.write_file(full_container_path, test_file['content'])
            full_host_path = repository.get_full_path(relative_path)
            os.makedirs(os.path.dirname(full_host_path), exist_ok=True)
            with open(full_host_path, "w") as f:
                f.write(test_file['content'])
            # Add to context if not already added
            if relative_path not in test_file_context._files:
                test_file_context.add_file(relative_path, content=test_file['content'])
    
    # Create TestAgent
        agent = TestAgent.create(
            repository=repository,
            code_index=code_index,
            model=model_name,
            runtime=docker_env,
            message_cache=True,
            **_completion_protocol_kwargs(model_name),
            **_model_kwargs(model_name, "COTEST_TEST_ENHANCEMENT_MAX_TOKENS"),
        )
        
        repo_structure = _repo_structure_for_model(repository.repo_path, model_name)
        
        # Get the current patch
        current_patch = file_context.generate_git_patch()
        
        # Build test enhancement message
        message = (
            f"<task>\n"
            f"Test Enhancement Task (Iteration {iteration})\n\n"
            f"The following code patch has been applied and has passed basic PoC and regression tests. "
            f"Your task is to generate comprehensive test cases to enhance the robustness of this patch.\n\n"
            f"Problem Statement:\n{memory.problem_statement}\n\n"
        )
        
        if current_patch:
            message += f"<Current_Patch>\n{current_patch}\n</Current_Patch>\n\n"
        
        if memory.test_files:
            poc_test_paths = [f"  - {test_file['path']}" for test_file in memory.test_files]
            poc_block = "<Poc_test>\n" + "\n".join(poc_test_paths) + "\n</Poc_test>"
            message += (
            f"The following PoC test file(s) exist:\n"
            f"{poc_block}\n\n"
            f"These PoC files are immutable evidence. Do not edit, replace, delete, "
            f"or append to them; create a separate test file for any new coverage.\n\n"
        )
        
        if memory.test_command:
            command_block = f"<Poc_command>\n{memory.test_command}\n</Poc_command>"
            message += (
                f"Use the following command to run the PoC test:\n"
                f"{command_block}\n\n"
            )
        
        message += (
            f"<Requirements>\n"
            f"1. Generate adversarial test cases that try to find weaknesses in the patch\n"
            f"2. Generate boundary value test cases (extreme values, edge cases, invalid inputs, error paths, overflow/underflow, empty/null values)\n"
            f"3. Ensure at least some test cases would fail on the buggy version and pass only because of the fix\n"
            f"4. Run all new tests and verify they pass\n"
            f"5. If any tests fail, provide detailed failure analysis\n"
            f"</Requirements>\n\n"
            f"Repo Structure:\n{repo_structure}\n\n"
            f"Please generate comprehensive test cases to enhance the robustness of the patch.\n"
            f"</task>\n"
        )
        
        persist_path = os.path.join(output_dir, f"test_enhancement_trajectory_{instance_name}_iter{iteration}.json")
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        test_enhancement_steps = _env_int("COTEST_TEST_ENHANCEMENT_STEPS", 30)
        logger.info(f"Test enhancement max iterations: {test_enhancement_steps}")
        metadata = {
            "instance_id": instance_name,
            "problem_statement": memory.problem_statement,
            "iteration": iteration,
            "result_persist_path": output_dir,
        }
        
        # Create loop
        loop = AgenticLoop.create(
            agent=agent,
            file_context=test_file_context,
            message=message,
            repository=repository,
            runtime=docker_env,
            code_index=code_index,
            metadata=metadata,
            persist_path=persist_path,
            max_iterations=test_enhancement_steps
        )
        
        loop.memory_module = None
        
        logger.info("Running test enhancement loop...")
        try:
            final_trajectory = loop.run()
        except Exception as e:
            logger.error(f"Test enhancement loop failed with exception: {e}", exc_info=True)
            from INTENT2FIX.exceptions import CompletionRuntimeError
            if isinstance(e, CompletionRuntimeError):
                logger.error(f"CompletionRuntimeError detected during test enhancement.")
            return TestEnhancementResult(
                success=False,
                enhanced_patch=None,
                failed_tests=[],
                failure_analysis=f"Test enhancement loop failed: {str(e)}",
                new_test_files=[],
                file_context=test_file_context
            )
        
        try:
            enhanced_patch = final_trajectory.file_context.generate_git_patch()
            
            # Check if tests passed by examining the final trajectory
            # Look for TestEnhancementFinish action in the final node
            failed_tests = []
            failure_analysis = None
            new_test_files = []
            
            # Extract information from the final node's action
            final_node = final_trajectory
            if final_node and final_node.action_steps:
                last_action = final_node.action_steps[-1].action
                action_name = last_action.name
                
                if action_name == "TestEnhancementNeedRepair":
                    success = False
                    failure_analysis = getattr(last_action, 'failure_analysis', "No analysis provided")
                    failed_tests.append({
                        "file": "Unknown",
                        "test_name": "NeedRepair",
                        "error": failure_analysis
                    })
                    logger.info(f"Test enhancement found bugs. Action: {action_name}. Triggering repair.")
                elif action_name == "TestEnhancementPassed":
                    success = True
                    logger.info(f"Test enhancement successful. Action: {action_name}. No repair needed.")
                else:
                    # Test enhancement has an explicit terminal protocol.  A
                    # tool action is not evidence that the new tests passed;
                    # accepting it can hide a failed regression or a truncated
                    # agent trajectory.
                    success = False
                    failure_analysis = (
                        f"Unexpected terminal action {action_name}; "
                        "the test-enhancement agent must finish with "
                        "TestEnhancementPassed or TestEnhancementNeedRepair."
                    )
                    failed_tests.append({
                        "file": "Unknown",
                        "test_name": "UnexpectedTerminalAction",
                        "error": failure_analysis,
                    })
                    logger.warning(f"Test enhancement rejected: {failure_analysis}")

                if hasattr(last_action, 'new_test_files'):
                    new_test_files = last_action.new_test_files
            
            if final_trajectory and enhanced_patch:
                logger.info(f"Test enhancement {'successful' if success else 'completed with failures'}!")
                return TestEnhancementResult(
                    success=success,
                    enhanced_patch=enhanced_patch,
                    failed_tests=failed_tests,
                    failure_analysis=failure_analysis,
                    new_test_files=new_test_files,
                    file_context=final_trajectory.file_context
                )
            else:
                logger.warning("Test enhancement was unsuccessful. No enhanced patch found.")
                return TestEnhancementResult(
                    success=False,
                    enhanced_patch=None,
                    failed_tests=[],
                    failure_analysis="No enhanced patch generated",
                    new_test_files=[],
                    file_context=test_file_context
                )
        except Exception as e:
            logger.error(f"Error processing test enhancement result: {e}", exc_info=True)
            return TestEnhancementResult(
                success=False,
                enhanced_patch=None,
                failed_tests=[],
                failure_analysis=f"Error processing result: {str(e)}",
                new_test_files=[],
                file_context=test_file_context
            )

def run_iterative_repair_step(
    instance_name: str,
    model_name: str,
    docker_env: DockerEnv,
    file_context: FileContext,
    memory: Memory,
    test_failures: List[Dict[str, str]],
    failure_analysis: str,
    output_dir: str,
    iteration: int = 1
) -> RepairResult:
    """
    运行迭代修复步骤，根据测试失败信息修复代码
    
    返回 RepairResult，包含：
    - success: 是否修复成功
    - enhanced_patch: 修复后的 patch
    - file_context: 更新后的 file_context
    """
    logger.info(f"--- Starting Iterative Repair Step (iteration {iteration}) for {instance_name} ---")
    _load_env()
    
    instance = get_benchmark_instance(instance_name)
    repository = create_repository(instance)
    
    # Set SWEBENCH_INSTANCE_NAME for entity_tools
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    if not index_store_dir:
        raise ValueError("INDEX_STORE_DIR environment variable is not set. Please set it in .env file.")
    
    # Load the index
    code_index = CodeIndex.from_index_name(
        index_name=instance_name,
        file_repo=repository,
        index_store_dir=index_store_dir
    )
    
    # Note: docker_env is already in a 'with' context from run_repair_step, so we don't need another one
    logger.info(f"Docker container '{docker_env.container_name}' is running for iterative repair.")
    
    # Create a new file context from the existing one
    repair_file_context = FileContext(repo=repository, runtime=docker_env)
    
    # Copy all files from the original file_context (not just patched files)
    # This ensures all files from repair phase are available, including those that may have been modified
    # IMPORTANT: We need to preserve patches and ensure content is correctly set
    for file_path, context_file in file_context._files.items():
        try:
            # Get content with patch applied (this is what agent should see)
            file_content = context_file.content
            
            # IMPORTANT: Preserve original base content from the existing context
            # Avoid re-reading from repo if it may already contain patched content.
            base_content = None
            if context_file.patch:
                base_content = getattr(context_file, "_cached_base_content", None)
                if base_content is None:
                    try:
                        base_content = context_file.get_base_content()
                    except Exception as e:
                        logger.warning(f"Failed to get base content from context_file for {file_path}: {e}")
                        # Fallback: try to get from repository (may already be patched)
                        if repository.file_exists(file_path):
                            base_content = repository.get_file_content(file_path)
            
            # Add to repair_file_context with content
            repair_context_file = repair_file_context.add_file(file_path, content=file_content)
            
            # IMPORTANT: Preserve the patch from original context_file if it exists
            # This ensures generate_git_patch() will include project code changes
            if context_file.patch:
                # Set the patch so generate_git_patch() can collect it
                repair_context_file.patch = context_file.patch
                # Set base_content to the original repository content (before repair)
                if base_content is not None:
                    repair_context_file._cached_base_content = base_content
                    repair_context_file._initial_patch = None  # No initial patch needed
            
            # Write to container to ensure agent sees patched code
            full_container_path = os.path.join(docker_env.work_dir, file_path)
            docker_env.write_file(full_container_path, file_content)
            
            # Do NOT write patched content back to host repo here; it can corrupt
            # base content for subsequent patch applications across stages.
            logger.debug(f"Copied patched file {file_path} to repair_file_context (with patch preserved) and container")
        except Exception as e:
            logger.warning(f"Failed to copy {file_path} to repair_file_context: {e}. Skipping this file.")
    
    # Also copy test files from memory
    if memory.test_files:
        for test_file in memory.test_files:
            relative_path = test_file['path'].lstrip('/')
            full_container_path = os.path.join(docker_env.work_dir, relative_path)
            docker_env.write_file(full_container_path, test_file['content'])
            full_host_path = repository.get_full_path(relative_path)
            os.makedirs(os.path.dirname(full_host_path), exist_ok=True)
            with open(full_host_path, "w") as f:
                f.write(test_file['content'])
            repair_file_context.add_file(relative_path, content=test_file['content'])
    
    # Create CodingAgent
    agent = CodingAgent.create(
        repository=repository,
        code_index=code_index,
        model=model_name,
        runtime=docker_env,
        message_cache=True,
        **_completion_protocol_kwargs(model_name),
        **_model_kwargs(model_name, "COTEST_ITERATIVE_REPAIR_MAX_TOKENS"),
    )
    
    repo_structure = _repo_structure_for_model(repository.repo_path, model_name)
    
    # Get the current patch
    current_patch = file_context.generate_git_patch()
    
    # Format test failures
    formatted_failures = _format_test_failures(test_failures)
    
    # Build iterative repair message
    message = (
        f"<task>\n"
        f"Iterative Repair Task (Iteration {iteration})\n\n"
        f"The following test cases failed after test enhancement. "
        f"Your task is to fix the code to make these tests pass.\n\n"
        f"Problem Statement:\n{memory.problem_statement}\n\n"
    )
    
    if current_patch:
        message += f"<Current_Patch>\n{current_patch}\n</Current_Patch>\n\n"
    
    message += f"<Failed_Tests>\n{formatted_failures}\n</Failed_Tests>\n\n"
    
    if failure_analysis:
        message += f"<Failure_Analysis>\n{failure_analysis}\n</Failure_Analysis>\n\n"
    
    message += (
        f"<Requirements>\n"
        f"1. Analyze why the tests failed\n"
        f"2. Fix the code (NOT the tests) to make the failing tests pass\n"
        f"3. Ensure your fix does not break existing functionality\n"
        f"4. Run RunRegression to verify no regressions\n"
        f"5. Run RunSingleTest to verify the failing tests now pass\n"
        f"</Requirements>\n\n"
        f"Repo Structure:\n{repo_structure}\n\n"
        f"Please fix the code to make the failing tests pass.\n"
        f"</task>\n"
    )
    
    persist_path = os.path.join(output_dir, f"iterative_repair_trajectory_{instance_name}_iter{iteration}.json")
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    iterative_repair_steps = _env_int("COTEST_ITERATIVE_REPAIR_STEPS", 40)
    logger.info(f"Iterative repair max iterations: {iterative_repair_steps}")
    metadata = {
        "instance_id": instance_name,
        "problem_statement": memory.problem_statement,
        "iteration": iteration,
        "result_persist_path": output_dir,
    }
    
    # Create loop
    loop = AgenticLoop.create(
        agent=agent,
        file_context=repair_file_context,
        message=message,
        repository=repository,
        runtime=docker_env,
        code_index=code_index,
        metadata=metadata,
        persist_path=persist_path,
        max_iterations=iterative_repair_steps
    )
    
    loop.memory_module = None
    
    logger.info("Running iterative repair loop...")
    try:
        final_trajectory = loop.run()
    except Exception as e:
        logger.error(f"Iterative repair loop failed with exception: {e}", exc_info=True)
        from INTENT2FIX.exceptions import CompletionRuntimeError
        if isinstance(e, CompletionRuntimeError):
            logger.error(f"CompletionRuntimeError detected during iterative repair.")
        return RepairResult(
            success=False,
            enhanced_patch=None,
            file_context=repair_file_context
        )
    
    try:
        enhanced_patch = final_trajectory.file_context.generate_git_patch()
        
        # Check if tests passed
        verification = getattr(final_trajectory.file_context, "verification", {}) or {}
        reg = verification.get("regression") or {}
        single = verification.get("single_test") or {}
        
        success = bool(reg.get("passed")) and bool(single.get("passed"))
        
        if final_trajectory and enhanced_patch:
            logger.info(f"Iterative repair {'successful' if success else 'completed but tests may still fail'}!")
            return RepairResult(
                success=success,
                enhanced_patch=enhanced_patch,
                file_context=final_trajectory.file_context
            )
        else:
            logger.warning("Iterative repair was unsuccessful. No enhanced patch found.")
            return RepairResult(
                success=False,
                enhanced_patch=None,
                file_context=repair_file_context
            )
    except Exception as e:
        logger.error(f"Error processing iterative repair result: {e}", exc_info=True)
        return RepairResult(
            success=False,
            enhanced_patch=None,
            file_context=repair_file_context
        )

def extract_instance_ids_from_jsonl(file_path: str) -> List[str]:
    instance_ids = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json_object = json.loads(line.strip())
                    if 'instance_id' in json_object:
                        instance_ids.append(json_object['instance_id'])
                except json.JSONDecodeError:
                    logger.warning(f"Skipping line due to JSON decode error: {line.strip()[:100]}...")
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
    return instance_ids

def _signal_handler(signum, frame):
    """Handle system signals that might terminate the process"""
    raise SystemExit(f"Process terminated by signal {signum}")


def _main_interrupt_handler(signum, frame):
    """Route terminal and service-manager stops through coordinated shutdown."""
    raise KeyboardInterrupt


def _build_issue_only_memory(
    instance_name: str,
    benchmark_instance: Dict[str, Any],
) -> Memory:
    return Memory(
        instance_id=instance_name,
        problem_statement=benchmark_instance["problem_statement"],
        reproduction_summary="",
        repair_summary="",
        solution_patch="",
        golden_patch="",
        test_command="",
        test_files=[],
        reproduction_result="",
        expected_reproduction_result="",
    )


def _fallback_intent_portfolio(memory: Memory) -> IntentPortfolio:
    contract = memory.repair_intent_contract or {}
    ledger = [
        item
        for item in contract.get("intent_ledger", [])
        if isinstance(item, dict)
    ]
    grounded_targets = list(dict.fromkeys(
        value.strip()
        for value in [
            *contract.get("target_behavior", []),
            *(item.get("supported_delta", "") for item in ledger),
        ]
        if isinstance(value, str) and value.strip()
    ))
    grounded_preserves = list(dict.fromkeys(
        value.strip()
        for value in contract.get("preserve_behavior", [])
        if isinstance(value, str) and value.strip()
    ))
    evidence_ids = list(dict.fromkeys(
        evidence_id
        for item in ledger
        for evidence_id in item.get("evidence_ids", [])
        if isinstance(evidence_id, str) and evidence_id.strip()
    ))
    fix_statement = (
        "\n".join(grounded_targets)
        if grounded_targets
        else memory.problem_statement
    )
    obligations = [
        IntentObligation(
            id="FIX-1",
            kind="fix",
            statement=fix_statement,
            authority="issue_explicit",
            strength="hard",
            evidence_basis=(
                ["Grounded Intent Scout contract", *evidence_ids[:6]]
                if grounded_targets
                else ["Original Issue"]
            ),
        )
    ]
    for index, statement in enumerate(grounded_preserves[:4], start=1):
        obligations.append(
            IntentObligation(
                id=f"PRESERVE-{index}",
                kind="preserve",
                statement=statement,
                authority="execution_backed",
                strength="soft",
                evidence_basis=["Grounded Intent Scout contract"],
            )
        )
    covered_ids = [item.id for item in obligations]
    return IntentPortfolio(
        obligations=obligations,
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title=(
                    "Grounded Scout contract fallback"
                    if grounded_targets
                    else "Complete literal Issue contract"
                ),
                branch_assumption="none",
                root_cause=(
                    contract.get("local_behavior_witnesses", [""])[0]
                    if contract.get("local_behavior_witnesses")
                    else ""
                ),
                repair_scope=[
                    "Repair every public surface explicitly named by this "
                    f"grounded target: {statement}"
                    for statement in grounded_targets
                ],
                covered_obligation_ids=covered_ids,
                hard_obligation_coverage={
                    "FIX-1": (
                        "Implement and verify the grounded Scout target on all "
                        "public surfaces it explicitly names."
                        if grounded_targets
                        else "Implement and verify the original Issue contract."
                    )
                },
                expected_behavior=grounded_targets,
                preservation_behavior=grounded_preserves,
                evidence_basis=(
                    ["Grounded Intent Scout contract", *evidence_ids[:6]]
                    if grounded_targets
                    else ["Original Issue"]
                ),
            )
        ],
        unresolved_dimensions=[],
    )


def _generate_intent_portfolio_with_verification(
    memory, model_name, output_dir, *, image_name, work_dir, base_commit,
    container_suffix,
):
    expanded = _env_bool("COTEST_PORTFOLIO_RERANK", False)
    expand_source = expanded and _env_bool("COTEST_PORTFOLIO_EXPAND_SOURCE", False)
    verify = expanded and _env_bool("COTEST_PORTFOLIO_VERIFY_PREMISES", False)
    if not (expand_source or verify):
        return _generate_intent_portfolio(memory, model_name, output_dir)
    with DockerEnv(
        image_name=image_name,
        container_name=f"hypothesis-audit-{memory.instance_id}-{container_suffix}",
        work_dir=work_dir,
        benchmark_base_commit=base_commit,
    ) as env:
        if expand_source:
            from INTENT2FIX.hypothesis_source import expand_source_evidence

            memory = memory.model_copy(update={"repair_intent_bundle": expand_source_evidence(
                env, memory.repair_intent_bundle or {}, memory.issue_evidence or {}, output_dir
            )})
        if verify:
            return _generate_intent_portfolio(
                memory, model_name, output_dir,
                audit_tools=HypothesisEvidenceTools(
                    env, output_dir,
                    scout_probe_snapshot=(memory.repair_intent_bundle or {}).get("scout_probe_snapshot"),
                ),
            )
    return _generate_intent_portfolio(memory, model_name, output_dir)


def _generate_intent_portfolio(
    memory: Memory,
    model_name: str,
    output_dir: str | None = None,
    audit_tools: HypothesisEvidenceTools | None = None,
) -> IntentPortfolio:
    expanded_pool = _env_bool("COTEST_PORTFOLIO_RERANK", False)
    portfolio_schema = IntentHypothesisPool if expanded_pool else IntentPortfolio
    audit_schema = HypothesisPoolAudit if expanded_pool else PortfolioCoverageAudit
    cache_dir = (
        "" if _graph_ablation_enabled()
        else os.getenv("COTEST_INTENT_PORTFOLIO_CACHE_DIR", "").strip()
    )
    if cache_dir and audit_tools is None and not (
        expanded_pool and (
            _env_bool("COTEST_PORTFOLIO_EXPAND_SOURCE", False)
            or _env_bool("COTEST_PORTFOLIO_POLICY_PROMPT", False)
        )
    ):
        cache_path = os.path.join(
            _resolve_jsonl_path(cache_dir),
            memory.instance_id,
            "intent_portfolio.json",
        )
        if os.path.isfile(cache_path):
            try:
                cached = portfolio_schema.model_validate(
                    _read_json_file(cache_path)
                )
                if expanded_pool:
                    cached_audit = HypothesisPoolAudit.model_validate(
                        _read_json_file(os.path.join(
                            os.path.dirname(cache_path), "intent_hypothesis_rerank.json"
                        ))
                    )
                    reviewed = select_hypothesis_pool(cached, cached_audit)
                    if reviewed.selected_hypothesis_ids != cached.selected_hypothesis_ids:
                        raise ValueError("Cached pool selection differs from its audit")
                cached.repair_hypotheses
                logger.info(
                    "Reused schema-valid Intent portfolio checkpoint: %s",
                    cache_path,
                )
                return cached
            except (ValidationError, ValueError):
                logger.warning(
                    "Ignoring invalid Intent portfolio checkpoint: %s",
                    cache_path,
                    exc_info=True,
                )
    contract = memory.repair_intent_contract or {}
    bundle = memory.repair_intent_bundle or {}
    open_decisions = (
        contract.get("open_decisions")
        or bundle.get("open_decisions")
        or []
    )
    merge_adjudication = HypothesisMergeAdjudication()
    if open_decisions and not expanded_pool:
        adjudication_prompt = build_hypothesis_merge_adjudication_prompt(
            issue=memory.problem_statement,
            issue_evidence=memory.issue_evidence,
            repair_intent_bundle=bundle,
            repair_intent_contract=contract,
        )
        try:
            extracted_adjudication = _run_structured_extraction(
                adjudication_prompt,
                HypothesisMergeAdjudication,
                model_override=model_name,
                max_tokens=_env_int(
                    "COTEST_PORTFOLIO_MERGE_MAX_TOKENS", 2400
                ),
                glm_reasoning_effort=_env_str(
                    "COTEST_GLM_REASONING_EFFORT", "low"
                ),
            )
            if isinstance(
                extracted_adjudication, HypothesisMergeAdjudication
            ):
                merge_adjudication = extracted_adjudication
            else:
                logger.warning(
                    "Hypothesis merge adjudication returned no validated "
                    "payload; preserving the original open decisions."
                )
        except Exception:
            logger.exception(
                "Hypothesis merge adjudication failed; preserving the "
                "original open decisions."
            )
        if output_dir:
            with open(
                os.path.join(
                    output_dir,
                    "intent_hypothesis_merge_adjudication.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    merge_adjudication.model_dump(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    prompt = build_hypothesis_prompt(
        issue=memory.problem_statement,
        issue_evidence=memory.issue_evidence,
        repair_intent_bundle=bundle,
        repair_intent_contract=contract,
        scout_summary=memory.repair_summary,
        merge_adjudication=merge_adjudication.model_dump(),
        expanded_pool=expanded_pool,
        policy_prompt=_env_bool("COTEST_PORTFOLIO_POLICY_PROMPT", False),
    )
    try:
        portfolio = _run_structured_extraction(
            prompt,
            portfolio_schema,
            model_override=model_name,
            max_tokens_key="COTEST_PORTFOLIO_HYPOTHESIS_MAX_TOKENS",
            glm_reasoning_effort=_env_str(
                "COTEST_GLM_REASONING_EFFORT", "low"
            ),
        )
        if not isinstance(portfolio, portfolio_schema):
            raise ValueError(
                "Intent portfolio extraction returned no validated payload"
            )
    except Exception:
        if expanded_pool:
            # The legacy fallback has no per-branch audit and cannot stand in
            # for a reviewed pool in this mode.
            raise
        logger.exception(
            "Intent portfolio extraction failed; using one grounded H0 branch."
        )
        return _fallback_intent_portfolio(memory)

    if output_dir:
        with open(
            os.path.join(output_dir, "intent_portfolio_initial.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(portfolio.model_dump(), f, indent=2, ensure_ascii=False)

    if expanded_pool:
        portfolio.selected_hypothesis_ids = []
        if output_dir:
            with open(os.path.join(output_dir, "intent_hypothesis_pool.json"), "w") as f:
                json.dump(portfolio.model_dump(), f, indent=2, ensure_ascii=False)

    audit_prompt = build_portfolio_coverage_audit_prompt(
        issue=memory.problem_statement,
        issue_evidence=memory.issue_evidence,
        repair_intent_bundle=bundle,
        repair_intent_contract=contract,
        scout_summary=memory.repair_summary,
        merge_adjudication=merge_adjudication.model_dump(),
        portfolio=portfolio,
    )
    coverage_audit: PortfolioCoverageAudit | None = None
    try:
        extracted_audit = _run_structured_extraction(
            audit_prompt,
            audit_schema,
            audit_pool=portfolio if expanded_pool else None,
            audit_tools=audit_tools,
            model_override=model_name,
            max_tokens=_env_int(
                "COTEST_PORTFOLIO_COVERAGE_AUDIT_MAX_TOKENS",
                16384 if expanded_pool else 3200,
            ),
            glm_reasoning_effort=_env_str(
                "COTEST_GLM_REASONING_EFFORT", "low"
            ),
        )
        if not isinstance(extracted_audit, audit_schema):
            raise ValueError(
                "Intent portfolio coverage Audit returned no validated payload"
            )
        coverage_audit = extracted_audit
        if expanded_pool:
            coverage_audit.validate_pool(portfolio)
        if output_dir:
            with open(
                os.path.join(
                    output_dir, "intent_portfolio_coverage_audit.json"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    coverage_audit.model_dump(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
    except Exception:
        logger.exception(
            "Intent portfolio coverage Audit failed; keeping initial portfolio."
        )

    if audit_tools is not None:
        bundle = {**bundle, "cited_repository_evidence": [
            *bundle.get("cited_repository_evidence", []), *audit_tools.evidence,
        ]}
    revised = False
    if coverage_audit is not None and coverage_audit.decision == "REVISE":
        revision_prompt = build_portfolio_revision_prompt(
            issue=memory.problem_statement,
            issue_evidence=memory.issue_evidence,
            repair_intent_bundle=bundle,
            repair_intent_contract=contract,
            scout_summary=memory.repair_summary,
            merge_adjudication=merge_adjudication.model_dump(),
            portfolio=portfolio,
            audit=coverage_audit,
            policy_prompt=_env_bool("COTEST_PORTFOLIO_POLICY_PROMPT", False),
        )
        try:
            revised_portfolio = _run_structured_extraction(
                revision_prompt,
                portfolio_schema,
                model_override=model_name,
                max_tokens_key="COTEST_PORTFOLIO_HYPOTHESIS_MAX_TOKENS",
                glm_reasoning_effort=_env_str(
                    "COTEST_GLM_REASONING_EFFORT", "low"
                ),
            )
            if not isinstance(revised_portfolio, portfolio_schema):
                raise ValueError(
                    "Intent portfolio revision returned no validated payload"
                )
            portfolio = revised_portfolio
            revised = True
        except Exception:
            logger.exception(
                "Intent portfolio revision failed; keeping initial portfolio."
            )

    if expanded_pool:
        portfolio.selected_hypothesis_ids = []
        if output_dir:
            with open(os.path.join(output_dir, "intent_hypothesis_pool.json"), "w") as f:
                json.dump(portfolio.model_dump(), f, indent=2, ensure_ascii=False)
        if revised:
            # IDs and policies may change during revision; never reuse stale ranks.
            coverage_audit = None
            try:
                extracted_audit = _run_structured_extraction(
                    build_portfolio_coverage_audit_prompt(
                        issue=memory.problem_statement,
                        issue_evidence=memory.issue_evidence,
                        repair_intent_bundle=bundle,
                        repair_intent_contract=contract,
                        scout_summary=memory.repair_summary,
                        merge_adjudication=merge_adjudication.model_dump(),
                        portfolio=portfolio,
                    ),
                    HypothesisPoolAudit,
                    audit_pool=portfolio,
                    audit_tools=audit_tools,
                    model_override=model_name,
                    max_tokens=_env_int(
                        "COTEST_PORTFOLIO_COVERAGE_AUDIT_MAX_TOKENS", 16384
                    ),
                    glm_reasoning_effort=_env_str(
                        "COTEST_GLM_REASONING_EFFORT", "low"
                    ),
                )
                if not isinstance(extracted_audit, HypothesisPoolAudit):
                    raise ValueError(
                        "Revised Hypothesis Audit returned no validated payload"
                    )
                coverage_audit = extracted_audit
            except Exception:
                logger.exception(
                    "Revised Hypothesis Audit failed; scheduling generated "
                    "branches for best-effort Repair."
                )

        selection_mode = "generated_pool_fallback"
        if isinstance(coverage_audit, HypothesisPoolAudit):
            if output_dir:
                with open(os.path.join(output_dir, "intent_hypothesis_rerank.json"), "w") as f:
                    json.dump(coverage_audit.model_dump(), f, indent=2, ensure_ascii=False)
            requested = [
                item_id for item_id in coverage_audit.selected_hypothesis_ids
                if item_id in {item.id for item in portfolio.hypotheses}
            ][:3]
            portfolio = select_hypothesis_pool(portfolio, coverage_audit)
            selection_mode = (
                "audit_ranked"
                if requested and requested == portfolio.selected_hypothesis_ids
                else "audit_review_fallback"
            )
        else:
            payload = portfolio.model_dump()
            payload["selected_hypothesis_ids"] = [
                item.id for item in portfolio.hypotheses[:3]
            ]
            portfolio = IntentHypothesisPool.model_validate(payload)

        if output_dir:
            audit_findings = {}
            selected_reviews = {}
            if isinstance(coverage_audit, HypothesisPoolAudit):
                audit_findings = {
                    field: list(getattr(coverage_audit, field))
                    for field in (
                        "uncovered_issue_obligations",
                        "repair_surface_gaps",
                        "causal_coverage_gaps",
                        "unsupported_hard_preserves",
                        "internal_conflicts",
                        "nonblocking_notes",
                    )
                }
                selected_reviews = {
                    item_id: coverage_audit.hypothesis_reviews[item_id].model_dump()
                    for item_id in portfolio.selected_hypothesis_ids
                }
            with open(
                os.path.join(output_dir, "intent_hypothesis_execution.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({
                    "audit_available": isinstance(
                        coverage_audit, HypothesisPoolAudit
                    ),
                    "audit_decision": (
                        coverage_audit.decision
                        if isinstance(coverage_audit, HypothesisPoolAudit)
                        else "unavailable"
                    ),
                    "selection_mode": selection_mode,
                    "selected_hypothesis_ids": portfolio.selected_hypothesis_ids,
                    "audit_reason": (
                        coverage_audit.reason
                        if isinstance(coverage_audit, HypothesisPoolAudit)
                        else "Final Hypothesis Audit was unavailable or invalid."
                    ),
                    "blocking_findings": audit_findings,
                    "selected_hypothesis_reviews": selected_reviews,
                }, f, indent=2, ensure_ascii=False)
        logger.info(
            "Hypothesis pool: %d proposed; scheduled %s; coverage=%s; mode=%s",
            len(portfolio.hypotheses), portfolio.selected_hypothesis_ids,
            (
                coverage_audit.decision
                if isinstance(coverage_audit, HypothesisPoolAudit)
                else "unavailable"
            ),
            selection_mode,
        )

    if output_dir:
        with open(
            os.path.join(output_dir, "intent_portfolio_coverage.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                portfolio.coverage_report(), f, indent=2, ensure_ascii=False
            )
    return portfolio


def _read_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _bounded_command_output(output: str, limit: int = 6000) -> str:
    if len(output) <= limit:
        return output
    half = limit // 2
    return (
        output[:half]
        + f"\n... [{len(output) - limit} characters omitted] ...\n"
        + output[-half:]
    )


def _portfolio_frozen_probes(memory: Memory) -> List[tuple[str, dict]]:
    raw_contract = getattr(memory, "repair_intent_contract", {}) or {}
    if not raw_contract:
        return []
    try:
        if "probes" in raw_contract:
            spec = raw_contract
        else:
            spec = RepairIntentContract.model_validate(
                raw_contract
            ).verification_spec()
    except ValidationError:
        logger.warning(
            "Portfolio could not validate the frozen intent contract; shared "
            "probe replay is unavailable.",
            exc_info=True,
        )
        return []
    probes = spec.get("probes") or {}
    ordered_ids = list(spec.get("acceptance_ids") or [])
    ordered_ids.extend(
        check_id
        for check_id in spec.get("diagnostic_ids") or []
        if check_id not in ordered_ids
    )
    return [
        (
            check_id,
            {
                **probes[check_id],
                "origin": "scout",
                "source_hypothesis_id": "",
            },
        )
        for check_id in ordered_ids
        if check_id in probes
    ]


def _portfolio_repair_probe_specs(
    candidates: List[PortfolioCandidate],
) -> List[tuple[str, dict]]:
    probes: List[tuple[str, dict]] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        source_candidate_id = (
            f"{candidate.hypothesis_id}@{candidate.variant_id}"
        )
        for probe in candidate.repair_probes:
            signature = (
                probe.command,
                json.dumps(probe.files, sort_keys=True),
                probe.expected_exit_code,
                probe.required_output_regex,
                probe.expected_outcome,
                probe.kind,
                probe.authority,
            )
            if signature in seen:
                continue
            seen.add(signature)
            probes.append(
                (
                    f"{probe.probe_id}@{candidate.variant_id}",
                    {
                        "authority": probe.authority,
                        "kind": probe.kind,
                        "origin": "repair",
                        "source_hypothesis_id": source_candidate_id,
                        "description": probe.description,
                        "expected_outcome": probe.expected_outcome,
                        "command": probe.command,
                        "files": probe.files,
                        "expected_exit_code": probe.expected_exit_code,
                        "required_output_regex": (
                            probe.required_output_regex
                        ),
                        "baseline_exit_code": None,
                    },
                )
            )
    return probes


def _executed_probe_status(
    output: str,
    exit_code: int,
    *,
    expected_exit_code: int = 0,
    required_output_regex: str = "",
) -> str:
    from INTENT2FIX.actions.output_sanitizer import OUTPUT_CAPTURE_TRUNCATION_MARKER

    if OUTPUT_CAPTURE_TRUNCATION_MARKER in output:
        return "unknown"
    if expected_exit_code == 0 and re.search(
        r"(?m)\b(?:EXIT|R[0-9]+|RC|STATUS)=([1-9][0-9]*)\b",
        output,
        flags=re.IGNORECASE,
    ):
        return "failed"
    if re.search(
        r"(?m)(?:^|\n)(?:[^\n]*:)?(?:fatal )?error:|"
        r"undefined reference|collect2: error|Segmentation fault|"
        r"AddressSanitizer:(?:DEADLYSIGNAL|[^\n]*ERROR)|"
        r"UndefinedBehaviorSanitizer:[^\n]*(?:ERROR|runtime error)|"
        r"(?:^|\n)sh: [0-9]+: [^\n]+: not found",
        output,
    ):
        return "failed"
    if exit_code != expected_exit_code:
        return "failed"
    if required_output_regex:
        try:
            if re.search(
                required_output_regex, output, flags=re.MULTILINE
            ) is None:
                return "failed"
        except re.error:
            return "failed"
    return "passed"


def _cross_execute_portfolio_candidate(
    *,
    instance_name: str,
    image_name: str,
    work_dir: str,
    base_commit: str,
    container_suffix: str,
    label: str,
    memory: Memory,
    patch: str | None,
    extra_probes: List[tuple[str, dict]] | None = None,
) -> CrossExecutionResult:
    command = getattr(memory, "test_command", "").strip()
    frozen_probes = _portfolio_frozen_probes(memory) + list(extra_probes or [])
    is_candidate = patch is not None
    docker_env = DockerEnv(
        image_name=image_name,
        container_name=(
            f"portfolio-check-{label}-{instance_name}-{container_suffix}"
        ),
        work_dir=work_dir,
        benchmark_base_commit=base_commit,
    )
    try:
        with docker_env:
            if is_candidate:
                patch_path = "/tmp/intent2fix-portfolio-candidate.patch"
                docker_env.write_file(patch_path, patch or "")
                apply_output, apply_code = docker_env.run_bash(
                    f"git apply --check {patch_path} && git apply {patch_path}",
                    timeout=_env_int("COTEST_PORTFOLIO_APPLY_TIMEOUT", 120),
                )
                if apply_code != 0:
                    return CrossExecutionResult(
                        status="apply_failed",
                        command=command,
                        exit_code=apply_code,
                        output=_bounded_command_output(apply_output),
                        patch_applied=False,
                    )

            for test_file in getattr(memory, "test_files", []):
                relative_path = str(test_file.get("path", "")).lstrip("/")
                if not relative_path:
                    continue
                docker_env.write_file(
                    os.path.join(work_dir, relative_path),
                    str(test_file.get("content", "")),
                )

            probe_results: List[ProbeExecutionResult] = []
            if command:
                output, exit_code = docker_env.run_bash(
                    command,
                    timeout=_env_int("COTEST_PORTFOLIO_CROSS_TEST_TIMEOUT", 900),
                )
                probe_results.append(
                    ProbeExecutionResult(
                        check_id="SHARED-REPRO",
                        authority="issue_reproduction",
                        kind="f2p",
                        origin="reproducer",
                        command=command,
                        status=_executed_probe_status(output, exit_code),
                        exit_code=exit_code,
                        output=_bounded_command_output(output),
                    )
                )

            for check_id, probe in frozen_probes:
                files = probe.get("files") or {}
                expected_exit_code = probe.get("expected_exit_code", 0)
                if not isinstance(expected_exit_code, int):
                    expected_exit_code = 0
                required_output_regex = str(
                    probe.get("required_output_regex", "") or ""
                )
                unsafe_path = next(
                    (
                        path
                        for path in files
                        if PurePosixPath(path).is_absolute()
                        or ".." in PurePosixPath(path).parts
                    ),
                    None,
                )
                if unsafe_path is not None:
                    probe_results.append(
                        ProbeExecutionResult(
                            check_id=check_id,
                            authority=str(probe.get("authority", "normative")),
                            kind=str(probe.get("kind", "")),
                            origin=str(probe.get("origin", "scout")),
                            source_hypothesis_id=str(
                                probe.get("source_hypothesis_id", "")
                            ),
                            expected_outcome=str(
                                probe.get("expected_outcome", "")
                            ),
                            description=str(probe.get("description", "")),
                            command=str(probe.get("command", "")),
                            baseline_expected_exit_code=probe.get(
                                "baseline_exit_code"
                            ),
                            expected_exit_code=expected_exit_code,
                            required_output_regex=required_output_regex,
                            status="infrastructure_error",
                            output=f"Unsafe frozen probe path: {unsafe_path}",
                        )
                    )
                    continue
                probe_command = str(probe.get("command", ""))
                if str(probe.get("origin", "scout")) == "scout":
                    rejection = frozen_probe_command_rejection(
                        probe_command, files
                    )
                    if rejection:
                        probe_results.append(
                            ProbeExecutionResult(
                                check_id=check_id,
                                authority=str(
                                    probe.get("authority", "normative")
                                ),
                                kind=str(probe.get("kind", "")),
                                origin="scout",
                                description=str(
                                    probe.get("description", "")
                                ),
                                command=probe_command,
                                baseline_expected_exit_code=probe.get(
                                    "baseline_exit_code"
                                ),
                                expected_exit_code=expected_exit_code,
                                required_output_regex=required_output_regex,
                                status="infrastructure_error",
                                output=(
                                    "Non-replayable frozen Scout probe: "
                                    + rejection
                                ),
                            )
                        )
                        continue
                for path, content in files.items():
                    docker_env.write_file(
                        os.path.join(work_dir, path), str(content)
                    )
                probe_command = RunIntentCheck._canonicalize_probe_sources(
                    probe_command, files
                )
                probe_command = RunIntentCheck._prepare_probe_output_directory(
                    probe_command
                )
                output, exit_code = docker_env.run_bash(
                    probe_command,
                    timeout=_env_int("COTEST_PORTFOLIO_CROSS_TEST_TIMEOUT", 900),
                )
                probe_results.append(
                    ProbeExecutionResult(
                        check_id=check_id,
                        authority=str(probe.get("authority", "normative")),
                        kind=str(probe.get("kind", "")),
                        origin=str(probe.get("origin", "scout")),
                        source_hypothesis_id=str(
                            probe.get("source_hypothesis_id", "")
                        ),
                        expected_outcome=str(
                            probe.get("expected_outcome", "")
                        ),
                        description=str(probe.get("description", "")),
                        command=probe_command,
                        baseline_expected_exit_code=probe.get(
                            "baseline_exit_code"
                        ),
                        expected_exit_code=expected_exit_code,
                        required_output_regex=required_output_regex,
                        status=_executed_probe_status(
                            output,
                            exit_code,
                            expected_exit_code=expected_exit_code,
                            required_output_regex=required_output_regex,
                        ),
                        exit_code=exit_code,
                        output=_bounded_command_output(output),
                    )
                )

            if not probe_results:
                return CrossExecutionResult(
                    status="unavailable",
                    output="No shared reproduction or contrast command was available.",
                    patch_applied=True if is_candidate else None,
                )
            binding_results = [
                item
                for item in probe_results
                if item.authority in {"normative", "issue_reproduction"}
            ]
            if not binding_results:
                return CrossExecutionResult(
                    status="unavailable",
                    command="\n".join(item.command for item in probe_results),
                    output="\n\n".join(
                        f"[{item.check_id}] {item.output}"
                        for item in probe_results
                    ),
                    patch_applied=True if is_candidate else None,
                    probe_results=probe_results,
                )
            statuses = {item.status for item in binding_results}
            overall_status = (
                "infrastructure_error"
                if "infrastructure_error" in statuses
                else "failed"
                if "failed" in statuses
                else "passed"
            )
            first_nonzero = next(
                (
                    item.exit_code
                    for item in binding_results
                    if item.exit_code not in {None, 0}
                ),
                0,
            )
            return CrossExecutionResult(
                status=overall_status,
                command="\n".join(item.command for item in probe_results),
                exit_code=first_nonzero,
                output="\n\n".join(
                    f"[{item.check_id}] {item.output}" for item in probe_results
                ),
                patch_applied=True if is_candidate else None,
                probe_results=probe_results,
            )
    except Exception as exc:
        logger.exception(
            "Portfolio cross-execution failed for %s (%s).",
            instance_name,
            label,
        )
        return CrossExecutionResult(
            status="infrastructure_error",
            command=command,
            output=str(exc),
            patch_applied=None,
        )


def _load_portfolio_repair_checkpoint(instance_name, instance, output_dir):
    """Resume an audited pool and its matching Scout; never regenerate on failure."""
    cache_dir = _env_str("COTEST_INTENT_PORTFOLIO_CACHE_DIR", "")
    if not cache_dir:
        raise ValueError("Repair-only requires COTEST_INTENT_PORTFOLIO_CACHE_DIR")
    case_dir = os.path.join(_resolve_jsonl_path(cache_dir), instance_name)
    required = {
        "intent_scout_checkpoint.json", "intent_grounding_result.json",
        "intent_portfolio.json",
    }
    hashes = _read_json_file(os.path.join(case_dir, "repair_checkpoint.json")).get(
        "artifact_sha256", {}
    )
    if not required <= hashes.keys() or hashes.keys() - required - {
        "intent_hypothesis_rerank.json",
        "hypothesis_source_evidence.json", "hypothesis_audit_evidence.json",
        "intent_hypothesis_execution.json",
    } or not ({"intent_hypothesis_rerank.json", "intent_hypothesis_execution.json"} & hashes.keys()):
        raise ValueError("Repair checkpoint has an incomplete artifact manifest")
    artifacts = {}
    for name, expected in hashes.items():
        with open(os.path.join(case_dir, name), "rb") as f:
            artifacts[name] = f.read()
        if hashlib.sha256(artifacts[name]).hexdigest() != expected:
            raise ValueError(f"Repair checkpoint hash mismatch: {name}")
    if json.loads(artifacts["intent_scout_checkpoint.json"]) != (
        _intent_scout_checkpoint_manifest(instance_name, instance)
    ):
        raise ValueError("Repair checkpoint does not match instance/base/Issue")
    raw_scout = json.loads(artifacts["intent_grounding_result.json"])
    # Repair consumes observations and frozen checks, not the old Scout adjudication.
    scout = IntentDialogueResult.model_validate({
        name: raw_scout[name]
        for name in ("issue_evidence", "intent_contract", "investigation", "exploration_plan")
        if name in raw_scout
    })
    portfolio = IntentHypothesisPool.model_validate_json(artifacts["intent_portfolio.json"])
    if "intent_hypothesis_rerank.json" in artifacts:
        audit = HypothesisPoolAudit.model_validate_json(artifacts["intent_hypothesis_rerank.json"])
        reviewed = select_hypothesis_pool(portfolio, audit)
        if reviewed.selected_hypothesis_ids != portfolio.selected_hypothesis_ids:
            raise ValueError("Repair checkpoint selection differs from final Audit")
    else:
        execution = json.loads(artifacts["intent_hypothesis_execution.json"])
        if (
            execution.get("audit_available") is not False
            or execution.get("selection_mode") != "generated_pool_fallback"
            or execution.get("selected_hypothesis_ids") != portfolio.selected_hypothesis_ids
        ):
            raise ValueError("Repair checkpoint lacks a valid generated-pool fallback")
    memory = _memory_from_intent_dialogue_result(
        instance_name=instance_name, problem_statement=instance["problem_statement"],
        issue_evidence=scout.issue_evidence, result=scout,
        output_dir=output_dir,
    )
    os.makedirs(output_dir, exist_ok=True)
    for name, data in artifacts.items():
        with open(os.path.join(output_dir, name), "wb") as f:
            f.write(data)
    logger.info("Reused frozen Scout and accepted Hypothesis checkpoint: %s", case_dir)
    return memory, portfolio


def _portfolio_scout_verification_spec(contract):
    spec = contract.verification_spec()
    unavailable = {}
    for check_id, probe in list(spec["probes"].items()):
        reason = frozen_probe_command_rejection(probe["command"], probe["files"])
        if reason:
            unavailable[check_id] = reason
            del spec["probes"][check_id]
    spec["diagnostic_ids"] = [item for item in spec["diagnostic_ids"] if item not in unavailable]
    spec["unavailable_checks"] = unavailable
    return spec


def _build_portfolio_branch_memory(memory, portfolio, hypothesis, output_dir):
    """Give Repair one final contract plus observations and unchanged diagnostic probes."""
    branch_memory = memory.model_copy(deep=True)
    scout_contract = RepairIntentContract.model_validate(memory.repair_intent_contract or {})
    diagnostic_contract = RepairIntentContract(acceptance_checks=[
        check.model_copy(update={"authority": "illustrative"})
        for check in scout_contract.acceptance_checks
    ])
    branch_memory.repair_intent_contract = diagnostic_contract.model_dump()
    branch_memory.repair_summary = render_hypothesis_for_repair(portfolio, hypothesis)
    execution = _read_json_file(
        os.path.join(output_dir, "intent_hypothesis_execution.json")
    )
    if execution and execution.get("audit_decision") != "ACCEPT":
        advisory = {
            "audit_decision": execution.get("audit_decision"),
            "selection_mode": execution.get("selection_mode"),
            "audit_reason": execution.get("audit_reason"),
            "this_branch_review": (
                execution.get("selected_hypothesis_reviews", {}).get(
                    hypothesis.id
                )
            ),
            "portfolio_findings": execution.get("blocking_findings", {}),
        }
        branch_memory.repair_summary += (
            "\n"
            "The Audit is diagnostic, not a Repair gate. Verify its concrete "
            "findings against the repository before editing. Address findings "
            "that apply to this branch, including necessary causal or closure "
            "scope, but do not inherit conflicts belonging only to another "
            "policy branch and do not replace the Issue contract.\n"
            "<hypothesis_audit_advisory>\n"
            + json.dumps(advisory, ensure_ascii=False)
            + "\n</hypothesis_audit_advisory>\n"
        )
    if diagnostic_contract.acceptance_checks:
        branch_memory.repair_summary += (
            "\n<repair_intent_contract>\n"
            "Frozen Scout checks are diagnostic observations for this branch. "
            "Their original oracles and files are unchanged; a disagreement must "
            "be reported, not used to replace the assigned Hypothesis. The original "
            "Issue and final <intent_hypothesis> define the required behavior. "
            "Checks listed as unavailable_checks lack a replayable fixture; do not "
            "invoke them or spend Repair steps reconstructing their missing setup.\n"
            + json.dumps(_portfolio_scout_verification_spec(diagnostic_contract), ensure_ascii=False)
            + "\n</repair_intent_contract>\n"
        )
    bundle = memory.repair_intent_bundle or {}
    observations = list(bundle.get("cited_repository_evidence", []))
    for name in ("hypothesis_source_evidence.json", "hypothesis_audit_evidence.json"):
        observations.extend(_read_json_file(os.path.join(output_dir, name)).get("evidence", []))
    branch_memory.repair_intent_bundle = {
        "source": "scout_and_hypothesis_observations",
        "cited_repository_evidence": observations,
        "authority": (
            "Observed code and executions are evidence, not additional repair requirements. "
            "Reuse exact source locations and observations for implementation checks. "
            "An observation supports only the relationship or executed call shape it shows; "
            "a search summary, source path, or failed lookup does not establish complete "
            "caller, overload, template, or generated-representation coverage."
        ),
    }
    if bundle.get("scout_probe_snapshot"):
        branch_memory.repair_intent_bundle["scout_probe_snapshot"] = copy.deepcopy(
            bundle["scout_probe_snapshot"]
        )
    branch_memory.reproduction_summary = (
        "Pre-repair observations follow; they do not supersede the final assigned contract."
    )
    branch_memory.expected_reproduction_result = ""
    return branch_memory


def _select_portfolio_candidate_independently(
    *, instance_name: str, base_commit: str, model_name: str,
    output_dir: str, candidates: List[PortfolioCandidate],
    memory: Memory, portfolio: IntentPortfolio,
) -> PortfolioCandidate | None:
    """Compare actual patches against public Issue evidence, without H-authored verdicts."""
    from scripts.run_portfolio_selector import prepare_case, run_case

    instance = get_benchmark_instance(instance_name)
    if instance["base_commit"] != base_commit:
        raise ValueError("Selector instance/base commit mismatch")
    public = {
        key: instance[key]
        for key in ("instance_id", "repo", "base_commit", "problem_statement")
    }
    repo = Path(_env_str("REPO_DIR", "/tmp/repos")) / instance_name
    scout = memory.repair_intent_contract or {}
    repair_intent = {
        "scout": {key: scout.get(key, []) for key in (
            "target_behavior", "preserve_behavior", "properties",
            "scope_evidence", "open_decisions", "validation_gaps",
        )},
        "common_obligations": [item.model_dump() for item in portfolio.obligations],
        "open_questions": [
            {"statement": item.statement, "impact": item.impact}
            for item in portfolio.unresolved_dimensions
        ],
    }
    case = prepare_case(public, [candidate.model_dump() for candidate in candidates], repo,
                        repair_intent=repair_intent,
                        hypotheses=[item.model_dump() for item in portfolio.hypotheses])
    result = run_case(
        case,
        Path(output_dir) / "portfolio_independent_selector",
        model_alias=model_name,
        execute=True,
        registry=_env_str("COTEST_DOCKER_REGISTRY", "6pqg5gclf6yaz1ch6g.xuanyuan.run"),
        reasoning_effort="low",
    )
    selected_sha = result.get("selected_patch_sha256")
    if selected_sha is None:
        return None
    selected = next((candidate for candidate in candidates if candidate.patch_sha256 == selected_sha), None)
    if selected is None:
        raise ValueError("Independent Selector returned a patch outside the frozen pool")
    return selected


def _save_portfolio_finalist(output_dir: str, selected: PortfolioCandidate) -> str:
    with open(os.path.join(output_dir, "solution.patch"), "w", encoding="utf-8") as f:
        f.write(selected.patch)
    with open(os.path.join(output_dir, "portfolio_finalist.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "hypothesis_id": selected.hypothesis_id,
                "variant_id": selected.variant_id,
                "patch_sha256": selected.patch_sha256,
                "basic_tests_passed": selected.basic_tests_passed,
                "branch_compliance": selected.branch_compliance,
                "discriminator_verified": selected.discriminator_verified,
                "cross_execution": selected.cross_execution.model_dump(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return selected.patch


def run_intent_portfolio_repair_step(
    *,
    instance_name: str,
    model_name: str,
    image_name: str,
    work_dir: str,
    base_commit: str,
    memory: Memory,
    output_dir: str,
    container_suffix: str,
    portfolio: IntentPortfolio | None = None,
) -> str | None:
    if portfolio is None:
        if _env_bool("COTEST_PORTFOLIO_REPAIR_ONLY", False):
            raise ValueError("Repair-only requires an accepted checkpoint portfolio")
        portfolio = _generate_intent_portfolio_with_verification(
            memory, model_name, output_dir,
            image_name=image_name, work_dir=work_dir, base_commit=base_commit,
            container_suffix=container_suffix,
        )
    portfolio_path = os.path.join(output_dir, "intent_portfolio.json")
    with open(portfolio_path, "w", encoding="utf-8") as f:
        json.dump(portfolio.model_dump(), f, indent=2, ensure_ascii=False)

    candidates: List[PortfolioCandidate] = []
    for hypothesis in portfolio.repair_hypotheses:
        branch_dir = os.path.join(output_dir, "portfolio", hypothesis.id)
        os.makedirs(branch_dir, exist_ok=True)
        branch_memory = _build_portfolio_branch_memory(
            memory, portfolio, hypothesis, output_dir
        )
        hypothesis_context = build_hypothesis_context(portfolio, hypothesis)
        branch_env = DockerEnv(
            image_name=image_name,
            container_name=(
                f"repair-{hypothesis.id}-{instance_name}-{container_suffix}"
            ),
            work_dir=work_dir,
            benchmark_base_commit=base_commit,
        )
        try:
            patch = run_repair_step(
                instance_name=instance_name,
                model_name=model_name,
                docker_env=branch_env,
                memory=branch_memory,
                output_dir=branch_dir,
                candidate_mode=True,
                candidate_hypothesis_context=hypothesis_context,
            )
        except Exception as exc:
            logger.exception(
                "Portfolio branch %s failed; attempting persisted-candidate "
                "recovery before continuing.",
                hypothesis.id,
            )
            with open(
                os.path.join(branch_dir, "branch_failure.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"error": str(exc)}, f, indent=2, ensure_ascii=False)
            patch = None
        status = _read_json_file(
            os.path.join(branch_dir, "portfolio_candidate_status.json")
        )
        if not patch and not status.get("admitted"):
            continue
        if not status.get("admitted"):
            logger.warning(
                "Portfolio branch %s returned a patch without candidate "
                "admission metadata; excluding it.",
                hypothesis.id,
            )
            continue
        variant_payloads = status.get("candidate_variants") or [
            {
                "variant_id": "final",
                "patch": patch,
                "admitted": status.get("admitted"),
                "kept_paths": status.get("kept_paths"),
                "excluded_paths": status.get("excluded_paths"),
                "finish_succeeded": status.get("finish_succeeded"),
                "basic_tests_passed": status.get("basic_tests_passed"),
                "branch_compliance": status.get("branch_compliance"),
                "discriminator_verified": status.get(
                    "discriminator_verified"
                ),
                "repair_probes": status.get("repair_probes"),
            }
        ]
        for variant in variant_payloads:
            if not variant.get("admitted") or not str(
                variant.get("patch") or ""
            ).strip():
                continue
            repair_probes = [
                RepairProbeEvidence.model_validate(item)
                for item in variant.get("repair_probes") or []
            ]
            candidates.append(
                PortfolioCandidate.create(
                    hypothesis_id=hypothesis.id,
                    variant_id=str(variant.get("variant_id") or "final"),
                    branch_assumption=hypothesis.branch_assumption,
                    repair_contract=build_candidate_repair_contract(
                        portfolio, hypothesis
                    ),
                    discriminator=render_assigned_discriminator(
                        portfolio, hypothesis
                    ),
                    predicted_outcome=render_assigned_outcome(
                        portfolio, hypothesis
                    ),
                    patch=str(variant["patch"]),
                    kept_paths=list(variant.get("kept_paths") or []),
                    excluded_paths=dict(
                        variant.get("excluded_paths") or {}
                    ),
                    finish_succeeded=bool(
                        variant.get("finish_succeeded")
                    ),
                    basic_tests_passed=bool(
                        variant.get("basic_tests_passed")
                    ),
                    branch_compliance=str(
                        variant.get("branch_compliance") or "not_checked"
                    ),
                    discriminator_verified=bool(
                        variant.get("discriminator_verified")
                    ),
                    repair_probes=repair_probes,
                    cross_execution=CrossExecutionResult(
                        status="unavailable",
                        output="Cross-execution has not run yet.",
                    ),
                )
            )

    candidates = deduplicate_candidates(candidates)
    if (
        _env_bool("COTEST_PORTFOLIO_REPAIR_ONLY", False)
        or _env_bool("COTEST_PORTFOLIO_STOP_AFTER_REPAIR", False)
    ):
        with open(os.path.join(output_dir, "portfolio_candidate_pool.json"), "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in candidates], f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "portfolio_repair_status.json"), "w", encoding="utf-8") as f:
            json.dump({
                "status": "completed", "hypothesis_count": len(portfolio.repair_hypotheses),
                "candidate_count": len(candidates), "selector_ran": False,
                "cross_execution_ran": False,
                "dual_layer_graph_ablation": _graph_ablation_enabled(),
            }, f, indent=2)
        logger.info("Repair candidates saved: %d; Selector deferred.", len(candidates))
        return None
    if not candidates:
        with open(
            os.path.join(output_dir, "portfolio_candidate_pool.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        logger.warning("No Finish-submitted portfolio candidate was admissible.")
        return None

    selector_backend = _env_str("COTEST_PORTFOLIO_SELECTOR_BACKEND", "legacy")
    if selector_backend not in {"legacy", "independent"}:
        raise ValueError(f"Unknown portfolio Selector backend: {selector_backend}")
    with open(os.path.join(output_dir, "portfolio_candidate_pool.json"), "w", encoding="utf-8") as f:
        json.dump([item.model_dump() for item in candidates], f, indent=2, ensure_ascii=False)
    if selector_backend == "independent":
        try:
            selected = _select_portfolio_candidate_independently(
                instance_name=instance_name,
                base_commit=base_commit,
                model_name=model_name,
                output_dir=output_dir,
                candidates=candidates,
                memory=memory,
                portfolio=portfolio,
            )
        except Exception as exc:
            logger.exception("Independent Selector failed; no build-verified finalist can be inferred.")
            selected = None
            with open(os.path.join(output_dir, "portfolio_independent_selector_failure.json"), "w", encoding="utf-8") as f:
                json.dump({"error_type": type(exc).__name__, "fallback": "none"}, f, indent=2)
        return _save_portfolio_finalist(output_dir, selected) if selected else None

    from INTENT2FIX.selector_execution import compile_candidate

    compile_rows = [
        compile_candidate({"candidate_id": f"{candidate.hypothesis_id}:{candidate.variant_id}",
                           "patch_sha256": candidate.patch_sha256, "patch": candidate.patch},
                          image=image_name, work_dir=work_dir, base_commit=base_commit)
        for candidate in candidates
    ]
    with open(os.path.join(output_dir, "portfolio_compile_gate.json"), "w", encoding="utf-8") as f:
        json.dump(compile_rows, f, indent=2, ensure_ascii=False)
    candidates = [candidate for candidate, row in zip(candidates, compile_rows)
                  if row["status"] == "passed"]
    if not candidates:
        logger.warning("No portfolio candidate passed the mandatory public build gate.")
        return None

    repair_probe_specs = _portfolio_repair_probe_specs(candidates)
    baseline_execution = _cross_execute_portfolio_candidate(
        instance_name=instance_name,
        image_name=image_name,
        work_dir=work_dir,
        base_commit=base_commit,
        container_suffix=container_suffix,
        label="baseline",
        memory=memory,
        patch=None,
        extra_probes=repair_probe_specs,
    )
    with open(
        os.path.join(output_dir, "portfolio_baseline_execution.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            baseline_execution.model_dump(), f, indent=2, ensure_ascii=False
        )

    cross_verified_candidates: List[PortfolioCandidate] = []
    for candidate in candidates:
        cross_execution = _cross_execute_portfolio_candidate(
            instance_name=instance_name,
            image_name=image_name,
            work_dir=work_dir,
            base_commit=base_commit,
            container_suffix=container_suffix,
            label=f"{candidate.hypothesis_id}-{candidate.variant_id}",
            memory=memory,
            patch=candidate.patch,
            extra_probes=repair_probe_specs,
        )
        candidate.cross_execution = cross_execution
        branch_dir = os.path.join(
            output_dir,
            "portfolio",
            candidate.hypothesis_id,
            "variants",
            candidate.variant_id,
        )
        os.makedirs(branch_dir, exist_ok=True)
        with open(
            os.path.join(branch_dir, "cross_execution.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                cross_execution.model_dump(), f, indent=2, ensure_ascii=False
            )
        if cross_execution.patch_applied is not True:
            logger.warning(
                "Portfolio branch %s could not rebuild its sanitized patch in "
                "a clean container; excluding it.",
                candidate.hypothesis_id,
            )
            continue
        cross_verified_candidates.append(candidate)

    candidates = cross_verified_candidates
    with open(
        os.path.join(output_dir, "portfolio_candidate_pool.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            [item.model_dump() for item in candidates],
            f,
            indent=2,
            ensure_ascii=False,
        )
    if not candidates:
        logger.warning("No candidate patch applied in clean cross-execution.")
        return None

    anonymous_candidates, candidate_mapping = anonymize_candidates(candidates)
    anonymous_baseline_execution = anonymize_baseline_execution(
        baseline_execution, candidates
    )
    audit_prompt = build_candidate_audit_prompt(
        issue=memory.problem_statement,
        portfolio=portfolio,
        anonymous_candidates=anonymous_candidates,
        baseline_execution=anonymous_baseline_execution,
        scout_repair_intent=memory.repair_intent_contract,
    )
    selected: PortfolioCandidate | None = None
    try:
        decision = _run_structured_extraction(
            audit_prompt,
            CandidateAuditDecision,
            model_override=model_name,
            max_tokens_key="COTEST_PORTFOLIO_AUDIT_MAX_TOKENS",
            glm_reasoning_effort=_env_str(
                "COTEST_GLM_REASONING_EFFORT", "low"
            ),
        )
        with open(
            os.path.join(output_dir, "portfolio_candidate_audit.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(decision.model_dump(), f, indent=2, ensure_ascii=False)
        selected = resolve_audit_selection(decision, candidate_mapping)
        if decision.selection == "SELECTED" and selected is None:
            logger.warning(
                "Candidate Audit selection failed deterministic safety checks; "
                "emitting no finalist."
            )
    except Exception as exc:
        logger.exception(
            "Candidate comparison Audit failed; no intent-verified finalist available."
        )
        with open(
            os.path.join(output_dir, "portfolio_candidate_audit_failure.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "error": str(exc),
                    "fallback": "none",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    if selected is None:
        return None
    return _save_portfolio_finalist(output_dir, selected)


def process_instance(instance_name: str, run_id: str | None = None) -> Dict[str, str] | None:
    """
    Process one benchmark instance through the configured evidence and repair stages.
    Configures logging to a file specific to the instance.
    """
    # Register signal handlers to catch system-level terminations
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        # Ensure worker process sees the same env as main, and override any exported defaults.
        _load_env()
        os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
        repair_model_name = _env_str("COTEST_REPAIR_MODEL", "claude-3.7-sonnet")
        repro_model_name = _env_str("COTEST_REPRO_MODEL", repair_model_name)
        sanitized_model_name = repair_model_name.replace("/", "_")
        results_prefix = results_path(
            _env_str("COTEST_RESULTS_DIR_PREFIX", "results")
        )
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        container_suffix = run_id or f"{time_stamp}_{os.getpid()}"
        result_dir = f"{results_prefix}_{sanitized_model_name}_cotest/{instance_name}"
        os.makedirs(result_dir, exist_ok=True)
        log_file_path = os.path.join(result_dir, f'repair_run_{time_stamp}.log')

        # Get the root logger for this process
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to prevent conflicts or duplicates from parent process
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optionally, add a console handler for this process's output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # --- End of per-instance logging setup ---

        logger.info(f"===== Starting repair process for instance: {instance_name} =====")
        logger.info(f"Logging for this instance will be saved to: {log_file_path}")
        logger.info(f"All results will be saved in: {result_dir}")
        if _graph_ablation_enabled():
            if not (
                _test_derived_intent_enabled()
                and _env_bool("COTEST_ENABLE_INTENT_PORTFOLIO", False)
                and _env_bool("COTEST_PORTFOLIO_STOP_AFTER_REPAIR", False)
            ):
                raise ValueError(
                    "Dual-layer graph ablation requires Scout, portfolio Repair, "
                    "and stop-after-Repair mode"
                )
            with open(os.path.join(result_dir, "graph_ablation.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "dual_layer_graph": "disabled",
                    "scout_graph_query": False,
                    "scout_test_anchor_retrieval": False,
                    "repair_indexed_tools": False,
                    "scout_hypothesis_cache_reuse": False,
                    "selector_ran": False,
                }, f, indent=2)
        test_derived_intent = _test_derived_intent_enabled()
        legacy_reproducer = _legacy_reproducer_enabled()
        repair_evidence_source = (
            "intent-evidence"
            if test_derived_intent
            else "reproduction"
            if legacy_reproducer
            else "issue-only"
        )
        logger.info(
            "Stage 1: %s; Stage 2: repair; Stage 3: %s.",
            (
                "test-derived Intent evidence"
                if test_derived_intent
                else "reproduction"
                if legacy_reproducer
                else "Issue-only input (legacy Reproducer disabled)"
            ),
            "patch surface review" if _patch_reviewer_enabled() else "review disabled",
        )
        
        instance_str = parse_instance_string(instance_name)
        if not instance_str:
            logger.error(f"Invalid instance name format: {instance_name}")
            return None
        benchmark_instance = get_benchmark_instance(instance_name)

        image_name = f'mswebench/{instance_str["org"]}_m_{instance_str["repo"]}:pr-{instance_str["number"]}'.lower()
        docker_registry = _env_str("COTEST_DOCKER_REGISTRY", "6pqg5gclf6yaz1ch6g.xuanyuan.run").rstrip("/")
        if docker_registry:
            image_name = f"{docker_registry}/{image_name}"
        
        if _env_bool("COTEST_PORTFOLIO_REPAIR_ONLY", False):
            memory, portfolio = _load_portfolio_repair_checkpoint(
                instance_name, benchmark_instance, result_dir
            )
            logger.info("===== Starting Stage 2: cached Hypothesis Repair for %s =====", instance_name)
            run_intent_portfolio_repair_step(
                instance_name=instance_name, model_name=repair_model_name,
                image_name=image_name, work_dir=f'/home/{instance_str["repo"]}',
                base_commit=benchmark_instance["base_commit"], memory=memory,
                output_dir=result_dir, container_suffix=container_suffix, portfolio=portfolio,
            )
            return None

        # --- Stage 1: Intent evidence induction or legacy reproduction. ---
        repro_docker_env = None
        memory_instance = None
        stage_name = (
            "Intent Evidence"
            if test_derived_intent
            else "Reproduction"
            if legacy_reproducer
            else "Issue-only Input"
        )
        logger.info(
            "===== Starting Stage 1: %s for %s =====",
            stage_name,
            instance_name,
        )
        try:
            if not test_derived_intent and not legacy_reproducer:
                logger.info(
                    "Legacy Reproducer is disabled. Passing the original Issue "
                    "directly to Repair without reproduction evidence."
                )
                memory_instance = _build_issue_only_memory(
                    instance_name,
                    benchmark_instance,
                )
            else:
                repro_docker_env = DockerEnv(
                    image_name=image_name,
                    container_name=(
                        f'intent-{instance_name}-{container_suffix}'
                        if test_derived_intent
                        else f'repro-{instance_name}-{container_suffix}'
                    ),
                    work_dir=f'/home/{instance_str["repo"]}',
                    benchmark_base_commit=benchmark_instance["base_commit"],
                )
                if test_derived_intent:
                    memory_instance = run_test_derived_intent_step(
                        instance_name=instance_name,
                        model_name=repro_model_name,
                        docker_env=repro_docker_env,
                        output_dir=result_dir,
                    )
                else:
                    memory_instance = run_reproduction_step(
                        instance_name=instance_name,
                        model_name=repro_model_name,
                        docker_env=repro_docker_env,
                        output_dir=result_dir,
                    )
        except IntentEmbeddingUnavailableError:
            raise
        except Exception as e:
            logger.error(
                "An error occurred during the %s stage for '%s': %s",
                stage_name.lower(),
                instance_name,
                e,
                exc_info=True,
            )

        missing_contrastive_witness = (
            test_derived_intent
            and _env_str("COTEST_INTENT_DIALOGUE_MODE", "contrastive")
            == "contrastive"
            and not _has_valid_contrastive_witness(memory_instance)
        )
        if missing_contrastive_witness and _reproducer_fallback_enabled():
            logger.info(
                "No valid contrastive witness was produced; starting "
                "explicitly enabled Reproducer fallback while retaining "
                "repository Intent evidence."
            )
            fallback_memory = None
            try:
                fallback_docker_env = DockerEnv(
                    image_name=image_name,
                    container_name=(
                        f'repro-fallback-{instance_name}-{container_suffix}'
                    ),
                    work_dir=f'/home/{instance_str["repo"]}',
                    benchmark_base_commit=benchmark_instance["base_commit"],
                )
                fallback_memory = run_reproduction_step(
                    instance_name=instance_name,
                    model_name=repro_model_name,
                    docker_env=fallback_docker_env,
                    output_dir=result_dir,
                    run_intent_analysis=False,
                )
            except Exception:
                logger.exception(
                    "Reproducer fallback failed; retaining available Intent "
                    "evidence."
                )
            memory_instance = _merge_intent_and_reproduction_memory(
                memory_instance,
                fallback_memory,
            )
            if fallback_memory is not None:
                repair_evidence_source = "intent+reproduction-fallback"
        elif missing_contrastive_witness:
            logger.info(
                "No valid contrastive witness was produced. Continuing to "
                "Repair with the original Issue and repository evidence; "
                "standalone Reproducer fallback is disabled."
            )
        # Note: No need to cleanup here - run_reproduction_step uses 'with docker_env:' which auto-cleans up

        if not memory_instance:
            logger.warning(
                "%s failed for %s. Using Issue-only memory for repair.",
                stage_name,
                instance_name,
            )
            memory_instance = _build_issue_only_memory(
                instance_name,
                benchmark_instance,
            )

        # --- Stage 2: Repair, optionally followed by TestAgent enhancement. ---
        repair_docker_env = None
        solution_patch = None
        portfolio_enabled = _env_bool("COTEST_ENABLE_INTENT_PORTFOLIO", False)
        portfolio_scout_only = _env_bool(
            "COTEST_PORTFOLIO_SCOUT_ONLY", False
        )
        logger.info(
            "Passing evidence to repair: instance=%s source=%s evidence=%s "
            "legacy_intent=%s dialogue_evidence=%s hypothesis_ready=%s "
            "(diagnostic_only=true)",
            instance_name,
            repair_evidence_source,
            "present" if memory_instance.reproduction_result else "missing",
            memory_instance.repair_intent_bundle.get("status", "missing"),
            "present" if memory_instance.repair_summary else "missing",
            memory_instance.repair_intent_bundle.get(
                "hypothesis_ready", "unavailable"
            ),
        )
        # Scout readiness communicates evidence confidence to Hypothesis; it is
        # intentionally not a condition for Hypothesis generation or Repair.
        if portfolio_scout_only:
            logger.info(
                "===== Starting portfolio hypothesis Scout for %s; Repair and "
                "all downstream stages are disabled =====",
                instance_name,
            )
            portfolio = _generate_intent_portfolio_with_verification(
                memory_instance,
                repair_model_name,
                result_dir,
                image_name=image_name, work_dir=f'/home/{instance_str["repo"]}',
                base_commit=benchmark_instance["base_commit"],
                container_suffix=container_suffix,
            )
            with open(
                os.path.join(result_dir, "intent_portfolio.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    portfolio.model_dump(), f, indent=2, ensure_ascii=False
                )
            logger.info(
                "===== Finished portfolio hypothesis Scout for %s: %d "
                "scheduled hypotheses, %d obligations =====",
                instance_name,
                len(portfolio.repair_hypotheses),
                len(portfolio.obligations),
            )
            return None
        try:
            test_enhancement_rounds = _env_int("COTEST_TEST_ENHANCEMENT_ROUNDS", 1)
            logger.info(
                "===== Starting Stage 2: %s for %s "
                "(test_enhancement_rounds=%d) =====",
                "Intent Portfolio Repair" if portfolio_enabled else "Repair",
                instance_name,
                test_enhancement_rounds,
            )
            if portfolio_enabled:
                solution_patch = run_intent_portfolio_repair_step(
                    instance_name=instance_name,
                    model_name=repair_model_name,
                    image_name=image_name,
                    work_dir=f'/home/{instance_str["repo"]}',
                    base_commit=benchmark_instance["base_commit"],
                    memory=memory_instance,
                    output_dir=result_dir,
                    container_suffix=container_suffix,
                )
            else:
                repair_docker_env = DockerEnv(
                    image_name=image_name,
                    container_name=f'repair-{instance_name}-{container_suffix}',
                    work_dir=f'/home/{instance_str["repo"]}',
                    benchmark_base_commit=benchmark_instance["base_commit"],
                )
                solution_patch = run_repair_step(
                    instance_name=instance_name,
                    model_name=repair_model_name,
                    docker_env=repair_docker_env,
                    memory=memory_instance,
                    output_dir=result_dir,
                )
            logger.info(
                "===== Finished Stage 2 for %s "
                "(test_enhancement_rounds=%d) =====",
                instance_name,
                test_enhancement_rounds,
            )
        except Exception as e:
            logger.error(f"An error occurred during the repair stage for '{instance_name}': {e}", exc_info=True)
        finally:
            if memory_instance.test_files:
                repository = create_repository(get_benchmark_instance(instance_name))
                _remove_host_reproduction_artifacts(
                    repository.repo_path,
                    [test_file["path"] for test_file in memory_instance.test_files],
                )
        # Note: No need to cleanup here - run_repair_step uses 'with docker_env:' which auto-cleans up
        
        if _env_bool("COTEST_PORTFOLIO_STOP_AFTER_REPAIR", False):
            logger.info("Portfolio Repair ended; candidates saved and Selector skipped.")
            return None
        if solution_patch:
            submission = {
                "instance_id": instance_name,
                "model_patch": solution_patch
            }
            return submission
        
        logger.info(f"===== End of processing for instance: {instance_name} ====")
        return None
    except Exception as e:
        # Global exception handler to prevent process crashes
        # Try to log to file if logger is set up, otherwise use print
        try:
            logger = logging.getLogger()
            logger.error(f"Unhandled exception in process_instance for '{instance_name}': {e}", exc_info=True)
        except Exception:
            print(f"CRITICAL: Unhandled exception in process_instance for '{instance_name}': {e}", file=sys.stderr)
        return None


def _shutdown_process_pool(
    executor: concurrent.futures.ProcessPoolExecutor,
    grace_seconds: float,
) -> None:
    """Cancel queued work, then terminate and finally kill running workers."""
    processes = list(getattr(executor, "_processes", {}).values())
    executor.shutdown(wait=False, cancel_futures=True)

    for process in processes:
        if process.is_alive():
            process.terminate()

    deadline = time.monotonic() + max(0.0, grace_seconds)
    try:
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            process.join(remaining)
    except KeyboardInterrupt:
        logging.warning("Second interrupt received; killing workers now.")

    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.kill()
    for process in alive:
        process.join(timeout=1)


def _cleanup_run_containers(run_id: str) -> None:
    """Force-remove containers owned by this runner invocation."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={run_id}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        container_ids = result.stdout.split()
        if container_ids:
            subprocess.run(
                ["docker", "rm", "-f", *container_ids],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            logging.info(
                "Removed %d container(s) from interrupted run %s.",
                len(container_ids),
                run_id,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        logging.warning("Failed to clean containers for run %s: %s", run_id, exc)


def main():
    _load_env()
    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

    parser = argparse.ArgumentParser(description="INTENT2FIX co-test (no memory module).")
    default_jsonl = _env_str("COTEST_JSONL", "INTENT2FIX/benchmark/multi_swebench_c_change.jsonl")
    parser.add_argument(
        "--jsonl",
        default=default_jsonl,
        help="Benchmark JSONL (relative paths resolve from repo root). Default: env COTEST_JSONL or C benchmark.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Default: env COTEST_MAX_WORKERS or 8.",
    )
    parser.add_argument(
        "--submission",
        default=_env_str("COTEST_SUBMISSION_FILE", "submission.json"),
        help=(
            "Output submission JSON path. Relative paths are written below "
            "submission/. Default: env COTEST_SUBMISSION_FILE or submission.json."
        ),
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated instance_id values to skip (merged with COTEST_SKIP_INSTANCE_IDS).",
    )
    parser.add_argument(
        "--portfolio-scout-only",
        action="store_true",
        help=(
            "Run shared Intent evidence plus hypothesis generation/coverage "
            "audit, then stop before Repair."
        ),
    )
    parser.add_argument(
        "--portfolio-repair-only", action="store_true",
        help="Resume a frozen accepted Hypothesis checkpoint; save Repair candidates and defer Selector.",
    )
    parser.add_argument(
        "--ablate-dual-layer-graph", action="store_true",
        help="Run fresh Scout, Hypothesis, and Repair without dual-layer graph access; save candidates and skip Selector.",
    )
    parser.add_argument(
        "--ablation-results-prefix",
        default=None,
        help="Result prefix for the graph ablation; default is a new timestamped directory.",
    )
    args = parser.parse_args()

    if args.portfolio_scout_only and args.portfolio_repair_only:
        parser.error("Scout-only and Repair-only are mutually exclusive")
    if args.ablate_dual_layer_graph and (args.portfolio_scout_only or args.portfolio_repair_only):
        parser.error("Graph ablation runs fresh Scout through Repair; it cannot use stage-only modes")
    if args.ablation_results_prefix and not args.ablate_dual_layer_graph:
        parser.error("--ablation-results-prefix requires --ablate-dual-layer-graph")
    if args.ablate_dual_layer_graph:
        os.environ.update({
            "COTEST_ABLATE_DUAL_LAYER_GRAPH": "1",
            "COTEST_ENABLE_INTENT_GROUNDING": "1",
            "COTEST_INTENT_DIALOGUE_MODE": "contrastive",
            "COTEST_ENABLE_INTENT_PORTFOLIO": "1",
            "COTEST_ENABLE_LEGACY_REPRODUCER": "0",
            "COTEST_ENABLE_REPRODUCER_FALLBACK": "0",
            "COTEST_PORTFOLIO_REPAIR_ONLY": "0",
            "COTEST_PORTFOLIO_SCOUT_ONLY": "0",
            "COTEST_PORTFOLIO_STOP_AFTER_REPAIR": "1",
            "COTEST_RESULTS_DIR_PREFIX": (
                args.ablation_results_prefix
                or "results/ablation_no_dual_layer_graph_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
        })
    if args.portfolio_repair_only:
        os.environ["COTEST_ENABLE_INTENT_PORTFOLIO"] = "1"
        os.environ["COTEST_PORTFOLIO_REPAIR_ONLY"] = "1"
        os.environ["COTEST_PORTFOLIO_SCOUT_ONLY"] = "0"

    if args.portfolio_scout_only:
        os.environ["COTEST_ENABLE_INTENT_PORTFOLIO"] = "1"
        os.environ["COTEST_PORTFOLIO_SCOUT_ONLY"] = "1"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    jsonl_file = _resolve_jsonl_path(args.jsonl)
    logging.info("Loading instance IDs from: %s", jsonl_file)

    test_instance_names = extract_instance_ids_from_jsonl(file_path=jsonl_file)
    if not test_instance_names:
        logging.warning("No instance names extracted. Exiting.")
        return

    skip_ids = _parse_skip_instances()
    if args.skip:
        skip_ids |= {x.strip() for x in args.skip.split(",") if x.strip()}
    valid_instance_names = [n for n in test_instance_names if n not in skip_ids]

    logging.info("Extracted %d instance IDs (%d after skip).", len(test_instance_names), len(valid_instance_names))

    max_workers = args.workers if args.workers is not None else _env_int("COTEST_MAX_WORKERS", 8)
    logging.info(
        "Starting concurrent processing for %d instances with up to %d workers.",
        len(valid_instance_names),
        max_workers,
    )

    submissions = []
    interrupted = False
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    signal.signal(signal.SIGINT, _main_interrupt_handler)
    signal.signal(signal.SIGTERM, _main_interrupt_handler)
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    try:
        future_to_instance = {
            executor.submit(process_instance, name, run_id): name
            for name in valid_instance_names
        }

        for future in concurrent.futures.as_completed(future_to_instance):
            instance_name = future_to_instance[future]
            try:
                result = future.result()
                if result:
                    submissions.append(result)
                    logging.info("Successfully processed and got result for instance: %s", instance_name)
                elif _env_bool("COTEST_PORTFOLIO_REPAIR_ONLY", False) or _env_bool("COTEST_PORTFOLIO_STOP_AFTER_REPAIR", False):
                    logging.info("Repair candidate worker ended for %s; see portfolio_repair_status.json for completion and candidate count.", instance_name)
                else:
                    logging.warning("Processing for instance %s did not return a result.", instance_name)
            except concurrent.futures.process.BrokenProcessPool as bpp_exc:
                logging.error(
                    "Process pool broken for instance %s. A worker process crashed: %s",
                    instance_name,
                    bpp_exc,
                    exc_info=True,
                )
                logging.warning(
                    "BrokenProcessPool detected. Shutting down executor and continuing with remaining futures."
                )
                break
            except Exception as exc:
                logging.error(
                    "Instance %s generated an exception in the worker process: %s",
                    instance_name,
                    exc,
                    exc_info=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        grace_seconds = float(_env_str("COTEST_SHUTDOWN_GRACE_SECONDS", "10"))
        logging.warning(
            "\nCaught Ctrl+C. Cancelling queued work; workers have %.1f seconds to exit.",
            grace_seconds,
        )
        _shutdown_process_pool(executor, grace_seconds)
        _cleanup_run_containers(run_id)
    finally:
        if not interrupted:
            executor.shutdown(wait=True, cancel_futures=False)

    if interrupted:
        logging.warning("Run interrupted; saving %d completed submission(s).", len(submissions))
    else:
        logging.info("All instances have been processed.")

    submission_file_path = submission_path(args.submission)
    os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)
    with open(submission_file_path, "w", encoding="utf-8") as f:
        json.dump(submissions, f, indent=2)
    logging.info("All submissions have been saved to %s", submission_file_path)
    return 130 if interrupted else 0

if __name__ == "__main__":
    raise SystemExit(main())
