from INTENT2FIX.actions.verified_finish import (
    VerifiedFinish,
    VerifiedFinishArgs,
    _missing_intent_checks,
)
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import InMemRepository


def _finish_args():
    return VerifiedFinishArgs(
        thoughts="done",
        finish_reason="fixed",
        test_verification="tests passed",
    )


def test_finish_rejects_silent_loss_of_new_production_file():
    context = FileContext(repo=InMemRepository())
    path = "include/helper.hpp"
    context.add_file(path).apply_changes("int value;\n")
    context.verification["ephemeral_paths"] = [path]
    result = VerifiedFinish().execute(_finish_args(), file_context=context)
    assert not result.terminal
    assert result.expect_correction
    assert result.properties["fail_reason"] == "submission_artifact_error"
    assert context.get_file(path).content == "int value;\n"


def test_finish_schema_describes_current_state_verification():
    schema = VerifiedFinishArgs.model_json_schema()["properties"]

    assert "original issue" in schema["finish_reason"]["description"]
    verification = schema["test_verification"]["description"]
    assert "RunSingleTest" in verification
    assert "RunRegression" in verification
    assert "RunIntentCheck" in verification
    assert "including failures or limitations" in verification
    assert "tests were added/modified" not in verification


def test_finish_allows_missing_verification_with_warnings():
    context = FileContext(repo=InMemRepository())
    context.add_file("src/fix.cpp", content="old").patch = "diff --git a/src/fix.cpp b/src/fix.cpp"
    context.change_seq = 1
    action = VerifiedFinish()

    result = action.execute(_finish_args(), file_context=context)

    assert result.terminal is True
    assert result.expect_correction is False
    warnings = result.properties["verification_warnings"]
    assert any("RunRegression" in warning for warning in warnings)
    assert any("RunSingleTest" in warning for warning in warnings)


def test_finish_accepts_tests_for_latest_change():
    context = FileContext(repo=InMemRepository())
    context.add_file("src/fix.cpp", content="old").patch = "diff --git a/src/fix.cpp b/src/fix.cpp"
    context.change_seq = 1
    context.verification = {
        "regression": {"passed": True, "change_seq": context.change_seq},
        "single_test": {"passed": True, "change_seq": context.change_seq},
    }

    result = VerifiedFinish().execute(_finish_args(), file_context=context)

    assert result.terminal is True


def test_temporary_probe_does_not_stale_regression_but_stales_focused_test():
    context = FileContext(repo=InMemRepository())
    context.add_file("src/fix.cpp", content="old").patch = (
        "diff --git a/src/fix.cpp b/src/fix.cpp"
    )
    context.change_seq = 1
    context.verification = {
        "regression": {"passed": True, "change_seq": 1},
        "single_test": {
            "passed": True,
            "change_seq": 1,
            "probe_change_seq": 0,
        },
        "probe_change_seq": 1,
    }

    result = VerifiedFinish().execute(_finish_args(), file_context=context)

    warnings = result.properties["verification_warnings"]
    assert not any("RunRegression" in warning for warning in warnings)
    assert any("RunSingleTest" in warning for warning in warnings)

    context.verification["single_test"]["probe_change_seq"] = 1
    current = VerifiedFinish().execute(_finish_args(), file_context=context)
    assert "verification_warnings" not in current.properties


def test_finish_blocks_missing_required_intent_checks():
    context = FileContext(repo=InMemRepository())
    context.add_file("src/fix.cpp", content="old").patch = (
        "diff --git a/src/fix.cpp b/src/fix.cpp"
    )
    context.change_seq = 2
    context.verification = {
        "regression": {"passed": True, "change_seq": 2},
        "single_test": {"passed": True, "change_seq": 2},
        "intent_contract": {
            "checks": {
                "TARGET-1": "Target works.",
                "PRESERVE-1": "Existing behavior remains valid.",
                "ACCEPT-1": "Exercise target and contrast.",
            },
            "target_ids": ["TARGET-1"],
            "preserve_ids": ["PRESERVE-1"],
            "acceptance_ids": ["ACCEPT-1"],
        },
        "intent_check_results": {
            "TARGET-1": {"passed": True, "change_seq": 2},
        },
    }

    result = VerifiedFinish().execute(_finish_args(), file_context=context)

    assert not result.terminal
    assert result.expect_correction
    assert "ACCEPT-1" in result.message
    assert "PRESERVE-1" not in result.message
    assert result.properties["fail_reason"] == (
        "required_intent_checks_incomplete"
    )


def test_portfolio_finish_treats_missing_intent_checks_as_soft_evidence():
    context = FileContext(repo=InMemRepository())
    context.add_file("src/fix.cpp", content="old").patch = (
        "diff --git a/src/fix.cpp b/src/fix.cpp"
    )
    context.change_seq = 2
    context.verification = {
        "finish_policy": "portfolio_soft_evidence",
        "intent_contract": {
            "checks": {"ACCEPT-1": "Exercise the disputed boundary."},
            "acceptance_ids": ["ACCEPT-1"],
        },
        "intent_check_results": {},
    }

    result = VerifiedFinish().execute(_finish_args(), file_context=context)

    assert result.terminal
    assert result.expect_correction is False
    assert any(
        "Intent acceptance evidence" in warning
        for warning in result.properties["verification_warnings"]
    )


def test_finish_accepts_current_intent_contract_checks():
    verification = {
        "intent_contract": {
            "checks": {
                "TARGET-1": "Target one.",
                "TARGET-2": "Target two.",
                "PRESERVE-1": "Preserve one.",
                "ACCEPT-1": "Acceptance one.",
            },
            "target_ids": ["TARGET-1", "TARGET-2"],
            "preserve_ids": ["PRESERVE-1"],
            "acceptance_ids": ["ACCEPT-1"],
        },
        "intent_check_results": {
            "TARGET-1": {"passed": True, "change_seq": 3},
            "TARGET-2": {"passed": True, "change_seq": 3},
            "PRESERVE-1": {"passed": True, "change_seq": 3},
            "ACCEPT-1": {"passed": True, "change_seq": 3},
        },
    }

    assert _missing_intent_checks(verification, 3) == []
    verification["intent_check_results"].pop("ACCEPT-1")
    assert any(
        item.startswith("ACCEPT-1:")
        for item in _missing_intent_checks(verification, 3)
    )
    assert _missing_intent_checks(verification, 4)


def test_finish_does_not_require_optional_diagnostic_witness():
    verification = {
        "intent_contract": {
            "checks": {"ACCEPT-1": "Local diagnostic witness."},
            "acceptance_ids": [],
            "diagnostic_ids": ["ACCEPT-1"],
            "probes": {
                "ACCEPT-1": {
                    "authority": "illustrative",
                    "command": "false",
                    "files": {"intent_probe.cpp": "int main() { return 1; }"},
                }
            },
        },
        "intent_check_results": {},
    }

    assert _missing_intent_checks(verification, 1) == []


def test_finish_warns_when_original_tests_fail_after_test_edit():
    context = FileContext(repo=InMemRepository())
    context.add_file("src/fix.cpp", content="old").patch = (
        "diff --git a/src/fix.cpp b/src/fix.cpp"
    )
    context.add_file("tests/fix.cpp", content="old test").patch = (
        "diff --git a/tests/fix.cpp b/tests/fix.cpp"
    )
    context.change_seq = 3
    context.verification = {
        "regression": {"passed": True, "change_seq": 3},
        "single_test": {"passed": True, "change_seq": 3},
        "production_regression": {"passed": False, "change_seq": 3},
    }

    result = VerifiedFinish().execute(_finish_args(), file_context=context)

    assert result.terminal
    assert result.expect_correction is False
    assert any(
        "original tests did not pass" in warning
        for warning in result.properties["verification_warnings"]
    )
