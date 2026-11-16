import json
import subprocess

from INTENT2FIX.hypothesis_source import (
    candidate_boundary_symbols,
    cpp_definitions,
    expand_source_evidence,
)


def test_cpp_definitions_keep_template_and_body_without_returning_uses():
    code = '''template<class T> struct is_ready : is_ready_impl<T> {};
template<class T> using is_valid = is_ready<T>;
bool check() { return is_ready<int>::value; }
template<class T> bool access(T x) { return x.ready(); }
'''
    rows = cpp_definitions("traits.hpp", code, {"is_ready", "is_valid", "access"})
    assert [r["symbol"] for r in rows] == ["is_ready", "is_valid", "access"]
    assert all(r["complete"] for r in rows)
    assert all(r["text"].startswith("template<class T>") for r in rows)
    assert "return x.ready();" in rows[-1]["text"]


def test_large_or_broken_definition_cannot_claim_complete_source():
    for code in ["struct is_ready {", "struct is_ready { /*" + "x" * 6000 + "*/ };",
                 "struct is_ready;"]:
        rows = cpp_definitions("traits.hpp", code, {"is_ready"})
        assert not any(r["complete"] for r in rows)


def test_partial_specialization_body_is_not_replaced_by_empty_primary_template():
    source = '''template<class T, class = void> struct is_ready_impl {};
template<class T> struct is_ready_impl<T, enable_if_t<has_decoder<T>::value>> {
  static constexpr bool value = true;
};'''
    rows = cpp_definitions('traits.hpp', source, {'is_ready_impl'})
    assert len(rows) == 2
    assert rows[1]['complete']
    assert 'has_decoder<T>::value' in rows[1]['text']


def test_candidate_boundary_symbols_skip_unstructured_descriptions():
    bundle = {"candidate_boundaries": [
        {"anchor_operation": "Owner::sibling"},
        {"anchor_operation": "patch_inplace (public via patch)"},
        {"anchor_operation": "single_include patch_inplace"},
        {"anchor_operation": "PR #3020 trait restructure"},
    ]}
    assert candidate_boundary_symbols(bundle) == ["sibling", "patch_inplace"]


def test_expansion_reads_candidate_boundary_not_mentioned_in_raw_text(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "owner.hpp").write_text('''struct Owner {
  int local() { return 1; }
  int sibling() { return shared_rule(); }
  int shared_rule() { return 2; }
};
''')
    subprocess.run(["git", "-C", str(tmp_path), "add", "owner.hpp"], check=True)

    class LocalReadOnlyEnv:
        def run_bash(self, command, timeout):
            result = subprocess.run(command, shell=True, cwd=tmp_path, timeout=timeout,
                                    capture_output=True, text=True)
            return result.stdout, result.returncode

    bundle = {
        "cited_repository_evidence": [{"id": "SEARCH-1", "observation": "Owner::local()"}],
        "candidate_boundaries": [{"anchor_operation": "Owner::sibling"}],
    }
    result = expand_source_evidence(LocalReadOnlyEnv(), bundle, {}, tmp_path)
    manifest = json.loads((tmp_path / "hypothesis_source_evidence.json").read_text())
    assert manifest["requested_symbols"][0] == "sibling"
    assert any(item["symbol"] == "sibling" and item["included"] for item in manifest["definitions"])
    assert any("int sibling()" in item["observation"] for item in result["cited_repository_evidence"])


def test_expansion_reads_referenced_trait_dependency_and_keeps_raw_input(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    header = tmp_path / "traits.hpp"
    header.write_text('''template<class T> struct is_ready : is_ready_impl<T> {};
template<class T> struct is_ready_impl {
  static constexpr bool value = has_encoder<T>::value;
};
''')
    subprocess.run(["git", "-C", str(tmp_path), "add", "traits.hpp"], check=True)

    class LocalReadOnlyEnv:
        def run_bash(self, command, timeout):
            result = subprocess.run(command, shell=True, cwd=tmp_path, timeout=timeout,
                                    capture_output=True, text=True)
            return result.stdout, result.returncode

    raw = {"id": "SEARCH-1", "observation": "enable_if<is_ready<T>::value> decode(T&);"}
    bundle = {"cited_repository_evidence": [raw], "repository_facts": ["old interpretation"]}
    result = expand_source_evidence(LocalReadOnlyEnv(), bundle, {}, tmp_path)
    assert bundle["cited_repository_evidence"] == [raw]
    assert result["cited_repository_evidence"][0] == raw
    manifest = json.loads((tmp_path / "hypothesis_source_evidence.json").read_text())
    assert manifest["requested_symbols"] == ["is_ready", "is_ready_impl"]
    assert {r["symbol"] for r in manifest["definitions"]} == {"is_ready", "is_ready_impl"}
    assert "has_encoder<T>::value" in result["cited_repository_evidence"][-1]["observation"]
    assert len(result["cited_repository_evidence"]) == 3
    assert subprocess.run(["git", "diff", "--exit-code"], cwd=tmp_path).returncode == 0
