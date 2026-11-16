from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace

import pytest

from INTENT2FIX import branch_context as bc
from INTENT2FIX.memory.memory import Memory


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for path, text in {
        "include/dom/value.h": "namespace dom { int access() { return 0; } }\n",
        "include/ondemand/value.h": "namespace ondemand { struct value { int access() { return next(); } int next() { return 1; } }; }\n",
        "include/overload.h": "namespace one { int get(int x) { return x; } }\nnamespace two { int get(int x) { return x; } }\n",
        "include/ondemand/reader.h": "namespace ondemand { struct reader { int resume(int x); }; }\n",
        "include/ondemand/reader-inl.h": "namespace ondemand { int reader::resume(int x) { return x + 1; } }\n",
        "tests/src/fuzzer.cpp": "int LLVMFuzzerTestOneInput() { return 0; }\n",
    }.items():
        file = repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Context test", "-c", "user.email=context@example.test", "commit", "-qm", "base"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return SimpleNamespace(repo_dir=str(repo)), commit


def contract(scope="include/ondemand/value.h"):
    return {"branch": {"id": "H0", "title": "OnDemand access", "repair_scope": [scope],
                       "expected_behavior": ["access calls next"], "preservation_behavior": ["preserve existing get"]},
            "hard_obligations": [{"id": "FIX-1", "statement": "Expose access"}]}


@pytest.mark.parametrize("declarator, expected", [
    ("simdjson_result<value> document::operator[]", "document::operator[]"),
    ("value_iterator document::resume_value_iterator", "document::resume_value_iterator"),
    ("simdjson_result<value> simdjson_result<document>::operator[]", "simdjson_result<document>::operator[]"),
    ("value::operator bool", "value::operator bool"),
])
def test_macro_recovery_does_not_treat_return_type_as_callable(declarator, expected):
    assert bc.callable_spelling(declarator) == expected


def test_macro_declarator_recovery_preserves_conversion_and_assignment_names():
    text = """struct document {
simdjson_really_inline document &operator=(const document &other) noexcept = delete;
simdjson_really_inline operator array() & noexcept(false);
};"""
    rows = bc.definitions("document.h", text, include_declarations=True)
    methods = [r for r in rows if r["definition_kind"].startswith("function_")]
    assert {r["symbol"] for r in methods} == {"document::operator=", "document::operator array"}
    assert all(r["parameters"] is not None for r in methods)


def test_base_source_and_qualified_owners_are_preserved(repository):
    repo, commit = repository
    file = __import__("pathlib").Path(repo.repo_dir) / "include/ondemand/value.h"
    file.write_text("int injected_from_another_branch() { return 999; }")
    rows, trace = bc.retrieve_source(repo.repo_dir, commit, "OnDemand access", contract())
    assert any(r["symbol"] == "ondemand::value::access" for r in rows)
    assert all("injected_from_another_branch" not in r["observation"] for r in rows)
    assert {"one::get", "two::get"} <= {r["symbol"] for r in rows}
    assert trace["anchors"] == ["include/ondemand/value.h"]
    with pytest.raises(AssertionError):
        bc.retrieve_source(repo.repo_dir, "0" * 40, "Issue", contract())


def test_graph_locations_are_checked_against_actual_source(repository, monkeypatch):
    repo, commit = repository
    leads = [
        {"id": "actual", "file": "include/ondemand/value.h", "line": 1, "name": "access", "provenance": []},
        {"id": "wrong", "file": "include/ondemand/value.h", "line": 1, "name": "missing", "provenance": []},
    ]
    monkeypatch.setattr(bc, "graph_candidates", lambda *args: (leads, {"status": "available"}))
    rows, trace = bc.retrieve_source(repo.repo_dir, commit, "Issue", contract(), object())
    assert any("graph" in r["channels"] for r in rows)
    assert {r["id"] for r in trace["rejected"]} == {"wrong"}
    assert all(g["id"] != "wrong" for r in rows for g in r.get("graph_leads", []))


def test_dedup_preserves_original_execution_and_different_owners():
    body = "int access() { /* repeated source with useful context */ return 1; }"
    sources = [{"id": "CODE-a", "path": "a.h", "kind": "code", "observation": body, "aliases": []},
               {"id": "CODE-b", "path": "b.h", "kind": "code", "observation": body, "aliases": []}]
    observations = [{"id": "SOURCE-1", "path": "a.h", "observation": body},
                    {"id": "RUN-1", "locator": "c++ -DCONFIG_A probe.cpp", "exit_code": 1, "observation": body + "\nfailed"},
                    {"id": "RUN-2", "locator": "c++ -DCONFIG_B probe.cpp", "exit_code": 0, "observation": "passed"}]
    before = deepcopy(observations)
    rows, aliases, originals = bc.deduplicate(observations, sources)
    assert aliases == {"SOURCE-1": "CODE-a"}
    assert {r["id"] for r in rows} >= {"CODE-a", "CODE-b", "RUN-1", "RUN-2"}
    run = next(r for r in rows if r["id"] == "RUN-1")
    assert run["exit_code"] == 1 and "failed" in run["observation"]
    assert body in run["observation"]  # Factoring waits until bodies survive delivery.
    assert originals == {r["id"]: r for r in before}
    assert observations == before


def test_paths_resolve_shorthand_urls_directories_and_reject_ambiguous_names():
    files = {"include/ondemand/document.h", "include/ondemand/document-inl.h",
             "include/dom/object.h", "include/ondemand/object.h"}
    paths, rejected = bc.resolve_paths(
        "include/ondemand/document.h/-inl.h https://github.com/x/y/blob/main/include/dom/object.h "
        "object.h /usr/include/nonrepo.h ../include/dom/object.h", files)
    assert paths == ["include/ondemand/document.h", "include/ondemand/document-inl.h", "include/dom/object.h"]
    assert {r["reason"] for r in rejected} == {"ambiguous_anchor", "anchor_not_in_base"}
    for scope in ("include/ondemand", "include/ondemand/*"):
        assert set(bc.resolve_paths(scope, files)[0]) == {p for p in files if "/ondemand/" in p}


def test_explicit_scope_allows_reading_fuzzer_but_issue_trace_does_not(repository):
    repo, commit = repository
    issue = "Crash in tests/src/fuzzer.cpp"
    rows, _ = bc.retrieve_source(repo.repo_dir, commit, issue, contract())
    assert all(not r["path"].startswith("tests/") for r in rows)
    rows, report = bc.retrieve_source(repo.repo_dir, commit, issue, contract("tests/src/fuzzer.cpp"))
    assert any(r["symbol"] == "LLVMFuzzerTestOneInput" for r in rows)
    assert report["explicit_scoped_nonproduction_files"] == ["tests/src/fuzzer.cpp"]
    assert not bc.source_file("tests/src/fuzzer.cpp")  # General production rule unchanged.


def test_graph_declaration_resolves_to_sibling_implementation(repository, monkeypatch):
    repo, commit = repository
    leads = [{"id": "resume", "file": "include/ondemand/reader.h", "line": 1,
              "type": "FUNC", "name": "resume", "provenance": []}]
    monkeypatch.setattr(bc, "graph_candidates", lambda *args: (leads, {"status": "available"}))
    rows, report = bc.retrieve_source(repo.repo_dir, commit, "Issue", contract(), object())
    tagged = [r for r in rows if "graph" in r["channels"]]
    assert len(tagged) == 1 and tagged[0]["path"] == "include/ondemand/reader-inl.h"
    assert tagged[0]["symbol"] == "ondemand::reader::resume"
    assert tagged[0]["graph_leads"][0]["source_resolution"] == "declaration_to_definition"
    assert not report["rejected"]


def test_graph_function_does_not_match_a_containing_class_or_substring(repository, monkeypatch):
    repo, commit = repository
    leads = [{"id": "class-prefix", "file": "include/ondemand/value.h", "line": 1,
              "name": "val", "type": "FUNC", "provenance": []},
             {"id": "member", "file": "include/ondemand/value.h", "line": 1,
              "name": "access", "type": "FUNC", "provenance": []}]
    monkeypatch.setattr(bc, "graph_candidates", lambda *args: (leads, {"status": "available"}))
    rows, report = bc.retrieve_source(repo.repo_dir, commit, "Issue", contract(), object())
    assert [r["symbol"] for r in rows if "graph" in r["channels"]] == ["ondemand::value::access"]
    assert [r["id"] for r in report["rejected"]] == ["class-prefix"]


@pytest.mark.parametrize("scope", ["include/ondemand/value.h", "include/"])
def test_repeated_graph_paths_do_not_use_up_file_slots(repository, monkeypatch, scope):
    from pathlib import Path
    repo, _ = repository
    for i in range(55):
        (Path(repo.repo_dir) / f"include/noise{i:02}.h").write_text("int unrelated() { return 0; }")
    (Path(repo.repo_dir) / "include/zz.h").write_text("int needed() { return 1; }")
    subprocess.run(["git", "-C", repo.repo_dir, "add", "."], check=True)
    subprocess.run(["git", "-C", repo.repo_dir, "-c", "user.name=Context test", "-c", "user.email=context@example.test",
                    "commit", "-qm", "file budget fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", repo.repo_dir, "rev-parse", "HEAD"], text=True).strip()
    lead = {"id": "member", "file": "include/ondemand/value.h", "line": 1, "name": "access", "provenance": []}
    leads = [lead] * 20 + [{"id": "needed", "file": "include/zz.h", "line": 1, "name": "needed", "provenance": []}]
    monkeypatch.setattr(bc, "graph_candidates", lambda *args: (leads, {"status": "available"}))
    text_rows, _ = bc.retrieve_source(repo.repo_dir, commit, "Issue", contract(scope))
    rows, report = bc.retrieve_source(repo.repo_dir, commit, "Issue", contract(scope), object())
    assert not any(r["symbol"] == "needed" for r in text_rows)
    assert any(r["symbol"] == "needed" and "graph" in r["channels"] for r in rows)
    assert len(report["read_files"]) == 48
    if scope == "include/":
        assert len(report["anchors"]) > 48 and not report["exact_file_anchors"]


def test_ineligible_and_duplicate_graph_seeds_do_not_consume_traversal(monkeypatch, tmp_path):
    from code_intent.code_intent import entity_tools, embedding_manifest
    from code_intent.code_intent.graph_builder import Graph
    functions = {
        "test": {"def_file": "tests/helper.cpp", "short_name": "access", "fq_name": "access", "def_start_line": 1},
        "target": {"def_file": "include/ondemand/value.h", "short_name": "access", "fq_name": "ondemand::value::access", "def_start_line": 1},
        "mirror": {"def_file": "include/ondemand/value.h", "short_name": "access", "fq_name": "expanded::value::access", "def_start_line": 1},
        **{f"other{i}": {"def_file": "include/dom/object.h", "short_name": f"unrelated{i}",
                         "fq_name": f"unrelated{i}", "def_start_line": i + 2} for i in range(4)},
    }
    hits = [( {"func_id": fid}, 1.0) for fid in ["test", "other0", "other1", "other2", "target"]]
    encoded, expanded = [], []
    rag = SimpleNamespace(
        func_index=SimpleNamespace(index_path=tmp_path / "func_index.faiss", search=lambda vector, k: hits),
        feature_index=SimpleNamespace(search=lambda vector, k: [({"feature_id": "feature"}, 1.0)]),
        graph_ctx=SimpleNamespace(functions=functions, classes={}, graph=Graph(), features={"feature": {"feature_name": "access"}},
                                  feature_to_funcs={"feature": list(functions)}),
        _encode_query=lambda query: encoded.append(query))
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: rag)
    monkeypatch.setattr(embedding_manifest, "load_embedding_manifest", lambda path: {
        "indexes": {k: {"model": "voyage-2"} for k in ("functions", "features")}})
    def neighbors(fid, **kwargs):
        expanded.append(fid)
        return {"neighbors": [{"id": "test", "type": "FUNC", "edge_types": ["CALLS"]}]}
    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    leads, report = bc.graph_candidates(SimpleNamespace(_activate_instance=lambda: None),
        "ondemand value access", ["include/ondemand/value.h"], {"include/ondemand/value.h", "include/dom/object.h"})
    assert len(encoded) == 1
    assert expanded[0] == "target" and len(expanded) == 4
    assert "test" not in report["seed_ids"] and "mirror" not in report["seed_ids"]
    assert all(r["id"] != "test" for r in leads)
    assert {r["channel"] for r in report["skipped"]} == {"function_vector", "intent_membership", "structural_neighbor"}


def test_dependent_type_member_requires_real_outward_edges_and_preserves_path():
    from code_intent.code_intent.graph_builder import Graph
    graph = Graph()
    classes = {name: {"fq_name": name, "short_name": name, "def_file": "include/state.h", "def_line": i + 1}
               for i, name in enumerate(["Receiver", "State", "Unrelated", "TemplateHub", "TooDeep"])}
    functions = {name: {"fq_name": name, "short_name": name, "def_file": "include/state.h", "def_start_line": i + 20}
                 for i, name in enumerate(["entry", "advance", "noise", "template_noise", "deep_noise"])}
    for fid, row in functions.items():
        graph.add_node(fid, "FUNC", row)
    for cid, row in classes.items():
        graph.add_node(cid, "CLASS", row)
    for source, target, edge in [("entry", "Receiver", "IN_CLASS"), ("Receiver", "State", "DEP_TYPE_REF"),
            ("advance", "State", "IN_CLASS"), ("Unrelated", "Receiver", "DEP_TYPE_REF"),
            ("noise", "Unrelated", "IN_CLASS"), ("Receiver", "TemplateHub", "DEP_TEMPLATE_REF"),
            ("template_noise", "TemplateHub", "IN_CLASS"), ("State", "TooDeep", "DEP_TYPE_REF"),
            ("deep_noise", "TooDeep", "IN_CLASS")]:
        graph.add_edge(source, target, edge)
    ctx = SimpleNamespace(graph=graph, functions=functions, classes=classes)
    leads, report = bc.dependent_type_members(ctx, ["entry"], lambda row: True, bc.words("advance state"), [])
    assert [r["id"] for r in leads] == ["advance"]
    assert leads[0]["provenance"][0]["path"] == [
        {"from": "entry", "to": "Receiver", "edge_type": "IN_CLASS", "direction": "out"},
        {"from": "Receiver", "to": "State", "edge_type": "DEP_TYPE_REF", "direction": "out"},
        {"from": "State", "to": "advance", "edge_type": "IN_CLASS", "direction": "in"}]
    assert "not_call_chain" in leads[0]["provenance"][0]["status"]
    assert report["member_lead_count"] == 1
    graph._edges_out["Receiver"]["DEP_TYPE_REF"].remove("State")
    assert bc.dependent_type_members(ctx, ["entry"], lambda row: True, set(), [])[0] == []


def test_dependent_types_share_member_budget_and_macro_alias_ranking():
    from code_intent.code_intent.graph_builder import Graph
    graph = Graph()
    classes, functions = {}, {}
    for cid in ["Receiver", "Left", "Right"]:
        classes[cid] = {"fq_name": cid, "short_name": cid, "def_file": f"include/{cid}.h", "def_line": 1}
        graph.add_node(cid, "CLASS", classes[cid])
    functions["entry"] = {"fq_name": "entry", "short_name": "entry", "def_file": "include/entry.h", "def_start_line": 1}
    graph.add_node("entry", "FUNC", functions["entry"])
    graph.add_edge("entry", "Receiver", "IN_CLASS")
    for cid in ["Left", "Right"]:
        graph.add_edge("Receiver", cid, "DEP_TYPE_REF")
        for i in range(80):
            fid = f"{cid}-{i:02}"
            functions[fid] = {"fq_name": fid, "short_name": f"method{i}", "def_file": f"include/{cid}.h", "def_start_line": i + 2}
            graph.add_node(fid, "FUNC", functions[fid])
            graph.add_edge(fid, cid, "IN_CLASS")
    functions["physical_alias"] = {**functions["Right-79"], "short_intent": "Advance cursor after consuming an object"}
    ctx = SimpleNamespace(graph=graph, functions=functions, classes=classes)
    leads, report = bc.dependent_type_members(ctx, ["entry"], lambda row: True, bc.words("advance cursor"), [])
    assert len(leads) == report["max_members"] == 64
    assert sum(r["id"].startswith("Left") for r in leads) == 32
    assert sum(r["id"].startswith("Right") for r in leads) == 32
    assert leads[1]["id"] == "Right-79"  # Ranking may use its alias; the edge still targets the real member.
    assert all(r["id"] != "physical_alias" for r in leads)


def test_dependent_type_declaration_is_materialized_as_an_implementation(repository, monkeypatch):
    from code_intent.code_intent.graph_builder import Graph
    repo, commit = repository
    graph = Graph()
    functions = {"entry": {"fq_name": "ondemand::value::access", "short_name": "access", "def_file": "include/ondemand/value.h", "def_start_line": 1},
                 "resume": {"fq_name": "ondemand::reader::resume", "short_name": "resume", "def_file": "include/ondemand/reader.h", "def_start_line": 1}}
    classes = {"value": {"fq_name": "ondemand::value", "short_name": "value", "def_file": "include/ondemand/value.h", "def_line": 1},
               "reader": {"fq_name": "ondemand::reader", "short_name": "reader", "def_file": "include/ondemand/reader.h", "def_line": 1}}
    for fid, row in functions.items():
        graph.add_node(fid, "FUNC", row)
    for cid, row in classes.items():
        graph.add_node(cid, "CLASS", row)
    for source, target, edge in [("entry", "value", "IN_CLASS"), ("value", "reader", "DEP_TYPE_REF"), ("resume", "reader", "IN_CLASS")]:
        graph.add_edge(source, target, edge)
    ctx = SimpleNamespace(graph=graph, functions=functions, classes=classes)
    monkeypatch.setattr(bc, "graph_candidates", lambda index, query, paths, files: bc.dependent_type_members(
        ctx, ["entry"], lambda row: row["def_file"] in files, bc.words(query), paths))
    rows, _ = bc.retrieve_source(repo.repo_dir, commit, "OnDemand access", contract(), object())
    implementation = next(r for r in rows if r["symbol"] == "ondemand::reader::resume")
    assert implementation["path"] == "include/ondemand/reader-inl.h"
    assert "graph" in implementation["channels"]
    assert implementation["graph_leads"][0]["source_resolution"] == "declaration_to_definition"


def test_type_extension_preserves_the_original_catalog_through_rerank(repository, monkeypatch, tmp_path):
    from pathlib import Path
    repo, _ = repository
    (Path(repo.repo_dir) / "include/bulk.h").write_text("\n".join(f"int keep{i}() {{ return {i}; }}" for i in range(270)))
    (Path(repo.repo_dir) / "include/zz_state.h").write_text("\n".join(f"int added{i}() {{ return {i}; }}" for i in range(64)))
    subprocess.run(["git", "-C", repo.repo_dir, "add", "."], check=True)
    subprocess.run(["git", "-C", repo.repo_dir, "-c", "user.name=Context test", "-c", "user.email=context@example.test",
                    "commit", "-qm", "catalog budget fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", repo.repo_dir, "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(bc, "graph_candidates", lambda *args: ([], {"status": "available"}))
    before, _ = bc.retrieve_source(repo.repo_dir, commit, "Issue", contract(), object())
    assert len(before) == 256 and not any(r["symbol"].startswith("added") for r in before)
    leads = [{"id": f"added{i}", "file": "include/zz_state.h", "line": i + 1, "type": "FUNC", "name": f"added{i}",
              "provenance": [{"channel": "dependent_type_member", "path": []}]} for i in range(64)]
    monkeypatch.setattr(bc, "graph_candidates", lambda *args: (leads, {"status": "available"}))
    after, report = bc.retrieve_source(repo.repo_dir, commit, "Issue", contract(), object())
    assert len(after) == 320 and report["catalog_extension_count"] == 64
    assert {r["id"] for r in before} <= {r["id"] for r in after}
    memory = Memory(instance_id="test__repo-1", problem_statement="Issue", repair_intent_bundle={
        "cited_repository_evidence": [{"id": f"RUN-{i}", "observation": "failed"} for i in range(40)]})
    def select(prompt, schema):
        candidates = json.loads(prompt.split("\nINPUT:\n", 1)[1])["candidates"]
        assert len(candidates) == 360
        assert {r["id"] for r in before} <= {r["id"] for r in candidates}
        row = next(r for r in candidates if r.get("symbol") == "added0")
        return schema(selected=[bc.EvidenceChoice(evidence_id=row["id"], purpose="State source")], missing_relationships=[])
    bc.prepare_branch_context(memory, contract(), repo, commit, tmp_path / "out", "graph", select, object())


def test_budget_omission_never_erases_observed_source_or_leaves_dangling_reference():
    body = "int access() { /* existing source body important to the repair */ return 1; }"
    sources = [{"id": "CODE-a", "path": "a.h", "kind": "code", "observation": body, "aliases": [],
                "metadata": "large " * 3000}]
    observations = [{"id": "RUN-1", "observation": body + "\ncompilation failed", "exit_code": 1}]
    rows, aliases, _ = bc.deduplicate(observations, sources)
    selected = bc.BranchEvidenceSelection(selected=[bc.EvidenceChoice(evidence_id=i, purpose="check")
        for i in ("RUN-1", "CODE-a")], missing_relationships=[])
    bundle, omitted = bc.assemble(selected, rows, aliases, "/tmp/archive.json", 1200)
    assert [r["id"] for r in bundle["cited_repository_evidence"]] == ["RUN-1"]
    assert body in bundle["cited_repository_evidence"][0]["observation"]
    assert not bundle["cited_repository_evidence"][0].get("code_references")
    assert omitted[0]["reason"] == "context_budget"
    rows[0].pop("metadata")
    bundle, _ = bc.assemble(selected, rows, aliases, "/tmp/archive.json", 1200)
    run = bundle["cited_repository_evidence"][0]
    assert body not in run["observation"] and run["code_references"] == ["CODE-a"]
    assert body == bundle["cited_repository_evidence"][1]["observation"]


def test_factored_reference_follows_a_delivered_container():
    body = "int access() { /* repeated source with useful context */ return 1; }"
    rows = [{"id": "RUN-1", "kind": "observation", "observation": body + " failed"},
            {"id": "CODE-member", "kind": "code", "path": "a.h", "observation": body},
            {"id": "CODE-class", "kind": "code", "path": "a.h", "observation": "struct A { " + body + " };"}]
    selected = bc.BranchEvidenceSelection(selected=[bc.EvidenceChoice(evidence_id=r["id"], purpose="check")
        for r in rows], missing_relationships=[])
    bundle, _ = bc.assemble(selected, rows, {}, "/tmp/archive.json", 2000)
    run = bundle["cited_repository_evidence"][0]
    assert body in run["observation"] or run.get("code_references") == ["CODE-class"]
    assert "CODE-member" not in run.get("code_references", [])
    assert bundle["evidence_aliases"]["CODE-member"] == "CODE-class"


def test_nested_member_only_aliases_to_delivered_container():
    rows = [{"id": "SOURCE-28", "path": "type.h", "kind": "code", "observation": "struct X { int get() { return 1; } };"},
            {"id": "SOURCE-29", "path": "type.h", "kind": "code", "observation": "int get() { return 1; }"}]
    selection = bc.BranchEvidenceSelection(selected=[bc.EvidenceChoice(evidence_id=r["id"], purpose="check") for r in rows], missing_relationships=[])
    bundle, _ = bc.assemble(selection, rows, {}, "/tmp/archive.json", 2000)
    assert [r["id"] for r in bundle["cited_repository_evidence"]] == ["SOURCE-28"]
    assert bundle["evidence_aliases"]["SOURCE-29"] == "SOURCE-28"
    rows[0]["observation"] += " huge" * 10000
    bundle, omitted = bc.assemble(selection, rows, {}, "/tmp/archive.json", 2000)
    assert [r["id"] for r in bundle["cited_repository_evidence"]] == ["SOURCE-29"]
    assert "SOURCE-29" not in bundle["evidence_aliases"]
    assert omitted[0]["reason"] == "context_budget"


@pytest.mark.parametrize("ids", [["unknown"], []])
def test_invalid_selections_are_not_silently_accepted(ids):
    selection = bc.BranchEvidenceSelection(selected=[bc.EvidenceChoice(evidence_id=i, purpose="check") for i in ids], missing_relationships=[])
    with pytest.raises(ValueError):
        bc.assemble(selection, [{"id": "RUN-1", "kind": "observation", "observation": "failed"}], {}, "/tmp/a.json", 2000)


def test_existing_aliases_duplicates_and_rank_limits_do_not_lose_valid_evidence():
    rows = [{"id": f"CODE-{i}", "kind": "observation", "observation": "failed"} for i in range(26)]
    selection = bc.BranchEvidenceSelection(selected=[
        bc.EvidenceChoice(evidence_id="SOURCE-1", purpose="Original contract evidence alias"),
        *[bc.EvidenceChoice(evidence_id=r["id"], purpose="check") for r in rows],
    ], missing_relationships=[])
    bundle, omitted = bc.assemble(selection, rows, {"SOURCE-1": "CODE-0"}, "/tmp/a.json", 8000)
    assert [r["id"] for r in bundle["cited_repository_evidence"]] == [r["id"] for r in rows[:24]]
    assert {r["reason"] for r in omitted} == {"duplicate_selection", "ranked_selection_limit"}
    selection.selected[0].evidence_id = "SOURCE-unknown"
    with pytest.raises(ValueError, match="Unknown evidence ID"):
        bc.assemble(selection, rows, {"SOURCE-1": "CODE-0"}, "/tmp/a.json", 8000)


def test_prepare_changes_only_evidence_and_archives_exact_records(repository, tmp_path):
    repo, commit = repository
    observations = [{"id": "RUN-1", "observation": "Existing caller fails", "exit_code": 1}]
    memory = Memory(instance_id="test__repo-1", problem_statement="OnDemand access", repair_summary="full assigned behavior",
                    repair_intent_contract={"probes": {"ACCEPT-1": "unchanged oracle"}},
                    repair_intent_bundle={"cited_repository_evidence": observations})
    before = memory.model_dump()
    def select(prompt, schema):
        data = json.loads(prompt.split("\nINPUT:\n", 1)[1])
        assert data["assigned_contract"] == contract()
        assert data["evidence_budget_tokens"] == 12000
        assert all(r["estimated_tokens"] > 0 for r in data["candidates"])
        row = next(r for r in data["candidates"] if r.get("symbol") == "ondemand::value::access")
        return schema(selected=[bc.EvidenceChoice(evidence_id=row["id"], purpose="access to next"),
                                bc.EvidenceChoice(evidence_id="RUN-1", purpose="failure observation")], missing_relationships=["Remaining callers not enumerated"])
    result, path, archive = bc.prepare_branch_context(memory, contract(), repo, commit, tmp_path / "out", "text", select)
    assert archive["records"]["RUN-1"] == observations[0]
    assert path.startswith("/tmp/intent2fix_context_")
    assert result.model_dump(exclude={"repair_intent_bundle"}) == memory.model_dump(exclude={"repair_intent_bundle"})
    assert memory.model_dump() == before
    report = json.loads((tmp_path / "out/branch_context_report.json").read_text())
    assert report["status"] == "selected" and report["output_tokens"] <= 12000 and report["budget"] == 12000
    def should_not_call(*args):
        raise AssertionError("Exact prepared input should reuse the saved selection")
    cached, _, _ = bc.prepare_branch_context(memory, contract(), repo, commit, tmp_path / "out", "text", should_not_call)
    assert cached == result
    changed = contract()
    changed["branch"]["expected_behavior"] = ["different policy"]
    with pytest.raises(ValueError, match="different inputs"):
        bc.prepare_branch_context(memory, changed, repo, commit, tmp_path / "out", "text", should_not_call)


def test_rerank_failure_is_recorded_with_nonempty_fallback(repository, tmp_path):
    repo, commit = repository
    memory = Memory(instance_id="test__repo-1", problem_statement="Issue", repair_intent_bundle={
        "cited_repository_evidence": [{"id": "RUN-1", "observation": "Existing call failed", "exit_code": 1}]})
    def failed(*args):
        raise RuntimeError("provider unavailable")
    result, _, _ = bc.prepare_branch_context(memory, contract(), repo, commit, tmp_path / "out", "text", failed)
    assert result.repair_intent_bundle["cited_repository_evidence"]
    report = json.loads((tmp_path / "out/branch_context_report.json").read_text())
    assert report["status"] == "fallback" and report["error_type"] == "RuntimeError"


def test_progressive_diagnostics_leave_contract_untouched_and_reuse_summaries(repository, tmp_path):
    repo, commit = repository
    raw = {"id": "RUN-1", "locator": "c++ -std=c++20 probe.cpp", "exit_code": 1,
           "observation": "template trace\n" * 1000 + "no member access"}
    memory = Memory(instance_id="test__repo-1", problem_statement="OnDemand access",
                    repair_intent_contract={"probes": {"ACCEPT-1": "unchanged oracle"}},
                    repair_intent_bundle={"cited_repository_evidence": [raw]})
    calls = []
    def summarize(prompt, schema):
        calls.append(prompt)
        return schema(conditions="C++20 probe", finding="Compilation fails: no member access",
                      limits="Runtime behavior unknown", evidence_quotes=["no member access"])
    def select(prompt, schema):
        return schema(selected=[bc.EvidenceChoice(evidence_id="RUN-1", purpose="Compilation evidence")], missing_relationships=[])
    options = dict(summarize=summarize, summary_model_key="test", diagnostic_cache_dir=tmp_path / "cache")
    for branch in ("H0", "H1"):
        result, _, archive = bc.prepare_branch_context(memory, contract(), repo, commit,
            tmp_path / branch, "text", select, **options)
        assert archive["records"]["RUN-1"] == archive["handoff_records"]["RUN-1"] == raw
        assert result.repair_intent_contract == memory.repair_intent_contract
        card = result.repair_intent_bundle["cited_repository_evidence"][0]
        assert card["summary_status"] == "model" and len(card["observation"]) < 1000
        assert result.repair_intent_bundle["archive_read_example"]["tool"] == "read_source"
    assert len(calls) == 1 and memory.repair_intent_bundle["cited_repository_evidence"] == [raw]
    with pytest.raises(ValueError, match="different inputs"):
        bc.prepare_branch_context(memory, contract(), repo, commit, tmp_path / "H0", "text", select,
            **{**options, "summary_model_key": "changed"})
