"""Read definitions referenced by Scout without generating another causal summary."""

import json
import re
import shlex
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_cpp


def referenced_predicates(text):
    return list(dict.fromkeys(re.findall(r"\b((?:is_|has_)\w+)\s*<", text)))


def candidate_boundary_symbols(bundle):
    symbols = []
    for boundary in bundle.get("candidate_boundaries", []):
        anchor = str(boundary.get("anchor_operation", "")).split("(", 1)[0].strip()
        qualified = re.match(r"^(?:[A-Za-z_]\w*::)+([A-Za-z_]\w*)\b", anchor)
        if qualified:
            symbols.append(qualified.group(1))
        elif re.fullmatch(r"[A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)?", anchor):
            symbols.append(anchor.split()[-1])
    return list(dict.fromkeys(symbols))


def cpp_definitions(path, content, symbols):
    source = content.encode()
    tree = Parser(Language(tree_sitter_cpp.language())).parse(source)
    stack = [tree.root_node]
    found = []
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.named_children))
        if node.type in {"struct_specifier", "class_specifier", "alias_declaration"}:
            name = node.child_by_field_name("name")
        elif node.type == "function_definition":
            name = node.child_by_field_name("declarator")
            while name is not None and name.child_by_field_name("declarator"):
                name = name.child_by_field_name("declarator")
            if name is not None and name.type == "qualified_identifier":
                name = name.child_by_field_name("name")
        else:
            continue
        if name is not None and name.type in {"template_type", "template_function"}:
            name = name.child_by_field_name("name")
        if name is None or name.text.decode() not in symbols:
            continue
        symbol = name.text.decode()
        body = node.child_by_field_name("body")
        has_body = (node.type not in {"class_specifier", "struct_specifier"}
                    or node.child_by_field_name("body") is not None)
        # Unknown project macros can cause parser errors inside a bounded body.
        # Require real opening/closing braces; never accept a missing close token.
        bounded_body = (body is not None and body.text.startswith(b"{")
                        and body.text.endswith(b"}")
                        and not body.children[-1].is_missing)
        if node.parent and node.parent.type == "template_declaration":
            node = node.parent
        # Never present a truncated or syntactically broken body as a definition.
        text = source[node.start_byte:node.end_byte].decode()
        found.append({"symbol": symbol, "path": path,
                      "start_byte": node.start_byte, "end_byte": node.end_byte,
                      "start_line": node.start_point.row + 1,
                      "end_line": node.end_point.row + 1,
                      "parser_errors": node.has_error,
                      "complete": has_body and (not node.has_error or bounded_body) and len(text) <= 6000,
                      "text": text if len(text) <= 6000 else ""})
    return found


def expand_source_evidence(env, bundle, issue_evidence, output_dir):
    observations = bundle.get("cited_repository_evidence", [])
    raw_text = "\n".join(row.get("observation", "") for row in observations)
    seeds = candidate_boundary_symbols(bundle)
    seeds += referenced_predicates(raw_text)
    seeds += [name.rsplit("::", 1)[-1] for name in issue_evidence.get("symbols", [])
              if re.fullmatch(r"(?:\w+::)*\w+", name)]
    seeds += re.findall(r"::([a-z_]\w*)\s*\(", raw_text)
    pending = list(dict.fromkeys(seeds))[:24]
    requested, definitions, seen = [], [], set()
    evidence = []
    rounds = []
    remaining_chars = 36000
    # The container supplies tracked base-commit source only. Parsing stays local.
    reader = '''import json, pathlib, re, subprocess, sys
symbols = json.loads(sys.argv[1])
args = ['git', 'grep', '-I', '-l', '-w']
for symbol in symbols: args += ['-e', symbol]
args += ['--', '*.h', '*.hpp', '*.hh', '*.hxx', '*.cpp', '*.cc', '*.inl', '*.ipp', '*.tpp']
result = subprocess.run(args, stdout=subprocess.PIPE, check=False)
root = pathlib.Path.cwd().resolve()
files = []
for raw in result.stdout.decode().splitlines():
    path = pathlib.Path(raw)
    if set(path.parts) & {'test', 'tests', 'benchmark', 'benchmarks', 'examples'}: continue
    try: path.resolve().relative_to(root)
    except ValueError: continue
    if not path.is_file(): continue
    if path.stat().st_size > 750000: continue
    files.append(path)
contents = {str(path): path.read_text(errors='replace') for path in files}
def priority(path):
    text = contents[str(path)]
    rank = next((i for i, symbol in enumerate(symbols)
                 if re.search(r'\\b' + re.escape(symbol) + r'\\b', text)), len(symbols))
    return rank, path.stat().st_size, str(path)
files.sort(key=priority)
print(json.dumps({'search_exit': result.returncode, 'matched_files': len(files),
                  'files': {str(path): contents[str(path)] for path in files[:16]}}))
'''
    for _ in range(2):
        if not pending or not remaining_chars:
            break
        requested.extend(pending)
        command = "python3 -c " + shlex.quote(reader) + " " + shlex.quote(json.dumps(pending))
        output, code = env.run_bash(command, timeout=120)
        if code:
            Path(output_dir, "hypothesis_source_error.json").write_text(json.dumps({
                "exit_code": code, "output": output[-2000:]}))
            raise ValueError("Source-definition retrieval failed")
        packet = json.loads(output)
        if packet["search_exit"] not in (0, 1):
            raise ValueError("Source-definition search failed")
        rounds.append({"symbols": pending, "matched_files": packet["matched_files"],
                       "read_files": list(packet["files"])})
        round_chars = min(18000, remaining_chars) if len(rounds) == 1 else remaining_chars
        dependencies = []
        candidates = [item for path, content in packet["files"].items()
                      for item in cpp_definitions(path, content, set(pending))]
        candidates.sort(key=lambda item: pending.index(item["symbol"]))
        for item in candidates:
            path = item["path"]
            key = (path, item["start_byte"], item["end_byte"])
            if key in seen:
                continue
            seen.add(key)
            text = item.pop("text")
            item["included"] = item["complete"] and len(text) <= round_chars
            definitions.append(item)
            if not item["included"]:
                continue
            remaining_chars -= len(text)
            round_chars -= len(text)
            evidence.append({"id": f"SOURCE-{len(evidence) + 1}", "path": path,
                             "locator": f"{path}:{item['start_line']}-{item['end_line']} ({item['symbol']})",
                             "observation": text})
            dependencies.extend(referenced_predicates(text))
        pending = [name for name in dict.fromkeys(dependencies) if name not in requested][:24]
    manifest = {"requested_symbols": requested, "definitions": definitions, "rounds": rounds,
                "unexpanded_dependency_symbols": pending,
                "evidence": evidence, "unresolved_symbols": sorted(set(requested) - {
                    item["symbol"] for item in definitions if item["included"]}),
                "limits": {"rounds": 2, "symbols_per_round": 24, "files_per_round": 16,
                           "file_bytes": 750000, "definition_chars": 6000,
                           "first_round_chars": 18000, "total_chars": 36000}}
    Path(output_dir, "hypothesis_source_evidence.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    return {**bundle, "cited_repository_evidence": [*observations, *evidence]}
