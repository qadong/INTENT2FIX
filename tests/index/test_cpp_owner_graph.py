import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from code_intent.code_intent.code_indexer import (
    _compile_args_with_working_directory,
    _clear_unindexed_anonymous_owner_ids,
    _collect_extra_translation_units,
    _deduplicate_edges,
    _extract_preprocessor_directives,
    _partition_graph_ready_edges,
    _qualify_multi_definition_classes,
    _qualify_multi_definition_functions,
    _qualify_multi_definition_template_instantiations,
    _reconcile_template_instantiation_targets,
    _reconcile_dependency_endpoints,
    _require_nonempty_index,
    _sanitize_args,
    index_repo,
)
from code_intent.code_intent import code_indexer
from code_intent.code_intent.graph_builder import (
    EDGE_CALLS,
    EDGE_DEP_CALL,
    EDGE_IN_CLASS,
    EDGE_IN_FILE,
    NODE_FILE,
    NODE_PREPROCESSOR,
    build_graph,
)


def test_extra_tu_discovery_skips_recursive_compile_stress_wrappers(tmp_path):
    normal = tmp_path / "project" / "normal.cpp"
    stress = (
        tmp_path
        / "projects"
        / "SelfTest"
        / "CompileTimePerfTests"
        / "100.tests.cpp"
    )
    normal.parent.mkdir(parents=True)
    stress.parent.mkdir(parents=True)
    normal.write_text("int normal() { return 1; }\n", encoding="utf-8")
    stress.write_text('#include "10.tests.cpp"\n' * 10, encoding="utf-8")

    discovered = _collect_extra_translation_units(str(tmp_path), set())

    assert normal in discovered
    assert stress not in discovered


def test_process_executor_matches_thread_executor(monkeypatch, tmp_path):
    library = Path("/usr/lib/llvm-14/lib/libclang.so.1")
    if not library.exists():
        pytest.skip("libclang 14 unavailable")
    repo = tmp_path / "repo"
    build = tmp_path / "build"
    repo.mkdir()
    build.mkdir()
    (repo / "common.hpp").write_text(
        "inline int increment(int value) { return value + 1; }\n",
        encoding="utf-8",
    )
    sources = []
    for name in ("a", "b"):
        source = repo / f"{name}.cpp"
        source.write_text(
            f'#include "common.hpp"\nint {name}() {{ return increment(1); }}\n',
            encoding="utf-8",
        )
        sources.append(source)
    commands = [
        {
            "directory": str(repo),
            "file": str(source),
            "arguments": [
                "clang++",
                "-std=c++17",
                f"-I{repo}",
                "-c",
                str(source),
            ],
        }
        for source in sources
    ]
    (build / "compile_commands.json").write_text(
        json.dumps(commands), encoding="utf-8"
    )
    monkeypatch.setenv("PROJECT_ROOT_OVERRIDE", str(repo))
    monkeypatch.setenv("CODE_INTENT_INDEX_NO_STANDALONE_HEADERS", "1")
    monkeypatch.setenv("CODE_INTENT_INDEX_NO_EXTRA_TU", "1")

    original_cwd = Path.cwd()
    outputs = {}
    try:
        for mode in ("process", "thread"):
            monkeypatch.setenv("CODE_INTENT_INDEX_EXECUTOR", mode)
            output = tmp_path / mode
            index_repo(str(build), str(output), str(library), max_workers=2)
            if mode == "process":
                assert Path.cwd() == original_cwd
            functions = {
                (row["id"], row["fq_name"], row["def_file"])
                for row in map(json.loads, (output / "functions.jsonl").read_text().splitlines())
            }
            edges = {
                (row.get("edge_type", "call"), row["caller"], row["callee"])
                for row in map(json.loads, (output / "callgraph.jsonl").read_text().splitlines())
            }
            outputs[mode] = (functions, edges)
    finally:
        os.chdir(original_cwd)

    assert outputs["process"] == outputs["thread"]


def test_translation_unit_keeps_anonymous_macro_namespace_variants(tmp_path):
    library = Path('/usr/lib/llvm-14/lib/libclang.so.1')
    if not library.exists():
        pytest.skip('libclang 14 unavailable')
    header = tmp_path / 'generic.h'
    source = tmp_path / 'main.cpp'
    header.write_text('''
namespace lib { namespace ARCH { namespace { namespace stage {
struct Box {
  int next() { return 1; }
  int run() { return next(); }
};
}}}}
''')
    source.write_text('''
#define ARCH alpha
#include "generic.h"
#undef ARCH
#define ARCH beta
#include "generic.h"
''')
    root = Path(__file__).resolve().parents[2]
    script = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from code_intent.code_intent import code_indexer as ci
ci.init_clang(sys.argv[2])
result = ci.extract_translation_unit(
    ci.Index.create(), sys.argv[3], ['-std=c++17'], sys.argv[4]
)
print(json.dumps(result))
'''
    process = subprocess.run(
        [sys.executable, '-c', script, str(root / 'code_intent/local_fix'),
         str(library), str(source), str(tmp_path)],
        cwd=root, text=True, capture_output=True, check=True,
    )
    result = json.loads(process.stdout.splitlines()[-1])
    runs = [row for row in result['functions'] if row['short_name'] == 'run']
    assert {row['fq_name'] for row in runs} == {
        'lib::alpha::stage::Box::run',
        'lib::beta::stage::Box::run',
    }
    assert len({row['id'] for row in runs}) == 1
    run_calls = {
        (edge['caller_name'], edge['callee_name'])
        for edge in result['call_edges']
        if edge['edge_type'] == 'call' and edge['caller_name'].endswith('::run')
    }
    assert run_calls == {
        ('lib::alpha::stage::Box::run', 'lib::alpha::stage::Box::next'),
        ('lib::beta::stage::Box::run', 'lib::beta::stage::Box::next'),
    }


def test_translation_unit_keeps_macro_template_instantiation_variants(tmp_path):
    library = Path('/usr/lib/llvm-14/lib/libclang.so.1')
    if not library.exists():
        pytest.skip('libclang 14 unavailable')
    header = tmp_path / 'generic.h'
    source = tmp_path / 'main.cpp'
    header.write_text('''
namespace lib { namespace ARCH { namespace { namespace simd {
template<class T> struct Box {};
template<> struct Box<int> {};
}}}}
''')
    source.write_text('''
#define ARCH alpha
#include "generic.h"
#undef ARCH
#define ARCH beta
#include "generic.h"
''')
    root = Path(__file__).resolve().parents[2]
    script = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from code_intent.code_intent import code_indexer as ci
ci.init_clang(sys.argv[2])
result = ci.extract_translation_unit(
    ci.Index.create(), sys.argv[3], ['-std=c++17'], sys.argv[4]
)
ci._qualify_multi_definition_functions([result], sys.argv[4])
ci._qualify_multi_definition_classes([result], sys.argv[4])
stats = ci._qualify_multi_definition_template_instantiations(
    [result], sys.argv[4]
)
print(json.dumps({'result': result, 'stats': stats}))
'''
    process = subprocess.run(
        [sys.executable, '-c', script, str(root / 'code_intent/local_fix'),
         str(library), str(source), str(tmp_path)],
        cwd=root, text=True, capture_output=True, check=True,
    )
    payload = json.loads(process.stdout.splitlines()[-1])
    rows = payload['result']['template_instantiations']
    edges = [
        edge for edge in payload['result']['call_edges']
        if edge['edge_type'] == 'instantiates'
    ]
    assert {row['fq_name'] for row in rows} == {
        'lib::alpha::simd::Box', 'lib::beta::simd::Box',
    }
    assert len({row['id'] for row in rows}) == 2
    assert {(edge['caller'], edge['callee']) for edge in edges} == {
        (row['id'], row['template_def_id']) for row in rows
    }
    assert payload['stats']['edge_rows_not_disambiguated'] == 0
    assert payload['stats']['edge_target_disagreements_after_reconciliation'] == 0


def test_translation_unit_empty_usr_template_edges_match_table_ids(tmp_path):
    library = Path('/usr/lib/llvm-14/lib/libclang.so.1')
    if not library.exists():
        pytest.skip('libclang 14 unavailable')
    source = tmp_path / 'main.cpp'
    source.write_text('template<class T> struct Box {};\ntemplate<> struct Box<int> {};\n')
    root = Path(__file__).resolve().parents[2]
    script = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from code_intent.code_intent import code_indexer as ci
ci.init_clang(sys.argv[2])
original = ci.get_usr
ci.get_usr = lambda cursor: '' if cursor.spelling == 'Box' else original(cursor)
result = ci.extract_translation_unit(
    ci.Index.create(), sys.argv[3], ['-std=c++17'], sys.argv[4]
)
print(json.dumps(result))
'''
    process = subprocess.run(
        [sys.executable, '-c', script, str(root / 'code_intent/local_fix'),
         str(library), str(source), str(tmp_path)],
        cwd=root, text=True, capture_output=True, check=True,
    )
    result = json.loads(process.stdout.splitlines()[-1])
    rows = result['template_instantiations']
    assert rows and {row['template_def_id'] for row in rows} == {'Box'}
    assert {(edge['caller'], edge['callee']) for edge in result['call_edges']
            if edge['edge_type'] == 'instantiates'} == {
        (row['id'], row['template_def_id']) for row in rows
    }


def test_translation_unit_edges_keep_their_definition_file(tmp_path):
    library = Path('/usr/lib/llvm-14/lib/libclang.so.1')
    if not library.exists():
        pytest.skip('libclang 14 unavailable')
    for arch in ('alpha', 'beta'):
        directory = tmp_path / arch
        directory.mkdir()
        (directory / 'shared.h').write_text(f'''
namespace lib {{ namespace {arch} {{ namespace {{
int next() {{ return 1; }}
int run() {{ return next(); }}
}} }} }}
''')
    source = tmp_path / 'main.cpp'
    source.write_text('#include "alpha/shared.h"\n#include "beta/shared.h"\n')
    root = Path(__file__).resolve().parents[2]
    script = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from code_intent.code_intent import code_indexer as ci
ci.init_clang(sys.argv[2])
result = ci.extract_translation_unit(
    ci.Index.create(), sys.argv[3], ['-std=c++17'], sys.argv[4]
)
ci._qualify_multi_definition_functions([result], sys.argv[4])
print(json.dumps(result))
'''
    process = subprocess.run(
        [sys.executable, '-c', script, str(root / 'code_intent/local_fix'),
         str(library), str(source), str(tmp_path)],
        cwd=root, text=True, capture_output=True, check=True,
    )
    result = json.loads(process.stdout.splitlines()[-1])
    functions = {row['id']: row for row in result['functions']}
    run_edges = [
        edge for edge in result['call_edges']
        if edge['edge_type'] == 'call' and edge['caller_name'].endswith('::run')
    ]
    assert len(run_edges) == 2
    for edge in run_edges:
        caller = functions[edge['caller']]
        assert caller['fq_name'] == edge['caller_name']
        assert caller['def_file'] == edge['caller_file']
        assert caller['def_start_line'] == edge['caller_definition_line']


def test_real_clang_template_owners_reach_distinct_class_nodes(tmp_path):
    library = Path('/usr/lib/llvm-14/lib/libclang.so.1')
    if not library.exists():
        pytest.skip('libclang 14 unavailable')
    source = tmp_path / 'owners.cpp'
    source.write_text('''
#include <algorithm>
#include <utility>
namespace n {
struct Helper { int target() { return 7; } };
struct DepTarget {
  template<class T> int is() { return 8; }
  template<class T> int get() { return 9; }
  template<class T> int get(T&) { return 10; }
  template<class T> int pick(T&) { return 11; }
  template<class T> int pick(T const&) { return 12; }
};
struct DepWrapper {
  DepTarget first;
  template<class T> int is() { return first.is<T>(); }
  template<class T> int get() { return first.get<T>(); }
  template<class T> int pick(T& value) { return first.pick<T>(value); }
};
template<bool, class, class> struct Choose;
template<bool B, class T, class F>
using ChooseT = typename Choose<B, T, F>::type;
template<class T> struct AliasTarget {
  template<class C> int parse(C&) { return 13; }
  template<class V, class C> int format(V&&, C&) { return 14; }
};
template<class T> struct NoMethods {};
template<class T>
using AliasTargetChoice =
    ChooseT<(sizeof(T) > 1), AliasTarget<T>, NoMethods<T>>;
template<class T> struct AliasWrapper {
  AliasTargetChoice<T> underlying;
  template<class C> int parse(C& ctx) { return underlying.parse(ctx); }
  template<class V, class C> int format(V&& value, C& ctx) {
    return underlying.format(value, ctx);
  }
};
template<class T, class E=void> struct Box {
  template<class C> int run(C&) { return 1; }
};
template<class T> struct Box<T*, void> {
  template<class C> int run(C&);
  int prepare() { return 2; }
  int call_prepare() { return prepare(); }
  Helper helper;
  int call_target() { return helper.target(); }
};
template<class T> template<class C> int Box<T*, void>::run(C&) { return 3; }
template<> struct Box<int, void> { int run() { return 4; } };
template<class T> int free_function(T&) { return 5; }
struct Friend { friend int free_friend() { return 6; } };
template<class T> struct CanonicalBox {
  void plain() {}
  template<class U> void templ(U) {}
};
template<class T> struct StaticFormat {
  static constexpr int bits();
};
template<> constexpr int StaticFormat<double>::bits() { return 52; }
void call_canonical(CanonicalBox<int>& box) {
  box.plain();
  box.templ(1);
}
template<class L> struct NestedFriend {
  template<class R> friend int nested_pick(NestedFriend&&, R&&) { return 16; } template<class R> friend int nested_pick(NestedFriend&&, R&) { return 17; }
};
int call_nested_friend() {
  NestedFriend<double> value;
  return nested_pick(std::move(value), 1);
}
int call_anonymous_handler() {
  struct { int operator()() { return 18; } } handler;
  return handler();
}
template<class T> struct TempReader {
  explicit TempReader(T&&) {}
  int sax_parse(int, int, int, int = 0) { return 15; }
};
template<class T> using TempReaderAlias = TempReader<T>;
template<class T> int call_typed_temp(T& value) {
  return TempReaderAlias<T>(std::move(value)).sax_parse(1, 2, 3);
}
}
''')
    root = Path(__file__).resolve().parents[2]
    # Keep the bundled binding isolated from other tests' clang installations.
    script = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from code_intent.code_intent import code_indexer as ci
ci.init_clang(sys.argv[2])
index = ci.Index.create()
tu = index.parse(sys.argv[3], args=['-std=c++17'])
assert not list(tu.diagnostics), list(tu.diagnostics)
result = ci.extract_translation_unit(index, sys.argv[3], ['-std=c++17'], sys.argv[4])
print(json.dumps(result))
'''
    process = subprocess.run(
        [sys.executable, '-c', script, str(root / 'code_intent/local_fix'),
         str(library), str(source), str(tmp_path)],
        cwd=root, text=True, capture_output=True, check=True,
    )
    result = json.loads(process.stdout.splitlines()[-1])
    for kind in ('functions', 'classes'):
        (tmp_path / f'{kind}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in result[kind]))
    (tmp_path / 'callgraph.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in result['call_edges']))
    ctx = build_graph(tmp_path)
    runs = [f for f in result['functions'] if f['short_name'] == 'run']
    assert len(runs) == 3
    assert len({f['owner_class_id'] for f in runs}) == 3
    for f in runs:
        assert f['is_method'] and f['classes'] == ['Box']
        assert f['fq_name'] == 'n::Box::run'
        assert ctx.graph.neighbors(f['id'], EDGE_IN_CLASS) == {f['owner_class_id']}
        assert f['owner_class_id'] in ctx.classes
    prepare = next(f for f in result['functions'] if f['short_name'] == 'prepare')
    call_prepare = next(f for f in result['functions'] if f['short_name'] == 'call_prepare')
    member_targets = {
        edge['callee'] for edge in result['call_edges']
        if edge['caller'] == call_prepare['id'] and edge['edge_type'] == 'member_ref'
    }
    assert prepare['id'] in member_targets
    target = next(f for f in result['functions'] if f['fq_name'] == 'n::Helper::target')
    call_target = next(f for f in result['functions'] if f['short_name'] == 'call_target')
    receiver_member_targets = {
        edge['callee'] for edge in result['call_edges']
        if edge['caller'] == call_target['id'] and edge['edge_type'] == 'member_ref'
    }
    assert target['id'] in receiver_member_targets
    out_of_line = next(f for f in runs if f['def_start_line'] > prepare['def_start_line'] and f['is_template'])
    assert out_of_line['owner_class_id'] == prepare['owner_class_id']
    for f in result['functions']:
        if f['short_name'].startswith('free_'):
            assert not f['is_method'] and not f['owner_class_id']
            assert not ctx.graph.neighbors(f['id'], EDGE_IN_CLASS)

    anonymous_handler = next(
        f for f in result['functions']
        if f['short_name'] == 'operator()' and 'return 18' in f['code']
    )
    assert anonymous_handler['is_method']
    assert anonymous_handler['classes'] == []
    assert not anonymous_handler['owner_class_id']
    assert not ctx.graph.neighbors(anonymous_handler['id'], EDGE_IN_CLASS)

    for name in ('is', 'get'):
        caller = next(
            f for f in result['functions']
            if f['fq_name'] == f'n::DepWrapper::{name}'
        )
        callee = next(
            f for f in result['functions']
            if f['fq_name'] == f'n::DepTarget::{name}'
            and not f['parameter_types']
        )
        recovered = [
            edge for edge in result['call_edges']
            if edge['caller'] == caller['id']
            and edge['callee'] == callee['id']
            and edge['edge_type'] == 'dependent_call'
        ]
        assert len(recovered) == 1
        assert recovered[0]['resolution']['method'] == 'libclang_overload_set'
        assert recovered[0]['resolution']['confidence'] == 'high'
        assert ctx.graph.neighbors(
            caller['id'], EDGE_DEP_CALL, direction='out'
        ) == {callee['id']}

    ambiguous_caller = next(
        f for f in result['functions'] if f['fq_name'] == 'n::DepWrapper::pick'
    )
    assert not [
        edge for edge in result['call_edges']
        if edge['caller'] == ambiguous_caller['id']
        and edge['edge_type'] == 'dependent_call'
    ]

    for name in ('parse', 'format'):
        caller = next(
            f for f in result['functions']
            if f['fq_name'] == f'n::AliasWrapper::{name}'
        )
        callee = next(
            f for f in result['functions']
            if f['fq_name'] == f'n::AliasTarget::{name}'
        )
        recovered = [
            edge for edge in result['call_edges']
            if edge['caller'] == caller['id']
            and edge['callee'] == callee['id']
            and edge['edge_type'] == 'dependent_call'
        ]
        assert len(recovered) == 1
        resolution = recovered[0]['resolution']
        assert resolution['method'] == 'receiver_type_method_family'
        assert resolution['confidence'] == 'medium'
        assert resolution['receiver'] == 'underlying'
        assert resolution['method_name'] == name
        assert resolution['family_candidate_count'] == 1
        assert ctx.graph.neighbors(
            caller['id'], EDGE_DEP_CALL, direction='out'
        ) == {callee['id']}

    canonical_caller = next(
        f for f in result['functions'] if f['fq_name'] == 'n::call_canonical'
    )
    canonical_targets = {
        f['short_name']: f
        for f in result['functions']
        if f['fq_name'] in {'n::CanonicalBox::plain', 'n::CanonicalBox::templ'}
    }
    assert set(canonical_targets) == {'plain', 'templ'}
    assert len(canonical_targets['templ']['parameter_types']) == 1
    canonical_edges = [
        edge for edge in result['call_edges']
        if edge['caller'] == canonical_caller['id']
        and edge['edge_type'] == 'call'
    ]
    assert {edge['callee'] for edge in canonical_edges} == {
        target['id'] for target in canonical_targets.values()
    }
    assert {edge['resolution']['method'] for edge in canonical_edges} == {
        'clang_specialized_template',
    }
    assert all(
        edge['resolution']['specialization_depth'] >= 1
        for edge in canonical_edges
    )
    assert ctx.graph.neighbors(
        canonical_caller['id'], EDGE_CALLS, direction='out'
    ) == {target['id'] for target in canonical_targets.values()}

    static_format = next(
        c for c in result['classes'] if c['fq_name'] == 'n::StaticFormat'
    )
    static_bits = next(
        f for f in result['functions']
        if f['fq_name'] == 'n::StaticFormat::bits' and f['def_start_line'] > 1
    )
    assert static_bits['owner_class_id'] == static_format['id']
    assert ctx.graph.neighbors(static_bits['id'], EDGE_IN_CLASS) == {
        static_format['id']
    }

    nested_caller = next(
        f for f in result['functions']
        if f['fq_name'] == 'n::call_nested_friend'
    )
    nested_targets = [
        f for f in result['functions'] if f['fq_name'] == 'n::nested_pick'
    ]
    nested_edges = [
        edge for edge in result['call_edges']
        if edge['caller'] == nested_caller['id']
        and edge['edge_type'] == 'call'
        and edge['callee_name'] == 'n::nested_pick'
    ]
    assert len(nested_targets) == 2
    assert len(nested_edges) == 1
    assert nested_edges[0]['callee'] in {
        target['id'] for target in nested_targets
    }
    assert nested_edges[0]['resolution']['method'] == (
        'clang_specialized_template'
    )
    assert nested_edges[0]['resolution']['specialization_depth'] == 2

    typed_temp_caller = next(
        f for f in result['functions'] if f['fq_name'] == 'n::call_typed_temp'
    )
    typed_temp_target = next(
        f for f in result['functions']
        if f['fq_name'] == 'n::TempReader::sax_parse'
    )
    typed_temp_edges = [
        edge for edge in result['call_edges']
        if edge['caller'] == typed_temp_caller['id']
        and edge['callee'] == typed_temp_target['id']
        and edge['edge_type'] == 'dependent_call'
    ]
    assert len(typed_temp_edges) == 1
    assert typed_temp_edges[0]['resolution']['method'] == (
        'typed_receiver_method_family'
    )
    assert typed_temp_edges[0]['resolution']['family_candidate_count'] == 1
    assert typed_temp_edges[0]['resolution']['target_required_argument_count'] == 3
    assert typed_temp_edges[0]['resolution']['target_parameter_count'] == 4
    assert ctx.graph.neighbors(
        typed_temp_caller['id'], EDGE_DEP_CALL, direction='out'
    ) == {typed_temp_target['id']}


def test_sanitize_compile_args_removes_compiler_and_header_input(tmp_path):
    source = tmp_path / 'input.h'
    source.write_text('inline int answer() { return 42; }\n')

    assert _sanitize_args(
        ['/usr/bin/clang++', '-std=c++17', '-include', 'config.h', str(source)],
        input_file=str(source),
    ) == ['-std=c++17', '-include', 'config.h']


def test_sanitize_compile_args_drops_compiler_generated_cotire_pch(tmp_path):
    cotire = tmp_path / 'cotire'
    cotire.mkdir()
    prefix = cotire / 'json_unit_CXX_prefix.hxx'
    prefix.write_text('#include "prefix.cxx"\n')
    Path(str(prefix) + '.gch').write_bytes(b'compiler-specific pch')

    assert _sanitize_args([
        '/usr/bin/c++', '-Winvalid-pch', '-include', str(prefix),
        '-std=gnu++11', 'unit.cpp',
    ], input_file='unit.cpp') == ['-std=gnu++11']


def test_task_executor_binds_cwd_without_process_chdir(tmp_path, monkeypatch):
    source = tmp_path / 'input.cpp'
    source.write_text('int answer() { return 42; }\n')
    original_cwd = os.getcwd()
    observed = {}

    monkeypatch.setattr(
        code_indexer, 'Index', SimpleNamespace(create=lambda: object())
    )

    def fake_extract(index, file_path, args, root):
        observed.update(cwd=os.getcwd(), args=args, file_path=file_path, root=root)
        return code_indexer.empty_result()

    monkeypatch.setattr(code_indexer, 'extract_translation_unit', fake_extract)
    code_indexer._task_executor((
        str(source),
        ['-Irelative', '-working-directory', '/stale'],
        str(tmp_path),
        str(tmp_path),
        None,
        None,
    ))

    assert os.getcwd() == original_cwd
    assert observed['cwd'] == original_cwd
    assert observed['args'] == [
        '-Irelative', f'-working-directory={tmp_path.resolve()}'
    ]
    assert _compile_args_with_working_directory(['-std=c++17'], None) == [
        '-std=c++17'
    ]


def test_index_build_rejects_silent_all_empty_result():
    with pytest.raises(RuntimeError, match='produced no entities'):
        _require_nonempty_index(
            indexed_file_count=4,
            function_count=0,
            class_count=0,
            macro_count=0,
        )
    with pytest.raises(RuntimeError, match='produced no entities'):
        _require_nonempty_index(
            indexed_file_count=4,
            function_count=0,
            class_count=0,
            macro_count=12,
        )


def test_edge_merge_deduplicates_compile_unit_reparsing_without_inflating_count():
    edge = {
        'caller': 'a',
        'callee': 'b',
        'edge_type': 'call',
        'count': 2,
    }
    rows, stats = _deduplicate_edges([
        {'call_edges': [edge]},
        {'call_edges': [{**edge, 'count': 1}]},
    ])

    assert rows == [edge]
    assert stats == {
        'raw_rows': 2,
        'unique_edges': 1,
        'duplicates_removed': 1,
        'by_type': {
            'call': {
                'raw_rows': 2,
                'unique_edges': 1,
                'duplicates_removed': 1,
            }
        },
    }


def test_multi_definition_usr_keeps_per_file_callers_and_targets():
    target = {
        'id': 'target', 'fq_name': 'target', 'def_file': 'target.cpp',
        'def_start_line': 1, 'def_end_line': 1, 'content_hash': 'target-hash',
    }
    results = []
    for file_name, content_hash in [('a.cpp', 'a-hash'), ('b.cpp', 'b-hash')]:
        duplicate = {
            'id': 'shared-usr', 'fq_name': 'entry', 'def_file': file_name,
            'def_start_line': 3, 'def_end_line': 5, 'content_hash': content_hash,
        }
        caller = {
            'id': f'caller-{file_name}', 'fq_name': f'caller_{file_name}',
            'def_file': file_name, 'def_start_line': 8, 'def_end_line': 9,
            'content_hash': f'caller-{content_hash}',
        }
        results.append({
            'functions': [duplicate, caller, dict(target)],
            'call_edges': [
                {'caller': 'shared-usr', 'callee': 'target', 'edge_type': 'call',
                 'caller_file': file_name, 'callee_file': 'target.cpp'},
                {'caller': caller['id'], 'callee': 'shared-usr', 'edge_type': 'call',
                 'caller_file': file_name, 'callee_file': file_name, 'callee_line': 3},
                {'caller': f'inst-{file_name}', 'callee': 'shared-usr',
                 'edge_type': 'instantiates', 'callee_file': file_name},
            ],
            'template_instantiations': [{
                'id': f'inst-{file_name}', 'template_def_id': 'shared-usr',
            }],
        })

    stats = _qualify_multi_definition_functions(results, '/repo')

    assert stats['collision_usr_count'] == 1
    assert stats['definition_variant_count'] == 2
    variant_ids = {
        result['functions'][0]['id'] for result in results
    }
    assert len(variant_ids) == 2
    for result in results:
        variant_id = result['functions'][0]['id']
        assert result['call_edges'][0]['caller'] == variant_id
        assert result['call_edges'][1]['callee'] == variant_id
        assert result['call_edges'][2]['callee'] == variant_id
        assert result['template_instantiations'][0]['template_def_id'] == variant_id
    assert stats['template_def_ids_disambiguated'] == 2
    assert stats['template_def_ids_not_disambiguated'] == 0


def test_same_usr_same_file_callers_use_definition_start_line():
    results = [{
        'functions': [
            {
                'id': 'shared-usr', 'fq_name': 'ns::convert',
                'def_file': 'convert.hpp', 'def_start_line': line,
                'def_end_line': line + 3, 'content_hash': f'hash-{line}',
            }
            for line in (97, 108)
        ] + [{
            'id': 'target', 'fq_name': 'target', 'def_file': 'target.cpp',
            'def_start_line': 1, 'def_end_line': 1, 'content_hash': 'target-hash',
        }],
        'call_edges': [
            {
                'caller': 'shared-usr', 'caller_name': 'ns::convert',
                'caller_file': 'convert.hpp', 'caller_definition_line': line,
                'callee': 'target', 'callee_name': 'target',
                'callee_file': 'target.cpp', 'edge_type': 'call',
            }
            for line in (97, 108)
        ],
        'template_instantiations': [],
    }]

    stats = _qualify_multi_definition_functions(results, '/repo')
    qualified_ids = {row['id'] for row in results[0]['functions'][:2]}
    edge_callers = {row['caller'] for row in results[0]['call_edges']}

    assert stats['collision_usr_count'] == 1
    assert stats['edge_endpoints_not_disambiguated'] == 0
    assert edge_callers == qualified_ids
    _partition_graph_ready_edges(results[0]['call_edges'], qualified_ids | {'target'})


def test_macro_namespace_definitions_at_same_source_site_stay_distinct():
    results = []
    for arch in ('fallback', 'haswell'):
        class_name = f'simdjson::{arch}::Box'
        function_name = f'{class_name}::run'
        caller_id = f'caller-{arch}'
        results.append({
            'classes': [{
                'id': 'shared-class-usr', 'fq_name': class_name,
                'def_file': 'generic.h', 'def_line': 2, 'def_end_line': 9,
                'content_hash': 'same-class-source',
            }],
            'functions': [
                {
                    'id': 'shared-function-usr', 'fq_name': function_name,
                    'def_file': 'generic.h', 'def_start_line': 5,
                    'def_end_line': 7, 'content_hash': 'same-function-source',
                    'owner_class_id': 'shared-class-usr',
                    'namespaces': ['simdjson', arch], 'classes': ['Box'],
                },
                {
                    'id': caller_id, 'fq_name': f'simdjson::{arch}::entry',
                    'def_file': f'{arch}.cpp', 'def_start_line': 1,
                    'def_end_line': 2, 'content_hash': f'caller-{arch}',
                },
            ],
            'call_edges': [
                {
                    'caller': caller_id, 'caller_name': f'simdjson::{arch}::entry',
                    'caller_file': f'{arch}.cpp', 'callee': 'shared-function-usr',
                    'callee_name': function_name, 'callee_file': 'generic.h',
                    'callee_line': 5, 'edge_type': 'call',
                },
                {
                    'caller': 'shared-function-usr', 'caller_name': function_name,
                    'caller_file': 'generic.h', 'callee': 'shared-class-usr',
                    'callee_name': class_name, 'callee_file': 'generic.h',
                    'edge_type': 'type_ref',
                },
            ],
            'template_instantiations': [],
        })

    function_stats = _qualify_multi_definition_functions(results, '/repo')
    class_stats = _qualify_multi_definition_classes(results, '/repo')

    assert function_stats['collision_usr_count'] == 1
    assert class_stats['collision_usr_count'] == 1
    assert len({result['functions'][0]['id'] for result in results}) == 2
    assert len({result['classes'][0]['id'] for result in results}) == 2
    for result in results:
        function = result['functions'][0]
        cls = result['classes'][0]
        assert result['call_edges'][0]['callee'] == function['id']
        assert result['call_edges'][1]['caller'] == function['id']
        assert result['call_edges'][1]['callee'] == cls['id']
        assert function['owner_class_id'] == cls['id']
    assert function_stats['edge_endpoints_not_disambiguated'] == 0
    assert class_stats['owner_ids_not_disambiguated'] == 0


def test_same_function_site_merges_compile_configuration_extents():
    results = [
        {
            'functions': [{
                'id': 'function-usr', 'fq_name': 'ns::configured',
                'def_file': 'configured.hpp', 'def_start_line': 3,
                'def_end_line': end, 'content_hash': content_hash,
                'code': code,
            }],
            'classes': [], 'call_edges': [], 'template_instantiations': [],
        }
        for end, content_hash, code in (
            (10, 'short-hash', 'short'), (20, 'long-hash', 'long')
        )
    ]

    stats = _qualify_multi_definition_functions(results, '/repo')

    assert stats['collision_usr_count'] == 0
    assert stats['configuration_variants_merged'] == 1
    assert {row['functions'][0]['id'] for row in results} == {'function-usr'}
    assert {row['functions'][0]['def_end_line'] for row in results} == {20}
    assert {row['functions'][0]['content_hash'] for row in results} == {'long-hash'}


def test_multi_definition_class_usr_keeps_per_file_owners():
    results = []
    for file_name, content_hash in [('split.hpp', 'split-hash'), ('single.hpp', 'single-hash')]:
        results.append({
            'classes': [{
                'id': 'shared-class-usr', 'fq_name': 'ns::Box',
                'def_file': file_name, 'def_line': 3, 'def_end_line': 20,
                'content_hash': content_hash,
            }],
            'functions': [{
                'id': f'method-{file_name}', 'fq_name': 'ns::Box::read',
                'def_file': file_name, 'def_start_line': 8, 'def_end_line': 10,
                'content_hash': f'method-{content_hash}',
                'owner_class_id': 'shared-class-usr', 'classes': ['Box'],
            }],
            'call_edges': [{
                'caller': f'inst-{file_name}', 'callee': 'shared-class-usr',
                'callee_file': file_name, 'edge_type': 'instantiates',
            }],
            'template_instantiations': [{
                'id': f'inst-{file_name}',
                'template_def_id': 'shared-class-usr',
            }],
        })

    stats = _qualify_multi_definition_classes(results, '/repo')

    assert stats['collision_usr_count'] == 1
    assert stats['definition_variant_count'] == 2
    class_ids = {result['classes'][0]['id'] for result in results}
    assert len(class_ids) == 2
    for result in results:
        cls = result['classes'][0]
        function = result['functions'][0]
        assert cls['canonical_usr'] == 'shared-class-usr'
        assert function['owner_class_id'] == cls['id']
        assert result['call_edges'][0]['callee'] == cls['id']
        assert result['template_instantiations'][0]['template_def_id'] == cls['id']
    assert stats['owner_ids_not_disambiguated'] == 0
    assert stats['edge_endpoints_not_disambiguated'] == 0


def test_multi_definition_class_owner_uses_lexical_variant_when_tu_has_both():
    result = {
        'classes': [
            {
                'id': 'shared-class-usr', 'fq_name': 'ns::Box',
                'def_file': file_name, 'def_line': 3, 'def_end_line': 20,
                'content_hash': content_hash,
            }
            for file_name, content_hash in (
                ('split.hpp', 'split-hash'), ('single.hpp', 'single-hash')
            )
        ],
        'functions': [
            {
                'id': f'method-{file_name}', 'fq_name': 'ns::Box::read',
                'def_file': file_name, 'def_start_line': 8, 'def_end_line': 10,
                'owner_class_id': 'shared-class-usr', 'classes': ['Box'],
            }
            for file_name in ('split.hpp', 'single.hpp')
        ],
        'call_edges': [],
        'template_instantiations': [],
    }

    stats = _qualify_multi_definition_classes([result], '/repo')

    classes_by_file = {row['def_file']: row['id'] for row in result['classes']}
    for function in result['functions']:
        assert function['owner_class_id'] == classes_by_file[function['def_file']]
    assert stats['owner_ids_not_disambiguated'] == 0


def test_same_class_site_merges_compile_configuration_extents():
    results = [
        {
            'classes': [{
                'id': 'class-usr', 'fq_name': 'ns::Configured',
                'def_file': 'configured.hpp', 'def_line': 3,
                'def_end_line': end, 'content_hash': content_hash,
                'code': code,
            }],
            'functions': [], 'call_edges': [], 'template_instantiations': [],
        }
        for end, content_hash, code in (
            (10, 'short-hash', 'short'), (20, 'long-hash', 'long')
        )
    ]

    stats = _qualify_multi_definition_classes(results, '/repo')

    assert stats['collision_usr_count'] == 0
    assert stats['configuration_variants_merged'] == 1
    assert {row['classes'][0]['id'] for row in results} == {'class-usr'}
    assert {row['classes'][0]['def_end_line'] for row in results} == {20}
    assert {row['classes'][0]['content_hash'] for row in results} == {'long-hash'}


def test_template_instantiation_usr_collision_keeps_source_triggers_and_edges():
    shared_id = 'template_instantiation:shared-specialization'
    results = []
    for architecture, target, line in (
        ('arm64', 'base-template', 109),
        ('icelake', 'special-template', 93),
    ):
        trigger_file = f'include/{architecture}/simd.h'
        target_file = f'include/{architecture}/base.h'
        results.append({
            'functions': [],
            'classes': [],
            'template_instantiations': [{
                'id': shared_id,
                'display_name': 'simd8',
                'fq_name': f'ns::{architecture}::simd8',
                'template_def_id': target,
                'template_def_name': f'ns::{architecture}::simd8',
                'template_def_file': target_file,
                'trigger_file': trigger_file,
                'trigger_line': line,
            }],
            'call_edges': [{
                'caller': shared_id,
                'caller_name': f'ns::{architecture}::simd8',
                'caller_file': trigger_file,
                'callee': target,
                'callee_name': f'ns::{architecture}::simd8',
                'callee_file': target_file,
                'edge_type': 'instantiates',
            }],
        })

    stats = _qualify_multi_definition_template_instantiations(results, '/repo')

    assert stats['collision_usr_count'] == 1
    assert stats['instantiation_variant_count'] == 2
    assert stats['edge_rows_not_disambiguated'] == 0
    assert stats['edge_target_disagreements_after_reconciliation'] == 0
    qualified_ids = {
        result['template_instantiations'][0]['id'] for result in results
    }
    assert len(qualified_ids) == 2
    for result in results:
        instantiation = result['template_instantiations'][0]
        edge = result['call_edges'][0]
        assert instantiation['canonical_usr'] == shared_id
        assert edge['caller'] == instantiation['id']
        assert edge['callee'] == instantiation['template_def_id']


def test_template_instantiation_compile_duplicates_keep_stable_id():
    row = {
        'id': 'template_instantiation:stable',
        'display_name': 'Box<int>',
        'template_def_id': 'box-template',
        'trigger_file': 'include/box.hpp',
        'trigger_line': 12,
    }
    results = [
        {
            'template_instantiations': [dict(row)],
            'call_edges': [{
                'caller': row['id'], 'callee': row['template_def_id'],
                'edge_type': 'instantiates',
            }],
        }
        for _ in range(2)
    ]

    stats = _qualify_multi_definition_template_instantiations(results, '/repo')

    assert stats['collision_usr_count'] == 0
    assert {result['template_instantiations'][0]['id'] for result in results} == {
        row['id']
    }
    assert stats['edge_target_disagreements_after_reconciliation'] == 0


def test_template_target_reconciles_unique_source_identity_from_edge():
    result = {
        'functions': [],
        'classes': [{
            'id': 'canonical-template', 'fq_name': 'ns::Box',
            'def_file': 'include/box.hpp', 'def_line': 4, 'def_end_line': 8,
        }],
        'template_instantiations': [{
            'id': 'template_instantiation:box-int',
            'display_name': 'Box<int>',
            'template_def_id': 'specialized-template-usr',
            'trigger_file': 'include/use.hpp',
            'trigger_line': 12,
        }],
        'call_edges': [{
            'caller': 'template_instantiation:box-int',
            'caller_name': 'Box<int>',
            'callee': 'specialized-template-usr',
            'callee_name': 'ns::Box',
            'callee_file': 'include/box.hpp',
            'callee_line': 4,
            'edge_type': 'instantiates',
        }],
    }

    stats = _reconcile_template_instantiation_targets([result], '/repo')

    row = result['template_instantiations'][0]
    edge = result['call_edges'][0]
    assert row['canonical_template_def_usr'] == 'specialized-template-usr'
    assert row['template_def_id'] == 'canonical-template'
    assert edge['callee'] == 'canonical-template'
    assert edge['resolution']['template_target_reconciliation']['method'] == (
        'unique_source_entity_identity'
    )
    assert stats['targets_reconciled'] == 1
    assert stats['edge_targets_reconciled'] == 1
    assert stats['unique_identity_misses_after_reconciliation'] == 0


def test_template_target_does_not_guess_between_source_overloads():
    result = {
        'functions': [
            {
                'id': f'candidate-{line}', 'fq_name': 'ns::make',
                'def_file': 'include/make.hpp', 'def_start_line': line,
                'def_end_line': line + 1,
            }
            for line in (4, 8)
        ],
        'classes': [],
        'template_instantiations': [{
            'id': 'template_instantiation:make-int',
            'display_name': 'make<int>',
            'template_def_id': 'specialized-template-usr',
            'template_def_file': 'include/make.hpp',
            'template_def_name': 'ns::make',
            'trigger_file': 'include/use.hpp',
            'trigger_line': 12,
        }],
        'call_edges': [],
    }

    stats = _reconcile_template_instantiation_targets([result], '/repo')

    assert result['template_instantiations'][0]['template_def_id'] == (
        'specialized-template-usr'
    )
    assert stats['targets_reconciled'] == 0
    assert stats['ambiguous_targets'] == 1
    assert stats['unique_identity_misses_after_reconciliation'] == 0


def test_cached_empty_usr_template_edges_keep_macro_definition_targets():
    name = 'ns::operator_dispatch::accessor'
    file_name = 'include/proxy.hpp'
    result = {
        'functions': [],
        'classes': [
            {'id': name, 'fq_name': name, 'def_file': file_name,
             'def_line': line, 'def_end_line': line}
            for line in (10, 20)
        ],
        'template_instantiations': [
            {'id': f'instantiation:{line}', 'fq_name': name,
             'template_def_id': name, 'template_def_name': name,
             'template_def_file': file_name, 'template_def_line': line,
             'template_def_column': 1, 'trigger_file': file_name,
             'trigger_line': line}
            for line in (10, 20)
        ],
        'call_edges': [
            {'caller': f'instantiation:{line}', 'caller_name': name,
             'callee': f'/repo/{file_name}:{line}:1', 'callee_name': name,
             'callee_file': file_name, 'callee_line': line, 'callee_column': 1,
             'caller_file': file_name, 'edge_type': 'instantiates'}
            for line in (10, 20)
        ],
    }

    _qualify_multi_definition_classes([result], '/repo')
    _reconcile_template_instantiation_targets([result], '/repo')
    stats = _qualify_multi_definition_template_instantiations([result], '/repo')

    assert stats['edge_target_disagreements_after_reconciliation'] == 0
    definitions = {row['def_line']: row['id'] for row in result['classes']}
    assert len(set(definitions.values())) == 2
    for row, edge in zip(result['template_instantiations'], result['call_edges']):
        assert row['template_def_id'] == definitions[row['template_def_line']]
        assert edge['callee'] == row['template_def_id']


def test_cached_template_edge_does_not_guess_between_local_records():
    result = {
        'functions': [],
        'classes': [
            {'id': target, 'fq_name': 'ns::Box', 'def_file': 'box.hpp',
             'def_line': 4, 'def_end_line': 4}
            for target in ('target-a', 'target-b')
        ],
        'template_instantiations': [
            {'id': 'shared-instantiation', 'template_def_id': target,
             'template_def_file': 'box.hpp', 'template_def_name': 'ns::Box',
             'template_def_line': 4, 'template_def_column': 1}
            for target in ('target-a', 'target-b')
        ],
        'call_edges': [{
            'caller': 'shared-instantiation', 'callee': '/repo/box.hpp:4:1',
            'callee_name': 'ns::Box', 'callee_file': 'box.hpp',
            'callee_line': 4, 'callee_column': 1, 'edge_type': 'instantiates',
        }],
    }

    _reconcile_template_instantiation_targets([result], '/repo')

    assert result['call_edges'][0]['callee'] == '/repo/box.hpp:4:1'


def test_dependency_endpoints_reconcile_only_unique_source_identity():
    results = [{
        'functions': [
            {
                'id': 'caller', 'fq_name': 'ns::caller', 'def_file': 'a.hpp',
                'def_start_line': 1, 'def_end_line': 2,
            },
            {
                'id': 'primary-template', 'fq_name': 'ns::convert',
                'def_file': 'a.hpp', 'def_start_line': 5, 'def_end_line': 8,
            },
            {
                'id': 'overload-a', 'fq_name': 'ns::overloaded',
                'def_file': 'a.hpp', 'def_start_line': 10, 'def_end_line': 11,
            },
            {
                'id': 'overload-b', 'fq_name': 'ns::overloaded',
                'def_file': 'a.hpp', 'def_start_line': 13, 'def_end_line': 14,
            },
        ],
        'classes': [],
        'call_edges': [
            {
                'caller': 'specialized-caller', 'caller_name': 'ns::caller',
                'caller_file': 'a.hpp', 'callee': 'specialized-template',
                'callee_name': 'ns::convert', 'callee_file': 'a.hpp',
                'edge_type': 'template_ref',
            },
            {
                'caller': 'caller', 'caller_name': 'ns::caller',
                'caller_file': 'a.hpp', 'callee': 'ambiguous-specialization',
                'callee_name': 'ns::overloaded', 'callee_file': 'a.hpp',
                'edge_type': 'decl_ref',
            },
            {
                'caller': 'caller', 'caller_name': 'ns::caller',
                'caller_file': 'a.hpp', 'callee': 'primary-template',
                'callee_name': 'ns::convert', 'callee_file': 'a.hpp',
                'edge_type': 'type_ref',
            },
        ],
    }]

    stats = _reconcile_dependency_endpoints(results, '/repo')

    first, ambiguous, valid = results[0]['call_edges']
    assert first['caller'] == 'caller'
    assert first['callee'] == 'primary-template'
    assert set(first['resolution']['dependency_endpoint_reconciliation']) == {
        'caller', 'callee',
    }
    assert ambiguous['callee'] == 'ambiguous-specialization'
    assert valid['callee'] == 'primary-template'
    assert stats['endpoints_reconciled'] == 2
    assert stats['unique_identity_misses_after_reconciliation'] == 0
    assert stats['ambiguous_endpoints'] == 1


def test_cached_anonymous_owner_is_cleared_without_hiding_named_owner_gap():
    results = [{
        'classes': [{'id': 'indexed-class'}],
        'functions': [
            {'id': 'anonymous-method', 'owner_class_id': 'anonymous-record', 'classes': []},
            {'id': 'indexed-method', 'owner_class_id': 'indexed-class', 'classes': ['Indexed']},
            {'id': 'missing-named-method', 'owner_class_id': 'missing-class', 'classes': ['Missing']},
        ],
    }]

    assert _clear_unindexed_anonymous_owner_ids(results) == 1
    functions = {row['id']: row for row in results[0]['functions']}
    assert functions['anonymous-method']['owner_class_id'] == ''
    assert functions['indexed-method']['owner_class_id'] == 'indexed-class'
    assert functions['missing-named-method']['owner_class_id'] == 'missing-class'


def test_executable_endpoint_partition_is_transparent_and_strict():
    complete = {'caller': 'a', 'callee': 'b', 'edge_type': 'call'}
    implicit = {
        'caller': 'a', 'callee': 'implicit', 'edge_type': 'call',
        'callee_has_body': False,
    }
    retained, unresolved, stats = _partition_graph_ready_edges(
        [complete, implicit], {'a', 'b'}
    )
    assert retained == [complete]
    assert unresolved[0]['unresolved_reason'] == (
        'missing_declaration_or_implicit_callee'
    )
    assert stats['executable_retained'] == 1
    assert stats['executable_unresolved'] == 1

    with pytest.raises(RuntimeError, match='missing_source_definition_callees=1'):
        _partition_graph_ready_edges([
            {
                'caller': 'a', 'callee': 'lost-body', 'edge_type': 'call',
                'callee_file': 'lost.hpp', 'callee_name': 'Box::lost',
                'callee_line': 12,
            }
        ], {'a'}, {('lost.hpp', 'Box::lost', 10, 14)})

    declaration_only = {
        'caller': 'a', 'callee': 'macro-specific-default-constructor',
        'callee_name': 'Result::Result', 'callee_file': 'result.hpp',
        'callee_line': 12, 'edge_type': 'call',
        'callee_is_definition': False, 'callee_has_body': False,
    }
    retained, unresolved, stats = _partition_graph_ready_edges(
        [declaration_only],
        {'a', 'variant-a', 'variant-b'},
        {('result.hpp', 'Result::Result', 12, 12)},
        {
            ('result.hpp', 'Result::Result', 12, 12, 'variant-a'),
            ('result.hpp', 'Result::Result', 12, 12, 'variant-b'),
        },
    )
    assert not retained
    assert unresolved[0]['unresolved_reason'] == (
        'missing_declaration_or_implicit_callee'
    )
    assert stats['by_reason'] == {'missing_declaration_or_implicit_callee': 1}


def test_executable_endpoint_partition_reconciles_unique_source_identity():
    edge = {
        'caller': 'caller',
        'callee': 'unstable-template-usr',
        'callee_name': 'Box::convert',
        'callee_file': 'box.hpp',
        'callee_line': 22,
        'edge_type': 'dependent_call',
        'resolution': {'method': 'libclang_overload_set'},
    }
    retained, unresolved, stats = _partition_graph_ready_edges(
        [edge],
        {'caller', 'canonical-template-usr'},
        {('box.hpp', 'Box::convert', 20, 24)},
        {('box.hpp', 'Box::convert', 20, 24, 'canonical-template-usr')},
    )

    assert not unresolved
    assert retained[0]['callee'] == 'canonical-template-usr'
    assert retained[0]['resolution']['endpoint_reconciliation'] == {
        'method': 'unique_source_definition_identity',
        'observed_target_usr': 'unstable-template-usr',
        'canonical_target_usr': 'canonical-template-usr',
        'target_file': 'box.hpp',
        'target_line': 22,
    }
    assert stats['source_definition_endpoints_reconciled'] == 1


def test_lexical_preprocessor_and_file_only_graph_nodes(tmp_path):
    source = tmp_path / 'macro_unscope.hpp'
    source.write_text('''
#ifndef PROJECT_MACRO_UNSCOPE_HPP
#define PROJECT_MACRO_UNSCOPE_HPP
#include "active.hpp"
#if 0
#define INACTIVE_VALUE 1
#endif
#undef ACTIVE_VALUE
#endif
''')
    directives = _extract_preprocessor_directives(str(source), str(tmp_path))
    assert {(row['directive'], row['name']) for row in directives} >= {
        ('define', 'INACTIVE_VALUE'), ('undef', 'ACTIVE_VALUE'),
        ('include', '"active.hpp"'),
    }

    entities = tmp_path / 'entities'
    entities.mkdir()
    (entities / 'files.jsonl').write_text(json.dumps({
        'file': 'macro_unscope.hpp', 'is_test': False, 'is_third_party': False,
    }) + '\n')
    (tmp_path / 'preprocessor_directives.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in directives)
    )
    (tmp_path / 'edge_endpoint_stats.json').write_text('{}')

    ctx = build_graph(tmp_path)
    file_id = 'file:macro_unscope.hpp'
    assert ctx.graph.nodes[file_id]['type'] == NODE_FILE
    undef = next(
        row for row in directives
        if row['directive'] == 'undef' and row['name'] == 'ACTIVE_VALUE'
    )
    assert ctx.graph.nodes[undef['id']]['type'] == NODE_PREPROCESSOR
    assert ctx.graph.neighbors(undef['id'], EDGE_IN_FILE) == {file_id}


@pytest.mark.parametrize('owner,expected', [('a', {'a'}), ('missing', set()), ('', set()), (None, set())])
def test_owner_ids_override_ambiguous_class_names_and_intent_metadata(tmp_path, owner, expected):
    classes = [dict(id=i, fq_name='Box', def_file='x.h', def_line=1, def_end_line=20) for i in ('a', 'b')]
    func = dict(id='f', classes=['Box'], is_method=True, def_file='x.h', def_start_line=5, def_end_line=8)
    if owner is not None:
        func['owner_class_id'] = owner
    for name, rows in [('classes', classes), ('functions', [func]),
                       ('function_intents', [dict(func_id='f', owner_class_id='b')])]:
        (tmp_path / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert build_graph(tmp_path).graph.neighbors('f', EDGE_IN_CLASS) == expected


def test_legacy_owner_requires_unique_name_and_contained_method(tmp_path):
    (tmp_path / 'classes.jsonl').write_text(json.dumps(dict(
        id='a', fq_name='Box', def_file='x.h', def_line=1, def_end_line=20)) + '\n')
    funcs = [dict(id='inside', classes=['Box'], is_method=True, def_file='x.h', def_start_line=5, def_end_line=8),
             dict(id='outside', classes=['Box'], is_method=True, def_file='x.h', def_start_line=25, def_end_line=28)]
    (tmp_path / 'functions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in funcs))
    graph = build_graph(tmp_path).graph
    assert graph.neighbors('inside', EDGE_IN_CLASS) == {'a'}
    assert not graph.neighbors('outside', EDGE_IN_CLASS)
