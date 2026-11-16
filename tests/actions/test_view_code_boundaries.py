from INTENT2FIX.actions.create_file import CreateFile, CreateFileArgs
from INTENT2FIX.actions.view_code import CodeSpan, ViewCode, ViewCodeArgs
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import InMemRepository


def _execute(files, repository):
    return ViewCode(repository=repository).execute(
        ViewCodeArgs(thoughts="", files=files),
        FileContext(repo=repository),
    )


def test_view_code_rejects_unscoped_large_file():
    repository = InMemRepository({"tests/abi/large.hpp": "\n".join(["line"] * 1000)})

    observation = _execute([CodeSpan(file_path="tests/abi/large.hpp")], repository)

    assert observation.properties["fail_reason"] == "whole_file_too_large"
    assert observation.expect_correction is True


def test_view_code_converts_search_span_to_bounded_line_range():
    repository = InMemRepository({
        "tests/abi/large.hpp": "\n".join(f"line {line}" for line in range(1, 1001))
    })

    observation = _execute(
        [CodeSpan(file_path="tests/abi/large.hpp", span_ids=["tests/abi/large.hpp:20-25"])],
        repository,
    )

    assert "line 20" in observation.message
    assert "line 25" in observation.message
    assert observation.properties.get("fail_reason") is None


def test_view_code_structured_limit_depends_on_size_not_path():
    repository = InMemRepository({
        "src/large.hpp": "\n".join(f"line {line}" for line in range(1, 1001)),
        "single_include/small.hpp": "struct Widget {};\n",
    })

    large_observation = _execute(
        [CodeSpan(file_path="src/large.hpp", span_ids=["Widget"])],
        repository,
    )
    small_observation = _execute(
        [CodeSpan(file_path="single_include/small.hpp", span_ids=["Widget"])],
        repository,
    )

    assert large_observation.properties["fail_reason"] == "structured_view_too_large"
    assert small_observation.properties.get("fail_reason") != "structured_view_too_large"


def test_view_code_reads_new_temporary_probe_from_current_context():
    repository = InMemRepository()
    file_context = FileContext(repo=repository)
    CreateFile(repository=repository).execute(
        CreateFileArgs(
            path="probe.cpp",
            file_text="int current_probe = 42;\n",
            artifact_kind="temporary_probe",
            scratch_pad="Create a focused probe",
        ),
        file_context,
    )

    observation = ViewCode(repository=repository).execute(
        ViewCodeArgs(thoughts="", files=[CodeSpan(file_path="probe.cpp")]),
        file_context,
    )

    assert observation.properties.get("fail_reason") is None
    assert "int current_probe = 42;" in observation.message


def test_view_code_limits_files_per_call():
    repository = InMemRepository({f"src/{index}.cpp": "int x;" for index in range(11)})

    observation = _execute(
        [CodeSpan(file_path=f"src/{index}.cpp", start_line=1, end_line=1) for index in range(11)],
        repository,
    )

    assert observation.properties["fail_reason"] == "too_many_files"


def test_view_code_keeps_multiple_explicit_line_ranges_in_context():
    repository = InMemRepository(
        {
            "src/first.cpp": "\n".join(f"first {line}" for line in range(1, 21)),
            "src/second.cpp": "\n".join(f"second {line}" for line in range(1, 21)),
        }
    )
    file_context = FileContext(repo=repository)

    observation = ViewCode(repository=repository).execute(
        ViewCodeArgs(
            thoughts="Compare both implementation sites",
            files=[
                CodeSpan(file_path="src/first.cpp", start_line=3, end_line=7),
                CodeSpan(file_path="src/second.cpp", start_line=11, end_line=15),
            ],
        ),
        file_context,
    )

    assert observation.properties.get("fail_reason") is None
    assert file_context.has_file("src/first.cpp")
    assert file_context.has_file("src/second.cpp")
    assert file_context.get_context_file("src/first.cpp").spans[0].start_line == 3
    assert file_context.get_context_file("src/second.cpp").spans[0].start_line == 11
