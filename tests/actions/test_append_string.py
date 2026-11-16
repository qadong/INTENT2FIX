from INTENT2FIX.actions.append_string import AppendString, AppendStringArgs
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import InMemRepository


def test_append_string_rejects_immutable_reproduction_artifact():
    repository = InMemRepository()
    path = "tests/repro_bug.cpp"
    repository.save_file(path, "int main() {}\n")
    file_context = FileContext(repo=repository)
    file_context.add_file(path, show_all_spans=True)
    file_context.verification["immutable_reproduction_paths"] = [path]

    observation = AppendString(repository=repository).execute(
        AppendStringArgs(path=path, new_str="int unsupported;\n"),
        file_context,
    )

    assert observation.properties["fail_reason"] == "immutable_reproduction_artifact"
    assert observation.expect_correction is True
    assert file_context.get_file(path).content == "int main() {}\n"


def test_append_string_rejects_parent_path_even_when_repository_contains_it():
    path = "../outside.cpp"
    repository = InMemRepository({path: "int value;\n"})
    file_context = FileContext(repo=repository)
    file_context.add_file(path, show_all_spans=True)

    observation = AppendString(repository=repository).execute(
        AppendStringArgs(path=path, new_str="int other;\n"),
        file_context,
    )

    assert observation.properties["fail_reason"] == "invalid_path"
    assert observation.expect_correction is True
    assert file_context.get_file(path).content == "int value;\n"


def test_append_string_rejects_existing_fixture_in_portfolio_mode():
    path = "fixtures/expected_output.txt"
    repository = InMemRepository({path: "int main() {}\n"})
    file_context = FileContext(repo=repository)
    file_context.add_file(path, show_all_spans=True)
    file_context.verification["finish_policy"] = "portfolio_soft_evidence"

    observation = AppendString(repository=repository).execute(
        AppendStringArgs(path=path, new_str="int changed_expectation;\n"),
        file_context,
    )

    assert observation.properties["fail_reason"] == "portfolio_repository_test_edit"
    assert observation.expect_correction is True
    assert file_context.get_file(path).content == "int main() {}\n"
