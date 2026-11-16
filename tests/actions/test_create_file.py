import pytest

from INTENT2FIX.actions.create_file import CreateFile, CreateFileArgs
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.file import FileRepository
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch


@pytest.fixture
def repository():
    repo = InMemRepository()
    return repo

@pytest.fixture
def file_context(repository):
    return FileContext(repo=repository)

def test_create_file_basic(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        artifact_kind="production",
        path="new_file.py",
        file_text='''def greet(name: str) -> str:
    return f"Hello, {name}!"
''',
        scratch_pad="Creating a new greeting function"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["success"]
    content = file_context.get_file("new_file.py").content
    assert "def greet" in content
    assert "diff" in observation.properties

def test_create_file_already_exists(repository, file_context):
    # First create a file
    repository.save_file("existing.py", "# existing content")
    
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        artifact_kind="production",
        path="existing.py",
        file_text="# new content",
        scratch_pad="Trying to create an existing file"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["fail_reason"] == "file_exists"
    assert observation.properties["success"] is False

def test_create_file_with_path(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        artifact_kind="production",
        path="utils/helpers/string_utils.py",
        file_text='''def capitalize_words(text: str) -> str:
    return " ".join(word.capitalize() for word in text.split())
''',
        scratch_pad="Creating utility function in new directory"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["success"]
    content = file_context.get_file("utils/helpers/string_utils.py").content
    assert "def capitalize_words" in content
    assert "diff" in observation.properties

def test_create_file_normalize_path(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        artifact_kind="production",
        path="/repo/test/normalize.py",  # Path starts with /repo/
        file_text="# normalized path test",
        scratch_pad="Testing path normalization"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["success"]
    content = file_context.get_file("test/normalize.py").content
    assert content == "# normalized path test\n"
    assert "diff" in observation.properties


def test_create_file_does_not_treat_repository_as_repo_prefix(repository, file_context):
    action = CreateFile(repository=repository)

    observation = action.execute(
        CreateFileArgs(
            artifact_kind="production",
            path="/repository/normalize.py",
            file_text="# keep complete first path component",
            scratch_pad="Testing exact repo prefix normalization",
        ),
        file_context,
    )

    assert observation.properties["success"]
    assert file_context.file_exists("repository/normalize.py")
    assert not file_context.file_exists("sitory/normalize.py")


@pytest.mark.parametrize("path", ["", "/repo", "../outside.py", "src/../../outside.py"])
def test_create_file_rejects_invalid_paths(repository, file_context, path):
    action = CreateFile(repository=repository)

    observation = action.execute(
        CreateFileArgs(
            artifact_kind="production",
            path=path,
            file_text="# invalid path",
            scratch_pad="Testing repository boundary",
        ),
        file_context,
    )

    assert observation.expect_correction is True
    assert observation.properties == {
        "fail_reason": "invalid_path",
        "success": False,
    }


def test_create_file_rejects_symlink_that_resolves_outside_repository(tmp_path):
    repo_root = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo_root.mkdir()
    outside.mkdir()
    (repo_root / "external").symlink_to(outside, target_is_directory=True)
    repository = FileRepository(repo_path=str(repo_root))
    file_context = FileContext(repo=repository)

    observation = CreateFile(repository=repository).execute(
        CreateFileArgs(
            artifact_kind="production",
            path="external/escaped.cpp",
            file_text="int escaped;\n",
            scratch_pad="Testing resolved repository containment",
        ),
        file_context,
    )

    assert observation.properties["fail_reason"] == "invalid_path"
    assert observation.expect_correction is True
    assert not (outside / "escaped.cpp").exists()

def test_create_file_empty_content(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        artifact_kind="production",
        path="empty.py",
        file_text="",
        scratch_pad="Creating empty file"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["success"]
    content = file_context.get_file("empty.py").content
    assert content == ""
    assert "diff" in observation.properties


def test_stale_portfolio_diagnostic_metadata_does_not_block_probe_creation(
    repository, file_context
):
    file_context.verification["portfolio_diagnostic_policy"] = {
        "anchor_command": "python full_repro.py",
        "required_action": "production_revision",
        "production_change_seq": file_context.change_seq,
    }

    observation = CreateFile(repository=repository).execute(
        CreateFileArgs(
            path="another_probe.py",
            file_text="print('probe')\n",
            artifact_kind="temporary_probe",
            scratch_pad="Trying another reduced probe",
        ),
        file_context,
    )

    assert observation.properties["success"] is True
    assert file_context.file_exists("another_probe.py")

def test_create_file_with_indentation(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        artifact_kind="production",
        path="indented.py",
        file_text='''class MyClass:
    def __init__(self):
        self.value = 0
        
    def increment(self):
        self.value += 1
        return self.value''',
        scratch_pad="Creating class with proper indentation"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["success"]
    content = file_context.get_file("indented.py").content
    assert "class MyClass:" in content
    assert "    def __init__(self):" in content
    assert "        self.value = 0" in content
    assert "diff" in observation.properties


def test_create_file_registers_temporary_probe(repository, file_context):
    action = CreateFile(repository=repository)
    observation = action.execute(
        CreateFileArgs(
            path="compile_probe.cpp",
            file_text="int main() {}",
            artifact_kind="temporary_probe",
            scratch_pad="Compile a diagnostic probe",
        ),
        file_context,
    )

    assert observation.properties["success"]
    assert file_context.verification["ephemeral_paths"] == [
        "compile_probe.cpp"
    ]
    assert file_context.change_seq == 0
    assert file_context.verification["probe_change_seq"] == 1


def test_create_file_requires_explicit_classification(repository, file_context):
    observation = CreateFile(repository=repository).execute(
        CreateFileArgs(
            path="one_off_check.cpp",
            file_text="int main() {}",
            scratch_pad="Create a one-off check without an explicit kind",
        ),
        file_context,
    )

    assert observation.expect_correction is True
    assert observation.properties["fail_reason"] == "artifact_classification_required"
    assert not file_context.file_exists("one_off_check.cpp")
    assert file_context.change_seq == 0
    assert "artifact_kind" in CreateFileArgs.model_json_schema()["required"]
    assert CreateFileArgs.model_json_schema()["properties"]["artifact_kind"]["enum"] == [
        "production", "temporary_probe"
    ]


@pytest.mark.parametrize("path", ["include/simdjson.h", "src/catch2/internal/catch_test_case_hashing.hpp"])
def test_production_directory_file_cannot_be_silently_marked_temporary(repository, file_context, path):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(path=path, file_text="int value;\n", artifact_kind="temporary_probe")
    refused = action.execute(args, file_context)
    assert refused.expect_correction
    assert not file_context.file_exists(path)
    accepted = action.execute(args.model_copy(update={"artifact_kind": "production"}), file_context)
    assert accepted.properties["success"]
    assert path in sanitize_candidate_patch(file_context).kept_paths


def test_legacy_misclassified_file_can_be_promoted_without_recreating_it(repository, file_context):
    path = "include/helper.hpp"
    file_context.add_file(path).apply_changes("int value;\n")
    file_context.verification["ephemeral_paths"] = [path]
    file_context.verification["production_regression"] = {"change_seq": 0, "passed": True}
    before = sanitize_candidate_patch(file_context)
    assert before.artifact_errors
    assert not before.patch.strip()
    result = CreateFile(repository=repository).execute(
        CreateFileArgs(path=path, file_text="int value;\n", artifact_kind="production"), file_context
    )
    after = sanitize_candidate_patch(file_context)
    assert result.properties["reclassified"]
    assert after.patch == before.raw_patch
    assert not after.artifact_errors
    assert file_context.change_seq == 1
    assert file_context.verification["production_regression"]["change_seq"] != file_context.change_seq


def test_create_file_replaces_registered_temporary_probe(repository, file_context):
    action = CreateFile(repository=repository)
    first = action.execute(
        CreateFileArgs(
            path="probe.cpp",
            file_text=(
                "// INTENT2FIX_DISCRIMINATOR_H0_PASS\n"
                "int main() { return 1; }\n"
            ),
            artifact_kind="temporary_probe",
            scratch_pad="Create a focused probe",
        ),
        file_context,
    )
    second = action.execute(
        CreateFileArgs(
            path="probe.cpp",
            file_text=(
                "// INTENT2FIX_DISCRIMINATOR_H0_PASS\n"
                "int main() { return 0; }\n"
            ),
            artifact_kind="temporary_probe",
            scratch_pad="Replace the focused probe",
        ),
        file_context,
    )

    assert first.properties["success"] is True
    assert second.properties["success"] is True
    assert "replaced" in second.message.lower()
    assert file_context.get_file("probe.cpp").content == (
        "// INTENT2FIX_DISCRIMINATOR_H0_PASS\n"
        "int main() { return 0; }\n"
    )
    assert file_context.verification["probe_change_seq"] == 2
    assert file_context.change_seq == 0


def test_create_file_does_not_replace_new_production_file(repository, file_context):
    action = CreateFile(repository=repository)
    first = action.execute(
        CreateFileArgs(
            path="src/deliverable.cpp",
            file_text="int original;\n",
            artifact_kind="production",
            scratch_pad="Create an Issue-required deliverable",
        ),
        file_context,
    )
    second = action.execute(
        CreateFileArgs(
            path="src/deliverable.cpp",
            file_text="int replacement;\n",
            artifact_kind="temporary_probe",
            scratch_pad="Try to replace a production file",
        ),
        file_context,
    )

    assert first.properties["success"] is True
    assert second.properties == {
        "fail_reason": "file_exists",
        "success": False,
    }
    assert file_context.get_file("src/deliverable.cpp").content == (
        "int original;\n"
    )


def test_create_file_reuses_one_portfolio_discriminator(repository, file_context):
    action = CreateFile(repository=repository)
    first = action.execute(
        CreateFileArgs(
            path="branch_probe.cpp",
            file_text=(
                'int main() { /* INTENT2FIX_DISCRIMINATOR_H0_PASS */ '
                "return 0; }"
            ),
            artifact_kind="temporary_probe",
            scratch_pad="Create the branch discriminator",
        ),
        file_context,
    )
    second = action.execute(
        CreateFileArgs(
            path="branch_probe2.cpp",
            file_text=(
                'int main() { /* INTENT2FIX_DISCRIMINATOR_H0_PASS */ '
                "return 0; }"
            ),
            artifact_kind="temporary_probe",
            scratch_pad="Try a replacement branch discriminator",
        ),
        file_context,
    )

    assert first.properties["success"] is True
    assert second.properties == {
        "fail_reason": "duplicate_portfolio_discriminator",
        "success": False,
    }
    assert second.expect_correction is True
    assert "edit and rerun branch_probe.cpp" in second.message
    assert not file_context.file_exists("branch_probe2.cpp")


def test_create_file_marks_production_change(repository, file_context):
    action = CreateFile(repository=repository)
    observation = action.execute(
        CreateFileArgs(
            path="src/new_feature.cpp",
            file_text="int feature() { return 1; }",
            artifact_kind="production",
            scratch_pad="Create a production source file",
        ),
        file_context,
    )

    assert observation.properties["success"]
    assert file_context.change_seq == 1
    assert "probe_change_seq" not in file_context.verification
