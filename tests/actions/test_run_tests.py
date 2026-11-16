from unittest.mock import Mock

from INTENT2FIX.actions.run_tests import RunTests, RunTestsArgs


def test_run_tests_reports_invalid_inputs_when_valid_test_remains():
    file_context = Mock()
    file_context.file_exists.side_effect = lambda path: path != "missing.cpp"
    file_context.is_directory.side_effect = lambda path: path == "tests"
    result = Mock(file_path="tests/valid.cpp")
    file_context.run_tests.return_value = [result]
    file_context.get_test_failure_details.return_value = ""
    file_context.get_test_summary.return_value = "1 passed"

    observation = RunTests().execute(
        RunTestsArgs(
            thoughts="run the valid test and report skipped inputs",
            test_files=["tests/valid.cpp", "missing.cpp", "tests"],
        ),
        file_context,
    )

    file_context.run_tests.assert_called_once_with(["tests/valid.cpp"])
    assert "Skipped invalid test inputs" in observation.message
    assert observation.properties == {
        "skipped_files": ["missing.cpp"],
        "skipped_directories": ["tests"],
    }
