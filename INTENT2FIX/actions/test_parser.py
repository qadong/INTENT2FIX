import re
from enum import Enum
from typing import Dict, List, Tuple, Pattern
from pydantic import BaseModel, Field
from textwrap import dedent # 导入 dedent
from INTENT2FIX.actions.output_sanitizer import OUTPUT_CAPTURE_TRUNCATION_MARKER

class LogType(str, Enum):
    TEST_RESULTS = "test_results"
    COMPILATION_ERROR = "compilation_error"
    NO_TESTS = "no_tests"
    MEMORY_LEAK = "memory_leak"
    OUTPUT_TRUNCATED = "output_truncated"
    UNKNOWN = "unknown"

class TestStatus(str, Enum):
    PASSED = 'passed'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    NONE = 'none'

ALL_PATTERNS: Dict[TestStatus, Tuple[Pattern[str], ...]] = {
    TestStatus.PASSED: (
        re.compile(r"Test\s+#\d+:\s+(.+?)\s*\.{2,}.*?Passed"),
        re.compile(r"^\d+/\d+\s+Test\s+\#\d+:\s+(.+)\s+\.+\s+Passed"),
        re.compile(r"^\[\s*OK\s*\]\s+([a-zA-Z0-9_\/.:]+)"),
        re.compile(r"^\[\s+OK\s+\]\s+(\S+?)(?:\s+\(.+\))?$"),
        re.compile(r"^\x1b\[0;32m\[\s+OK\s+\]\x1b\[0m\s+(\S+?)(?:\s+\(.+\))?$"),
        re.compile(r"^(?:\[doctest\]\s+)?PASSED!\s+test case:\s*\"([^\"]+)\""),
        re.compile(r"^.+?:\d+:\s+passed:\s+(.+)$"),
        re.compile(r"^PASS\s*:\s*([\w:]+)"),
        re.compile(r"^\s*--- PASS:\s+(\S+)(?:\s+\(.+\))?"),
        re.compile(r"^\s*[✔✓]\s+(.*?)(?:\s*\d+ms)?$"),
        re.compile(r"^test\s+(\S+)\s+\.\.\.\s+ok$"),
        re.compile(r"^(\S+)\s+\.\.\.\s+ok$"),
        re.compile(r"^(.+?:\d+):\s+PASSED:"),
        re.compile(r"^\[\s+PASSED\s+\]\s+(\S+?)(?:\s+\(.+\))?$"),
    ),
    TestStatus.FAILED: (
        re.compile(r"Test\s+#\d+:\s+(.+?)\s*\.{2,}.*?(?:Failed|Timeout)"),
        re.compile(r"^\d+/\d+\s+Test\s+\#\d+:\s+(.+)\s+\.+\s+\*+Failed", re.IGNORECASE),
        re.compile(r"^\d+/\d+\s*Test\s*\#\d+:\s*(.*?)\s*\.+\s*\*+Exception:"),
        re.compile(r"^\[\s*FAILED\s*\]\s+([a-zA-Z0-9_\/.:]+)"),
        re.compile(r"^\[\s+FAILED\s+\]\s+(\S+?)(?:\s+\(.+\))?$"),
        re.compile(r"^\x1b\[0;31m\[\s+FAILED\s+\]\x1b\[0m\s+(\S+?)(?:\s+\(.+\))?$"),
        re.compile(r"^(?:\[doctest\]\s+)?FAILED!\s+test case:\s*\"([^\"]+)\""),
        re.compile(r"^.+?:\d+:\s+failed:\s+(.+)$"),
        re.compile(r"^FAIL!\s*:\s*([\w:]+)"),
        re.compile(r'error in "([^\"]+)"'),
        re.compile(r"^--- FAIL: (\S+)"),
        re.compile(r"^\s*[✘×𐄂]\s+(.*?)(?:\s*\d+ms)?$"),
        re.compile(r"^test\s+(\S+)\s+\.\.\.\s+FAILED$"),
        re.compile(r"^(\S+)\s+\.\.\.\s+FAILED$"),
        re.compile(r"^(.+?:\d+):\s+FAILED:"),
        re.compile(r"^\[\s+FAILED\s+\]\s+(\S+?)(?:\s+\(.+\))?$"),
    ),
    TestStatus.SKIPPED: (
        re.compile(r"Test\s+#\d+:\s+(.+?)\s*\.{2,}.*?Not Run"),
        re.compile(r"^(?:SKIP|XFAIL)\s*:\s*([\w:]+)"),
        re.compile(r"^\s*-\s+(.*)$"),
        re.compile(r"^\[\s+SKIPPED\s+\]\s+(\S+?)(?:\s+\(.+\))?$"),
    )
}

# --- 关键修复 1: NO_TESTS 已被移除 ---
LOG_TYPE_PATTERNS: Dict[LogType, Pattern[str]] = {
    LogType.COMPILATION_ERROR: re.compile(
        # IMPORTANT: avoid matching runtime outputs like "Error: [json.exception...]".
        # Only treat as compilation/build error if it resembles compiler/linker diagnostics
        # or build-system markers.
        r"(?im)(^|\n)(?:"
        r".+?:\d+:\d+:\s+(?:fatal\s+)?error:\s|"      # file:line:col: error:
        r".+?:\d+:\s+(?:fatal\s+)?error:\s|"           # file:line: error:
        r"compilation terminated\b|"
        r"[^\n]*linker command failed\b|"
        r"[^\n]*undefined reference to\b|"
        r"collect2:\s+error:\s|"
        r"ld:\s+.*error:\s"
        r")"
    ),
    LogType.MEMORY_LEAK: re.compile(
        r"(LeakSanitizer: detected memory leaks|AddressSanitizer:.*leaked)",
        re.IGNORECASE
    )
}

# --- 关键修复 2: NO_TESTS 模式现在是独立的 ---
NO_TESTS_PATTERN: Pattern[str] = re.compile(
    r"(No tests were found|No tests ran|0 tests ran|No test configuration file found!)",
    re.IGNORECASE
)

_DOCTEST_RUNTIME_ERROR = re.compile(
    r"^.+?:\d+:\s+ERROR:\s+(?:test case (?:THREW|CRASHED)\b|(?:CHECK|REQUIRE)(?:_\w+)?\s*\()"
)


class TestResult(BaseModel):
    name: str
    status: TestStatus
    details: str = ""

class TestSummary(BaseModel):
    log_type: LogType = LogType.UNKNOWN
    results: List[TestResult] = Field(default_factory=list)
    summary_line: str = ""
    raw_output: str

def _is_test_result_line(line: str) -> bool:
    """Check if a line indicates the start of any test result."""
    for patterns in ALL_PATTERNS.values():
        for pattern in patterns:
            if re.search(pattern, line):
                return True
    return False

# --- 关键修复 3: 完整的、修复了循环逻辑的函数 ---
def parse_test_output(output: str) -> TestSummary:
    if OUTPUT_CAPTURE_TRUNCATION_MARKER in output:
        return TestSummary(
            log_type=LogType.OUTPUT_TRUNCATED,
            summary_line="Command output capture was truncated; verification is incomplete.",
            raw_output=output,
        )
    lines = output.splitlines()
    # Doctest reports runtime assertions/exceptions as file:line: ERROR, which
    # resembles a compiler diagnostic. Preserve them as test failure details.
    diagnostic_lines = ["" if _DOCTEST_RUNTIME_ERROR.search(line) else line for line in lines]
    diagnostic_output = "\n".join(diagnostic_lines)
    max_context_lines = 20  # Max lines for failure details

    # 1. 首先，只检查 致命 错误 (编译, 内存泄漏)
    for log_type, pattern in LOG_TYPE_PATTERNS.items():
        if re.search(pattern, diagnostic_output if log_type == LogType.COMPILATION_ERROR else output):
            summary_line = log_type.value.replace('_', ' ').title()
            
            if log_type == LogType.COMPILATION_ERROR:
                error_line_index = -1
                
                for i, line in enumerate(diagnostic_lines):
                    # Prefer compiler-like diagnostics; avoid generic "Error:" runtime messages.
                    if re.search(r"(?i).+?:\d+:\d+:\s+(?:fatal\s+)?error:\s", line) or re.search(
                        r"(?i).+?:\d+:\s+(?:fatal\s+)?error:\s", line
                    ):
                        error_line_index = i
                        break
                
                if error_line_index == -1:
                    for i, line in enumerate(lines):
                        if (
                            "make: ***" in line
                            or "linker command failed" in line
                            or "ninja: build stopped" in line.lower()
                            or "undefined reference to" in line.lower()
                            or "collect2: error:" in line.lower()
                        ):
                            error_line_index = i
                            break

                if error_line_index != -1:
                    start = max(0, error_line_index - 10)
                    end = min(len(lines), error_line_index + 11)
                    context_lines = lines[start:end]
                    summary_line = f"Compilation Error Detected. Context:\n---\n"
                    summary_line += "\n".join(context_lines)
                    summary_line += "\n---"
                else:
                    summary_line = "Compilation Error: No specific error line found."

            return TestSummary(log_type=log_type, summary_line=summary_line, raw_output=output)

    # 2. 如果没有致命错误, 则继续解析各个测试
    found_tests: List[TestResult] = []
    i = 0
    
    # --- *** 循环逻辑修正 *** ---
    while i < len(lines):
        line = lines[i]
        test_matched = False
        
        for status, patterns in ALL_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    test_name = match.group(1).strip() if match.groups() else "Unknown Test"
                    details = ""
                    
                    if status == TestStatus.FAILED:
                        details_lines = []
                        j = i + 1
                        while j < len(lines) and not _is_test_result_line(lines[j]) and len(details_lines) < max_context_lines:
                            details_lines.append(lines[j])
                            j += 1
                        
                        details = "\n".join(details_lines).strip()
                        
                        if j < len(lines) and not _is_test_result_line(lines[j]):
                            details += "\n[... details truncated]"

                        # 修正: 我们明确地将 i 设置为下一行 (j)
                        i = j 
                    
                    else:
                        # 修正: 对于 PASSED 或 SKIPPED, 我们只前进 1
                        i += 1

                    found_tests.append(TestResult(name=test_name, status=status, details=details))
                    test_matched = True
                    break # 跳出 'pattern' 循环
            if test_matched:
                break # 跳出 'status' 循环
        
        # 修正: 只有在这一行*没有*匹配到测试时, 我们才前进 1
        if not test_matched:
            i += 1
    # --- *** 循环逻辑修正结束 *** ---


    # 4. 如果没有 "No tests" 消息, 且我们什么也没解析到
    if NO_TESTS_PATTERN.search(output):
        return TestSummary(log_type=LogType.NO_TESTS, summary_line="No tests executed.", raw_output=output)
    if not found_tests:
        last_lines = "\n".join(lines[-50:])
        summary = f"Showing last 50 lines of output:\n---\n{last_lines}\n---"
        return TestSummary(log_type=LogType.UNKNOWN, summary_line=summary, raw_output=output)

    passed_count = sum(1 for r in found_tests if r.status == TestStatus.PASSED)
    failed_count = sum(1 for r in found_tests if r.status == TestStatus.FAILED)
    skipped_count = sum(1 for r in found_tests if r.status == TestStatus.SKIPPED)

    summary_line = f"{passed_count} passed, {failed_count} failed, {skipped_count} skipped."
    if failed_count > 0:
        first_failed_test = next((r.name for r in found_tests if r.status == TestStatus.FAILED), "Unknown")
        summary_line += f" First failure: '{first_failed_test}'. See details in results."

    return TestSummary(
        log_type=LogType.TEST_RESULTS,
        results=found_tests,
        summary_line=summary_line,
        raw_output=output
    )


# --- 单元测试 (使用 dedent 修复了缩进) ---

if __name__ == "__main__":
    
    # 定义一个辅助函数来运行测试
    def run_test(test_function):
        test_name = test_function.__name__
        try:
            test_function()
            print(f"[ PASS ] {test_name}")
        except AssertionError as e:
            import sys
            _, _, tb = sys.exc_info()
            print(f"[ FAIL ] {test_name} (at line {tb.tb_lineno})")
            print(f"       AssertionError: {e}")
        except Exception as e:
            print(f"[ ERROR ] {test_name}")
            print(f"         Exception: {e}")

    # --- 测试用例 ---

    def test_global_compilation_error():
        log_input = dedent("""
            In file included from main.cpp:1:
            header.h:5:10: fatal error: 'missing.h' file not found
            #include "missing.h"
                     ^~~~~~~~~~~
            1 error generated.
            make: *** [build] Error 1
        """)
        summary = parse_test_output(log_input)
        assert summary.log_type == LogType.COMPILATION_ERROR
        assert "fatal error: 'missing.h' file not found" in summary.summary_line

    def test_global_linker_error():
        log_input = dedent("linker command failed with exit code 1 (use -v to see invocation)")
        summary = parse_test_output(log_input)
        assert summary.log_type == LogType.COMPILATION_ERROR
        assert "linker command failed" in summary.summary_line

    def test_global_memory_leak():
        log_input = "LeakSanitizer: detected memory leaks"
        summary = parse_test_output(log_input)
        assert summary.log_type == LogType.MEMORY_LEAK

    def test_global_explicit_no_tests():
        """这个测试现在会正确地失败（因为'No tests ran'优先）"""
        log_input = "[ OK ] Test.ShouldBeIgnored\nNo tests ran."
        summary = parse_test_output(log_input)
        assert summary.log_type == LogType.NO_TESTS
        assert len(summary.results) == 0

    def test_no_parsable_tests():
        log_input = "Starting application...\nApplication running."
        summary = parse_test_output(log_input)
        assert summary.log_type == LogType.NO_TESTS

    def test_empty_log():
        log_input = ""
        summary = parse_test_output(log_input)
        assert summary.log_type == LogType.NO_TESTS

    def test_complex_log_parsing():
        log_input = dedent("""
            [  OK  ] Test.Pass.Basic
            Test #1: Module.Test.AnotherPass ... Passed
            test sub::module::test_pass ... ok
            [ FAILED ] Test.Fail.WithDetails
              Assertion failed: (1 == 2)
            [  OK  ] Test.Pass.AfterFailure
            [ FAILED ] Test.Fail.NoDetails
            [ SKIPPED ] Test.Skip.Reason
            Test #4: Module.Test.Timeout ... Timeout
            test sub::module::test_fail ... FAILED
              More failure details
        """)
        summary = parse_test_output(log_input)
        assert summary.log_type == LogType.TEST_RESULTS
        assert len(summary.results) == 9
        assert summary.summary_line == "4 passed, 4 failed, 1 skipped. First failure: 'Test.Fail.WithDetails'. See details in results."

    def test_failure_detail_capture():
        """测试失败详情是否被正确捕获 (这是失败的测试)"""
        log_input = dedent("""
            [ FAILED ] Test.Fail.WithDetails
              Assertion failed: (1 == 2)
              at /path/to/file.cpp:42
            [  OK  ] Test.Pass.AfterFailure
            [ FAILED ] Test.Fail.NoDetails
            [ FAILED ] test::sub::fail
              More failure details
              another line
        """)
        summary = parse_test_output(log_input)
        results_dict = {r.name: r for r in summary.results}

        # 1. 检查 'Test.Fail.WithDetails'
        fail_with_details = results_dict["Test.Fail.WithDetails"]
        assert fail_with_details.status == TestStatus.FAILED
        assert "Assertion failed: (1 == 2)" in fail_with_details.details
        assert "[  OK  ] Test.Pass.AfterFailure" not in fail_with_details.details

        # 2. 检查 'Test.Fail.NoDetails'
        fail_no_details = results_dict["Test.Fail.NoDetails"]
        assert fail_no_details.status == TestStatus.FAILED
        assert fail_no_details.details == ""
        
        # 3. 检查 'test::sub::fail'
        fail_rust = results_dict["test::sub::fail"]
        assert fail_rust.status == TestStatus.FAILED
        assert fail_rust.details == "  More failure details\n  another line"

    def test_failure_detail_truncation():
        detail_line = "this is a long error line\n"
        log_input = dedent("[ FAILED ] Test.LongFailure\n" + (detail_line * 25) + "[ OK ] Test.Next")
        summary = parse_test_output(log_input)
        assert summary.log_type == LogType.TEST_RESULTS
        failed_test = summary.results[0]
        assert failed_test.details.count("this is a long error line") == 20
        assert failed_test.details.endswith("[... details truncated]")

    def test_helper_is_test_result_line():
        assert _is_test_result_line("[ OK ] a.test") == True
        assert _is_test_result_line("  [ OK ] a.test") == False
        assert _is_test_result_line("  Assertion failed") == False

    # --- 运行所有测试 ---
    print("Running Log Parser Unit Tests...\n")
    all_tests = [
        test_global_compilation_error,
        test_global_linker_error,
        test_global_memory_leak,
        test_global_explicit_no_tests,
        test_no_parsable_tests,
        test_empty_log,
        test_complex_log_parsing,
        test_failure_detail_capture,
        test_failure_detail_truncation,
        test_helper_is_test_result_line,
    ]
    for test in all_tests:
        run_test(test)
    print("\n...Test run finished.")
