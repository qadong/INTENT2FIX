import re
import json
import logging
from enum import Enum
from typing import Tuple, Dict, Optional, Pattern

logger = logging.getLogger(__name__)

class TestCase:
    """
    A data class to hold the status of a single test case across all stages.
    """
    def __init__(self, name: str):
        self.name = name
        self.baseline_status: Optional[str] = None
        self.pre_patch_status: Optional[str] = None
        self.post_patch_status: Optional[str] = None

    def __repr__(self) -> str:
        return (f"TestCase(name={self.name}, "
                f"run={self.baseline_status}, "
                f"test={self.pre_patch_status}, "
                f"fix={self.post_patch_status})")

class LogType(Enum):
    """Defines the category of the log content."""
    TEST_RESULTS = "test_results"
    COMPILATION_ERROR = "compilation_error"
    UNKNOWN = "unknown"

class TestStatus(Enum):
    """Standardized test status values."""
    PASSED = 'passed'
    FAILED = 'failed'
    SKIPPED = 'skipped'  # 'Not Run' will map to this status
    NONE = 'none'

# Patterns now ONLY include actual unit tests.
# All "-- Performing Test" patterns have been removed.
ALL_PATTERNS: Dict[TestStatus, Tuple[Pattern[str], ...]] = {
    TestStatus.PASSED: (
        re.compile(r"Test\s+#\d+:\s+(.+?)\s*\.{2,}.*?Passed"),
        re.compile(r"^\d+/\d+\s+Test\s+\#\d+:\s+(.+)\s+\.+\s+Passed"),
        re.compile(r"^\[\s*OK\s*\]\s+([a-zA-Z0-9_\/.:]+)"),
        re.compile(r"^\[\s+OK\s+\]\s+(\S+?)(?:\s+\(.+\))?$"),
        re.compile(r"^\x1b\[0;32m\[\s+OK\s+\]\x1b\[0m\s+(\S+?)(?:\s+\(.+\))?$"),
        re.compile(r"^(.+?:\d+):\s+PASSED:"),
        re.compile(r'test case "([^"]+)" passed'),
        re.compile(r"^\s*--- PASS:\s+(\S+)(?:\s+\(.+\))?"),
        re.compile(r"^\s*[✔✓]\s+(.*?)(?::\s*\d+ms)?$"),
        re.compile(r"^\x1b\[92m---- Passed: (\S+?)\x1b\[0m$"),
        re.compile(r"^test\s+(\S+)\s+\.\.\.\s+ok$"),
        re.compile(r"^(\S+)\s+\.\.\.\s+ok$"),
    ),
    TestStatus.FAILED: (
        re.compile(r"Test\s+#\d+:\s+(.+?)\s*\.{2,}.*?(?:Failed|Timeout)"),
        re.compile(r"^\d+/\d+\s+Test\s+\#\d+:\s+(.+)\s+\.+\s+\*+Failed", re.IGNORECASE),
        re.compile(r"^\[\s*FAILED\s*\]\s+([a-zA-Z0-9_\/.:]+)"),
        re.compile(r"^\[\s+FAILED\s+\]\s+(\S+?)(?:\s+\(.+\))?$"),
        re.compile(r"^\x1b\[0;31m\[\s+FAILED\s+\]\x1b\[0m\s+(\S+?)(?:\s+\(.+\))?$"),
        re.compile(r"^\d+/\d+\s*Test\s*\#\d+:\s*(.*?)\s*\.+\s*\*+Exception: Illegal"),
        re.compile(r"^\d+/\d+\s+Test\s+\#\d+:\s+(.+?)\s+\.+\s*Subprocess\s+aborted\*+\s*Exception:"),
        re.compile(r"^\d+/\d+\s+Test\s+\#\d+:\s+(.+?)\s+\.+\s*Child\s+aborted\*+\s*Exception:"),
        re.compile(r"^\d+/\d+\s*Test\s*\#\d+:\s*(.*?)\s*\.+\s*\*+Exception: SegFault"),
        re.compile(r"^(.+?:\d+):\s+FAILED:"),
        re.compile(r'error in "([^"]+)"'),
        re.compile(r"^--- FAIL: (\S+)"),
        re.compile(r"^\s*[✘×𐄂]\s+(.*?)(?::\s*\d+ms)?$"),
        re.compile(r"^\x1b\[91m\*\*\*\* FAILED: (\S+?)\x1b\[0m$"),
        re.compile(r"^test\s+(\S+)\s+\.\.\.\s+FAILED$"),
        re.compile(r"^(\S+)\s+\.\.\.\s+FAILED$"),
    ),
    TestStatus.SKIPPED: (
        re.compile(r"Test\s+#\d+:\s+(.+?)\s*\.{2,}.*?Not Run"),
        re.compile(r"^\s*-\s+(.*)$"),
    )
}

def parse_log(log: str) -> Dict[str, str]:
    """
    Parses test logs for unit test results using a predefined set of regex patterns.
    """
    test_status_map: Dict[str, str] = {}

    for line in log.split("\n"):
        matched_line = False
        for status_enum, patterns in ALL_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    try:
                        test_name = match.group(1).strip()
                        if test_name not in test_status_map or test_status_map[test_name] != TestStatus.FAILED.value:
                            test_status_map[test_name] = status_enum.value
                        matched_line = True
                        break
                    except IndexError:
                        pass
            if matched_line:
                break

    summary_marker = "The following tests FAILED:"
    if summary_marker in log:
        summary_block = log.split(summary_marker, 1)[1]
        summary_pattern = re.compile(r"\d+\s+-\s+([\w\d_-]+)\s+\(([^)]+)\)")
        for line in summary_block.split("\n"):
            match = summary_pattern.search(line)
            if match:
                test_name, reason = match.groups()
                test_name = test_name.strip()
                reason_lower = reason.lower()
                if reason_lower == "not run":
                    test_status_map[test_name] = TestStatus.SKIPPED.value
                else:
                    test_status_map[test_name] = TestStatus.FAILED.value
                    
    return test_status_map

def analyze_log(log: str) -> Tuple[LogType, Optional[Dict[str, str]]]:
    """
    Analyzes a log to determine its type and parse it for unit tests.
    """
    results = parse_log(log)
    if results:
        logger.info("✅ Log successfully parsed as TEST_RESULTS.")
        return LogType.TEST_RESULTS, results

    compile_error_pattern = re.compile(
        r"(error:|fatal error:|compilation terminated|make: \*\*\*|linker command failed)", 
        re.IGNORECASE
    )
    if compile_error_pattern.search(log):
        logger.info("✅ Log identified as COMPILATION_ERROR.")
        return LogType.COMPILATION_ERROR, None

    logger.warning("❌ Could not determine log type or no unit tests found.")
    return LogType.UNKNOWN, None

# ==============================================================================
# Example Usage with your provided log
# ==============================================================================
if __name__ == "__main__":
    # Configure logger to display INFO level and above logs on the console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # You can replace this example log with any of the formats
    # covered by the patterns above to test the parser.
    ctest_log_example = """
INFO: Verifying the fix...
+ cd /home/fmt
+ echo 'INFO: Verifying the fix...'
+ git checkout 06f1c0d725855861535e9e65cd4d502aca7c61ed
HEAD is now at 06f1c0d7 Clarify that calling non-const format is deprecated
Applying both fix and test patches...
+ echo 'Applying both fix and test patches...'
+ git apply /home/fix.patch
+ git apply /home/test.patch
Building project with fixes...
+ echo 'Building project with fixes...'
+ rm -rf build
+ mkdir build
+ cd build
+ cmake .. -DFMT_TEST=ON
-- CMake version: 3.22.1
-- The CXX compiler identification is GNU 11.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Version: 10.1.1
-- Build type: Release
-- Target 'doc' disabled (requires doxygen)
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- Looking for strptime
-- Looking for strptime - found
-- FMT_PEDANTIC: OFF
-- Configuring done
-- Generating done
-- Build files have been written to: /home/fmt/build
++ nproc
+ make -j16
[  1%] Building CXX object CMakeFiles/fmt.dir/src/format.cc.o
[  4%] Building CXX object test/gtest/CMakeFiles/gtest.dir/gmock-gtest-all.cc.o
[  4%] Building CXX object CMakeFiles/fmt.dir/src/os.cc.o
[  5%] Linking CXX static library libfmt.a
[  5%] Built target fmt
[  7%] Linking CXX static library libgtest.a
[  7%] Built target gtest
[ 10%] Building CXX object test/CMakeFiles/format-impl-test.dir/format-impl-test.cc.o
[ 10%] Building CXX object test/CMakeFiles/test-main.dir/test-main.cc.o
[ 13%] Building CXX object test/CMakeFiles/compile-fp-test.dir/compile-fp-test.cc.o
[ 13%] Building CXX object test/CMakeFiles/compile-fp-test.dir/test-main.cc.o
[ 14%] Building CXX object test/CMakeFiles/compile-fp-test.dir/gtest-extra.cc.o
[ 17%] Building CXX object test/CMakeFiles/format-impl-test.dir/header-only-test.cc.o
[ 17%] Building CXX object test/CMakeFiles/test-main.dir/util.cc.o
[ 20%] Building CXX object test/CMakeFiles/format-impl-test.dir/gtest-extra.cc.o
[ 20%] Building CXX object test/CMakeFiles/compile-fp-test.dir/util.cc.o
[ 22%] Building CXX object test/CMakeFiles/format-impl-test.dir/__/src/os.cc.o
[ 23%] Building CXX object test/CMakeFiles/test-main.dir/gtest-extra.cc.o
[ 29%] Building CXX object test/CMakeFiles/format-impl-test.dir/util.cc.o
[ 29%] Building CXX object test/CMakeFiles/posix-mock-test.dir/posix-mock-test.cc.o
[ 29%] Building CXX object test/CMakeFiles/compile-fp-test.dir/__/src/os.cc.o
[ 31%] Building CXX object test/CMakeFiles/unicode-test.dir/unicode-test.cc.o
[ 29%] Building CXX object test/CMakeFiles/format-impl-test.dir/test-main.cc.o
[ 32%] Building CXX object test/CMakeFiles/unicode-test.dir/test-main.cc.o
[ 34%] Building CXX object test/CMakeFiles/posix-mock-test.dir/__/src/format.cc.o
[ 35%] Building CXX object test/CMakeFiles/unicode-test.dir/util.cc.o
[ 37%] Building CXX object test/CMakeFiles/unicode-test.dir/gtest-extra.cc.o
[ 38%] Linking CXX static library libtest-main.a
[ 38%] Built target test-main
[ 40%] Building CXX object test/CMakeFiles/unicode-test.dir/__/src/os.cc.o
[ 41%] Building CXX object test/CMakeFiles/args-test.dir/args-test.cc.o
[ 43%] Building CXX object test/CMakeFiles/posix-mock-test.dir/test-main.cc.o
[ 44%] Building CXX object test/CMakeFiles/assert-test.dir/assert-test.cc.o
[ 46%] Building CXX object test/CMakeFiles/chrono-test.dir/chrono-test.cc.o
[ 47%] Linking CXX executable ../bin/assert-test
[ 49%] Building CXX object test/CMakeFiles/color-test.dir/color-test.cc.o
[ 49%] Built target assert-test
[ 50%] Building CXX object test/CMakeFiles/posix-mock-test.dir/gtest-extra.cc.o
[ 52%] Linking CXX executable ../bin/args-test
[ 53%] Building CXX object test/CMakeFiles/posix-mock-test.dir/util.cc.o
[ 53%] Built target args-test
[ 55%] Building CXX object test/CMakeFiles/core-test.dir/core-test.cc.o
[ 56%] Building CXX object test/CMakeFiles/gtest-extra-test.dir/gtest-extra-test.cc.o
[ 58%] Building CXX object test/CMakeFiles/format-test.dir/format-test.cc.o
[ 59%] Building CXX object test/CMakeFiles/ostream-test.dir/ostream-test.cc.o
[ 61%] Building CXX object test/CMakeFiles/compile-test.dir/compile-test.cc.o
[ 62%] Building CXX object test/CMakeFiles/printf-test.dir/printf-test.cc.o
[ 64%] Linking CXX executable ../bin/compile-fp-test
[ 65%] Linking CXX executable ../bin/posix-mock-test
[ 65%] Built target posix-mock-test
[ 65%] Built target compile-fp-test
[ 67%] Building CXX object test/CMakeFiles/ranges-test.dir/ranges-test.cc.o
[ 68%] Building CXX object test/CMakeFiles/scan-test.dir/scan-test.cc.o
[ 70%] Building CXX object test/CMakeFiles/std-test.dir/std-test.cc.o
[ 71%] Building CXX object test/CMakeFiles/ranges-test.dir/ranges-odr-test.cc.o
[ 73%] Linking CXX executable ../bin/unicode-test
[ 74%] Building CXX object test/CMakeFiles/xchar-test.dir/xchar-test.cc.o
[ 76%] Building CXX object test/CMakeFiles/enforce-checks-test.dir/enforce-checks-test.cc.o
[ 76%] Built target unicode-test
[ 77%] Building CXX object test/CMakeFiles/os-test.dir/os-test.cc.o
[ 79%] Linking CXX executable ../bin/color-test
[ 79%] Built target color-test
[ 80%] Linking CXX executable ../bin/format-impl-test
[ 80%] Built target format-impl-test
[ 82%] Linking CXX executable ../bin/scan-test
[ 82%] Built target scan-test
[ 83%] Linking CXX executable ../bin/gtest-extra-test
[ 83%] Built target gtest-extra-test
[ 85%] Linking CXX executable ../bin/ostream-test
[ 85%] Built target ostream-test
[ 86%] Linking CXX executable ../bin/os-test
[ 86%] Built target os-test
[ 88%] Linking CXX executable ../bin/enforce-checks-test
[ 88%] Built target enforce-checks-test
[ 89%] Linking CXX executable ../bin/std-test
[ 89%] Built target std-test
[ 91%] Linking CXX executable ../bin/compile-test
[ 91%] Built target compile-test
[ 92%] Linking CXX executable ../bin/ranges-test
[ 92%] Built target ranges-test
[ 94%] Linking CXX executable ../bin/core-test
[ 94%] Built target core-test
[ 95%] Linking CXX executable ../bin/xchar-test
[ 95%] Built target xchar-test
[ 97%] Linking CXX executable ../bin/chrono-test
[ 97%] Built target chrono-test
[ 98%] Linking CXX executable ../bin/format-test
[ 98%] Built target format-test
[100%] Linking CXX executable ../bin/printf-test
[100%] Built target printf-test
Running tests to verify fix...
+ echo 'Running tests to verify fix...'
+ ctest --output-on-failure
Test project /home/fmt/build
      Start  1: args-test
 1/20 Test  #1: args-test ........................   Passed    0.00 sec
      Start  2: assert-test
 2/20 Test  #2: assert-test ......................   Passed    0.00 sec
      Start  3: chrono-test
 3/20 Test  #3: chrono-test ......................   Passed    0.01 sec
      Start  4: color-test
 4/20 Test  #4: color-test .......................   Passed    0.00 sec
      Start  5: core-test
 5/20 Test  #5: core-test ........................   Passed    0.00 sec
      Start  6: gtest-extra-test
 6/20 Test  #6: gtest-extra-test .................   Passed    0.00 sec
      Start  7: format-test
 7/20 Test  #7: format-test ......................   Passed    0.01 sec
      Start  8: format-impl-test
 8/20 Test  #8: format-impl-test .................   Passed    0.01 sec
      Start  9: ostream-test
 9/20 Test  #9: ostream-test .....................   Passed    0.00 sec
      Start 10: compile-test
10/20 Test #10: compile-test .....................   Passed    0.00 sec
      Start 11: compile-fp-test
11/20 Test #11: compile-fp-test ..................   Passed    0.00 sec
      Start 12: printf-test
12/20 Test #12: printf-test ......................   Passed    0.00 sec
      Start 13: ranges-test
13/20 Test #13: ranges-test ......................   Passed    0.00 sec
      Start 14: scan-test
14/20 Test #14: scan-test ........................   Passed    0.00 sec
      Start 15: std-test
15/20 Test #15: std-test .........................   Passed    0.00 sec
      Start 16: unicode-test
16/20 Test #16: unicode-test .....................   Passed    0.00 sec
      Start 17: xchar-test
17/20 Test #17: xchar-test .......................   Passed    0.00 sec
      Start 18: enforce-checks-test
18/20 Test #18: enforce-checks-test ..............   Passed    0.00 sec
      Start 19: posix-mock-test
19/20 Test #19: posix-mock-test ..................   Passed    0.00 sec
      Start 20: os-test
20/20 Test #20: os-test ..........................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 20

Total Test time (real) =   0.07 sec
All tests passed - fix verified.
+ echo 'All tests passed - fix verified.'
    """

    logger.info("--- Testing log with the new generic parser ---")
    log_type, parsed_results = analyze_log(ctest_log_example)
    
    if parsed_results:
        # Format JSON results and log as a single message for readability
        formatted_results = json.dumps(parsed_results, indent=4, sort_keys=True)
        logger.info("Parsed Results:\n%s", formatted_results)
        logger.info("Total test cases parsed: %d", len(parsed_results))