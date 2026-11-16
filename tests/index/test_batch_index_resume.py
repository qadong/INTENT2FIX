import subprocess
import sys


def test_prefixed_instance_matches_completed_and_failed_log_ids(tmp_path):
    log = tmp_path / "summary.log"
    log.write_text(
        "[SUCCESS] swe-bench_fmtlib__fmt-974 - Completed\n"
        "[FAILURE] swe-bench_catchorg__Catch2-2187 - Exit code 1\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "-c", """
import sys
from pathlib import Path
from code_intent import batch_index

log = Path(sys.argv[1])
completed = batch_index.load_completed_instances_from_log(log)
failed = batch_index.load_failed_instances_from_log(log)
assert batch_index._instance_in_log('swe-bench_fmtlib__fmt-974', completed)
assert batch_index._instance_in_log('fmtlib__fmt-974', completed)
assert not batch_index._instance_in_log('swe-bench_fmtlib__fmt-974', failed)
assert batch_index._instance_in_log('swe-bench_catchorg__Catch2-2187', failed)
""", str(log)],
        check=True,
    )
