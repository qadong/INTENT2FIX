from pathlib import Path
import subprocess
import sys


def test_repair_cli_cold_start():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_no_memory_cotest", "--help"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--portfolio-repair-only" in result.stdout


def test_reviewer_public_import_preserved():
    from INTENT2FIX.patch_review import PatchSurfaceReviewer
    from INTENT2FIX.patch_review.reviewer import PatchSurfaceReviewer as Reviewer

    assert PatchSurfaceReviewer is Reviewer
