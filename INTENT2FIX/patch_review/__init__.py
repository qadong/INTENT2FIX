from INTENT2FIX.patch_review.sanitizer import (
    CandidatePatch,
    sanitize_candidate_patch,
)
from INTENT2FIX.patch_review.schema import PatchReviewResult, SurfaceCheck


def __getattr__(name):
    # Actions use the sanitizer while ActionAgent is still being imported.
    if name == "PatchSurfaceReviewer":
        from INTENT2FIX.patch_review.reviewer import PatchSurfaceReviewer
        return PatchSurfaceReviewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CandidatePatch",
    "PatchReviewResult",
    "PatchSurfaceReviewer",
    "SurfaceCheck",
    "sanitize_candidate_patch",
]
