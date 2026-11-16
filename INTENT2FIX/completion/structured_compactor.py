import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


_COMPACTABLE_ERROR_TYPES = {
    "extra_forbidden",
    "string_too_long",
    "too_long",
}
_REFERENCE_ID = re.compile(
    r"\b[A-Z][A-Z0-9_]*(?:-[A-Z0-9_]+)*-\d+\b"
)
_NON_SEMANTIC_ROOT_FIELDS = {"thoughts", "scratch_pad"}


@dataclass(frozen=True)
class CompactionAudit:
    mutable_paths: tuple[str, ...]
    reference_ids: tuple[str, ...]
    strategy: str = "llm"


class StructuredCompactionError(ValueError):
    pass


def parse_json_object(raw: str | Any) -> Any:
    if not isinstance(raw, str):
        return copy.deepcopy(raw)

    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        from dirtyjson import loads as dirty_json_loads

        return dirty_json_loads(text)


def is_compactable_validation_error(error: ValidationError) -> bool:
    errors = error.errors(include_url=False)
    return bool(errors) and all(
        item.get("type") in _COMPACTABLE_ERROR_TYPES for item in errors
    )


def _path_text(path: tuple[Any, ...]) -> str:
    return ".".join(str(item) for item in path) or "<root>"


def _delete_path(root: Any, path: tuple[Any, ...]) -> None:
    if not path:
        return
    current = root
    try:
        for item in path[:-1]:
            current = current[item]
        key = path[-1]
        if isinstance(current, dict):
            current.pop(key, None)
        elif isinstance(current, list) and isinstance(key, int):
            current[key] = None
    except (KeyError, IndexError, TypeError):
        return


def _read_path(root: Any, path: tuple[Any, ...]) -> Any:
    current = root
    for item in path:
        current = current[item]
    return current


def _replace_path(root: Any, path: tuple[Any, ...], value: Any) -> None:
    if not path:
        raise StructuredCompactionError("Cannot replace the payload root.")
    current = root
    try:
        for item in path[:-1]:
            current = current[item]
        current[path[-1]] = value
    except (KeyError, IndexError, TypeError) as exc:
        raise StructuredCompactionError(
            f"Compactor returned an invalid replacement path: {_path_text(path)}."
        ) from exc


def _reference_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.update(_REFERENCE_ID.findall(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.update(_reference_ids(str(key)))
            found.update(_reference_ids(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_reference_ids(item))
    return found


def _semantic_payload(value: Any) -> Any:
    payload = copy.deepcopy(value)
    if isinstance(payload, dict):
        for field in _NON_SEMANTIC_ROOT_FIELDS:
            payload.pop(field, None)
    return payload


def _assert_immutable_fields_unchanged(
    source: Any,
    compacted: Any,
    mutable_paths: list[tuple[Any, ...]],
) -> None:
    source_fixed = _semantic_payload(source)
    compacted_fixed = _semantic_payload(compacted)
    for path in mutable_paths:
        _delete_path(source_fixed, path)
        _delete_path(compacted_fixed, path)
    if source_fixed != compacted_fixed:
        raise StructuredCompactionError(
            "Compactor changed fields outside the schema-rejected paths."
        )


def _build_prompt(
    source: Any,
    error: ValidationError,
) -> str:
    failures = error.errors(include_url=False)
    failure_paths = [tuple(item.get("loc", ())) for item in failures]
    mutable_paths = [_path_text(path) for path in failure_paths]
    field_budgets = []
    for item in failures:
        path = _path_text(tuple(item.get("loc", ())))
        error_type = item.get("type")
        maximum = item.get("ctx", {}).get("max_length")
        if not isinstance(maximum, int):
            continue
        if error_type == "string_too_long":
            # Providers do not always count characters exactly as the local
            # validator does. Leave enough headroom for one-pass semantic
            # compaction instead of accepting a borderline replacement that
            # immediately fails the same schema again.
            target = max(1, int(maximum * 0.5))
            field_budgets.append(
                f"- `{path}`: hard maximum {maximum} characters; target at "
                f"most {target} characters so validation is not borderline."
            )
        elif error_type == "too_long":
            field_budgets.append(
                f"- `{path}`: hard maximum {maximum} list items. Merge related "
                "items while preserving all distinct facts and reference IDs."
            )
    budget_instructions = (
        "\n".join(field_budgets)
        if field_budgets
        else "- No numeric length limit was reported; follow the validation errors exactly."
    )
    source_values = []
    remove_paths = []
    for item, path in zip(failures, failure_paths, strict=True):
        if item.get("type") == "extra_forbidden":
            remove_paths.append(list(path))
            continue
        try:
            value = _read_path(source, path)
        except (KeyError, IndexError, TypeError):
            value = item.get("input")
        source_values.append({"path": list(path), "value": value})

    return f"""# Single-Turn Structured Payload Compactor

You are not an investigator and must not reinterpret the task. Return a small
field-replacement patch for the schema-rejected values below. The caller will
merge it into the authoritative source payload and validate the result.

Rules:
- Modify only these validation-failing paths: {json.dumps(mutable_paths)}.
- Obey these path-specific output budgets exactly:
{budget_instructions}
- Keep every other field and value exactly unchanged.
- Preserve the exact set of evidence/object IDs. Never invent or remove an ID.
- For an overlong string, remove repetition and compress wording; preserve every
  distinct decision, fact, constraint, uncertainty, and qualification.
- For an overlong list, merge closely related items when necessary. Do not just
  drop the tail.
- Omit a forbidden extra field only when it is redundant metadata. Do not alter
  a locked valid field to absorb it; an invalid result is safer than information
  loss and will be returned to the original Agent for correction.
- Do not add facts, resolve open decisions, strengthen evidence, or change status.
- Return exactly one JSON object with no markdown or commentary in this form:
  {{"replacements": [{{"path": ["field", 0], "value": "replacement"}}],
    "remove_paths": [["forbidden_field"]]}}
- Include exactly one replacement for every non-extra failing path and no other
  path. Use the exact path components and preserve their string/integer types.
- Set `remove_paths` to exactly the forbidden-extra paths listed below. Do not
  copy valid fields into the response.

Validation errors:
{json.dumps(failures, ensure_ascii=False)}

Required forbidden-extra removals:
{json.dumps(remove_paths, ensure_ascii=False)}

Authoritative failing values:
{json.dumps(source_values, ensure_ascii=False)}
"""


def _apply_compaction_patch(
    source: Any,
    response: Any,
    failures: list[dict[str, Any]],
) -> Any:
    if not isinstance(response, dict) or "replacements" not in response:
        # Backward-compatible acceptance for providers that returned the full
        # object despite the field-patch instruction.
        return response

    # Some reasoning providers add transport metadata beside the requested
    # patch. It cannot affect the authoritative payload because only these two
    # allowlisted fields are read and every replacement path is checked below.
    replacements = response.get("replacements")
    remove_paths = response.get("remove_paths")
    if not isinstance(replacements, list) or not isinstance(remove_paths, list):
        raise StructuredCompactionError("Compactor patch fields must be lists.")

    replacement_paths = {
        tuple(item.get("loc", ()))
        for item in failures
        if item.get("type") != "extra_forbidden"
    }
    expected_remove_paths = {
        tuple(item.get("loc", ()))
        for item in failures
        if item.get("type") == "extra_forbidden"
    }
    proposed_remove_paths = {
        tuple(path) for path in remove_paths if isinstance(path, list)
    }
    if proposed_remove_paths != expected_remove_paths:
        raise StructuredCompactionError(
            "Compactor patch did not return the exact forbidden-extra paths."
        )

    proposed_replacements: dict[tuple[Any, ...], Any] = {}
    for item in replacements:
        if not isinstance(item, dict) or set(item) != {"path", "value"}:
            raise StructuredCompactionError(
                "Each compactor replacement requires only path and value."
            )
        raw_path = item["path"]
        if not isinstance(raw_path, list):
            raise StructuredCompactionError("Replacement path must be a list.")
        path = tuple(raw_path)
        if path in proposed_replacements:
            raise StructuredCompactionError(
                f"Duplicate compactor replacement path: {_path_text(path)}."
            )
        proposed_replacements[path] = item["value"]
    if set(proposed_replacements) != replacement_paths:
        raise StructuredCompactionError(
            "Compactor patch did not return exactly the schema-rejected paths."
        )

    compacted = copy.deepcopy(source)
    for path, value in proposed_replacements.items():
        _replace_path(compacted, path, value)
    for path in proposed_remove_paths:
        _delete_path(compacted, path)
    return compacted


def _merge_oversized_string_lists(
    source: Any,
    failures: list[dict[str, Any]],
) -> Any | None:
    """Losslessly fit string-only lists when the semantic compactor misformats its patch."""
    if not failures or any(item.get("type") != "too_long" for item in failures):
        return None
    compacted = copy.deepcopy(source)
    for item in failures:
        path = tuple(item.get("loc", ()))
        maximum = item.get("ctx", {}).get("max_length")
        try:
            values = _read_path(source, path)
        except (KeyError, IndexError, TypeError):
            return None
        if (not isinstance(maximum, int) or maximum < 1
                or not isinstance(values, list)
                or len(values) <= maximum
                or not all(isinstance(value, str) for value in values)):
            return None
        replacement = values[: maximum - 1] + [
            "\n".join(values[maximum - 1 :])
        ]
        _replace_path(compacted, path, replacement)
    return compacted


def compact_structured_payload_once(
    source: str | Any,
    schema: type[BaseModel],
    complete: Callable[[str], str],
) -> tuple[BaseModel, CompactionAudit]:
    source_data = parse_json_object(source)
    try:
        return schema.model_validate(source_data), CompactionAudit((), ())
    except ValidationError as validation_error:
        if not is_compactable_validation_error(validation_error):
            raise StructuredCompactionError(
                "Payload has non-compaction validation errors."
            ) from validation_error
        source_validation_error = validation_error

    failures = source_validation_error.errors(include_url=False)
    mutable_paths = [tuple(item.get("loc", ())) for item in failures]
    prompt = _build_prompt(source_data, source_validation_error)
    strategy = "llm"
    try:
        compacted_raw = complete(prompt)
        if not compacted_raw or not compacted_raw.strip():
            raise StructuredCompactionError("Compactor returned an empty response.")
        compacted_response = parse_json_object(compacted_raw)
        compacted_data = _apply_compaction_patch(
            source_data,
            compacted_response,
            failures,
        )
    except Exception:
        compacted_data = _merge_oversized_string_lists(source_data, failures)
        if compacted_data is None:
            raise
        strategy = "deterministic_string_list_merge"

    _assert_immutable_fields_unchanged(source_data, compacted_data, mutable_paths)
    source_ids = _reference_ids(_semantic_payload(source_data))
    compacted_ids = _reference_ids(_semantic_payload(compacted_data))
    if compacted_ids != source_ids:
        missing = sorted(source_ids - compacted_ids)
        added = sorted(compacted_ids - source_ids)
        raise StructuredCompactionError(
            f"Compactor changed reference IDs (missing={missing}, added={added})."
        )

    validated = schema.model_validate(compacted_data)
    return validated, CompactionAudit(
        mutable_paths=tuple(_path_text(path) for path in mutable_paths),
        reference_ids=tuple(sorted(source_ids)),
        strategy=strategy,
    )
