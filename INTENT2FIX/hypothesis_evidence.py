"""Bounded repository evidence for the existing pre-Repair hypothesis Audit."""

import json
import hashlib
import re
from pathlib import Path

from pydantic import ValidationError

from INTENT2FIX.actions.schema import Observation
from INTENT2FIX.completion.structured_compactor import parse_json_object
from INTENT2FIX.intent_dialogue.developer_agent import (
    InvestigationSession,
    ReadOnlySearch,
    ReplaceInvestigationText,
    RunInvestigationCommand,
    WriteInvestigationFile,
)


class AuditInvestigationSession(InvestigationSession):
    def record(self, *args, **kwargs):
        kwargs["prefix"] = "AUDIT-" + kwargs.get("prefix", "SEARCH")
        return super().record(*args, **kwargs)


class HypothesisEvidenceTools:
    def __init__(self, docker_env, output_dir, max_actions=8, scout_probe_snapshot=None):
        self.env = docker_env
        self.output_path = Path(output_dir) / "hypothesis_audit_evidence.json"
        self.session = AuditInvestigationSession()
        self.remaining_actions = max_actions
        self.argument_refine_used = False
        self.actions = {
            action.__name__: action(docker_env=docker_env, session=self.session)
            for action in (ReadOnlySearch, WriteInvestigationFile,
                           ReplaceInvestigationText, RunInvestigationCommand)
        }
        self._assert_clean()
        self.restored_scout_snapshot = {}
        if scout_probe_snapshot:
            self.restored_scout_snapshot = {
                "evidence_id": scout_probe_snapshot["evidence_id"],
                "command": scout_probe_snapshot.get("command", ""),
                "restored_files": {}, "skipped_files": {},
            }
            writer = self.actions["WriteInvestigationFile"]
            for path, content in scout_probe_snapshot.get("files", {}).items():
                observation = writer.execute(writer.args_schema.model_validate({
                    "path": path, "content": content,
                    "purpose": f"Restore frozen Scout artifact from {scout_probe_snapshot['evidence_id']}",
                }))
                if observation.expect_correction:
                    self.restored_scout_snapshot["skipped_files"][path] = observation.message
                else:
                    self.restored_scout_snapshot["restored_files"][path] = hashlib.sha256(content.encode()).hexdigest()
            self._assert_clean()
            self._save()

    def _assert_clean(self):
        _, code = self.env.run_bash(
            "git diff --quiet && git diff --cached --quiet", timeout=30
        )
        if code != 0:
            raise ValueError("Hypothesis evidence must not modify tracked repository files")

    @property
    def evidence(self):
        exits = {row.get("evidence_id"): row.get("exit_code")
                 for row in self.session.actions if "evidence_id" in row}
        return [{**item.model_dump(), "observation": item.observation[:6000],
                 "exit_code": exits.get(item.id)}
                for item in self.session.evidence]

    def _save(self):
        self.output_path.write_text(json.dumps({
            "evidence": self.evidence,
            "actions": self.session.actions,
            "probe_files": self.session.investigation_files,
            "remaining_actions": self.remaining_actions,
            "argument_refine_used": self.argument_refine_used,
            "restored_scout_snapshot": self.restored_scout_snapshot,
        }, indent=2, ensure_ascii=False))

    def _refine_replace_arguments(self, source, completion_kwargs):
        """Locate the already requested edit; never generate a new probe change."""
        if (self.argument_refine_used or not isinstance(source, dict)
                or not all(isinstance(source.get(k), str) and source[k] for k in ("old", "new"))):
            return None
        files = self.session.investigation_files
        if not files or len(json.dumps(files)) + len(json.dumps(source)) > 24000:
            return None
        prompt = (
            'Fix only path/old arguments of an existing ReplaceInvestigationText call. '
            'Return {"path":"an existing probe path","old":"exact unique current block"}. '
            'Locate the block already requested; old may differ only in whitespace/comments, '
            'never code, strings, types, APIs, fixtures or assertions. Keep a supplied path. '
            'Do not return new or any new command. If impossible, return {}.\n'
            + json.dumps({"requested": {k: source[k] for k in ("path", "old") if k in source},
                          "current_probe_files": files}, ensure_ascii=False)
        )
        self.argument_refine_used = True
        import litellm
        kwargs = {k: v for k, v in completion_kwargs.items()
                  if k not in {"stream", "stream_options", "response_format", "tools", "tool_choice"}}
        kwargs.update(max_tokens=2048, max_retries=0, timeout=60)
        if "glm-5.3-flash" in kwargs.get("model", ""):
            kwargs["extra_body"] = {"thinking": {"type": "enabled", "clear_thinking": False},
                                    "reasoning_effort": "low"}
        try:
            response = litellm.completion(**kwargs, messages=[{"role": "user", "content": prompt}])
            patch = parse_json_object(response.choices[0].message.content or "")
            if not isinstance(patch, dict) or set(patch) != {"path", "old"}:
                raise ValueError("Expected only path and old")
            path, old = patch["path"], patch["old"]
            if (not isinstance(path, str) or path not in files
                    or (source.get("path") and path != source["path"])
                    or not isinstance(old, str) or not old or files[path].count(old) != 1):
                raise ValueError("Refined edit must uniquely match an existing probe")
            # Only comments, blank lines and trailing whitespace may drift.
            # Preserve code-line boundaries, indentation and literal contents;
            # multiline raw strings and line splicing need the full Audit.
            def code_lines(value):
                if 'R"' in value or "\\\n" in value:
                    raise ValueError("Complex literal requires the original Audit")
                clean = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/',
                               lambda m: " " if m[0].startswith(("//", "/*")) else m[0], value)
                return [line.rstrip() for line in clean.splitlines() if line.strip()]
            if code_lines(old) != code_lines(source["old"]):
                raise ValueError("Refine changed the intended code block")
            args = self.actions["ReplaceInvestigationText"].args_schema.model_validate({**source, **patch})
            self.session.actions.append({"action": "RefineInvestigationArguments", "status": "validated",
                                         "path": path, "new_unchanged": True})
            return args
        except Exception as error:
            self.session.actions.append({"action": "RefineInvestigationArguments", "status": "rejected",
                                         "error_type": type(error).__name__})
            return None

    def complete(self, completion_kwargs, prompt, submit_schema):
        import litellm

        self._assert_clean()
        submit = {"type": "function", "function": {
            "name": "SubmitHypothesisAudit",
            "description": "Submit the complete evidence-based audit and budget assignment.",
            "parameters": submit_schema,
        }}
        evidence_tools = [{"type": "function", "function": {
            "name": name, "description": action.args_schema.__doc__ or name,
            "parameters": action.args_schema.model_json_schema(),
        }} for name, action in self.actions.items()]
        instructions = """
You can now verify premises in the actual repository before submitting this
same Audit. Inspect the complete definition behind a decisive predicate, not
just its name or the candidate's explanation. For a disputed compile/runtime
prediction, use a standalone minimal probe on the exact original types and
call shape, with a control that separates the proposed cause from the symptom.
Do not write a product repair or change tracked repository files. A disposable
probe may model a proposed access/dispatch rule, but label what it proves and
what it does not. Do not use gold, hidden tests, git history, or network access.
Do not spend this budget rerunning a full project build or copying a test runner.
Check a proposed additive API's argument count against its scoped signature.
If a claimed outcome remains unverified, say so; do not manufacture success.
New AUDIT-* records are observations, not product intent. Cite them in the
original audit fields. Preserve the original fixtures and assigned predictions;
all corrections must be explicit REVISE feedback for upstream regeneration.
Use evidence tools before your first submission. The remaining tool budget is
shared with any retries and the final audit after revision. Submit the complete
audit when enough evidence exists or the tool budget is exhausted.
Call exactly one tool per response.
"""
        artifact_instructions = """
This is a fresh base checkout. Only the Scout snapshot listed below has been
restored, at the exact recorded RUN version; other historical probe paths and
build outputs are not present. Do not search for missing old probes. Inspect
or rebuild restored sources, or create a focused probe from the original Issue.
Restoration is not execution and does not establish that a prior result holds
for a different file version or a proposed repair. Preserve original fixtures.
Restored Scout artifact manifest:
""" + json.dumps(self.restored_scout_snapshot)
        messages = [{"role": "user", "content": prompt + instructions + artifact_instructions +
                     "\nPrior verification observations:\n" + json.dumps(self.evidence)}]
        # The action budget is shared across batches and audit retries.
        for _ in range(self.remaining_actions + 2):
            tools = [submit, *evidence_tools] if self.remaining_actions else [submit]
            response = litellm.completion(
                **completion_kwargs, messages=messages, tools=tools,
                tool_choice="auto", timeout=600.0, max_retries=0,
            )
            response_usage = getattr(response, "usage", None)
            if completion_kwargs.get("stream"):
                stream = response
                if (isinstance(response, litellm.CustomStreamWrapper)
                        and response.custom_llm_provider == "openai"):
                    # This installed wrapper drops reasoning-only chunks. Read
                    # the existing SDK stream before that lossy transformation.
                    stream = response.completion_stream
                try:
                    chunks = list(stream)
                finally:
                    if stream is not response and hasattr(stream, "close"):
                        stream.close()
                if not chunks:
                    raise ValueError("Audit returned an empty completion stream")
                # The installed LiteLLM builder mixes arguments across tool
                # indexes and drops simultaneous calls. Assemble each index
                # independently before any tool can execute.
                streamed_calls = {}
                content = []
                reasoning = []
                finish_reason = None
                usage = None
                for chunk in chunks:
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    finish_reason = chunk.choices[0].finish_reason or finish_reason
                    delta = chunk.choices[0].delta
                    content.append(delta.content or "")
                    reasoning.append(getattr(delta, "reasoning_content", None) or "")
                    for call in delta.tool_calls or []:
                        if call.index is None:
                            raise ValueError("Streamed audit tool call has no index")
                        combined = streamed_calls.setdefault(call.index, {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if call.id:
                            if combined["id"] and combined["id"] != call.id:
                                raise ValueError("Streamed audit tool index changed its call ID")
                            combined["id"] = call.id
                        if call.function:
                            for field in ("name", "arguments"):
                                combined["function"][field] += getattr(call.function, field) or ""
                calls = [streamed_calls[index] for index in sorted(streamed_calls)]
                if any(not c["id"] or not c["function"]["name"] for c in calls):
                    raise ValueError("Streamed audit tool call is missing its ID or name")
                response = litellm.ModelResponse(usage=usage, choices=[{
                    "finish_reason": finish_reason, "message": {
                    "role": "assistant", "content": "".join(content) or None,
                    "reasoning_content": "".join(reasoning) or None,
                    "tool_calls": calls or None,
                }}])
                response_usage = usage
            message = response.choices[0].message
            calls = message.tool_calls or []
            # Retain public submissions even when downstream schema validation
            # fails. Never persist provider kwargs or private reasoning text.
            with self.output_path.with_name("hypothesis_audit_responses.jsonl").open("a") as log:
                log.write(json.dumps({
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response_usage.model_dump() if response_usage is not None else None,
                    "content": message.content,
                    "tool_calls": [call.model_dump() for call in calls],
                    "reasoning_chars": len(getattr(message, "reasoning_content", None) or ""),
                    "remaining_actions": self.remaining_actions,
                }, ensure_ascii=False) + "\n")
            if not calls:
                if self.session.evidence:
                    return message.content or ""
                raise ValueError("Audit submitted without any repository verification")
            submissions = [call for call in calls
                           if call.function.name == "SubmitHypothesisAudit"]
            if submissions:
                if len(calls) != 1:
                    raise ValueError("Audit submission must be separate from evidence calls")
                if not self.session.evidence:
                    raise ValueError("Audit submitted without any repository verification")
                return submissions[0].function.arguments
            if len(calls) > self.remaining_actions:
                raise ValueError("Audit evidence action budget exhausted")
            assistant = message.model_dump(exclude_none=True)
            messages.append({key: value for key, value in assistant.items()
                             if key in {"role", "content", "tool_calls", "reasoning_content"}})
            for call in calls:
                self.remaining_actions -= 1
                try:
                    action = self.actions.get(call.function.name)
                    if action is None:
                        raise ValueError("Unknown hypothesis evidence tool")
                    source = None
                    try:
                        source = json.loads(call.function.arguments)
                        args = action.args_schema.model_validate(source)
                    except (json.JSONDecodeError, ValidationError) as error:
                        detail = (str(error) if isinstance(error, json.JSONDecodeError) else
                                  json.dumps([{"loc": e["loc"], "type": e["type"], "msg": e["msg"]}
                                              for e in error.errors()]))
                        observation = Observation(message="Tool arguments invalid; no action executed. " + detail,
                                                  expect_correction=True)
                    else:
                        observation = action.execute(args)
                    if observation.expect_correction and call.function.name == "ReplaceInvestigationText":
                        refined = self._refine_replace_arguments(source, completion_kwargs)
                        if refined is not None:
                            observation = action.execute(refined)
                    if observation.expect_correction:
                        self.session.actions.append({"action": call.function.name, "status": "needs_correction",
                                                     "message": observation.message})
                    self._assert_clean()
                finally:
                    self._save()
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": observation.message +
                                 f"\nEvidence actions remaining: {self.remaining_actions}"})
        raise ValueError("Audit did not submit within its bounded evidence budget")
