from __future__ import annotations

import logging
from typing import ClassVar, Mapping, Type

from pydantic import ConfigDict, Field, PrivateAttr

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.agent.agent import ActionAgent
from INTENT2FIX.completion.base import BaseCompletionModel, LLMResponseFormat
from INTENT2FIX.exceptions import RejectError
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.intent_dialogue.developer_agent import (
    DeveloperGraphQuery,
    DeveloperSession,
    ReadOnlySearch,
)
from INTENT2FIX.message_history import MessageHistoryGenerator
from INTENT2FIX.model_config import get_model_config
from INTENT2FIX.node import Node
from INTENT2FIX.patch_review.schema import PatchReviewResult, SurfaceCheck
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.schema import MessageHistoryType


logger = logging.getLogger(__name__)


class SubmitPatchReviewArgs(ActionArguments):
    approved: bool = Field(
        ...,
        description="True only when no evidence-supported surface omission remains.",
    )
    report: str = Field(
        default="",
        description=(
            "Concise actionable report. Required when approved is false; name "
            "the missing surface and repository pattern that proves it."
        ),
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional review-level evidence IDs. Evidence cited by individual "
            "surface_checks is also accepted and persisted automatically."
        ),
    )
    surface_checks: list[SurfaceCheck] = Field(
        ...,
        description=(
            "Disposition for each concrete public surface whose relationship to "
            "the changed abstraction is supported by repository evidence. Use "
            "advisory when the relation is worth recording but does not prove a "
            "blocking omission. A required item must state the concrete public "
            "counterexample that remains wrong. Use [] when no such relationship "
            "is found."
        ),
    )

    model_config = ConfigDict(title="SubmitPatchReview", extra="forbid")


class SubmitPatchReview(Action):
    args_schema: ClassVar[Type[ActionArguments]] = SubmitPatchReviewArgs
    _session: DeveloperSession = PrivateAttr()

    def __init__(self, session: DeveloperSession, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    def execute(
        self,
        args: SubmitPatchReviewArgs,
        **kwargs,
    ) -> Observation:
        valid_ids = {item.id for item in self._session.evidence}
        cited_ids = list(args.evidence_ids)
        cited_ids.extend(
            evidence_id
            for check in args.surface_checks
            for evidence_id in check.evidence_ids
        )
        unknown = [item for item in cited_ids if item not in valid_ids]
        if unknown:
            return Observation(
                message="Unknown review evidence IDs: " + ", ".join(unknown),
                expect_correction=True,
            )
        if not cited_ids:
            return Observation(
                message=(
                    "Inspect at least one relevant repository pattern before "
                    "submitting the review, then cite its SEARCH-N or GRAPH-N ID."
                ),
                expect_correction=True,
            )
        persisted_evidence_ids = list(dict.fromkeys(cited_ids))
        if not args.approved and not args.report.strip():
            return Observation(
                message="A false review decision requires an actionable report.",
                expect_correction=True,
            )
        required_surfaces = [
            check.surface
            for check in args.surface_checks
            if check.disposition == "required"
        ]
        if args.approved and required_surfaces:
            return Observation(
                message=(
                    "approved=true conflicts with required missing surfaces: "
                    + ", ".join(required_surfaces)
                    + ". Reject the patch or revise the dispositions."
                ),
                expect_correction=True,
            )
        result = PatchReviewResult(
            approved=args.approved,
            report=args.report,
            evidence_ids=persisted_evidence_ids,
            surface_checks=args.surface_checks,
            actions=list(self._session.actions),
        )
        return Observation(
            message=args.report or "Patch surface review passed.",
            terminal=True,
            properties={"patch_review": result.model_dump()},
        )


class PatchSurfaceReviewer:
    def __init__(
        self,
        agent: ActionAgent,
        session: DeveloperSession,
        repository: Repository,
        *,
        max_steps: int = 50,
    ):
        self.agent = agent
        self.session = session
        self.repository = repository
        self.max_steps = max(2, max_steps)
        self.trajectory: Node | None = None

    @classmethod
    def create(
        cls,
        repository: Repository,
        docker_env: DockerEnv,
        code_index: CodeIndex | None,
        *,
        model: str,
        max_steps: int = 50,
        message_history_type: MessageHistoryType | None = None,
        **kwargs,
    ) -> "PatchSurfaceReviewer":
        model_config = get_model_config(model).copy()
        model_config.pop("compact_native_prompt", None)
        model_config.pop("message_history_max_tokens", None)
        model_config.pop("few_shot_examples", None)
        configured_history_type = model_config.pop("message_history_type", None)
        model_config.update(kwargs)
        completion_model = BaseCompletionModel.create(**model_config)

        if message_history_type is None:
            message_history_type = configured_history_type
        if message_history_type is None:
            message_history_type = (
                MessageHistoryType.MESSAGES
                if completion_model.response_format == LLMResponseFormat.TOOLS
                else MessageHistoryType.REACT
            )

        session = DeveloperSession()
        actions: list[Action] = [
            ReadOnlySearch(docker_env=docker_env, session=session),
        ]
        if code_index is not None:
            actions.append(
                DeveloperGraphQuery(code_index=code_index, session=session)
            )
        actions.append(SubmitPatchReview(session=session))

        system_prompt = f"""You are a read-only Patch Surface Reviewer.

The repair workspace is `{docker_env.work_dir}`. The candidate patch supplied
to you has been sanitized: harness probes, reproduction artifacts, and test-only
edits are excluded from it. Excluded files can still exist in this disposable
workspace for verification; ignore them and do not report findings about them.
Decide only whether the production patch
omitted an obviously related structural surface required by the protected issue
request or its explicit preservation boundary. Do not
re-evaluate product intent, redesign the repair, edit files, run Git history, or
seek future patches. Use repository searches and graph queries to inspect:

- parallel public APIs only when they forward to the changed abstraction or are
  explicitly covered by the issue;
- wrappers, result/reference adapters, and forwarding entry points that expose
  the changed behavior;
- for a newly added public operation with a clear existing sibling operation,
  the sibling's established declaration and forwarding surface, without
  assuming that every similarly named type must gain the operation;
- declaration/definition or source/header pairs;
- generated, amalgamated, or single-header mirrors when the repository has an
  explicit generation pattern;
- CMake, install, export, package, and configuration paths when the changed
  feature participates in those patterns;
- template parameter and policy forwarding already supported by the changed API.

Report only omissions supported by a clear repository pattern. Absence of a
current implementation is not proof that the issue excludes a new behavior.
Likewise, an accepted target probe is only evidence for the behavior it
exercises. An Issue detail or compatibility question left unresolved by that
probe is not an out-of-scope declaration. Preserve uncertainty: do not turn
"not established" into "must not change" or "need not support".
The decision boundary is a concrete counterexample, not confidence. If the
candidate leaves a source-backed public call, state combination, overload
direction, dispatch branch, declaration/definition pair, or generated mirror
observably inconsistent with the Issue, record that exact case in
`counterexample`, mark it `required`, and reject. It cannot remain advisory
merely because the primary witness passes. Advisory is reserved for a nearby
relationship for which no concrete wrong public observation is established.
Intent-routed surface questions in the task are prioritized investigation
leads, not required findings. Inspect each one against the candidate patch and
repository evidence; do not dismiss it merely because Intent left it open, and
do not require it merely because it was routed here.
For every concrete surface with an evidenced forwarding, generation, declaration,
or build relationship to the changed abstraction, include one `surface_checks`
entry in SubmitPatchReview. An open product decision by itself is not a surface
and must not be adjudicated here. Mark a surface `required` only when the
evidence shows that omitting it makes the Issue behavior inaccessible or
incorrect through an established public entry path, breaks declaration/definition
or generated-source synchronization, or leaves branches of the same semantic
overload/dispatch/error family observably inconsistent. A nearby sibling,
similar name, apparent symmetry, or forwarding pattern on another operation is
not sufficient by itself; record it as `advisory` unless the concrete user-visible
failure from omission is established. When that failure is established, include
it in `counterexample`; advisory is no longer valid. Advisory findings do not
reject a patch.

When changed code participates in overload selection, lookup, argument-category
handling, or error mapping, inspect sibling branches that share the exact helper,
trait, dispatch site, or public error contract. This check is about inconsistent
behavior within the same requested semantic family, not a demand to modify every
overload or wrapper. A `not_required`
disposition needs positive repository evidence of an intentional API
difference. The following are not sufficient evidence for `not_required`: the
wording uses a singular noun, the local probe did not exercise the surface, the
intent contract lists it as unresolved, or the current implementation lacks it.
When an established sibling operation is forwarded through a wrapper or
reference type and users normally obtain the changed abstraction through that
surface, inspect whether omission actually prevents that same requested
operation or behavior from working through the wrapper. Do not require the
wrapper solely to reproduce the sibling's shape.
For a newly added public operation, explicitly trace one closest established
sibling operation through every result adapter, value/document/reference
facade, declaration/definition pair, and generated public mirror. For each
surface, decide whether callers can hold the target receiver through that
surface and invoke sibling operations there. If yes, omission of the new
operation is blocking when it makes the requested behavior unavailable through
that established public entry path. The Issue need not name every forwarding
adapter, and an Intent open decision cannot downgrade this concrete
accessibility failure to advisory.
Judge accessibility at the same public abstraction and invocation level used by
the sibling operation. A caller being able to extract, convert to, or manually
obtain a lower-level receiver and call the new operation there does not make the
operation available on the wrapper or facade itself. Treat that workaround as
non-equivalent unless repository evidence documents an intentional asymmetry for
the target operation family.
Do not reject merely because a different implementation is possible. Existing
tests may be read as behavioral evidence, but missing test or documentation
changes are never a blocking finding. Do not invent sibling support or broaden
the requested product scope. Before asserting semantics from a helper, trait,
or wrapper name, inspect its definition or a direct behavioral test. Names and
proximity alone do not establish forwarding, customization, or dispatch rules.

Submit exactly one binary decision with SubmitPatchReview. Include
`surface_checks=[]` only when no evidenced adjacent structural surface exists.
Use approved=true when no evidence-supported required omission remains, even if
advisory checks are present. Use
approved=false with a concise report naming each missing surface and the
repository pattern establishing it.
"""
        message_generator = MessageHistoryGenerator.create(
            message_history_type=message_history_type,
            include_file_context=False,
            thoughts_in_action=False,
        )
        agent = ActionAgent(
            completion=completion_model,
            actions=actions,
            system_prompt=system_prompt,
            message_generator=message_generator,
            use_few_shots=False,
            disable_thoughts=True,
        )
        return cls(
            agent=agent,
            session=session,
            repository=repository,
            max_steps=max_steps,
        )

    def review(
        self,
        *,
        issue: str,
        repair_intent_contract: str,
        patch: str,
        file_context: FileContext,
        excluded_paths: Mapping[str, str] | None = None,
    ) -> PatchReviewResult:
        self.session.reset()
        excluded_lines = [
            f"- {path}: {reason}"
            for path, reason in sorted((excluded_paths or {}).items())
        ]
        excluded_section = "\n".join(excluded_lines) or "(none)"
        current = Node.stub(
            user_message=(
                "Review the candidate patch for structural surface omissions.\n\n"
                f"<original_issue>\n{issue}\n</original_issue>\n\n"
                f"{repair_intent_contract}\n\n"
                "<excluded_harness_paths>\n"
                f"{excluded_section}\n"
                "</excluded_harness_paths>\n\n"
                f"<candidate_patch>\n{patch}\n</candidate_patch>"
            ),
            file_context=file_context,
        )
        root = current

        for step in range(self.max_steps):
            try:
                self.agent.run(current)
            except RejectError as exc:
                if str(exc) != "No action found":
                    raise
                retry = Node.stub(
                    parent=current,
                    user_message=(
                        "Your response did not call a tool. Call exactly one "
                        "available tool now, or SubmitPatchReview if the review "
                        "is complete."
                    ),
                    file_context=current.file_context,
                )
                current.add_child(retry)
                current = retry
                continue
            if current.is_terminal():
                break
            if step == self.max_steps - 2 and current.observation:
                current.observation.message += (
                    "\n\nOne step remains. SubmitPatchReview now with your best "
                    "binary decision and evidence-supported report."
                )
            child = Node.stub(
                parent=current,
                file_context=current.file_context,
            )
            current.add_child(child)
            current = child

        self.trajectory = root
        data = (
            current.observation.properties.get("patch_review")
            if current.observation
            else None
        )
        if data:
            return PatchReviewResult.model_validate(data)
        return PatchReviewResult(
            approved=False,
            report=(
                "Patch Reviewer exhausted its step budget without submitting "
                "a decision. Treat the candidate as unreviewed."
            ),
            evidence_ids=[item.id for item in self.session.evidence],
            actions=list(self.session.actions),
        )
