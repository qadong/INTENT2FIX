import json

from INTENT2FIX.intent_dialogue.schema import (
    ContrastiveIntentDecision,
    DeveloperAnswer,
    EvidenceSufficiencyReview,
    IntentBoundaryTestAssessment,
    IntentContractSemanticAudit,
    IntentEvidenceAdjudication,
    IntentEvidenceResolution,
    IntentInvestigationPlan,
    IntentInvestigationTask,
    IntentPropertyExtraction,
    IntentValidationRouting,
    RepairIntentContract,
    IntentSynthesisSelection,
    TestAnchorRetrieval,
    TestAnchorSelection,
)
from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame, IssueIntentUnit


def _schema(model_type) -> str:
    return json.dumps(model_type.model_json_schema(), ensure_ascii=False)


def build_intent_task_plan_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    reproduction: dict,
) -> str:
    return f"""You are the Evidence TODO Planner in an Intent Grounding pipeline.

The original Issue is authoritative. Plan only repository investigations that
can clarify or complete its user-visible repair contract. Do not decide the
intent yet and do not invent A/B hypotheses before searching the repository.

Return `mode: evidence_todos` and zero to ten non-overlapping TODOs. Zero is
correct when the Issue already specifies enough observable behavior and no
adjacent public contract needs checking. Ten is a hard upper bound, not a
target. Prefer fewer high-value TODOs over exhaustive coverage.

Each TODO must have a concrete evidence goal and one type:
- behavior_boundary: accepted inputs, outputs, state changes, or defaults;
- compatibility_boundary: an adjacent public behavior that may need preserving;
- representation_or_error_contract: public representation, error category, or
  diagnostic convention;
- acceptance_example: an Issue-entitled public scenario that can be checked;
- repository_convention: a documented or consistently exposed public API
  convention that gives meaning to an Issue term.

Assign IDs TASK-1, TASK-2, and so on. Leave `hypotheses` empty and
`discriminator` empty. `related_issue_ids` may reference only supplied claim
IDs. `related_intent_ids` must reference the atomic INTENT-N obligations whose
boundary the TODO could clarify. Do not merge unrelated INTENT-N units into one
generic TODO. `rationale` must explain how the missing evidence could change the
observable repair contract. `done_when` must describe sufficient evidence or
an explicit conclusion that the repository does not settle the question.

Prioritize evidence in public docs, tests, examples, public headers, adjacent
APIs, deprecation/compatibility policy, and small public probes. Current
implementation and graph structure may guide search, but they do not by
themselves define new product intent.

Do not create TODOs for root cause, patch design, files to edit, complete symbol
inventories, wrappers, generated mirrors, or build closure. Those belong to
Repair and Patch Reviewer. Do not ask whether an explicit Issue requirement
should be ignored. Do not broaden one named API to every similarly named type.

Good TODOs:
- Determine whether adjacent FILE-oriented public formatting APIs consistently
  place FILE* before the format argument, and record counterexamples.
- Determine which argument category and error wording are promised by nearby
  documented diagnostics for the exact operation named by the Issue.
- Determine whether a named counting operation is documented for one public API
  family or as a cross-family guarantee.
- Establish a minimal public example directly entailed by the Issue and one
  adjacent behavior that must remain unchanged.

Bad TODOs:
- Enumerate every overload, wrapper, generated header, or affected source file.
- Decide the true intent from the current buggy implementation.
- Find the smallest patch or implementation mechanism.
- Compare arbitrary alternatives that the Issue never leaves open.

Return one JSON object matching:
{_schema(IntentInvestigationPlan)}

Original issue:
{issue}

Grounded issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Reproduction attempt context:
{json.dumps(reproduction, ensure_ascii=False)}

Only treat reproduction context as validated failure evidence when
`reproduction_status` is `reproduced` and `evidence_validated` is true. When it
is `not_reproduced`, commands and outputs are investigation history only; they
cannot establish either observed or expected behavior.
"""


# Compatibility names for callers outside this optional module.
build_exploration_plan_prompt = build_intent_task_plan_prompt
build_executor_prompt = build_intent_task_plan_prompt


def build_repository_investigation_task(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    reproduction: dict,
    plan: IntentInvestigationPlan,
) -> str:
    return f"""Investigate the intent evidence TODOs in the live repository.

Original issue:
{issue}

Protected issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Reproduction context:
{json.dumps(reproduction, ensure_ascii=False)}

If `reproduction_status` is `not_reproduced`, its commands and outputs are
investigation history, not confirmed behavior. Files created by the Reproducer
are investigation artifacts, not original repository evidence.

Intent evidence TODO plan:
{plan.model_dump_json(indent=2)}

For each TASK-N, collect the most decision-relevant observations with stable
evidence IDs. Search public docs, tests, examples, headers, and adjacent public
APIs before following implementation details. Look for counterexamples instead
of treating the first matching pattern as a convention. You may create and run
a minimal standalone probe to check current public behavior. Temporary edits
may test a conjecture, but do not produce the repair patch.

Share prior searches, graph results, probe files, and command observations.
Submit one final report after every TODO is supported, contradicted, or shown
repository-unresolvable. Report repository facts and candidate preservation or
acceptance evidence; do not rewrite the Issue or decide the final intent.
"""


def build_test_anchor_selection_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    retrieval: TestAnchorRetrieval,
) -> str:
    return f"""You are the Test Anchor Reranker. Rerank the Recall@5 repository
test/example candidates for a later Investigator. You do not decide repair
intent, infer implementation scope, or design a patch.

Assess every retrieved candidate. Return at most three suitable candidate IDs
in strict semantic preference order and make the first one primary. Rank by
whether the test exercises the same public operation, has comparable
setup/input/state, and contains an observable oracle relevant to an Issue
claim. Prefer an anchor that can be changed along one observable dimension into
the Issue-requested case. For an Issue that adds a named public operation,
distinguish a behavioral contrast from evidence about the target API family. A
test for an already equivalent operation can anchor output/state invariants, but
merely renaming that call does not show that its family should gain a redundant
alias. Prefer a candidate from a family with an evidenced missing counterpart
whose naming and contract match the requested operation. When plausible
independent public families are represented in Recall@5, retain their best
candidates in the shortlist so the Investigator can compare them; do not let
one locally mutable test erase that family-level ambiguity. A symbol-name,
source-file, or call-graph match alone is insufficient. A useful analogy may be
selected when no focal test exists, but label that limitation. Do not claim a
candidate passes: the Investigator must execute the checked-out baseline before
it can become ANCHOR-1.

Use `no_suitable_candidate` when every snippet is unrelated or lacks a usable
public-behavior oracle. Include exactly one assessment for every retrieved
candidate so rejected alternatives remain auditable. Candidate retrieval
scores only control recall; they are not evidence of semantic fitness and must
not determine the reranked order by themselves.

When `intent_groups` are present, emit exactly one `intent_selections` entry for
every INTENT-N group. Select up to three candidates independently for that
atomic intent, using only IDs listed by its retrieval group. An empty selection
is valid when that intent has no faithful anchor. The legacy top-level shortlist
identifies the best first witness from a `target` group across all groups; a
`preserve` group supplies compatibility evidence and P2P candidates but must not
replace the Issue-guided target mutation. The top-level shortlist does not merge
or discharge the remaining intents.

Return JSON matching:
{_schema(TestAnchorSelection)}

Original issue:
{issue}

Protected issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Retrieved candidates:
{retrieval.model_dump_json(indent=2)}
"""


def build_poc_investigation_task(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    reproduction: dict,
) -> str:
    return f"""Investigate the Issue in the checked-out repository. Build an
evidence packet for Hypothesis, not a test-derived repair contract. Find the
original public trigger, observed result, causal decision or state transition,
and repository-connected scope. Preserve materially different explanations
when the evidence does not settle them.

Start with the Issue and focused source, tests, and public-contract searches.
Use the relation graph when an actual call, forwarding, ownership, or build
relation could change the cause or scope. Check a concrete premise rather than
repeating a broad search. Existing tests can show compatibility behavior, but
an old passing result is not automatically a post-repair requirement.
Once a concrete public entry and likely fault site are found, compare the
issue-relevant callers, siblings, shared helpers, and packaged forms before
calling the repair scope local. Follow a useful returned GREF to the next
definition, then confirm the relation in source or execution; if no graph edge
helps, use source search and explain why. A diagnostic POC cannot close this
scope question.
Treat this as a recall pass, not an edit list or a final repair judgment: retain every materially
Issue-related surface for which source or graph evidence shows a plausible
shared type, state, helper, public contract, or shipped representation. A
sibling need not call the fault site to share its failure mode. Inspect the
relevant body or build relation when obtainable; if still uncertain, record
the observed connection and the precise missing check in a separate
`candidate_boundaries` lead. Do not decide whether each lead must ultimately
be edited; Hypothesis will compare the mechanisms and complete repair scopes.
Drop a suspicious lead only when source evidence contradicts its proposed
connection, not because the first POC omitted it or no direct call edge exists.
Do not inventory unrelated graph neighbors.

When execution would clarify the reported failure, write the smallest faithful
POC and run it against the unchanged baseline. Prefer the Issue's own example
and oracle. A baseline-failing POC should fail for the reported behavior, not a
build/setup mistake. Preserve the Issue's essential input type, character
representation, and end-to-end public operation. When the Issue explicitly
requires an outcome, assert that outcome or return a meaningful failure code;
a program that merely prints the wrong value and returns zero has observed
the symptom, not verified the requested behavior. Keep the exact fixture and
command in the cited RUN-N so downstream Repair can reuse it as diagnostic
evidence; its baseline result is not a post-repair preservation requirement.
Preserve its supported build mode too. If a proxy changes one of these,
classify its failure as a fixture/compatibility question first; do not promote
that proxy's behavior into an Issue requirement or use it to discard another
repair scope. If the Issue leaves a default, opt-in API, wording, or
other outcome unresolved, observe the symptom without inventing an assertion;
it is acceptable to submit no failing POC. A passing comparison can be observed
in the same POC or an existing test when it answers a causal question. Do not
construct an ANCHOR-1/TARGET-1 pair or spend steps searching for a passing
anchor merely to satisfy a template. Never modify production files.
Do not keep refining a speculative fixture after it fails because the fixture
itself violates unrelated type or build requirements. Use a compatible
repository-supported example when one is available, or record that a faithful
POC is unavailable. Once the reported failure is located well enough to
explain it, spend the remaining investigation on the causal relation and
connected repair scope; a more elaborate POC does not close an uninspected
caller, sibling operation, or packaged definition.

Use `repository_facts` for cited SEARCH-N/RUN-N observations, and
`investigation_paths` plus `candidate_boundaries` for the causal and scope
relations that matter. Each `candidate_boundaries` entry is a possible scope,
not a requirement: name concrete observed forms, cite the relation, and state
what would establish or rule out their participation. Keep unresolved product
intent in `open_decisions`, separate from an unverified repository relation.
Set `contrastive_witness` to null and `acceptance_checks` to an empty list:
the POC is diagnostic evidence, not a frozen acceptance gate. Cite the POC's
RUN-N in the report when it is useful. Submit `partial` when a material fact
remains unknown; a truthful partial report is preferable to an invented oracle.

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Reproduction context:
{json.dumps(reproduction, ensure_ascii=False)}
"""


def build_contrastive_investigation_task(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    reproduction: dict,
    dynamic_evidence_tools: set[str] | None = None,
    anchor_retrieval: TestAnchorRetrieval | None = None,
    anchor_selection: TestAnchorSelection | None = None,
) -> str:
    dynamic_evidence_tools = (
        {"coverage", "xray"}
        if dynamic_evidence_tools is None
        else dynamic_evidence_tools
    )
    if dynamic_evidence_tools == {"coverage", "xray"}:
        dynamic_guidance = (
            "Coverage and XRay collectors are available. Use coverage for "
            "execution locations and XRay only for unresolved call order."
        )
    elif "coverage" in dynamic_evidence_tools:
        dynamic_guidance = (
            "Coverage collection is available; XRay is unavailable."
        )
    elif "xray" in dynamic_evidence_tools:
        dynamic_guidance = "XRay collection is available; coverage is unavailable."
    else:
        dynamic_guidance = (
            "LLVM dynamic collection is unavailable; use source and ordinary "
            "execution evidence only."
        )
    retrieval_context = (
        "No pre-retrieved test candidates were available. Search for an anchor "
        "using the repository tools."
    )
    if anchor_retrieval is not None and anchor_retrieval.candidates:
        retrieval_context = (
            "Pre-retrieved test/example candidates (ranking is a recall hint, "
            "not intent evidence):\n"
            + anchor_retrieval.model_dump_json(indent=2)
        )
        if anchor_selection is not None:
            retrieval_context += (
                "\n\nTest Anchor Selector recommendation (must still be inspected "
                "and baseline-validated):\n"
                + anchor_selection.model_dump_json(indent=2)
            )
        else:
            retrieval_context += (
                "\n\nThe selector was unavailable. Compare the bounded candidates "
                "yourself before searching beyond the shortlist."
            )

    return f"""Apply the Repository-and-Test Intent Evidence workflow from the
system prompt to this Issue. Your primary objective is to leave a factual packet
from which Hypothesis can construct every repository-grounded, Issue-compatible
repair interpretation without guessing how the checked-out code works. Do not
prematurely choose one cause or patch. Produce one auditable evidence report
with two independent channels:

1. Repository contract evidence: direct public docs, formal API declarations,
   explicit test/TODO contract text, and focused examples or behavior tests.
   Normative text can clarify the target or establish preservation; ordinary
   passing behavior can expose a compatibility risk but cannot create a new
   requirement.
2. An optional executable witness: at most one faithful passing anchor and its
   minimal Issue-guided failing mutation.

The repository channel remains useful when no honest executable pair exists.
Current implementation shape, graph proximity, and merely passing old behavior
are descriptive facts, not authority for a new requirement.
Do not turn the path selected on the failing baseline into desired behavior.
A foreign provider, overload, callback, route, or intermediate result that
reaches the error localizes the failure; it does not show that Repair must
preserve or accept that result family.

For every operation family whose breadth is explicitly authorized by the Issue
and supported by a repeated repository rule, inspect its public ingress, seed,
route, or dispatcher as well as its inner members. If an ingress uses the same
operation token or dispatch hook and admits the Issue's input domain, record it
in the same `candidate_boundaries` entry with direct source evidence. This is
the intersection of an evidenced family and its selection chain, not permission
to inventory unrelated callers. If the same selection rule could hijack direct
entry, prefer one direct-entry discriminator over additional equivalent sibling
samples. Do not invent an ingress when the checked-out repository shows none.

For the public source/header that contains the causal rule, inspect the native
build or test route far enough to identify which concrete representation it
actually compiles, links, imports, or packages. Record a separate atomic
packaging/build fact when the repository has a generated, amalgamated, vendored,
or otherwise mirrored representation that is shipped or independently built.
Also record when the available regression route exercises that mirror instead
of the editable source. This is provenance for later structural closure and
verification, not authority to require every nearby copy: do not infer a mirror
or add patch scope without direct repository/build evidence.

An executable pair is useful lower-bound evidence, not the stopping condition.
Before submitting, ask what repository fact Hypothesis would still have to
guess and whether another supported causal or scope interpretation would
disappear from this packet. Continue when that fact could change the causal
locus, public behavior, or repair scope. Stop when the remaining uncertainty is
a genuine intent choice whose alternatives are already factually grounded.
Inspect relationship-connected public, forwarding, generated, build, or
compatibility surfaces when they materially distinguish those alternatives;
do not inventory unrelated neighbors or equivalent examples.

Use the pre-retrieved shortlist as a starting point, not a semantic authority.
Before extracting a standalone witness, inspect and execute the selected
repository test or example in its native test/build context and confirm that it
passes on the checked-out baseline. Derive ANCHOR-1 faithfully from that exact
test slice: preserve its public setup, relevant fixture state, operation,
language/build assumptions, and oracle. If a standalone form cannot preserve
those conditions, reject that anchor instead of silently replacing them.
Preserve test semantics rather than copying its harness. Translate a focused
framework test into a self-contained source with its own entry point and simple
assertions or exit status. Never compile framework runners, framework
amalgamation sources, or repository test-main utilities into the standalone
probe. Execute the original framework test only through an existing repository
build target; if none is available, report that limitation instead of rebuilding
the harness manually.
The frozen command must remain replayable in a clean Repair container: it may
write only its output binary under /tmp and must not reference prebuilt /tmp
libraries, headers, generated files, include directories, or source files.
A freezeable compile command starts with the compiler and has exactly one
compiler invocation. Compile and link all required probe and product sources in
that invocation, optionally followed by `&&` running the binary. Do not prepend
`cd`, create intermediate objects, or append `echo`, pipes, or redirections.
Use literal command arguments only. `$()` command substitution, backticks, `;`,
and `||` make a RUN diagnostic-only and therefore ineligible for
`contrastive_witness` or `acceptance_checks`.
After one setup, compile, or link failure, classify the failure before editing
the probe again. A missing framework main or symbol means harness dependencies
remain and the probe must be simplified, not linked with more framework source.
Other missing declarations, unresolved product symbols, include paths, and
language-mode errors permit one focused inspection of the native target
configuration; repeated build-command guessing does not investigate repair
intent.

Create TARGET-1 by changing one observable predicate requested or clearly
entailed by the Issue. It must be self-verifying and fail on the baseline for
the Issue reason. Before freezing it, identify plausible wrong repairs for that
same cited claim: accepting only the sampled input, selecting the wrong public
lookup or dispatch behavior, changing a default instead of an opt-in path, or
losing state/error behavior required by that claim. When the anchor fixture and
public oracle can distinguish one of these without adding a separate Issue
claim, refine TARGET-1 so that it does. This is discrimination within one claim,
not permission to add unrelated requirements. Do not add assertions to settle a
different default, compatibility, representation, error, state, or API-family
decision; record those choices as limitations for adjudication.

When the Issue explicitly asks a public operation to participate in a lookup,
dispatch, customization, conversion, or extension mechanism, make that
participation observable whenever the repository permits it: construct the
same-claim target so the old path and the requested mechanism are both viable
but yield distinguishable public outcomes. Do not freeze the old winner as a
preservation rule merely because the baseline passes. An analogy named by the
Issue supports the behavior it explicitly invokes, not every precedence or
fallback rule of the analogous language construct or API. Do not invent an
opt-in API, new public type, or compatibility alternative unless the Issue or
checked-out public contract exposes that alternative concretely.
Treat a mechanism named directly by the Issue as part of the observable target,
not merely as an implementation suggestion. When candidate selection or
precedence matters, use a small independent compiler/runtime probe to establish
the mechanism's semantics and construct the target accordingly. Do not call a
language-defined selection rule an open product decision merely because the
baseline uses a different helper. An author asking whether the old behavior is
"by design" questions the baseline rationale; it does not authorize an opt-in
or gated substitute for an otherwise explicit target.
Keep a discriminator experimentally controlled: competing paths must use
equivalent public inputs and differ only in the semantic dimension named by the
Issue. Otherwise unrelated conversion, fixture, or environment differences can
masquerade as the requested behavior.
For a standardized language or API mechanism, distinguish protocol conformance
from fallback availability. A target where only the requested path is viable
proves availability but can be passed by an old-path-first fallback. When the
Issue names the mechanism and its public selection semantics are independently
observable, compare the requested and competing paths in one controlled fixture.

Do not silently choose a broad meaning for an Issue term and then add more
samples under that chosen meaning. If two plausible meanings would require
different public behavior, make that semantic choice explicit in
`discrimination_analysis`. A useful refinement contrasts those alternatives;
it does not merely add more members of one assumed input category. Existing
repository terminology may bound a current API's meaning, but it cannot be
redefined or promoted into the requested new contract without Issue or public
contract support.

Treat author-proposed causes or mechanisms in `implementation_suggestions` as
active hypotheses, not requirements. Inspect or probe them when they could
explain why the requested behavior is missing. A first compiler diagnostic can
establish one failure without disproving another author hypothesis. Report an
unverified suggestion as unresolved instead of silently dropping it. Compare
nearby public tests or examples when they reveal a useful contrast case, but do
not turn their implementation shape into patch scope. The pair is a local
lower-bound witness, not the full Issue contract or a patch-surface inventory.

Compile-time diagnostics can establish a contrast. For runnable probes, collect
dynamic evidence only when it answers a remaining question; it describes the
execution boundary but never dictates a patch location.

{dynamic_guidance}

Report direct public-contract observations and cite their SEARCH-N records even
if the pair cannot be constructed. Also report the exercised Issue claims,
selected source reference, exact mutation, preserved context, first observable
divergence, and unresolved choices when a pair exists. Record the shortlist
`anchor_candidate_id` when applicable. If the pair can be passed by a different
observable behavior that does not satisfy the cited Issue claim, first refine
its oracle when the distinction is already entailed by that same claim;
otherwise report the discriminating limitation and do not broaden TARGET-1 to
decide other claims. Leave `contrastive_witness` null for setup failures,
unsupported oracles, or an anchor that was not baseline-validated, but still
submit useful repository facts, open decisions, and decisive evidence IDs. A
later adjudicator independently decides what each evidence channel can
contribute to Repair.
Keep `repository_facts` atomic: use a separate item for each public entry/seed,
family rule, downstream symptom, packaging/build fact, or preservation surface.
Do not combine independent structural relations in one sentence, because later
coverage must include or exclude each one separately.

Bind the pair to `issue_intent_ids` as well as source claim IDs. Include only
INTENT-N units whose complete atomic behavior is exercised by the pair. A test
that demonstrates one unit cannot stand for another unit merely because both
come from the same Issue sentence or use the same API.

Original issue:
{issue}

Protected issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Reproduction context:
{json.dumps(reproduction, ensure_ascii=False)}

Test anchor retrieval context:
{retrieval_context}
"""


def build_next_intent_witness_task(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    intent_unit: IssueIntentUnit,
    anchor_retrieval: TestAnchorRetrieval | None = None,
    anchor_selection: TestAnchorSelection | None = None,
) -> str:
    retrieval_group = None
    if anchor_retrieval is not None:
        retrieval_group = next(
            (
                group
                for group in anchor_retrieval.intent_groups
                if group.intent_id == intent_unit.id
            ),
            None,
        )
    selection_group = None
    if anchor_selection is not None:
        selection_group = next(
            (
                group
                for group in anchor_selection.intent_selections
                if group.intent_id == intent_unit.id
            ),
            None,
        )

    return f"""Continue the same shared Intent Evidence trajectory for one
atomic Issue obligation that no accepted test pair has exercised yet.

Focused atomic intent:
{intent_unit.model_dump_json(indent=2)}

Produce an independent evidence report for this INTENT-N. Reuse prior searches,
probe files, build knowledge, and repository observations, but do not copy the
previous pair and relabel it. Inspect the candidates selected for this exact
intent first. If they are unsuitable, search for a closer public test or example.

When an honest executable contrast exists:
- baseline-validate the repository anchor in its native context;
- derive a faithful standalone ANCHOR-1;
- create TARGET-1 by changing only an observable predicate entailed by this
  focused intent;
- execute both independently;
- bind `issue_intent_ids` to `{intent_unit.id}` only, plus any other atomic
  intent that the same exact setup and oracle genuinely exercises in full.

This pair is independent from earlier attempts even though its local candidate
IDs are again ANCHOR-1 and TARGET-1. Do not claim that it settles another
INTENT-N through API proximity, shared implementation, or wording overlap. If
no faithful pair exists, submit `contrastive_witness: null` with the repository
evidence inspected and the concrete limitation. Do not spend the remaining
budget manufacturing a weak analogy.

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Recall candidates for this atomic intent:
{retrieval_group.model_dump_json(indent=2) if retrieval_group else "unavailable"}

Reranked candidates for this atomic intent:
{selection_group.model_dump_json(indent=2) if selection_group else "unavailable"}
"""


def build_evidence_sufficiency_review_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    report: dict,
    repository_evidence: list[dict],
    executions: list[dict],
    report_history: list[dict] | None = None,
    prior_adjudications: list[dict] | None = None,
    prior_reviews: list[dict] | None = None,
) -> str:
    return f"""You are the independent Evidence Sufficiency Reviewer between
Scout and Hypothesis. Do not choose a patch, infer the hidden reference change,
or prefer one unresolved product policy.

Answer one question: if Hypothesis receives only this evidence packet, can it
construct every repository-grounded, Issue-compatible repair-intent branch
without guessing a repository fact that could change the causal locus, public
behavior, or required scope?

Use `ready` when remaining uncertainty is a genuine intent or policy choice and
the packet contains enough factual support to state each live alternative. Use
`continue` only when one missing repository fact can still materially change a
branch. In that case return the single highest-value investigation question and
an existing anchor when available. Do not prescribe grep, graph, compilation,
or any other tool; Scout chooses how to answer it. Do not request exhaustive
inventory, stylistic implementation details, or evidence that cannot affect a
complete repair interpretation.

Review the report against raw observations rather than trusting its causal
summary. Pay particular attention to unexplained jumps between the original
public expression and a downstream symptom, distinctions between ingress-made
wrappers and inner receivers, private or copied helpers without a production
caller, competing fixes that imply different public contracts, and related
public or packaged forms connected by an observed forwarding/build relation.
These are examples of reasoning, not mandatory checklist items. Preserve every
supported alternative in `preserved_alternatives`, including alternatives that
share the requested final output but differ in causal locus or repair scope.
Check whether the report carried each material source-backed suspicious surface
into `candidate_boundaries` or found source evidence contradicting its proposed
connection. Do not require Scout to choose the eventual edit set.
An absent direct call to the fault site does not rule out an independent public
operation using the same type, state, helper, or shipped representation. Do not
accept a local-scope conclusion based only on that absent call; ask for the
most consequential obtainable source or execution fact instead. A candidate
boundary is a recall lead, not evidence that the edit is required.
When a GRAPH observation supplies a `scope_frontier`, treat its linked
definitions as questions, not as required edits. If the report claims a local
scope while an Issue-relevant connected public operation or packaged copy is
still unread, return `continue` for the single most consequential relation to
inspect or exclude. Do not demand inspection of every graph neighbor, and do
not promote an edge alone into a causal conclusion.
An owning class in that frontier may contain public members not listed among
the bounded search hits. A report's phrase "no other inspected operation"
does not establish that an Issue-relevant producer or consumer is absent. If
the reported failure concerns a representation shared across operations, ask
for one focused check of that possible sibling before closing local scope.
Reading only a related operation's name or signature establishes that it
exists, not whether it shares the reported failure. If an obtainable source or
execution fact could determine whether that material sibling uses the same
faulty representation or decision, request that fact before treating its
inclusion as a mere policy choice. Once the shared mechanism is established or
excluded, leave any remaining ambiguity about desired scope to Hypothesis.

Do not equate being able to describe several plausible branches with having
enough evidence to ground them. For each material branch, ask whether the raw
observations actually connect the original public input and receiver to the
state, selection, or forwarding decision that causes the behavior, and whether
the observed repository relations sufficiently bound the affected public,
sibling, mirrored, generated, or configured surfaces. A downstream failure
site, nearby analogue, or Issue-suggested implementation can motivate a branch,
but does not by itself establish that branch's cause or scope.

Before deciding, rehearse each live interpretation as a minimal repair-intent
branch without designing an edit. The branch must be able to state from cited
observations: the original public trigger and observable result; the causal
decision, state transition, or forwarding relation that a repair would address;
material default, error, fallback, and compatibility behavior; and the bounded
public or packaged scope connected to that cause. If any of those statements
would require guessing an obtainable repository fact, return `continue`. Do not
require every category when it is irrelevant, and do not turn this rehearsal
into a fixed checklist or a predicted patch.

Apply one counterfactual to each branch: if only its narrowest supported causal
locus changed, would an observed caller, sibling entry, wrapper, mirror,
generated form, configuration, or fallback remain inconsistent with the Issue?
Follow only relations suggested by the supplied evidence. A current passing
test is an observation and a possible compatibility risk, not automatically a
preservation requirement when the Issue intentionally changes that behavior;
hard preservation needs independent Issue, documentation, declaration, or
adjacent-contract support.

Examples of the judgment, not repository-specific rules:
- If an ingress returns a wrapper while only an inner receiver was exercised,
  a branch is not grounded until evidence connects the ingress-made wrapper to
  its actual return/error path; inner success alone cannot settle that branch.
- If one public declaration is forwarded into a packaged or generated form, a
  local implementation branch is not scope-complete until the observed
  forwarding/build relation shows whether that form changes with it or needs a
  separate compatibility action. Similar names alone do not create that scope.
- If a focused baseline test pins the old default while the Issue asks to change
  that default, do not preserve the old result merely because the test passes;
  inspect explicit modes, fallback behavior, or public contract evidence that
  determines whether it is a compatibility branch or a test to update.

Before returning `ready`, challenge the report with the most consequential
unanswered causal-or-scope question suggested by the Issue and observations. If
one obtainable repository fact could relocate the cause, contradict or remove a
branch, or materially add or remove an affected surface, return `continue` and
ask for that fact. Return `ready` when such questions are already answered or
the remainder is genuinely a product-policy choice. Exact future edit shape,
new symbol names, and exhaustive file inventory are not required.

When earlier focused Scout reports or adjudications are supplied, review them
as one cumulative investigation rather than treating only the last atomic
intent as the whole handoff. An adjudicator's unresolved/refinement finding is
a concrete critique, not an automatic veto and not authority over product
intent. Before returning `ready`, identify the evidence that resolved each
still-material critique; return `continue` if the critique exposes an obtainable
causal or scope fact that is still absent. Ignore a stale critique when later raw
evidence directly answers it. Do not relabel an uninspected discriminator as a
policy choice merely because the report can narrate both outcomes.

Return only JSON matching this schema:
{_schema(EvidenceSufficiencyReview)}

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Candidate Scout report:
{json.dumps(report, ensure_ascii=False)}

Focused Scout report history:
{json.dumps(report_history or [], ensure_ascii=False)}

Prior contrastive adjudications:
{json.dumps(prior_adjudications or [], ensure_ascii=False)}

Prior evidence-sufficiency reviews:
{json.dumps(prior_reviews or [], ensure_ascii=False)}

Executed contrast, when available:
{json.dumps(executions, ensure_ascii=False)}

Raw repository observations available to Hypothesis:
{json.dumps(repository_evidence, ensure_ascii=False)}
"""


def build_evidence_sufficiency_refinement_task(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    previous_report: dict,
    review: EvidenceSufficiencyReview,
) -> str:
    return f"""Continue the same Scout investigation after an independent
evidence-sufficiency review. The reviewer did not select a repair and did not
select a tool. Use repository search, execution, or the relation graph according
to your own judgment, starting from the supplied anchor when it is useful.

Investigate the one material missing fact below. Retain prior evidence and all
still-supported alternatives. Submit one complete replacement report when the
question is answered or when direct repository evidence shows that the choice
is genuinely policy-ambiguous. Do not collapse alternatives merely to finish,
and do not broaden into unrelated source inventory.

Evidence review:
{review.model_dump_json(indent=2)}

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Previous Scout report:
{json.dumps(previous_report, ensure_ascii=False)}
"""


def build_poc_intent_decision_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    report: dict,
    repository_evidence: list[dict],
) -> str:
    return f"""Review Scout's repository evidence before Hypothesis. The
original Issue is authoritative; source, graph, and POC observations explain
the current implementation and plausible causal/scope branches. A Scout-made
POC is not an acceptance oracle and its failure does not choose an unresolved
default, API shape, compatibility policy, or repair location.

Return the ContrastiveIntentDecision schema for compatibility with the existing
handoff, but there is no test pair: set witness_valid=false,
boundary_resolved=false, next_action=leave_open, acceptance_authority=illustrative,
all anchor/target IDs and test-derived deltas empty, and all witness coverage
lists empty. Leave followup_request null. Use repository_effects only for
independently supported public-contract qualifiers, preservation boundaries,
compatibility risks, or genuine open choices. Each effect needs an exact quote
from a cited repository observation. Do not promote a POC assertion or the
baseline's current path into a required outcome. Hypothesis will receive the
raw report and observations for causal and scope reasoning.

Return only JSON matching this schema:
{_schema(ContrastiveIntentDecision)}

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Scout report:
{json.dumps(report, ensure_ascii=False)}

Cited repository observations:
{json.dumps(repository_evidence, ensure_ascii=False)}
"""


def build_contrastive_intent_decision_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    reproduction: dict,
    report: dict,
    repository_evidence: list[dict],
    executions: list[dict],
    anchor_retrieval: TestAnchorRetrieval | None = None,
    anchor_selection: TestAnchorSelection | None = None,
    pair_status: str = "validated",
    prior_decision: ContrastiveIntentDecision | None = None,
) -> str:
    return f"""You are the sole Repository-and-Test Intent Evidence
Adjudicator. Judge two independent evidence channels without redesigning the
patch or deciding unspecified product policy.

The exact Issue spans and their typed claims outrank paraphrased INTENT-N text.
An INTENT-N is an indexing aid, not a second normative source. If its paraphrase
imports ordering, precedence, fallback, state, or error semantics from an Issue
example that the exact expected-behavior span does not state, remove that imported
qualifier. Preserve only the relation explicitly illustrated by the example and
use a direct probe of the requested mechanism itself for any remaining choice.
Resolve the Issue's rhetorical structure before comparing behaviors. When a
title or direct request says operation O must use mechanism M, while the body
mentions operation A only as an example that also uses M, M is the target and A
is illustrative evidence. Evaluate O against M's independent public semantics;
do not replace M with A's additional transformation, precedence, or fallback.
In the supplied typed claims, only `expected_behavior` can name the target
mechanism. A claim whose kind is `example` is never the Issue-named mechanism:
it may supply a fixture, but cannot define the oracle or a selection winner.

Repository contract channel:
- Emit zero or more `repository_effects`. Each effect must bind one statement
  to exact quotes copied from its cited SEARCH-N observations.
- Bind each effect to the INTENT-N units it actually clarifies or bounds. Leave
  `issue_intent_ids` empty only when the fact is useful context but does not
  resolve or constrain an atomic Issue intent.
- `target_qualifier`: direct normative repository text clarifies an observable
  part of the Issue target on a public surface already selected by the Issue,
  without replacing, weakening, or expanding an explicit Issue clause. A
  nearby API's documentation may clarify the meaning of an operation, but it
  cannot transfer the requested operation onto that API family.
- `preserve`: direct normative repository text establishes a public invariant
  that the requested change must retain.
- `compatibility_risk`: documented or directly tested public behavior could be
  changed accidentally. This is a warning, not a requirement.
- `open_decision`: cited public evidence exposes concrete observable
  alternatives but does not choose between them.
- Ordinary passing tests may support a compatibility warning. Without explicit
  contract text, they cannot create a target qualifier or hard preservation.
- When the Issue explicitly requires participation in a public lookup,
  dispatch, customization, conversion, or extension mechanism, a witness that
  makes the requested mechanism and the old path disagree can establish the
  target behavior. The old dispatch winner is not a preservation boundary just
  because it passes on the baseline. An Issue analogy transfers only the
  behavior it states, not all precedence or fallback rules of the analogue.
- Separate a named mechanism from one repository implementation of that
  mechanism. A private helper, trait, or adjacent operation may demonstrate a
  feasible repair hypothesis, but its extra ordering, precedence, fallback, or
  receiver policy is not a `target_qualifier` unless direct public contract text
  states that same observable policy for the Issue operation. When an Issue
  says "like X does Y" or uses X as an example of Y, only the explicitly named
  relation Y is authorized; do not import the rest of X's selection algorithm.
- Report an `open_decision` only when two concrete, observable, Issue-compatible
  alternatives are exposed by the Issue or cited public contract. Do not invent
  default-versus-opt-in, new-type, or compatibility branches from generic
  caution.
- An author question about whether the current absence is "by design" is not a
  second product alternative. When an exact Issue span already requests the
  behavior on the existing public operation, do not reinterpret that question
  as permission for a gated or opt-in API.
- When an explicit target names a language or public mechanism, that
  mechanism's essential externally observable participation is part of the
  target. A focused compiler/runtime probe may establish whether that
  participation occurs. It cannot decide among multiple precedence or fallback
  policies that all provide the requested participation unless the Issue or
  direct public contract also chooses one. Do not preserve or copy a baseline
  helper's stronger policy merely because it is nearby.
- Interpret participation at the public operation's selection point. A helper
  that consults the requested mechanism only when the old path is unavailable
  does not establish participation when the old path and requested mechanism
  coexist. For a directly named mechanism, that coexistence case is a missing
  discriminator, not an extra product-policy decision. This excludes bypassing
  the named mechanism without importing unrelated fallback rules.
- Same-named overloads, wrappers, or forms that cannot exhibit the named target
  mechanism are not semantic scope alternatives merely because their spelling
  matches. Leave their structural coverage to Patch Reviewer.
- This channel remains admissible when no valid test pair exists.
- Implementation structure, graph relations, baseline RUN-N behavior, and test
  similarity cannot independently create a requirement.
- Repository-supported `BOUNDARY-N` entries are intentionally broader than the
  likely repair. They expose plausible public forms for Repair to compare; they
  are not normative evidence for a `target_qualifier`. If multiple boundary
  families remain viable and the Issue does not name one, report the receiver
  choice as an `open_decision` or descriptive `scope_evidence`, not as a target.

Optional test-pair channel:
- Mechanical pair status is `{pair_status}`. Only `validated` permits
  `witness_valid: true`; any other status requires `witness_valid: false`.
- A valid pair may establish a local witnessed delta and, only when all public
  choices are resolved, a supported behavioral delta and executable check.
- Pair failure or absence invalidates only witness-derived content. It does not
  erase independently supported repository contract evidence.

Judge the evidence holistically:
- Does the cited repository reference genuinely establish the anchor behavior?
- Does the selected anchor agree with the bounded candidate comparison, or did
  the Investigator provide a sound reason to search beyond it? Retrieval rank
  is not semantic authority.
- Does the automatic source diff show that TARGET-1 changes only behavior
  requested or clearly entailed by the Issue?
- Do the ordinary executions show a controlled baseline pass/fail contrast for
  the Issue reason rather than setup failure?
- When coverage or XRay evidence is present, does it support a descriptive
  execution boundary without being promoted into a mandatory patch location?
- Which cited Issue claims does this single execution pair actually exercise,
  and which remain unresolved? Do not treat omission from the probe as evidence
  that an Issue claim is out of scope.
- Which atomic INTENT-N units does the pair actually exercise? Populate
  `covered_issue_intent_ids` only from the proposal's `issue_intent_ids`, and
  list every other Issue intent in `unresolved_issue_intent_ids`. Whole-claim
  coverage is not a substitute for atomic-intent coverage.

First set `witness_valid`: true only when the pair is a controlled,
Issue-related witness for at least one cited claim. A valid witness can still be
too local to specify the repair. Fill `covered_issue_claim_ids` and
`unresolved_issue_claim_ids` explicitly for a valid witness. Also emit
`witnessed_delta` as the narrowest observable difference established by this
exact source mutation and execution pair. It must describe only what the pair
demonstrates, without generalizing one input, literal, overload, wrapper, error,
or public path into the complete requested behavior. Always emit the field; use
an empty string when `witness_valid` is false.

Then set `boundary_resolved`: true only when the supplied Issue and public
contract evidence resolve every observable semantic decision stated in
`supported_delta`. Apply this counterfactual test: could TARGET-1 pass through a
different observable behavior that violates the cited Issue claim, or do
default-vs-opt-in, compatibility, representation, state, or error-contract
choices needed by that delta remain open? If either is true, set
`boundary_resolved` false. Do not consider files, wrappers, generated mirrors,
call-chain size, or patch location here; Patch Reviewer owns structural
completeness. A successful local target is not itself evidence that the full
behavior boundary is resolved.

Before emitting `supported_delta`, remove every clause that comes only from a
candidate implementation mechanism. If the witness establishes "mechanism M
participates" but does not compare or normatively settle M's precedence against
another viable path, the supported delta may require M's participation but must
not prescribe that precedence. Keep such implementation choices available to
Repair rather than presenting them as developer intent.
For a mechanism named directly by the Issue, do not count a conditional
fallback to M as participation at the operation's selection point. If the old
path can bypass M when both are viable, either refine the witness to distinguish
that coexistence case or retain it as an explicit validation gap; do not mark
the mechanism boundary complete from an M-only fixture.
When the Issue directly requires mechanism M and independent language or public
repository-contract evidence establishes M's observable semantics in the
supported configuration, a different mechanism that merely makes TARGET-1 pass
is not an equally supported product-policy alternative. Do not leave the
boundary open solely for that outcome-compatible alternative unless the Issue
or a direct public contract independently authorizes it.
Also remove clauses that merely preserve the path taken by the failing baseline.
When a foreign overload, provider, callback, route, or intermediate result wins
and the later rejection exposes the bug, that winner is causal localization,
not post-fix intent. Unless an Issue span or independent public contract says
that result family itself must be accepted, `supported_delta` may require the
Issue's end result but must not require downstream acceptance of the foreign
result. Keep an upstream selection repair available.

Choose one `next_action` after making that judgment:
- `accept_boundary` only when `boundary_resolved` is true;
- `refine_test_pair` when a focused same-claim executable contrast can resolve
  one remaining observable choice without inventing a requirement;
- `seek_repository_contract` when direct checked-out public contract evidence
  can resolve one remaining observable choice;
- `leave_open` when the checked-out repository cannot determine the choice
  without inventing maintainer policy;
- `reject_witness` when the pair is not a faithful Issue-related contrast.

Request a follow-up only when it can materially change the observable repair
boundary. More examples of behavior already established by the current pair
are not a valid reason to refine.
Prefer one follow-up over `leave_open` when the remaining question is the
breadth or composition of a finite repository-evidenced public family and one
bounded same-operation matrix, or direct contract search over that family, can
distinguish the alternatives. Do not request a repository-wide family survey,
and do not infer family scope merely because the first witness sampled one
member.

For `refine_test_pair` or `seek_repository_contract`, provide exactly one
`followup_request`. It must identify the affected claim IDs, why current
evidence is insufficient, what observation could resolve the choice, and what
Issue-authorized context must remain fixed. Do not request a patch,
implementation mechanism, file inventory, wrapper inventory, or
generated-artifact audit. A follow-up exists to recover repair intent, not to
perform Repair or Patch Review. For every other action leave
`followup_request` null.

Emit `supported_delta` only when `boundary_resolved` is true. Repository effects
remain independent of that all-or-nothing test-pair decision. Exact citation
quotes must occur verbatim in the supplied observation. Do not cite a RUN-N
baseline, implementation, graph edge, or test similarity as normative contract
authority. Behavior merely shared by ANCHOR-1 and TARGET-1 is not automatically
required after repair. Use `acceptance_authority: normative` only
when the Issue or an explicit public contract requires the target's exact
oracle. Otherwise use `illustrative`, including when a literal, diagnostic
wording, representation, or one selected public path merely demonstrates the
behavior. Emit optional diagnostic `scope_evidence` and material `uncertainty`.
`uncertainty` is reserved for choices exposed by a valid witness; express
repository-only open choices as cited `open_decision` effects. Do not convert
uncertainty into a non-goal or narrowed scope. When the witness is invalid,
leave selected IDs, `witnessed_delta`, `supported_delta`, covered/unresolved
witness claim and intent IDs, `scope_evidence`, and `uncertainty` empty. Independently
supported `repository_effects` may still be returned. When the witness is valid
but the boundary is unresolved, retain `witnessed_delta`, selected IDs,
coverage, repository effects, uncertainty, and scope evidence,
but leave `supported_delta` empty. Phrase uncertainty as the concrete unresolved
choice and its observable alternatives. The original Issue contract remains
authoritative.
Return JSON matching:
{_schema(ContrastiveIntentDecision)}

Original issue:
{issue}

Protected issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Reproduction context:
{json.dumps(reproduction, ensure_ascii=False)}

Investigator proposal:
{json.dumps(report, ensure_ascii=False)}

Repository observations:
{json.dumps(repository_evidence, ensure_ascii=False)}

Candidate executions, including exact commands, exit codes, and frozen files:
{json.dumps(executions, ensure_ascii=False)}

Pre-investigation test candidate retrieval:
{anchor_retrieval.model_dump_json(indent=2) if anchor_retrieval else "unavailable"}

Test Anchor Selector recommendation:
{anchor_selection.model_dump_json(indent=2) if anchor_selection else "unavailable"}

Prior adjudication from the same evidence trajectory:
{prior_decision.model_dump_json(indent=2) if prior_decision else "none"}

When a prior adjudication is supplied, return one complete replacement
decision. Retain its repository effects and local witness only when they remain
supported after considering the follow-up evidence; revise or remove them when
the new evidence directly changes their status.
"""


def build_contrastive_refinement_task(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    previous_report: dict,
    previous_decision: ContrastiveIntentDecision,
    pair_status: str,
) -> str:
    request = previous_decision.followup_request
    return f"""Continue the existing Intent Evidence investigation. The
Adjudicator found that the current evidence is useful but does not yet support
a complete observable repair boundary.

This is not a request to design or implement the production patch. Investigate
only the focused intent-evidence objective below, reuse prior SEARCH-N and RUN-N
evidence, and submit one replacement final report when the objective is
answered or shown repository-unresolvable.

Adjudicator action: {previous_decision.next_action}
Mechanical pair status: {pair_status}
Focused follow-up request:
{request.model_dump_json(indent=2) if request else "unavailable"}

If the action is `refine_test_pair`:
- keep the existing passing anchor, its public setup, and its cited RUN-N
  binding unless new evidence proves it invalid;
- replace TARGET-1 with a new independently executed target that distinguishes
  the observable alternatives named by the focused request within the same
  cited Issue claim;
- change the fixture or oracle along that discriminating dimension; do not
  merely add more examples that behave identically under both alternatives;
- when the Issue requires a lookup, dispatch, customization, conversion, or
  extension mechanism, prefer a target where that requested mechanism and the
  old path are simultaneously viable and observably disagree. Do not preserve
  the old winner without independent normative authority, and do not import
  unstated precedence rules from an analogy;
- if the Issue names the mechanism directly, first establish its observable
  selection semantics with an independent compiler/runtime probe when needed,
  then use those semantics in the same-fixture contrast. Do not turn the
  baseline helper's different precedence into an unresolved product policy;
- when the focused request concerns breadth or composition within one finite,
  already-evidenced public family, use one bounded matrix over its concrete
  members and combined modes instead of a sequence of one-off probes;
- do not add assertions for a different default, API family, representation,
  error, state, compatibility, or patch-surface decision;
- resubmit the full contrastive witness with the preserved ANCHOR-1 binding and
  the new TARGET-1 RUN-N binding.

If the action is `seek_repository_contract`, search direct public contract
evidence for the specified choice. Preserve the previous witness when it
remains valid, but do not manufacture a stronger target from descriptive
implementation behavior. If the repository does not settle the choice,
resubmit it as an open decision with the decisive evidence inspected. Prefer
the two strongest distinct direct sources: the named public API's documentation
or declaration contract, then an existing behavior test that asserts the exact
choice. Once both are silent, do not keep mining changelogs, unrelated APIs, or
implementation history to prove an absence; submit the bounded negative result.

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Previous Investigator report:
{json.dumps(previous_report, ensure_ascii=False)}

Previous Adjudicator decision:
{previous_decision.model_dump_json(indent=2)}
"""


def build_intent_boundary_test_task(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    contract: RepairIntentContract,
    decision: ContrastiveIntentDecision,
    routing: IntentValidationRouting,
) -> str:
    return f"""Continue the existing Intent Evidence investigation by turning
the already established observable contract into executable boundary checks.
This is test materialization, not another intent decision and not patch design.

The Validation Router has already selected how each property may be checked.
Do not override its mode or invent an executable check for `structural_review`
or `text_only`. Create zero to four small standalone public-API checks. Prefer
one check that covers a coherent matrix over several overlapping examples:
- `f2p`: directly encodes a confirmed target behavior, fails on the checked-out
  buggy baseline for that behavior, and should pass after a correct repair;
- `p2p`: directly encodes a confirmed preservation boundary, passes on the
  checked-out baseline, and must continue to pass after repair.

Only materialize properties and ROUTE-N plans already present below. An `f2p`
check must exercise at least one `role: target` PROP-N; a `p2p` check must
exercise at least one `role: preserve` PROP-N. Every submitted candidate must
list both the exact `covered_property_ids` and `covered_route_ids`. Do not turn
a compatibility warning, local witness, implementation fact, or open decision
into a required check.

Apply each executable route as follows:
- `generative_property`: exercise the routed evidence-backed dimensions and
  oracle over multiple generated values. Prefer an already available repository
  property framework such as RapidCheck. Never fetch dependencies. If no such
  framework is available, a deterministic finite generator or explicit product
  of routed representatives is acceptable only when it still covers every
  routed dimension. Do not silently collapse this route to one example.
- `fixed_example`: encode only the exact Issue or public-contract example and
  its stated observable result.
- `compile_matrix`: compile a bounded matrix of routed public types, overloads,
  language modes, macros, or configurations. The oracle is compile success or
  failure for each listed cell, not runtime output from one cell.
- `structural_review` and `text_only`: create no executable candidate. These
  remain non-executable evidence for later review or repair reasoning.

For generated checks, use a fixed seed and print the seed and counterexample on
failure. Preserve any framework-provided shrinking output. A generated check is
not faithful if its domain comes only from the implementation or if its oracle
chooses an unresolved policy. A fixed example is not evidence for an untested
domain. A compile matrix must include the routed cells rather than one convenient
instantiation.
Treat representatives as seeds for the routed evidence-backed partitions, not
as permission to repeat one fixture with different literals. A generative check
must vary every dimension named by its route, exercise boundary/equivalence
classes rather than cosmetic values, and include cells that reject the plausible
narrow repair identified by the route. When the route cannot supply such cells,
do not materialize it as a generative acceptance check.
For a routed lookup, dispatch, customization, conversion, or extension
mechanism, an independent old-path control is only a preservation check. It is
not a discriminator. The target check must make the old path and requested path
coexist on the same public input and produce distinguishable outcomes, when the
Issue or cited public contract establishes which outcome is required. If that
winner remains an open decision, omit the target check rather than silently
choosing it.
Do not choose an unspecified API name, overload family, default-versus-opt-in
policy, representation, diagnostic wording, state policy, or compatibility
policy merely to make a probe concrete. When a confirmed statement is not
executable without making such a choice, omit it.

Use WriteInvestigationFile for a new standalone source or script and
RunInvestigationCommand to compile/run it against the unchanged baseline. Do
not edit production files. Every check must be self-verifying: an F2P exits
nonzero on the baseline for the intended missing behavior; a P2P exits zero.
Submit one report with `contrastive_witness` null and an `acceptance_checks`
entry for each candidate. Set each entry's `kind` to `f2p` or `p2p`, cite its
RUN-N in `evidence_ids`, list the exact property and route IDs, and list only
Issue claim IDs and INTENT-N IDs that the exact check exercises. An empty
`acceptance_checks`
list is correct when no faithful executable route can be materialized.

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Proposed repair intent contract:
{contract.model_dump_json(indent=2)}

Validation routes:
{routing.model_dump_json(indent=2)}

Final evidence adjudication:
{decision.model_dump_json(indent=2)}
"""


def build_intent_boundary_test_refinement_task(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    contract: RepairIntentContract,
    decision: ContrastiveIntentDecision,
    routing: IntentValidationRouting,
    previous_report: DeveloperAnswer,
    assessment: IntentBoundaryTestAssessment,
) -> str:
    return f"""Revise the previously materialized Intent boundary checks using
the Semantic Admission Assessor's feedback. Continue in the same repository
and evidence trajectory. This is one bounded correction pass within the
existing investigation budget; it is not a new intent decision or patch task.

Preserve the proposed contract, PROP-N predicates, ROUTE-N modes, and all open
decisions. Do not create a stronger requirement to make a check executable.
Replace only candidates that the Assessor found non-normative,
non-discriminating, or scope-incomplete. The new candidate must directly rule
out the stated `surviving_wrong_repair` by supplying the stated
`missing_discriminator`. When the missing discriminator cannot be constructed
from the Issue and supplied public-contract evidence, omit that candidate.

Interpret an Assessor phrase such as "follow the Issue-named mechanism" from
the original Issue, not from a nearby analogy. If the title or direct request
names mechanism M and the body cites operation A merely as an example of M,
construct the oracle by independently evaluating M. Do not make the replacement
candidate follow A's extra precedence, transformation, or fallback behavior.
Treat typed `example` claims as fixture evidence only. Never describe an
example operation as the Issue-named mechanism or use its additional selection
rules as the replacement oracle.

For mechanism, dispatch, lookup, customization, conversion, or extension
behavior, two independent probes are not a discriminator. Put the requested
path and the plausible competing path in the same public fixture and make the
selected behavior observable. Keep an accepted preservation candidate only
when it remains useful; do not resubmit a rejected candidate unchanged.

Use WriteInvestigationFile and RunInvestigationCommand against the unchanged
baseline, then submit one complete report. Each replacement candidate must cite
its new RUN-N and retain the exact valid PROP-N, ROUTE-N, Issue claim, and
INTENT-N bindings. An empty candidate list is correct when the Assessor's gap
cannot be resolved without choosing an open product decision.

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Proposed repair intent contract:
{contract.model_dump_json(indent=2)}

Validation routes:
{routing.model_dump_json(indent=2)}

Final evidence adjudication:
{decision.model_dump_json(indent=2)}

Previous materialization report:
{previous_report.model_dump_json(indent=2)}

Semantic Admission feedback:
{assessment.model_dump_json(indent=2)}
"""


def build_intent_property_extraction_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    contract: RepairIntentContract,
    repository_evidence: list[dict],
    executions: list[dict],
) -> str:
    return f"""You are the Intent Property Extractor. Convert the supplied
Issue and repository investigation into a small ledger of observable behavior
properties that can guide repair and executable checks. This is one semantic
extraction call. Do not propose a patch.

Use three evidence roles:
- The Issue authorizes target predicates: behavior that should become true.
- Explicit repository public-contract evidence may establish preservation
  predicates and may clarify the public scope or terminology of a target.
- A passing/failing test pair is a local witness that a target is currently
  violated. It may illustrate a property, but one witness does not define the
  complete input domain, API surface, default policy, representation, state
  policy, or error wording.
- The receiver or facade used by a mutated nearby test is experimental setup,
  not Issue authority. When `BOUNDARY-N` entries expose multiple viable public
  families and no cited Issue span or normative public contract selects one,
  do not emit a property whose predicate or scope chooses that family. Keep the
  operation-level target in the protected intent ledger and leave executable
  routing to Repair-time discrimination.

Treat the test pair and repository investigation as parallel evidence channels,
not as successive authorities. The pair anchors the direction of one concrete
change; repository evidence may support additional dimensions or independent
preservation boundaries. Neither channel may erase an Issue target, and a
property check is validation derived from the resulting contract rather than a
new source of product intent.

A property is a compact, quantified behavior relation, not a restatement of one
execution. Before emitting one, perform this abstraction in order:
1. Identify the public operation and externally observable result, state, error,
   compile-time capability, or relation that matters to the user.
2. Identify the smallest evidence-supported domain over which the same relation
   is required. Do not write `all`, `any`, or `for every` unless the Issue or an
   explicit public repository contract supports that quantifier.
3. State the relation independently of the implementation and of the candidate
   repair. A diagnostic cause, private helper, internal trait value, or proposed
   representation is not a target property. For example, require the public
   concept or call to be valid, not a particular typedef that might make it so.
4. State an independent oracle. It may be an explicit public postcondition, a
   relation between before/after observations, a metamorphic relation, or a
   comparison with an independently established public result. Do not compute
   the expected value by calling the same behavior under test.
5. Apply a counterfactual discriminator: consider one plausible narrow or
   incompatible repair. If it would satisfy the predicate and oracle, refine the
   evidence-backed dimensions or keep the result as an exact example instead of
   claiming a broader property.

Return zero to ten atomic properties. Each property must state one externally
observable predicate, its public scope, its role (`target` or `preserve`), its
authority, and exact evidence IDs. Also state an implementation-independent
`oracle` when the evidence establishes what observable result would decide the
predicate. Leave `oracle` empty rather than inventing a missing expected value,
error wording, default, opt-in policy, representation, or compatibility rule.
Use `issue_explicit` for a property directly
entailed by a cited Issue claim. Use `repository_contract` only for preservation
behavior supported by a public contract or by an unaffected observable relation
whose continued validity is independently established. A currently passing test
is descriptive evidence, not by itself a preservation requirement: do not freeze
its exact value, wording, default, or dispatch choice when it exercises the same
behavior family that the Issue asks to change. Focused tests and examples may
corroborate a preservation boundary when their relation to the target is clear.
A repository helper or adjacent operation that implements the requested
mechanism can suggest a repair hypothesis, but cannot establish that the Issue
operation must preserve the helper's additional precedence or fallback. A newly
invented fixture with two viable dispatch paths likewise cannot create a
preservation oracle merely by passing on the baseline. Preserve that winner
only when pre-existing public documentation or tests directly assert the same
observable selection relation for the Issue operation.
A target that explicitly requires a public lookup, dispatch, customization,
conversion, or extension mechanism should express observable participation in
that mechanism when evidence provides a discriminator; it must not preserve the
old winner simply because the old path is callable. An analogy in the Issue does
not authorize importing unstated precedence or fallback semantics.
A mechanism named directly by the Issue is different from an analogy: its
externally observable language/API selection semantics belong in the target
predicate. A focused independent compiler/runtime probe may ground the
discriminator and oracle without becoming patch-location authority. Do not
treat same-named overloads that cannot exercise that mechanism as unresolved
target scope.
When the Issue's title or direct request names mechanism M and the prose cites
another operation A as an example of M, the property is conformance to M, not
equivalence to every selection rule of A. Use A only to identify an observable
fixture; use an independent evaluation of M as the oracle.
Only an `expected_behavior` claim can establish that target. A typed `example`
claim may guide fixture construction but cannot establish the oracle,
precedence, fallback order, or winning path.
Do not reduce a directly named mechanism to fallback availability. If an older
path can bypass it, include the coexistence case in the target predicate or
retain a validation gap until a focused discriminator establishes the public
selection behavior. This is still one target property, not a preservation
property for the old winner.
In particular, a well-formedness property over inputs where only the requested
path exists is not a complete abstraction of an Issue that says the public
operation *uses* a named selection mechanism. Add one bounded coexistence
dimension whose oracle is the independently evaluated mechanism result.
A target property must cite at least one Issue
claim ID. Do not derive a target solely from current implementation or a passing
test. Do not emit a property for an unresolved choice or silently select among
alternatives in `open_decisions`.
Cover every protected target INTENT-N with at least one target property whenever
that intent can be expressed as an externally observable predicate. An
unsupported `open_decision` must not erase a protected target. In particular,
an author asking whether the baseline is "by design" does not create a gated or
opt-in alternative to behavior explicitly requested on the existing operation.

Keep behavioral requirements separate from repair hypotheses. Do not emit both
"the public operation satisfies contract C" and "internal mechanism M has value
V" as target properties when M is merely one possible way to satisfy C. Do not
promote a guessed default, opt-in policy, exact diagnostic wording, return type,
or public surface family into a property. If repository evidence establishes a
shared public family or state protocol, express its observable relation and
scope; otherwise leave the decision open. Callability across an
evidence-established public facade/type family is observable compile-time
behavior and may be a property. Internal forwarding implementation, duplicate
declaration/definition sites, file mirroring, and generated artifact closure
remain structural-review concerns rather than runtime properties.

For each property, record zero to five evidence-backed dimensions such as input
partition, public type/API form, receiver state, configuration, value category,
or operation sequence. A dimension's `domain` states only the supported range
or categories; `representatives` are bounded values explicitly present in the
Issue or cited repository contract. Representatives seed validation but do not
by themselves define the domain. One witness alone supports an exact example;
one representative may seed a broader generator only when the Issue or an
explicit public contract independently quantifies that broader domain. Do not
fabricate a second value merely to make a property look generative. Keep
independent dimensions separate so later validation can build a product matrix
and include boundary partitions instead of repeating one fixture with cosmetic
literal changes.

Prefer relations that survive changes of concrete examples: acceptance across
an evidenced input partition, round-trip or order relations, before/after state
invariants, error-category relations, or compile-time capability across an
evidenced public type family. Exact outputs and exact diagnostic strings are
properties only when their authority pins those exact values. Otherwise state
the supported semantic relation or leave the oracle empty.

Write implementation-independent predicates such as "after operation X, the
observable value satisfies Y" or "existing input class X remains accepted".
Do not name source files, private helpers, candidate patch mechanisms, or
symbols merely because they appeared in a trace. Assign provisional IDs
PROP-1, PROP-2, and so on; the caller will normalize their order.
Bind every property to the exact `issue_intent_ids` it validates. One local
witness may justify a property for its bound intent without providing evidence
for any other INTENT-N. Leave the binding empty for repository context that
does not validate an atomic Issue obligation.

Return JSON matching:
{_schema(IntentPropertyExtraction)}

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Current repair intent contract:
{contract.model_dump_json(indent=2)}

Cited repository evidence:
{json.dumps(repository_evidence, ensure_ascii=False)}

Validated contrastive executions:
{json.dumps(executions, ensure_ascii=False)}
"""


def build_intent_validation_routing_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    contract: RepairIntentContract,
    repository_evidence: list[dict],
    executions: list[dict],
) -> str:
    return f"""You are the Intent Validation Router. Select one validation
form for every supplied PROP-N. This is one semantic routing call. Do not alter
the property, resolve an open product decision, propose a patch, or name files
to edit. Routing allocates validation only: it cannot raise or lower a property's
authority, remove an Issue target, or turn a passing baseline example into a
preservation requirement.

Choose exactly one mode per property:
- `generative_property`: runtime values vary over at least one evidenced domain
  and a stable executable oracle can decide every generated case;
- `fixed_example`: the authority establishes one or more exact examples, but
  does not justify a broader generated domain;
- `compile_matrix`: the observable contract varies over C++ types, overloads,
  macros, language modes, headers, or build configurations and is decided by
  compile/link success or a finite compile-time result;
- `structural_review`: the property concerns forwarding surfaces, generated or
  amalgamated artifacts, installation/export/configuration paths, or another
  structural closure that cannot be faithfully established by one standalone
  behavioral oracle;
- `text_only`: the property is useful repair guidance, but no sound executable
  or structural check follows from current evidence.

Use `generative_property` only when both the property dimensions and oracle are
grounded and the same oracle decides multiple meaningfully different cells in
the domain. At least one routed dimension must genuinely vary by value, state,
sequence, or public runtime type; changing only names, formatting, or copies of
one fixture is not property testing. Randomizing one Issue example is not
property testing. Use `fixed_example` rather than broadening an exact example.
Use `compile_matrix` for concepts, overload availability, public facade/type
forms, language modes, and other compile-time variation even if the resulting
executable could print a value. Public facade callability belongs here when its
family scope is evidence-established. Use `structural_review` for internal
forwarding implementation, declaration/definition synchronization, generated
mirrors, or distribution closure that has no public compile/runtime oracle.
Route unresolved policy to `text_only`; lack of an oracle is a normal outcome,
not permission to invent one.

When a target explicitly names a lookup, dispatch, customization, conversion,
or extension mechanism, route a mechanism-discriminating cell whenever the
evidence supports one. Put the requested mechanism and its plausible competing
path in the same public fixture and give them different observable outcomes.
Separate cells that only show each path is individually callable do not reveal
which path the operation actually selects. If the authoritative contract does
not establish the winner, route the unresolved choice to `text_only` instead
of turning one winner into an oracle.
The mechanism's own standardized selection semantics count as an authoritative
oracle when the Issue explicitly names that mechanism. Do not route a valid
coexistence cell to `text_only` merely because the Issue does not restate the
mechanism's standard selection rules.

For executable modes, copy or conservatively refine the supplied observable
oracle into `executable_oracle` and describe a bounded behavior-level
`generation_plan`. For a generative route, the plan must identify the varying
dimensions, the bounded generator or finite partition, the independent oracle,
and one plausible narrow repair that the generated cells should reject. Prefer
boundary and equivalence-class representatives over arbitrary random values.
For non-executable modes, leave `executable_oracle` empty and describe what
later review or reasoning can establish. Return one route per property, with
provisional ROUTE-N IDs.

Return JSON matching:
{_schema(IntentValidationRouting)}

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Repair contract with extracted properties:
{contract.model_dump_json(indent=2)}

Cited repository evidence:
{json.dumps(repository_evidence, ensure_ascii=False)}

Validated contrastive executions:
{json.dumps(executions, ensure_ascii=False)}
"""


def build_intent_boundary_test_assessment_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    contract: RepairIntentContract,
    decision: ContrastiveIntentDecision,
    routing: IntentValidationRouting,
    repository_evidence: list[dict],
    report: dict,
    executions: list[dict],
) -> str:
    return f"""You are the Semantic Admission Assessor for executable repair
intent checks. This is one evidence-grounded audit call, not a new intent
synthesis and not a patch review. The contract is a proposal, not an oracle.
Judge it against the original Issue spans, supplied repository observations, the
actual anchor/target decision, candidate source, and execution output.

For each candidate make three separate judgments:

1. `normative`: the exact observable oracle must hold under every behavior
   interpretation still compatible with the Issue and repository contract. A
   check that selects an unresolved default, opt-in API, representation, exact
   wording, state policy, or compatibility choice is not normative.
2. `discriminating`: the exact check rules out a plausible narrower or
   incompatible repair. Ask which wrong repair could still pass this source and
   oracle. If one survives, record it and the missing discriminator.
3. `scope_resolved`: the public receiver, API family, default/opt-in choice,
   signature, and state policy encoded by this exact source and oracle are
   settled by explicit Issue text or independent public-contract evidence.
   A passing anchor or Issue-mutated local witness proves behavior only on the
   form it executes; it does not prove that this form is the requested product
   surface. Repository-supported `BOUNDARY-N` entries are deliberately
   conservative upper bounds. When they contain multiple viable public forms,
   a check selecting one form has unresolved scope unless stronger evidence
   distinguishes it.

A check can be normative but incomplete after its scope is settled. For
example, an Issue may explicitly name one public receiver and exact example
while the candidate fails to cover an evidence-backed input partition on that
receiver. Keep `scope_resolved: true`, set `boundary_complete: false`, and name
the missing dimension. This is different from a local witness that happens to
use one of several viable receiver/API families: that check has
`scope_resolved: false` and cannot become a Repair requirement. Conversely, a
broad randomized probe does not become complete when it varies irrelevant
values inside one semantic partition.

The Issue remains the authority for the direction of the requested change. A
question-form Issue may still contain an explicit expected behavior. Preserve
an explicit F2P example when the Issue also pins its public receiver and oracle.
Do not infer that the receiver selected to make a nearby repository test
executable was itself requested. Downgrade when the candidate adds any choice
not entailed by the target, such as selecting one public API family, a default
type instead of an opt-in type, or one wrapper/facade among repository-supported
alternatives. Likewise, do not mark a P2P check normative merely because it
passes on the baseline; require independent evidence that its observable
relation lies outside the behavior the Issue is changing.
For every `repository_contract` preservation property and P2P candidate,
locate pre-existing public documentation or a repository test that asserts the
same observable oracle. Evidence that only shows a private helper's
implementation, an adjacent API's policy, ordinary baseline execution, or a
newly constructed ambiguous fixture is not independent authority for that
oracle. Such evidence may justify a compatibility warning or diagnostic probe,
but the candidate is not normative. This applies especially when the P2P fixes
which lookup, dispatch, customization, conversion, or fallback path must win
while the Issue asks to change that behavior family.

For an Issue that explicitly requires participation in a public lookup,
dispatch, customization, conversion, or extension mechanism, assess whether a
candidate makes that participation observable when the old path is also viable.
Do not admit a candidate that merely preserves the old dispatch winner or only
proves callability when both behaviors remain indistinguishable. Treat an
opt-in, new-type, or compatibility branch as open only when the Issue or cited
public contract actually exposes that concrete alternative.

Set `contract_consistent: false` when any proposed candidate requirement
contradicts another supplied clause or chooses a branch explicitly retained in
`open_decisions`. Set `boundary_complete: true` only when the combined
normative and discriminating candidates cover the Issue's evidence-backed
public behavior dimensions. This does not require covering every imaginable
input. When evidence is insufficient, abstain: use false rather than filling a
gap from model knowledge.

Audit the authority of `open_decisions` rather than treating their presence as
an oracle. Only an alternative concretely exposed by the Issue or cited public
contract can invalidate a target check. A question about whether baseline
behavior is "by design" does not create a gated/opt-in branch that overrides an
explicit target. Likewise, when the Issue directly names a language or public
mechanism, a focused compiler/runtime probe may establish that mechanism's
observable selection semantics; do not downgrade a faithful check because an
implementation helper currently uses different precedence. Same-named public
forms that cannot exhibit the mechanism are not missing semantic dimensions.
Determine which mechanism is directly named from the Issue's rhetorical
structure. If the title or request says operation O must use M and the prose
cites operation A only as an example of M, conformance to M is normative while
A is illustrative. A discriminator must compare O with an independent
evaluation of M, not with A's additional precedence or transformation rules.
Apply the typed-claim authority literally: `expected_behavior` may name the
target mechanism; `example` never does. If an assessment reason calls an
example the "Issue-named mechanism", that assessment is invalid. Use the
independent semantics of the expected mechanism instead.

Audit the supplied PROP-N predicates as well as their executable candidates.
Put a PROP-N in `diagnostic_property_ids` when its predicate itself selects an
unresolved policy branch or claims a public scope unsupported by its cited
authority. Every property covered by a check with `scope_resolved: false` must
be diagnostic when the property's predicate encodes that same unresolved
choice. Do not downgrade a valid explicit local requirement merely because
additional dimensions within its already-settled scope are missing; report that
as an incomplete boundary instead. Diagnostic properties remain visible in the
persisted extraction for audit but must not instruct Repair.

Evidence authority is scoped. Explicit Issue behavior can establish a target.
Public documentation, existing tests, and stable adjacent public APIs can
establish preservation or compatibility boundaries. A test pair proves only
the behavior it executes. Existing tests are descriptive unless the Issue and
public contract establish that their particular expectation must remain true.
Implementation and graph evidence explain mechanism
or surface relationships but cannot independently create product intent.

An F2P candidate must bind a target PROP-N and fail on the baseline for the
missing behavior. A P2P candidate must bind a preservation PROP-N and pass on
the baseline. `normative` and `discriminating` do not depend only on exit-code
direction; inspect the exact source and oracle.

The candidate must also faithfully implement every cited ROUTE-N. A
`generative_property` candidate must vary all routed dimensions over multiple
values and apply the routed oracle to each value; one hard-coded example is not
enough. A `fixed_example` candidate must not claim a broader domain. A
`compile_matrix` candidate must execute the routed finite compile cells.
Candidates citing `structural_review` or `text_only` routes are invalid because
those modes are intentionally non-executable. Reject source that reports
success without checking the routed oracle, ignores generated failures, or
tests only a strict subset while claiming the full route.

The evidence packet marks which observations the proposed contract already
cited. Uncited observations are included so you can detect an omitted public
behavior family; they are not automatically authoritative. Return one item for
every submitted `candidate_id`, preserving that exact CHECK-N ID, its RUN-N,
and its proposed kind. Multiple candidates may intentionally share one RUN-N;
their CHECK-N IDs keep their property and route judgments independent. Never
merge or overwrite them by RUN-N. Cite only
evidence IDs present below. Each item must cite the observations
used for its normative and discrimination judgments. `covered_property_ids`
must contain only supplied PROP-N IDs that the exact check exercises, and
`covered_route_ids` only matching ROUTE-N plans. `covered_issue_claim_ids` may
contain only supplied Issue claim IDs. Do not redesign production code. A
single concise missing discriminator is allowed because it is an evidence need,
not a patch instruction.

Keep a missing discriminator's experimental fixture separate from its oracle.
Name the competing public behaviors that must coexist and the observation that
would distinguish them. State which behavior must win only when an exact Issue
span or independently cited public contract already chooses that outcome. Do
not copy the winner from an analogous helper, passing anchor, current
implementation, or the Assessor's own expectation. Otherwise describe the
oracle as "follow the Issue-named mechanism" and leave implementation policy
out of the diagnostic lead.
When an Issue explicitly names a standardized mechanism, its independently
evaluated selection result is an oracle rather than implementation policy. A
check that proves only fallback availability is incomplete if an old-path-first
repair can still pass. In that case set `boundary_complete=false` and request a
single coexistence refinement. Do not return `boundary_complete=true` together
with a missing mechanism discriminator.

Return JSON matching:
{_schema(IntentBoundaryTestAssessment)}

Original Issue:
{issue}

Protected Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Proposed repair intent contract (may contain omissions or conflicts):
{contract.model_dump_json(indent=2)}

Contrastive decision and unresolved alternatives:
{decision.model_dump_json(indent=2)}

Validation routes:
{routing.model_dump_json(indent=2)}

Repository evidence packet (contract-cited observations first):
{json.dumps(repository_evidence, ensure_ascii=False)}

Submitted boundary-test report:
{json.dumps(report, ensure_ascii=False)}

Executed candidates:
{json.dumps(executions, ensure_ascii=False)}
"""


def build_intent_contract_semantic_audit_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    clauses: list[dict],
    evidence: list[dict],
) -> str:
    """Build the single final authority audit over the merged contract."""
    return f"""You are the final Repair Intent Contract Adjudicator.

The original Issue is the normative authority. Repository observations explain
existing public contracts and implementation facts. A passing test, current
implementation, graph edge, or locally mutated failing test is evidence only
for the behavior it directly observes; none of them independently decides a
new product policy.

Classify every supplied clause exactly once:

- `binding`: the exact observable behavior, scope, receiver/API choice, and
  oracle are required by an explicit Issue intent or an independently evidenced
  public contract. For an executable check, it must also distinguish the
  requested behavior from a plausible incomplete repair.
- `diagnostic`: the clause is useful localization or local witness evidence,
  but its exact literals, API spelling, representation, policy choice, or scope
  are not fully authoritative. Diagnostic clauses must not constrain Repair.
- `reject`: the clause contradicts the Issue, chooses an explicitly open
  alternative, invents a public API or policy, preserves the reported defect,
  or is unsupported by the supplied evidence.

Apply authority monotonically. Never weaken or replace an atomic Issue intent.
Do not promote a baseline sibling behavior to a preservation requirement merely
because it currently passes. Do not generalize one witness to an API family or
input domain unless the Issue or public contract independently establishes that
scope. A mutation may instantiate an Issue requirement, but a name, overload,
default/opt-in choice, exact wording, or representation introduced only by the
mutation is at most diagnostic. Conversely, do not reject a local requirement
when the Issue itself explicitly fixes that exact surface and oracle.
Treat the provider, overload, route, callback, or intermediate result observed
only on the failing path as diagnostic. Do not make its downstream acceptance a
binding clause merely because accepting it could satisfy the end-result probe;
binding needs an Issue span or independent public contract for that result
family. Likewise, when Issue wording says that the same defect or repair applies
similarly to named operations or all members of a family, preserve that scope
separately from the optional implementation recipe even when both share a
sentence or proposed-fix section.

Set `consistent` false when any candidate binding clause conflicts with another
binding clause or an unresolved decision. Cite only IDs present in the supplied
evidence or a clause's `source_evidence_ids`. Missing authority means abstain by
using diagnostic or reject; do not fill gaps from model knowledge. This call
audits the merged contract only. Do not propose patches, files, mechanisms, or
new requirements.

Return one JSON object matching:
{_schema(IntentContractSemanticAudit)}

Original Issue:
{issue}

Protected atomic Issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Merged candidate clauses:
{json.dumps(clauses, ensure_ascii=False)}

Bounded evidence packet:
{json.dumps(evidence, ensure_ascii=False)}
"""


def build_intent_resolution_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    plan: IntentInvestigationPlan,
    report: dict,
    repository_evidence: list[dict],
    candidates: dict,
) -> str:
    return f"""You are the Evidence-Grounded Intent Resolver.

Resolve each TASK-N only after considering the supplied repository evidence.
Return exactly one resolution per task. You may add a concise observable
qualifier to the repair contract, or select an exact supplied preservation or
acceptance candidate. You may not propose implementation work, files, symbols,
wrappers, generated artifacts, or a patch.

Set `resolved: true` only with one of these evidence types:
- explicit_contract: public docs, explicit policy, formal requirement, or a
  test/TODO that states the intended public contract;
- consistent_public_api_convention: multiple relevant adjacent public APIs show
  one coherent convention, no material counterexample was found, and applying
  it is compatible with the Issue;
- issue_example_alignment: a concrete Issue example plus repository evidence
  establishes a narrow observable boundary without generalizing beyond it.

Ordinary buggy behavior, a single implementation occurrence, call frequency,
graph proximity, absence of an API, or the easiest patch is insufficient.
Repository evidence cannot weaken, replace, narrow, or expand an explicit Issue
clause. If evidence leaves a real product choice open, return `resolved: false`,
`resolution_kind: no_effect`, and `evidence_type: insufficient`. Unresolved
alternatives are audit information and must not become Repair instructions.

For `behavior_qualifier`, write one concise public behavior in `contract_text`
and leave `contract_candidate_id` empty. For `preservation_boundary` or
`acceptance_check`, select an exact candidate ID and copy its exact text or
description into `contract_text`. Cite exact quotes copied from observations.
List at most three genuine alternatives considered; form them from evidence,
not before investigation.

Return JSON matching:
{_schema(IntentEvidenceResolution)}

Original issue:
{issue}

Protected issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Evidence TODO plan:
{plan.model_dump_json(indent=2)}

Investigator report:
{json.dumps(report, ensure_ascii=False)}

Repository evidence observations:
{json.dumps(repository_evidence, ensure_ascii=False)}

Selectable candidates:
{json.dumps(candidates, ensure_ascii=False)}
"""


def build_intent_evidence_adjudication_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    plan: IntentInvestigationPlan,
    report: dict,
    repository_evidence: list[dict],
) -> str:
    return f"""You are the Intent Evidence Adjudicator.

For each supplied DIM-N, make exactly one binary decision: did the checked-out
repository evidence disambiguate the dimension? You may not rewrite the Issue,
add hypotheses, create repair requirements, propose a patch, or choose the most
convenient implementation.

Return `disambiguated: true` only when the cited evidence makes exactly one of
the declared hypotheses the uniquely supported interpretation. Set
`selected_hypothesis_id` to that hypothesis and cite the evidence with exact
quotes copied from the supplied observations. Several evidence items may be
combined when their conjunction is necessary.

Return `disambiguated: false` when another declared hypothesis remains
reasonable, the evidence only describes the current buggy implementation, the
argument depends on code size/call frequency/similar names, or the repository
does not encode the maintainer's new product decision. In that case leave
`selected_hypothesis_id` empty. Explain the missing discriminator in `reason`;
evidence citations are optional.

Evidence authority rules:
- Explicit Issue-referenced policy, public documentation defining the ambiguous
  term, disabled/XFAIL/TODO text stating a future contract, explicit API
  equivalence, and formal concept/requirement declarations may disambiguate.
- Ordinary current behavior, passing tests for the old behavior, RUN probes,
  GRAPH relations, implementation structure, and absence of an API cannot by
  themselves determine a new intended behavior. They may corroborate an
  explicit normative contract but cannot replace one.
- An explicit Issue clause is protected. Repository evidence cannot eliminate
  an interpretation by contradicting or weakening that clause.

Return one decision for every DIM-N and no other prose. Return JSON matching:
{_schema(IntentEvidenceAdjudication)}

Original issue:
{issue}

Protected issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Declared intent version space:
{plan.model_dump_json(indent=2)}

Investigator report:
{json.dumps(report, ensure_ascii=False)}

Repository evidence observations:
{json.dumps(repository_evidence, ensure_ascii=False)}
"""


def build_developer_task(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    reproduction: dict,
    question: IntentInvestigationTask,
) -> str:
    """Compatibility wrapper for the legacy one-task API."""
    return build_repository_investigation_task(
        issue,
        issue_evidence,
        reproduction,
        IntentInvestigationPlan(tasks=[question]),
    )


def build_developer_followup_task(question: IntentInvestigationTask) -> str:
    """Compatibility prompt for callers still running one task per round."""
    return f"""Continue the shared repository-evidence trajectory for this TODO.

Reuse existing evidence, command results, graph queries, and probe files. Do not
repeat completed searches. Answer only what the repository establishes and do
not revise the Issue's explicit request.

Investigation TODO:
{question.model_dump_json()}
"""


def build_intent_synthesis_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    objectives: list[dict],
    objective_results: list[dict],
    candidates: dict,
) -> str:
    return f"""You are the Intent Evidence Synthesizer.

The original issue is authoritative and its explicit target behavior is
immutable. You cannot rewrite, narrow, expand, or replace it.
Default to an empty selection.

The runtime applies a candidate-level evidence gate after this selection.
Unresolved findings, warnings, and open decisions are audit records and do not
change what Repair sees. A partial report may expose preservation candidates
only when every TODO was completed as supported with repository evidence and
the report is partial solely because repository policy leaves a product choice
open. Select such a preservation only when it does not assume or resolve that
open decision. Executable acceptance candidates require a fully supported
report.

The investigation report's `open_decisions` are authoritative limitations on
repository evidence. Never select an acceptance candidate that resolves,
chooses a side of, or assumes an answer to an open decision. Open decisions stay
in the investigation artifact; they are not Repair requirements.

Normally select at most two preserve candidates and two acceptance candidates
across the entire investigation. Never select candidates merely because they
are well supported. Select one only when it fills an intent-relevant gap,
defines an observable boundary around an explicit Issue request, or rules out a
plausible incomplete repair. A contrast between alternatives is useful when it
arises naturally from the evidence, but is not required.

Reject candidates whose real purpose is to tell Repair where or how to edit:
- lists of files, declarations, classes, overloads, wrappers, forwarding paths,
  generated artifacts, template requirements, or internal call chains;
- claims that every sibling/public type must receive an API because names or
  navigation methods look parallel;
- implementation mechanisms such as arg mapping, parser dispatch, traits, or
  helper functions;
- the current buggy behavior, broad compatibility inventories, and behavior
  already stated explicitly by the Issue.
- a change to a documented default when the report also identifies a plausible
  additive or opt-in API and repository policy does not choose between them.

Among the remaining candidates, select only those that materially improve the
repair contract:
- Preserve candidates describe stable, observable public behavior directly
  adjacent to the requested change. They cannot preserve the reported bug.
- Acceptance candidates discriminate the requested behavior from a plausible
  wrong or incomplete repair, have an executable frozen probe, and exercise a
  scenario directly entailed by an explicit Issue target. Current behavior does
  not become intended post-repair behavior merely because a probe observes it.
  A candidate for a broad formal property is incomplete when it omits a concrete
  observable condition explicitly named in the Issue. Do not replace a named
  condition with a related abstraction unless explicit evidence establishes
  their equivalence.
- Warning candidates report a repository conflict or evidence limitation. They
  never create a new product decision and never override the Issue.

Do not close an ambiguous term by combining every repository mechanism into a
stronger requirement. Several mechanisms producing similar output do not prove
that one new public contract must cover all of them. Preserve the ambiguity
instead of selecting a broader acceptance candidate.

It is valid and preferable to select nothing when the investigation added facts
but no decision-relevant behavioral boundary. Structural completeness belongs
to Patch Reviewer, not this contract. Return candidate IDs exactly as provided;
do not generate new requirements or prose.

Return one JSON object matching:
{_schema(IntentSynthesisSelection)}

Original issue:
{issue}

Protected issue evidence:
{issue_evidence.model_dump_json(indent=2)}

Investigation TODOs:
{json.dumps(objectives, ensure_ascii=False)}

Investigation report:
{json.dumps(objective_results, ensure_ascii=False)}

Selectable candidates:
{json.dumps(candidates, ensure_ascii=False)}
"""
