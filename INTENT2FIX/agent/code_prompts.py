AGENT_ROLE = """You are an autonomous AI assistant with superior programming skills. As you're working autonomously, 
you cannot communicate with the user but must rely on information you can get from the available functions.
"""


def generate_compact_native_prompt(
    has_runtime: bool = False,
    soft_verification_finish: bool = False,
    source_only_search: bool = False,
) -> str:
    localization_tools = (
        """     * Locate the relevant module, definitions, callers, macros, and template uses with
       focused `grep` or `find` through `RunCommand`; inspect exact source ranges
       with `ViewCode`. Follow calls and state transitions in source, and use
       focused compilation when overload or template selection is uncertain.
     * If an identifier is absent from text search, inspect neighboring public
       implementations and declarations instead of assuming the API is absent.
   - Source search is a navigation lead, not proof of a call, required edit, or
     behavior. Verify the relevant source and call shape. Use `RunCommand` for
     focused search and `ViewCode` for substantial code reads."""
        if source_only_search else
        """     * Module: if ownership is unclear, use `EntityIntentSearch` to find candidate features,
       then resolve a relevant member with symbol search. A feature cluster describes existing
       code; it is not a namespace, a verified call chain, or the Issue's required behavior.
     * Structure: use `EntitySearchSymbol` for the relevant class or narrowly scoped namespace
       skeleton. Disambiguate by namespace, owning class, or file; refine truncated results by
       file or symbol. A shortlist does not enumerate the whole module.
     * Implementation: resolve relevant definitions with `EntitySearchSymbol` and read them
       with `ViewCode`. Use `EntityFunctionSemanticSearch` for a needed behavior whose symbol
       or location is unknown, with a known file scope if useful. Verify outgoing calls in source.
     * Impact: use `FindEntityUsage` to fill missing incoming caller/reference evidence for a
       changed shared function/type. It does not enumerate outgoing callees or all overloads;
       check the returned target and inspect relevant usages. For a concrete template question,
       use `FindTemplateInstantiation` with observed names/arguments, never an invented definition
       ID. For a named macro, use `GetMacroUsages` for its definition and indexed function users;
       use text search for preprocessor-only references. Compile to verify overload/template
       selection when needed; graph records alone cannot establish the selected C++ path.
     * A new API is absent from the base index: inspect its existing owner, public analogue,
       and forwarding pattern instead of repeatedly searching for the proposed name.
   - A search hit is a navigation lead, not proof of a call, required edit, or behavior. Verify the
     relevant source and call shape. Empty index results do not prove absence. Use these tools
     to complete the assigned Hypothesis's implementation, not to rewrite or reselect it. Do not
     call every tool or repeat evidence already established by Scout or the current trajectory.
   - If indexes miss an identifier, use `grep` or `find` via `RunCommand`; `rg` is unavailable."""
    )
    if has_runtime and soft_verification_finish:
        verification = """5. Verification is evidence, not a completion gate in portfolio mode. Run the supplied
   focused check and required ACCEPT-N checks when applicable. Once the production candidate is stable, run
   `RunRegression` so the selector receives repository-wide compile/regression evidence. Avoid repeating the
   full suite on an unchanged patch; if it exposes a likely regression and you revise production code, rerun it
   once for the revised state. Report every pass, failure, timeout, unavailable check, and limitation honestly.
   A legacy or intent-check failure may reflect an Issue-authorized behavior change or a stale hypothesis, so
   it is soft evidence: do not broaden the production patch merely to force it green. Once the implementation
   relationships and relevant preservation checks have been examined for the final production state,
   call `Finish` even when some checks remain failed, unavailable,
   or inconclusive; the candidate selector will compare that evidence against other independently generated
   candidates. Soft evidence is not permission to explain a failure away. A compile/link failure in an existing
   repository caller is a candidate defect unless the Issue explicitly removes that caller contract. Likewise,
   an executed preservation check may be classified as an authorized conflict only from the Issue or a cited
   public contract, never from the candidate's own assertion that the old behavior is stale.
"""
    elif has_runtime:
        verification = """5. Required intent checks are completion obligations. After the final production-code
   edit, run `RunSingleTest`, `RunRegression`, then call `RunIntentCheck` once for every required
   ACCEPT-N listed in `<repair_intent_contract>`, including all `[F2P]` and `[P2P]` checks. Use the
   exact listed IDs; their source and commands are frozen. Every required check must pass for the
   latest production state. If any required ACCEPT-N is unrun, stale, unavailable, or failed, do
   not call `Finish`; diagnose it and continue repairing production code. Any later production edit
   invalidates the previous results, so rerun every required ACCEPT-N. Diagnostic-only ACCEPT-N
   witnesses are optional and non-binding. If no required ACCEPT-N is listed, do not call
   `RunIntentCheck` merely to invent one.
"""
    else:
        verification = ""
    if source_only_search and soft_verification_finish:
        verification = verification.replace(
            "so the selector receives", "so the candidate pool records"
        ).replace(
            "the candidate selector will compare that evidence against other independently generated\n   candidates",
            "the evidence will be saved with this candidate for later comparison",
        )
    finish_step = 6 if has_runtime else 5
    finish_requirement = (
        "is implemented and the available final-state evidence is reported "
        "honestly. In portfolio mode, failed or missing checks do not by "
        "themselves prohibit Finish"
        if soft_verification_finish
        else "is implemented and every required final-state check above has passed"
    )
    immediate_finish = (
        "After examining the necessary implementation relationships and final-state "
        "verification evidence, call `Finish`. Report unresolved links and unavailable "
        "checks honestly; do not spend remaining portfolio steps repeating equivalent "
        "diagnostics or trying to satisfy an Issue-conflicting legacy expectation."
        if soft_verification_finish
        else "Once `RunRegression`, the focused `RunSingleTest`, and all required "
        "`RunIntentCheck` calls pass for the latest production state, call `Finish` "
        "immediately."
    )
    search_result_note = (
        "2. Use focused text searches as navigation leads; read the relevant `ViewCode` source before editing."
        if source_only_search else
        "2. Search results are skeletons; `ViewCode` implementations before editing."
    )

    return f"""
# Repair Policy

Fix the defect with the smallest complete production change. The Issue is authoritative; the
repair contract may clarify but cannot override or narrow it. Satisfy TARGET/PRESERVE clauses.
Do not treat an unresolved choice as excluded or as a new requirement. If code must choose,
follow the Issue and public contracts; use minimum semantic impact only as a
tie-breaker and state the assumption. Do not reduce required behavior to remove ambiguity.
On conflict, follow the Issue.

Enforce an ACCEPT probe's exact oracle only when normative. Illustrative literals, wording,
representation, and API path are non-binding. A diagnostic may expose only one requirement.

1. Localize from evidence with a top-down implementation map:
   - Before the first production edit, establish the relevant public operation -> owning
     type/module -> implementation or dispatch point -> affected callers/wrappers. Reuse cited
     Scout evidence that establishes these links; a file location alone does not establish them.
     Hypothesis root-cause explanations and coverage rationales are implementation leads, not
     evidence that these links are complete. For each relevant link, identify supporting source
     evidence or the concrete gap to inspect; include the resulting facts in the visible response.
     Fill missing links in the order below, starting at the deepest level already supported by
     source. This checks how to implement the assigned Hypothesis, not which behavior to choose.
   - When `<repair_intent_contract>` contains an `Immediate repair brief`, reuse it. Inspect the
     cited implementation location and closest public analogue. After checking the required
     implementation chain, form and test the smallest Issue-consistent candidate. Do not repeat
     anchor-test discovery, broad repository searches, or an already executed baseline witness.
     Extend localization when a required implementation, affected caller, overload, or definition
     remains unknown, or current observations contradict the supplied locations.
     After a candidate exists, use at most one listed discriminator when it can change that
     implementation. You are responsible for the completeness of the production change; do not
     defer necessary implementation checks to a later reviewer.
   - Work down the relevant implementation chain:
{localization_tools}
   - Before editing, include a brief factual map of source locations and unverified links in
     visible text alongside the next tool call. Do not put it in tool arguments or create a file
     for it. Reuse already established links without repeating searches. Stop expansion at the
     chain needed to implement and preserve the assigned behavior; do not map unrelated modules.
   - Use only the base checkout. Benchmark patches, future/upstream history, network sources,
     `curl`, `wget`, and remote Git are unavailable.
{search_result_note}
3. Only independent read/search calls may be batched. Never batch an edit, `RunCommand`, test,
   or `Finish` with another call.
   - Route commands by effect: use `StringReplace`/`CreateFile` for direct repository edits;
     use `RunCommand` without capture only for read-only search or inspection; use
     `RunSingleTest` only for a focused compile/test command that does not write tracked source;
     and use `RunCommand(capture_generated_changes=true)` only for a pre-existing,
     repository-owned artifact generator. If no such generator exists, edit the tracked mirror
     directly. Do not create a temporary transformation script and execute it as a test or use it
     to bypass the editing actions.
   - A generated artifact is not an authoritative edit surface. If it needs another change after
     a generator has run, locate and edit the source/template that the generator reads, rerun the
     same generator after the final source change, and verify the regenerated consumer. Compiling
     an artifact that a later repository build will overwrite does not establish build closure.
4. Make the narrowest mechanism-level production change.
   - When the Issue explicitly requires a lookup, dispatch, customization,
     conversion, or extension mechanism, implement that mechanism at the
     public operation's actual selection point. A helper that invokes the
     requested mechanism only after an older path is unavailable is not an
     equivalent implementation. When the Issue or a cited public contract states
     which path wins when old and requested mechanisms are both viable, use at
     most one focused probe to verify that precedence. When no authoritative
     winner exists, do not invent one or spend steps on competing-path probes;
     implement the explicit request with minimum semantic impact and record the
     unresolved precedence. Do not infer unrelated fallback policy.
   - Preserve public signatures, overload ranking, template deduction, iterator requirements,
     conversions, and exceptions unless changed by the Issue. For overloaded or templated APIs,
     inspect relevant directions, language branches, and generic abstractions. When changing a public return
     type, wrapper, or trait, compile representative existing call shapes found in the repository; a focused
     new-API probe cannot establish compatibility for unchanged callers.
   - When changing a lookup, dispatch, or customization selection point, a conflict witness proves
     which competing provider wins but does not prove that the old input domain still participates.
     Inspect existing callers/tests and run one preservation check for evidence-backed fallback-only
     categories admitted by the old path (for example built-in, member-only, or standard-library
     forms). Preserve those categories unless the Issue explicitly removes them.
   - A reduced lookup, overload, or dispatch probe is valid evidence only when it preserves the
     competing providers and enclosing call shape that trigger the Issue. A one-provider mechanism
     demo cannot outweigh a failing full-context Issue reproduction. If the full-context check still
     fails, use at most one reduced diagnostic to answer one named uncertainty. A passing reduced
     diagnostic is not repair success: next revise the production mechanism (or abandon it), then
     rerun the same full-context witness before creating or running another reduced variant. Repair
     that behavior or correct and rerun a behavior-equivalent full-context check.
   - Every additional production file or module must be justified by concrete evidence: a
     compiler/linker error, observed call chain, requested API, or generated mirror.
   - If an edit introduces unrelated errors, revert or narrow that edit rather than stacking
     adapters, buffers, branches, or dispatch layers .
   - Distinguish declarations used only for compile-time detection from functions selected and
     called at runtime. Every newly selected runtime helper must have a reachable definition.
     An undefined reference naming a helper introduced by the current patch is an implementation
     defect in that helper, not a reason to create another probe or adjust unrelated constraints.
   - Do not edit CMake, source lists, or build flags for compile/template/API failures unless the
     Issue concerns build integration or evidence shows a required source is absent.
   - Before changing a generic core header or multiple implementation modules, inspect a working
     sibling and run a focused public-API compile probe.
   - Do not modify repository tests, fixtures, outputs, or registration to evade failure. Change a
     test only when the Issue changes its contract; otherwise fix production code. Keep declarations,
     definitions, namespaces, headers, and amalgamation aligned. Verify `<issue_evidence>` forms.
   Prefer a supplied PoC. If none exists, create one repository-relative `repro*.cpp` for the
   required `RunSingleTest`. Frozen ACCEPT witnesses are separate contract checks; do not copy an
   illustrative witness's exact non-binding oracle into the PoC.

{verification}{finish_step}. Call `Finish` only when the smallest complete repair satisfying the Issue
   {finish_requirement}. Report optional diagnostic
   limitations honestly; do not broaden the repair merely to force a diagnostic-only witness green.
   {immediate_finish} Once these checks are complete, do not spend further steps on redundant
   searches, replacement probes, numbered probe variants, cleanup, or speculative validation.

"""


def generate_react_guidelines(disable_thoughts: bool = False) -> str:
    if not disable_thoughts:
        return """# Action and ReAct Guidelines

1. **Analysis First**
   - Review all previous actions and their observations
   - Understand what has been done and what information you have

2. **Document Your Thoughts**
   - ALWAYS write your reasoning in `<thoughts>` tags before any action
   - Explain what you learned from previous observations
   - Justify why you're choosing the next action
   - Describe what you expect to learn/achieve

3. **STRICT Single Action Execution**
   - You MUST run EXACTLY ONE action at a time
   - Choose from the available functions
   - NEVER attempt to execute multiple actions at once
   - NEVER plan next actions before receiving the observation

4. **Wait for Observation**
   - After executing an action, you MUST STOP
   - You MUST wait for the observation (result) to be returned
   - You MUST NOT plan or execute any further actions until you receive and analyze the observation
   - Only after receiving and analyzing the observation can you proceed with your next thought and action"""
    else:
        return """# Action Guidelines

1. **Analysis and Action**
   - Review previous actions and observations
   - Choose ONE specific action based on available information
   - Include your analysis and reasoning in the action itself

2. **STRICT Single Action Execution**
   - You MUST run EXACTLY ONE action at a time
   - Document your thoughts in the action
   - Choose from the available functions
   - NEVER attempt to execute multiple actions at once
   - NEVER plan next actions before receiving the observation

3. **Wait for Observation**
   - After executing an action, you MUST STOP
   - You MUST wait for the observation (result) to be returned
   - You MUST NOT plan or execute any further actions until you receive and analyze the observation
   - Only after receiving and analyzing the observation can you proceed with your next action"""


REACT_CORE_OPERATION_RULES = """
# Core Operation Rules

1. EVERY response MUST follow EXACTLY this format:
   Thought: Your reasoning and analysis
   Action: ONE specific action to take
   
   NO OTHER FORMAT IS ALLOWED.

2. **STRICT Single Action and Observation Flow:**
   - You MUST execute EXACTLY ONE action at a time
   - After each Action you MUST wait for an Observation
   - You MUST NOT plan or execute further actions until you receive and analyze the Observation
   - Only after analyzing the Observation can you proceed with your next Thought and Action

3. Your Thought section MUST include:
   - Analysis of previous Observations and what you learned
   - Clear justification for your chosen action
   - What you expect to learn/achieve
   - Any risks to watch for
   
4. NEVER:
   - Execute multiple actions at once
   - Plan next steps before receiving the Observation
   - Skip the Thought section
   - Deviate from the Thought -> Action -> Observation cycle"""

REACT_CORE_OPERATION_RULES_NO_THOUGHTS = """
# Core Operation Rules

1. EVERY response must follow EXACTLY this format:
   Action: ONE specific action to take

2. After each Action you will receive an Observation to inform your next step.

3. Focus on:
   - Using previous Observations to inform your next action
   - Choosing specific actions that progress toward the goal
   - Being precise and accurate in your actions
"""

SUMMARY_CORE_OPERATION_RULES = """
# Core Operation Rules

First, analyze the provided history which will be in this format:
<history>
## Step {counter}
Thoughts: Previous reasoning
Action: Previous function call
Observation: Result of the function call

Code that has been viewed:
{filename}
```
{code contents}
```
</history>

Then, use WriteThoughts to document your analysis and reasoning:
1. Analysis of history:
   - What actions have been taken so far
   - What code has been viewed
   - What we've learned from observations
   - What gaps remain

2. Next steps reasoning:
   - What we need to do next and why
   - What we expect to learn/achieve
   - Any risks to consider

Finally, make ONE function call to proceed with the task.

After your function call, you will receive an Observation to inform your next step.
"""


def generate_workflow_prompt(actions, has_runtime: bool = False, source_only_search: bool = False) -> str:
    """Generate the workflow prompt based on available actions."""
    if source_only_search:
        return """# Source-only Repair workflow

Use the original Issue and assigned Hypothesis to identify the relevant public
entry, implementation, affected callers, and valid uses. Search exact names and
phrases with focused `grep` or `find` via `RunCommand`; inspect source with
`ViewCode`. Verify call chains and template/overload behavior in source or by
compilation, not by assuming a search hit proves reachability. Use the editing
actions for production changes; use `RunSingleTest`, `RunIntentCheck`, and
`RunRegression` for final-state verification. Report failed or unavailable
checks honestly and call `Finish` once a nonempty candidate is ready."""
    search_actions = []
    modify_actions = []
    other_actions = []

    # Define search action descriptions
    search_descriptions = {
        "EntitySearchSymbol": "Locate indexed symbols or a namespace skeleton by name; disambiguate and inspect the selected source with ViewCode.",
        "EntityFunctionSemanticSearch": "Find ranked function candidates by implementation behavior when a function/helper name or location is unknown; inspect source before editing.",
        "EntityIntentSearch": "Locate a feature/module by existing-code behavior. Returns feature descriptions and member labels, not desired behavior or full source.",
        "FindEntityUsage": "Find indexed incoming callers/references for a selected function/class. Results are incomplete and do not enumerate outgoing callees or all overloads.",
        "GetMacroUsages": "Find a named macro's indexed definitions and function users; use text search for unindexed or preprocessor-only references.",
        "FindTemplateInstantiation": "Find recorded template instances/specializations by observed name or argument; this is not compiler instantiation or overload resolution.",
    }

    # Define modify action descriptions
    modify_descriptions = {
        "StringReplace": "Replace exact text strings in files. **CRITICAL:** `old_str` must NOT be empty. If the file is empty or you want to add new content, use `CreateFile` or `AppendString` instead.",
        "CreateFile": "Create a new file with complete content and explicit artifact_kind: production for an intended repository deliverable, temporary_probe for a disposable diagnostic excluded from submission. A registered temporary probe can be replaced or promoted with unchanged content; existing repository/production files cannot be overwritten.",
        "InsertLine": "Insert new lines at specific positions in files.",
        "AppendString": "Add content to the end of files. Use this for adding code to the end of a file or populating an empty file.",
        "ClaudeEditTool": "Use the str_replace_editor tool to update the code.",
    }

    for action in actions:
        action_name = action.__class__.__name__
        if action_name in search_descriptions:
            search_actions.append((action_name, search_descriptions[action_name]))
        elif action_name in modify_descriptions:
            modify_actions.append((action_name, modify_descriptions[action_name]))
        elif action_name not in ["Finish", "Reject", "ListFiles"]:
            other_actions.append(action_name)

    if not search_actions and not modify_actions:
        raise ValueError("No search or modify actions found")

    # --- MODIFICATION 1: Enhanced Workflow ---
    prompt = f"""
# Workflow Overview

1. **Understand the Task**
  * **Review the Task:** Carefully read the task provided in <task>.
  * **Identify Code to Change:** Analyze the task to determine ALL potential parts of the codebase that need to be changed. A bug or feature might span multiple files or functions.
  * **Identify Necessary Context:** Determine what additional parts of the codebase are needed to understand how to implement the changes. Consider dependencies, related components, and any code that interacts with the affected areas. Ensure you have a COMPREHENSIVE understanding before applying fixes.

2. **Locate Code & Explore (Top-Down Approach)**
  * **Step 2.1: Choose the strongest search anchor:** If the Issue names an exact symbol, diagnostic, file, or API, start with symbol search or focused `grep`. Use `EntityIntentSearch` when the Issue describes behavior without a reliable identifier and a high-level feature cluster would narrow the repository.
  * **Step 2.2: Understand Module Structure (Namespace Skeleton):** Once you've identified a relevant module or namespace, use `EntitySearchSymbol(type="namespace", name="...")` to get a **skeleton view**. This lists all classes and functions in that namespace without their full implementation, allowing you to see the "map" of the module at a low token cost.
  * **Step 2.3: Locate Implementation Sites:** Use the skeleton or semantic search to pinpoint targets:
      * Use `EntityFunctionSemanticSearch` to find specific functions when you only know the intent.
      * Use `EntitySearchSymbol` to find exact DEFINITIONS of classes, functions, or macros.
      * **Deduction & Refinement:** If a search returns too many results, refine it by adding a `file_pattern` or using a more specific sub-namespace (e.g., `fmt::detail` instead of `fmt`).
  * **Step 2.4: Trace Relationships & Side Effects:** Use `GetMacroUsages` for macros or `FindTemplateInstantiation` for templates. Explore context to understand the impact on callers.
  * **Step 2.5: Deep Dive:** Use `ViewCode` to retrieve the COMPLETE implementation of the specific targets you've located.
  * **Strategy: efficient_code_exploration**
      * **Search Before You Scroll:** Do NOT use `ViewCode` to blindly hunt for a definition inside large files.
      * Prefer targeted symbol searches first:
          * For macros/constants: use `EntitySearchSymbol(type="macro", name="...")`.
          * For types/templates: use `EntitySearchSymbol(type="class", name="...")` and then `ViewCode` to confirm the real definition/signature.
      * **Fallback Search (when symbol search fails):** use `RunCommand` to locate plain identifiers (e.g., `constexpr` variables, enums, or non-macro constants):
          * Use `grep -RInF -- "inline_buffer_size" include/`; ripgrep is unavailable.
      * **Analyze Compiler Errors Deeply:** If compilation fails, pay close attention to “redefinition” / “previous declaration” notes (often indicates duplicated code or unclean state) and template-parameter mismatch errors (stop and look up the original definition immediately; do NOT guess).
"""

    if search_actions:
        prompt += """
  * **Search Tool Reference:**"""
        for action_name, description in search_actions:
            prompt += f"\n      * {action_name} - {description}"

    if "ViewCode" in [a.__class__.__name__ for a in actions]:
        prompt += """
  * **Contextual Reading (ViewCode):** Only use when you need to see:
      * Additional context not returned by searches.
      * Specific line ranges you discovered (ViewCode will automatically provide +-10 lines of context).
      * Code referenced in error messages or test failures.
      * **LIMITATION:** There is a limit of 20,000 tokens per request. If the code is too large, specify smaller line ranges or search for specific symbols first.
"""

    if has_runtime:
        prompt += """
  * **Environment Inspection:** Use `RunCommand` (e.g., `ls -R`, `cat`, `find`) to explore the file system or check generated log files."""



    if modify_actions:
        prompt += """
  
3. **Modify Code**
  * **Fix Task:** Make necessary code changes to resolve the task requirements
  * **Prerequisite:** ALWAYS use `ViewCode` to see the current content of the file before using `StringReplace` to ensure your `old_str` is accurate.
  * **Rule: verifiable_file_modification (CRITICAL)**
      * **Read Before Write (Mandatory):** If you have modified a file earlier in this session, you MUST use `ViewCode` again to verify the file’s CURRENT content immediately before applying ANY further modification to that same file (StringReplace / AppendString / InsertLine / CreateFile overwrite).
      * Do NOT rely on memory of the file state from earlier steps. Assume it may have changed due to your own edits or previous patches.
  * **NOTE (Single-Include Consistency):** If the project has a `single_include` / amalgamated header, keep it consistent with the granular (individual) headers:
      * If you modify a `single_include` header, also update the corresponding granular headers.
      * If you modify granular headers, also update the corresponding `single_include` header.
      * When the repository provides a script or build target that derives committed artifacts from edited sources, run it with `RunCommand(capture_generated_changes=true)`. The tool synchronizes the current patch first and imports changed tracked text outputs into the patch; do not manually duplicate generated content.
  * **Apply Changes:**"""
        for action_name, description in modify_actions:
            prompt += f"\n    * {action_name} - {description}"

    # --- MODIFICATION 2: Added new Step 4 and re-numbered steps 5-9 ---
    if has_runtime:
        prompt += """

4. **Run Initial PoC Test**
  * If a proof-of-concept (PoC) test script is provided in `<Poc_test>`, first run it using `RunSingleTest` to confirm the initial bug (command is in `<Poc_command>`).
  * Frozen ACCEPT-N witnesses are separate contract checks. Run required ones with `RunIntentCheck`, but do not copy an illustrative witness's exact non-binding oracle into a PoC.
  * If no PoC is supplied, create one focused check for the required `RunSingleTest` before attempting a fix:
      * **Write an F2P (Fail-to-Pass) test**: it must **fail on the current buggy code** and **pass after your fix**.
      * Prefer a focused repository-relative standalone `repro*.cpp` verification artifact.
        Do not modify an existing repository test merely to create a PoC.
      * Keep it focused and deterministic. Avoid flaky tests, timeouts, randomness, and external dependencies unless required.
  * After creating (or being given) the PoC, run it with `RunSingleTest` and analyze the output to understand the failure. The `RunSingleTest` action is described in more detail in step 7.

5. **Test and PoC Policy**
 * The goal of verification in this phase is to reproduce the defect, distinguish the intended behavior, and detect regressions.
 * **CRITICAL REQUIREMENT:** Fix the **underlying project/source code**.
    * Do **NOT** modify existing regression tests to evade the problem or simply make failures pass.
    * An existing test may be updated only when the original issue explicitly changes that test's behavioral contract. Keep such an update minimal and do not weaken unrelated assertions.
    * You may refine a standalone PoC if it has technical errors, but must never weaken it to mask a remaining bug.
 * Adding or modifying a repository test is not a completion requirement. A focused standalone PoC is sufficient for `RunSingleTest`.
 * After any test edits, you **MUST** re-run `RunSingleTest` (PoC/F2P), then run `RunRegression` to confirm stability.

6. **Iterative Repair (When Test Failures Are Reported)**
  * If you receive test failure information, you **MUST**:
     * Analyze the failure information carefully to understand why the tests failed
     * Fix the **code** (NOT the tests) to make the failing tests pass
     * Run `RunRegression` to ensure no regressions
     * Run `RunSingleTest` to verify the previously failing tests now pass
     * Do not modify tests to make them pass. The only exception is an explicit Issue requirement that changes the test's behavioral contract.

7. **Verify Changes**
  * **Step 1: Run regression tests after each coherent repair iteration and after the final production edit.** This is your primary tool to ensure that your changes have not broken other parts of the application; do not spend a full regression run between incomplete edits that cannot yet compile.
    * **Action: `RunRegression`**
      * This action takes no parameters. It will automatically apply your latest modifications from the context to the container and run the full test suite.
      * **State Persistence:** The container environment is PERSISTENT. Intermediate build artifacts (like `.o` files or binaries) will remain in the container between actions.
      * **NOTE:** `RunRegression` does not replace the focused PoC. Use `RunSingleTest` for that verification.
      * Analyze the output carefully:
        * If all tests pass and the original issue is resolved, you can finish the task.
        * If regression tests FAIL, you MUST follow this decision process (do NOT “fix” by editing tests blindly to evade the problem):
          1) Identify whether the failure is a **real regression** caused by your code change (most common). If yes, fix the production code (project code) and re-run `RunRegression`.
          2) If the failure appears unrelated/flaky/environmental, reselect a more appropriate regression subset if available (or re-run once to confirm flakiness), and document why. Do not mask failures.
          3) Update a regression test only if the original Issue explicitly changes its contract. Update it minimally and verify the intended behavior with a focused `RunSingleTest`.
        * You MUST NOT reduce test strictness, disable failing tests, or modify assertions to get a green run without fixing the underlying project code.

  * **Step 2: Once all regression tests pass, run the reproduction scripts to confirm you have fixed the issue.** 
    * **Action: `RunSingleTest`**
      * **Persistence & Sync:** The environment is PERSISTENT. This action will sync your modified files from context to the container before running the command.
      * **Flexible Command:** Since the environment is persistent, you can rely on previous build steps. However, ensure your `command` is sufficient to run the test (e.g., `make my_test && ./my_test` for incremental builds).
      * If the intended repaired behavior is a controlled nonzero exit (for example, a test case is expected to fail with a specific report), set `expected_exit_code` and a precise `required_output_regex`. Do not rewrite the shell command to hide or invert its real exit status.
      * Use the detailed output from `RunSingleTest` to verify the issue is fixed:
          * If `<Poc_test>` is provided, re-run it after the fix. 
          * If you created your own PoC test (because none was provided), you MUST run that PoC after the fix. 
          * If `<repair_intent_contract>` lists required ACCEPT-N oracles, run each one with `RunIntentCheck`. Their commands and source files are frozen by Intent Grounding; do not replace them with self-authored verification. Correct a required-oracle failure before finishing. A diagnostic-only ACCEPT-N witness is optional evidence: run it only when it helps explain the current patch, and never broaden or redirect the repair merely to force its exact literal, wording, representation, or selected API path to pass.
      * **CRITICAL:** Ensure the success is driven by **source code** improvements, not by weakening tests.

  * **FINAL VERIFICATION:** After the final production-code edit, run `RunRegression`, `RunSingleTest`, and every required frozen ACCEPT-N oracle listed in `<repair_intent_contract>`. Do not call `VerifiedFinish` while any required ACCEPT-N is unrun, stale, unavailable, or failed. A later production edit invalidates prior intent-check results, so rerun every required ACCEPT-N. If no required ACCEPT-N ID is listed, `RunIntentCheck` is optional. A diagnostic-only witness is not a completion condition; do not broaden the repair merely to force one to exit zero.

  * **Strategy: Systematic Compilation Error Analysis & Localization (CRITICAL)**
      * When compilation fails, systematically analyze errors to locate the root cause:
      * **1. Parse Error Messages Systematically:**
          * **Read errors from top to bottom, but focus on the FIRST error** (later errors are often cascading from the first).
          * **Extract key information from each error:**
              * **File path and line number:** Note the exact location mentioned (e.g., `include/format.h:123:45`).
              * **Error type:** Syntax, type mismatch, template, linker, preprocessor.
              * **Symbol/identifier mentioned:** Function name, class name, variable, macro, etc.
              * **Error context:** The code snippet shown in the error message.
      * **2. Locate the Actual Problem (Not Just the Error Report Location):**
          * **Important:** The file/line in the error message may NOT be where the problem actually is. C++ errors often report at the point of use, not the point of definition.
          * **For type/template errors:**
              * If error says "no matching function" or "template argument deduction failed", the problem is likely in the **call site** (where you're calling it), not the definition.
              * Use `EntitySearchSymbol` to find the function definition, then `ViewCode` to see both the definition AND the call site.
              * Compare: expected signature (from definition) vs actual arguments (at call site).
          * **For "incomplete type" / "forward declaration" errors:**
              * The error location is where the incomplete type is used, but the fix is usually adding the correct `#include` at the top of that file.
              * Use `EntitySearchSymbol(type="class", name="...")` to find where the complete type is defined, then add the include.
          * **For "redefinition" / "multiple definition" errors:**
              * Check if you accidentally added the same code twice (see `verifiable_file_modification` rule).
              * Use `ViewCode` on the file mentioned to see if there are duplicate definitions.
              * Check if a header is included multiple times without include guards.
          * **For "undefined reference" (linker errors):**
              * The error is at link time, but the problem is a missing implementation.
              * Use `EntitySearchSymbol` to find the declaration, then search for the implementation (may be in a `.cc/.cpp` file).
              * Check if the implementation file is included in the build.
          * **For template instantiation errors:**
              * Template errors show the instantiation point, but the problem is often in the template definition or the types being used.
              * Look at the "required from here" chain to trace back to where the template is instantiated.
              * Use `ViewCode` to examine the template definition and the types passed to it.
      * **3. Use Tools to Navigate from Error to Root Cause:**
          * **From error message → code location:**
              * Extract file path and line number → Use `ViewCode(path="...", start_line=..., end_line=...)` to see the exact code.
          * **From symbol in error → definition:**
              * Extract symbol name from error → Use `EntitySearchSymbol(type="function"/"class", name="...")` to find definition.
              * Then use `ViewCode` to see the full definition and compare with usage.
          * **From macro error → macro definition:**
              * Extract macro name → Use `EntitySearchSymbol(type="macro", name="...")` or `GetMacroUsages` to find definition.
          * **From "previous declaration" → find the previous declaration:**
              * The error message often shows the previous declaration location. Use `ViewCode` to see it.
              * Compare the two declarations to see what's different (parameters, return type, template params, etc.).
      * **4. Common Error Patterns & Fix Locations:**
          * **"no matching function for call"** → Problem at call site: wrong arguments, wrong types, missing conversion. Fix: adjust call or add overload.
          * **"template argument deduction/substitution failed"** → Problem in template definition or types used. Fix: check template constraints and actual types.
          * **"use of undeclared identifier"** → Missing include or forward declaration. Fix: add `#include` at top of file.
          * **"redefinition of X"** → Duplicate definition. Fix: remove duplicate or use include guards.
          * **"incomplete type"** → Type used before definition. Fix: add include or move usage after definition.
          * **"undefined reference to X"** → Missing implementation. Fix: ensure implementation exists and is compiled.
      * **5. Verify Your Fix:**
          * After making a change, recompile to see if the error is resolved.
          * If new errors appear, they may be cascading from the original fix. Address them systematically.
          * If the same error persists, re-examine: you may have fixed the wrong location or misunderstood the error.

  * **C++-Specific Guardrails (CRITICAL):**
      * Do NOT “fix” regressions by weakening correctness checks: do not remove/relax assertions, change comparison semantics, or broaden expected ranges unless you can justify it as the correct behavior per spec and update/add tests accordingly.
      * Do NOT hide build problems: do not add `-w`, disable warnings-as-errors globally, or change build flags to silence failures. Fix the underlying code.
      * If you change a public header or the single-header amalgamation, ensure consistency (e.g., update corresponding granular headers when required by the project).
      * Avoid introducing undefined behavior (UB) just to satisfy tests; prefer safe, well-defined C++ constructs.
      * **Guideline: cpp_dependency_management (CRITICAL):**
          * **Exact Signature Matching:** Before writing ANY forward declaration for a class/struct/template, you MUST locate the original definition first (prefer `EntitySearchSymbol(type="class", name="...")` and then `ViewCode`). Never guess template parameters or namespaces.
          * **Use Forward Declarations Safely:** Only use forward declarations when the type is used purely as a pointer/reference or in function declarations. If you need the complete type (sizeof, by-value members, inheritance, template instantiation), include the correct header or move implementation to a `.cc/.cpp`.
          * **Avoid Circular Dependencies:** Respect the project’s include layering. Do NOT include high-level headers into low-level headers if it creates cycles. Prefer moving implementations to `.cc/.cpp`, and keep headers minimal.
          * **Overload Resolution:** When resolving overload ambiguity, first identify which component owns the distinguishing type or policy. Prefer an owner-local correction when evidence supports it; change a shared core header only when the behavior is genuinely shared.

  * **Note on `RunCommand`**: This is your versatile tool for environment inspection and broad searching. 
      * **Fallback Search:** Use `grep` via `RunCommand` whenever symbol-based search (`EntitySearchSymbol`) fails to locate a code snippet or usage.
      * **Usage Search:** It's the most reliable way to find all occurrences of a string or pattern.
      * **System Inspection:** Use it for `ls -R`, `find`, or checking log files and documentation.
      * **RESTRICTION:** Do **NOT** use `RunCommand` (e.g., `cat`, `grep -C 100`) to read large **source code** blocks or entire code files. ALWAYS use `ViewCode` for reading code. (However, `cat` is perfectly fine for inspecting logs, test outputs, or small non-code documents).
      * **CRITICAL RESTRICTION:** Do **NOT** use `RunCommand` (e.g., `cat > file`, `echo "content" > file`, `sed`) to create or modify files. You **MUST** use `CreateFile` to create new files and `StringReplace` (or other provided modification tools) to edit existing code. The only exception is a repository-owned script or build target that derives committed artifacts: invoke it with `capture_generated_changes=true` so changed tracked text outputs enter the patch.
      * Git mutation commands such as `git checkout`, `git restore`, or `git reset` are also prohibited in `RunCommand`. To undo an edit, inspect the current file and use a provided file editing action so the reversion is recorded in the current patch context.
      * It does NOT reset the environment between calls.
      * **Do NOT use `RunCommand` to run tests.** Always use `RunRegression` or `RunSingleTest` for testing and verification.

8. **Iterate as Needed**
  * Based on the test results, loop back to step 3 to make further code modifications until `RunRegression` passes.

9. **Complete Task**
  * Use `Finish` when the smallest complete repair for the original Issue and TARGET/PRESERVE clauses is implemented and all runtime-required checks pass. Report optional or unavailable diagnostics honestly; never claim a required check passed when it did not.
  * **MANDATORY:** You MUST provide a detailed `test_verification` in the `Finish` action, explaining:
      1. Which verification commands/checks were run (`RunRegression`, `RunSingleTest`, and any `RunIntentCheck`).
      2. How they confirm the fix in the **project source code** (not by modifying the tests themselves).
      3. How they confirm no regressions were introduced.
  * Explain why the task is complete and how your changes address the problem."""
    
    # This 'else' block handles the case where no runtime/testing is available.
    else:
        prompt += """

4. **Iterate as Needed**
  * Continue the process until all changes are complete.

5. **Complete Task**
  * Use Finish when confident all changes are correct and complete."""
    # --- END: Corrected workflow ---

    return prompt


def generate_guideline_prompt(has_runtime: bool = False, thoughts_in_action: bool = True) -> str:
    prompt = """
# Important Guidelines

 * **Focus on the Specific Task**
  - Implement requirements exactly as specified, without additional changes.
  - Do not modify code unrelated to the task.

 * **Code Context and Changes**
   - Limit code changes to files in the code you can see.
   - If you need to examine more code, use ViewCode to see it.
   - **Sufficient Exploration:** Inspect the evidence needed to justify the repair and its directly affected public surfaces. Stop broad exploration once those questions are answered.
"""

    if has_runtime:
        prompt += """

 * **Testing**
   - Tests run after you make code change.
   - If tests fail, analyze the output and do necessary corrections.
   - **CRITICAL:** Fix the **underlying project code** to solve the problem. Do NOT “fix” failing regression tests by editing or disabling them to evade the issue or make the run green.
     * Only update an existing test when the original Issue explicitly changes that test's behavioral contract; verify the intended behavior with a focused `RunSingleTest`.
     * Never mask failures by skipping tests, commenting out assertions, weakening expectations, or changing outputs without strong evidence."""

    if thoughts_in_action:
        prompt += """

 * **Task Completion**
   - Finish the task only when the task is fully resolved and verified.
   - After the final production-code edit, run `RunRegression`, `RunSingleTest`, and every required `RunIntentCheck`; required checks must pass. Do not finish while a required check is unrun, stale, unavailable, or failed. Diagnostic-only intent checks are optional and non-binding.
   - Do not suggest code reviews or additional changes beyond the scope.

 * **State Management**
   - Keep a detailed record of all code sections you have viewed and actions you have taken.
   - Before performing a new action, check your history to ensure you are not repeating previous steps.
  - **Verifiable File State:** If you are about to modify a file that you already modified earlier in this session, you MUST re-run `ViewCode` on that file immediately beforehand to confirm its current content and avoid duplicated/contradicting edits.
   - Use the information you've already gathered to inform your next steps without re-fetching the same data.
"""
    else:
        prompt += """

 * **Task Completion**
   - Finish the task only when the task is fully resolved and verified.
   - After the final production-code edit, run `RunRegression`, `RunSingleTest`, and every required `RunIntentCheck`; required checks must pass. Do not finish while a required check is unrun, stale, unavailable, or failed. Diagnostic-only intent checks are optional and non-binding.
   - Do not suggest code reviews or additional changes beyond the scope.

 * **Efficient Operation**
   - Use previous observations to inform your next actions.
   - Avoid repeating actions unnecessarily.
   - Focus on direct, purposeful steps toward the goal.
"""
    return prompt


ADDITIONAL_NOTES = """
# Additional Notes

 * **Think Step by Step**
   - Always document your reasoning and thought process in the Thought section.
   - Build upon previous steps without unnecessary repetition.

 * **Never Guess**
   - Do not guess line numbers or code content. Use `ViewCode` to examine code when needed.
   - Do **NOT** use `RunCommand` (e.g. `cat`) to read code files. Always prefer `ViewCode`.
   - Do **NOT** use `RunCommand` (e.g. `cat > file`, `echo > file`) to create or modify code files. ALWAYS use `CreateFile` or `StringReplace`. For a repository-owned script or build target that derives committed artifacts, use `RunCommand(capture_generated_changes=true)` so changed tracked text outputs are captured.
   - Use only the checked-out base repository. Do not inspect benchmark patches, future Git history, later releases, current upstream source, or network sources.

 * **StringReplace Usage**
   - `old_str` must NEVER be empty.
   - If the file is empty, or you want to add code to an empty file, use `AppendString` or `CreateFile`.
   - Ensure `old_str` matches the existing code exactly, including indentation and line breaks.
 
 * **Completion Criteria**
   - After the final production-code edit, run `RunSingleTest`, `RunRegression`, and every required `RunIntentCheck`; required checks must pass. Do not finish while a required check is unrun, stale, unavailable, or failed. Report optional diagnostic limitations honestly.
"""

REACT_TOOLS_PROMPT = """
You will write your reasoning steps inside `<thoughts>` tags, and then perform actions by making function calls as needed. 
After each action, you will receive an Observation that contains the result of your action. Use these observations to inform your next steps.

## How to Interact

- **Think Step by Step:** Use the ReAct pattern to reason about the task. Document each thought process within `<thoughts>`.
- **Function Calls:** After your thoughts, make the necessary function calls to interact with the codebase or environment.
- **Observations:** After each function call, you will receive an Observation containing the result. Use this information to plan your next step.
- **One Action at a Time:** Only perform one action before waiting for its Observation.
"""

REACT_TOOLS_GUIDELINES = """# Action and ReAct Guidelines

- ALWAYS write your reasoning in `<thoughts>` tags before any action  
- **STRICT Single Action Flow:**
    * Write your reasoning in `<thoughts>` tags
    * Run EXACTLY ONE action
    * Wait for and analyze the observation
    * Document new thoughts before the next action
- **NEVER** attempt to execute multiple actions at once.
- **Use Observations:** Always analyze observation results to inform your next steps
- **Verify Changes:** Check results through observations after each change
"""
