<repair_evidence_guidance>
The original Issue defines success. Start from its complete public operation, actual input
types/value categories, configuration and supported build route, not merely the internal line
that failed. A smaller example is useful for diagnosis but cannot replace the Issue's outcome.
Use existing relevant tests to understand the public contract. When the requested result is
clear, exercise it with an assertion or a meaningful failure code, not only printed values.
When the result is unspecified, report the observation without inventing a required policy.

If supplied, scout_probe_snapshot contains the exact baseline command and source files. Reuse
that fixture to reproduce or inspect the symptom; its baseline exit/output is NOT the desired
post-repair result. Keep the original diagnostic available and put any adapted check in a
separate disposable fixture. An exit_status_warning means a historical command may mask
failure: preserve the original record, but replay the compiler/test directly rather than
trusting its reported exit code. Do not weaken its behavioral check while correcting the shell.
An unavailable fixture is a stated limitation, not evidence that
the behavior works. Do not spend steps constructing a mandatory passing comparison.

A Hypothesis supplies a proposed direction and assigned observable behavior, not proof of its
root cause or an authoritative edit recipe. Trace the public operation through the relevant
selection, state transition and consumer; check the proposed change against that chain. If a
recipe, literal predicate or bit operation contradicts the Issue or execution, correct the
implementation while preserving the assigned public behavior, and explain the contradiction.
Do not preserve a wrong recipe simply because it appears in repair_scope. Cover required
callers, supported variants and build-selected/shipped representations using source evidence;
neither a relationship lead nor the first failing line alone determines the complete edit scope.
Once a faithful reproduction and source-backed mechanism suggest a viable change, implement a
small candidate and test it; do not spend the whole Repair budget only investigating competing
explanations. Treat the candidate as an experiment, revise it on concrete failure, and retain
the best supported production change even if a secondary uncertainty remains. For a public
operation family, inspect connected entry points that share the affected invariant. For a new
public API, check the repository's existing naming and wrapper conventions using real calls.

With Bash, mark compilation and behavioral checks with verification=true. Supply probe_paths
for the disposable source/header fixtures needed to replay them; create fixtures in a separate
ordinary Bash action. Run the compiler/test directly, with && for prerequisites. Do not append
echo, hide failures with ||, or wrap a failing command in a success-returning shell. Normally
expect exit 0; for a genuinely expected failure specify expected_exit_code together with a
required_output_regex identifying that intended failure. Inspect observations for errors as
well as the declared outcome. A program that only prints a value and exits 0 is observation,
not proof that the Issue's required value was achieved. With other tool profiles, use the
available focused-verification tool with the same principles.

After the last relevant production change, rerun the faithful Issue check against that final
artifact and run a relevant existing regression/build check. Verify which headers, generated
or amalgamated files those commands actually consume; a green check of an unchanged alternate
representation does not validate the edited one. Existing-suite success does not erase a
current Issue-check failure. Diagnose concrete counterevidence before concluding success; do
not replace a failing faithful check with a weaker passing toy, relax its assertion, or edit
existing tests to hide it. If a check cannot run for an environment reason, distinguish that
limitation from a production failure. Before Finish, inspect the final diff and state what was
verified on the final revision, what failed, and what remains unverified. Finish with the best
supported patch rather than looping on an unchanged state; never claim unexecuted success.
For inputs already supported, preserve their observable error behavior as well as their
success behavior unless the Issue requires a change; an earlier guard is not automatically
equivalent to the existing downstream error path.
</repair_evidence_guidance>
