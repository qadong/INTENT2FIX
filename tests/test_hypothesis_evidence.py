import copy
import json
from datetime import datetime

import litellm
import pytest
from openai.types.chat import ChatCompletionChunk

from INTENT2FIX.hypothesis_evidence import HypothesisEvidenceTools


class FakeContainer:
    def __init__(self):
        self.work_dir = "/repo"
        self.files = {}
        self.commands = []
        self.dirty = False

    def run_bash(self, command, timeout):
        self.commands.append(command)
        if command.startswith("git diff"):
            return "", int(self.dirty)
        if command.startswith("test -e"):
            return "", 1
        if command.startswith("sed -n"):
            return "enable_if<is_string<T>::value> set_debug_format(bool);", 0
        if command == "change_tracked_file":
            self.dirty = True
            return "changed", 0
        return "error: no matching function; type constraint is false", 1

    def write_file(self, path, content):
        self.files[path] = content


def tool_call(name, arguments):
    return litellm.ModelResponse(choices=[{"message": {
        "role": "assistant", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }}])


def streamed_tool_call(name, arguments):
    encoded = json.dumps(arguments, ensure_ascii=False)
    split = len(encoded) // 2
    for index, part in enumerate([encoded[:split], encoded[split:]]):
        call = {'index': 0, 'function': {'arguments': part}}
        if index == 0:
            call.update(id='call_stream', type='function')
            call['function']['name'] = name
        yield litellm.ModelResponse(model='gpt-4o', stream=True, choices=[{
            'index': 0, 'delta': {'role': 'assistant', 'tool_calls': [call]},
            'finish_reason': 'tool_calls' if index == 1 else None,
        }])


def streamed_batch(calls, layout):
    headers, tails = [], []
    for index, (name, arguments) in enumerate(calls):
        encoded = json.dumps(arguments, ensure_ascii=False)
        split = len(encoded) // 2 if layout == 'interleaved' else len(encoded)
        headers.append({'index': index, 'id': f'call_{index}', 'type': 'function',
                        'function': {'name': name, 'arguments': encoded[:split]}})
        tails.append({'index': index, 'function': {'arguments': encoded[split:]}})
    groups = [[call] for call in headers] if layout == 'successive_chunks' else [headers]
    if layout == 'interleaved':
        groups.extend([[tail] for tail in reversed(tails)])
    for group in groups:
        yield litellm.ModelResponse(model='gpt-4o', stream=True, choices=[{
            'index': 0, 'delta': {'role': 'assistant', 'tool_calls': group},
        }])


def test_streamed_tool_continuation_preserves_reasoning_and_logs_public_response(monkeypatch, tmp_path):
    verifier = HypothesisEvidenceTools(FakeContainer(), tmp_path, max_actions=1)
    reasoning_parts = ['synthetic reasoning with "quotes"\n', '第二段\\exact']
    chunks = list(streamed_tool_call('ReadOnlySearch', {
        'command': 'sed -n 1,5p include/base.hpp'}))
    for chunk, part in zip(chunks, reasoning_parts):
        chunk.choices[0].delta.reasoning_content = part
    chunks.append(litellm.ModelResponse(model='gpt-4o', stream=True, choices=[],
                                      usage={'prompt_tokens': 23, 'completion_tokens': 41,
                                             'total_tokens': 64}))
    payload = {'reason': 'incomplete public audit'}
    responses = iter([iter(chunks), streamed_tool_call('SubmitHypothesisAudit', payload)])
    requests = []

    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)

    monkeypatch.setattr(litellm, 'completion', complete)
    assert json.loads(verifier.complete({
        'stream': True, 'api_key': 'never-log-provider-key',
        'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}},
    }, 'Audit', {})) == payload
    assistant = requests[1]['messages'][1]
    assert assistant['reasoning_content'] == ''.join(reasoning_parts)
    assert json.loads(assistant['tool_calls'][0]['function']['arguments']) == {
        'command': 'sed -n 1,5p include/base.hpp'}
    saved = (tmp_path / 'hypothesis_audit_responses.jsonl').read_text()
    rows = [json.loads(line) for line in saved.splitlines()]
    assert rows[0]['finish_reason'] == 'tool_calls'
    assert rows[0]['usage']['completion_tokens'] == 41
    assert rows[0]['reasoning_chars'] == len(''.join(reasoning_parts))
    assert json.loads(rows[1]['tool_calls'][0]['function']['arguments']) == payload
    assert 'never-log-provider-key' not in saved
    assert 'synthetic reasoning' not in saved


def test_empty_audit_submission_retains_finish_reason_and_usage(monkeypatch, tmp_path):
    verifier = HypothesisEvidenceTools(FakeContainer(), tmp_path, max_actions=1)
    responses = iter([
        tool_call('ReadOnlySearch', {'command': 'sed -n 1,5p include/base.hpp'}),
        litellm.ModelResponse(choices=[{'finish_reason': 'length', 'message': {
            'role': 'assistant', 'content': '', 'reasoning_content': 'synthetic-only',
        }}], usage={'prompt_tokens': 10, 'completion_tokens': 50, 'total_tokens': 60}),
    ])
    monkeypatch.setattr(litellm, 'completion', lambda **kwargs: next(responses))
    assert verifier.complete({}, 'Audit', {}) == ''
    rows = [json.loads(line) for line in
            (tmp_path / 'hypothesis_audit_responses.jsonl').read_text().splitlines()]
    assert rows[-1]['finish_reason'] == 'length'
    assert rows[-1]['usage']['completion_tokens'] == 50
    assert rows[-1]['content'] == ''
    assert rows[-1]['tool_calls'] == []


@pytest.mark.parametrize('submission', ['tool', 'text', 'empty'])
def test_raw_openai_stream_keeps_reasoning_only_chunks_and_unknown_usage(monkeypatch, tmp_path, submission):
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    class RawStream:
        def __init__(self, chunks):
            self.chunks = iter(chunks)
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.chunks)

        def close(self):
            self.closed = True

    def chunk(delta, finish_reason=None):
        return ChatCompletionChunk(id='sdk_call', created=1, model='glm-5.3-flash',
                                   object='chat.completion.chunk', choices=[{
                                       'index': 0, 'delta': delta, 'finish_reason': finish_reason}])

    raw = RawStream([
        chunk({'role': 'assistant', 'reasoning_content': 'synthetic first\n'}),
        chunk({'reasoning_content': '第二段\\exact'}),
        chunk({'tool_calls': [{'index': 0, 'id': 'sdk_tool', 'type': 'function',
                              'function': {'name': 'ReadOnlySearch', 'arguments': json.dumps({
                                  'command': 'sed -n 1,5p include/base.hpp'})}}]}),
        chunk({}, 'tool_calls'),
    ])
    logging = Logging(model='glm-5.3-flash', messages=[], stream=True,
                      call_type='completion', start_time=datetime.now(),
                      litellm_call_id='synthetic-test', function_id='synthetic-test')
    wrapped = CustomStreamWrapper(raw, 'glm-5.3-flash', logging,
                                  custom_llm_provider='openai')
    payload = json.dumps({'decision': 'REVISE'})
    final_delta = {'tool_calls': [{'index': 0, 'id': 'sdk_submit', 'type': 'function',
                                  'function': {'name': 'SubmitHypothesisAudit', 'arguments': payload}}]}
    if submission != 'tool':
        final_delta = {'content': payload} if submission == 'text' else {}
    finish = {'tool': 'tool_calls', 'text': 'stop', 'empty': 'length'}[submission]
    final_raw = RawStream([chunk({'reasoning_content': 'synthetic final'}),
                           chunk(final_delta), chunk({}, finish)])
    final_wrapped = CustomStreamWrapper(final_raw, 'glm-5.3-flash', logging,
                                        custom_llm_provider='openai')
    responses = iter([wrapped, final_wrapped])
    requests = []

    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)

    monkeypatch.setattr(litellm, 'completion', complete)
    verifier = HypothesisEvidenceTools(FakeContainer(), tmp_path, max_actions=1)
    result = verifier.complete({'stream': True}, 'Audit', {})
    assert result == ('' if submission == 'empty' else payload)
    assert requests[1]['messages'][1]['reasoning_content'] == 'synthetic first\n第二段\\exact'
    assert raw.closed and final_raw.closed
    rows = [json.loads(line) for line in (tmp_path / 'hypothesis_audit_responses.jsonl').read_text().splitlines()]
    assert rows[0]['usage'] is None
    assert rows[0]['finish_reason'] == 'tool_calls'
    assert rows[-1]['finish_reason'] == finish
    assert rows[-1]['reasoning_chars'] == len('synthetic final')


@pytest.mark.parametrize('layout', ['successive_chunks', 'same_chunk', 'interleaved'])
def test_streamed_audit_keeps_each_tool_call_and_its_arguments(monkeypatch, tmp_path, layout):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=2)
    content = '#include <cstdio>\nint main() { puts("中文\\n"); }\n'
    calls = [
        ('WriteInvestigationFile', {'path': 'probe.cpp', 'content': content}),
        ('ReadOnlySearch', {'command': 'sed -n 1,5p include/base.hpp'}),
    ]
    responses = iter([
        streamed_batch(calls, layout),
        streamed_tool_call('SubmitHypothesisAudit', {'decision': 'REVISE'}),
    ])
    requests = []
    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)
    monkeypatch.setattr(litellm, 'completion', complete)
    assert json.loads(verifier.complete({'stream': True}, 'Audit', {})) == {'decision': 'REVISE'}
    assert env.files['/repo/probe.cpp'] == content
    assert verifier.remaining_actions == 0
    message = requests[1]['messages'][1]
    assert [(call['function']['name'], json.loads(call['function']['arguments']))
            for call in message['tool_calls']] == calls
    assert [m['tool_call_id'] for m in requests[1]['messages'] if m['role'] == 'tool'] == [
        'call_0', 'call_1']


@pytest.mark.parametrize('submission', [False, True])
def test_streamed_batch_preserves_pre_execution_budget_and_submission_guards(monkeypatch, tmp_path, submission):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=1)
    calls = [('WriteInvestigationFile', {'path': 'probe.cpp', 'content': 'must not execute'})]
    calls.append(('SubmitHypothesisAudit', {'decision': 'ACCEPT'}) if submission else
                 ('ReadOnlySearch', {'command': 'sed -n 1,5p include/base.hpp'}))
    monkeypatch.setattr(litellm, 'completion', lambda **kwargs: streamed_batch(calls, 'same_chunk'))
    with pytest.raises(ValueError, match='separate from evidence|budget exhausted'):
        verifier.complete({'stream': True}, 'Audit', {})
    assert env.files == {}
    assert verifier.remaining_actions == 1
    assert not verifier.evidence


@pytest.mark.parametrize('malformed', ['missing_index', 'missing_id', 'missing_name', 'changed_id'])
def test_ambiguous_stream_call_identity_cannot_execute(monkeypatch, tmp_path, malformed):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=2)
    chunks = list(streamed_tool_call('WriteInvestigationFile', {
        'path': 'probe.cpp', 'content': 'must not execute'}))
    first = chunks[0].choices[0].delta.tool_calls[0]
    if malformed == 'missing_index':
        first.index = None
    elif malformed == 'missing_id':
        first.id = None
    elif malformed == 'missing_name':
        first.function.name = None
    else:
        chunks[1].choices[0].delta.tool_calls[0].id = 'different_call'
    monkeypatch.setattr(litellm, 'completion', lambda **kwargs: iter(chunks))
    with pytest.raises(ValueError, match='index|ID or name'):
        verifier.complete({'stream': True}, 'Audit', {})
    assert env.files == {}
    assert verifier.remaining_actions == 2


def test_streamed_text_submission_after_evidence_is_preserved(monkeypatch, tmp_path):
    verifier = HypothesisEvidenceTools(FakeContainer(), tmp_path, max_actions=1)
    audit = json.dumps({'decision': 'REVISE', 'reason': 'Exact "quoted" 中文 content.'}, ensure_ascii=False)
    text_chunks = [litellm.ModelResponse(model='gpt-4o', stream=True, choices=[{
        'index': 0, 'delta': {'role': 'assistant', 'content': part},
    }]) for part in [audit[:15], audit[15:]]]
    responses = iter([
        streamed_tool_call('ReadOnlySearch', {'command': 'sed -n 1,5p include/base.hpp'}),
        iter(text_chunks),
    ])
    monkeypatch.setattr(litellm, 'completion', lambda **kwargs: next(responses))
    assert verifier.complete({'stream': True}, 'Audit', {}) == audit


def test_streamed_audit_preserves_complete_tool_arguments(monkeypatch, tmp_path):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=2)
    content = '#include <cstdio>\nint main() { puts("中文\\n"); }\n'
    audit = {'decision': 'REVISE', 'reason': 'Keep from_json and quoted "API" bytes.'}
    responses = iter([
        streamed_tool_call('WriteInvestigationFile', {'path': 'probe.cpp', 'content': content}),
        streamed_tool_call('ReadOnlySearch', {'command': 'sed -n 1,5p include/base.hpp'}),
        streamed_tool_call('SubmitHypothesisAudit', audit),
    ])
    requests = []

    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)

    monkeypatch.setattr(litellm, 'completion', complete)
    assert json.loads(verifier.complete({'stream': True}, 'Audit', {})) == audit
    assert env.files['/repo/probe.cpp'] == content
    assert verifier.remaining_actions == 0
    assert all(request['stream'] is True for request in requests)
    assert json.loads(requests[1]['messages'][1]['tool_calls'][0]['function']['arguments'])['content'] == content


@pytest.mark.parametrize('interrupted', [False, True])
def test_empty_or_interrupted_stream_cannot_execute_partial_tool(monkeypatch, tmp_path, interrupted):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=1)

    def chunks():
        if interrupted:
            yield next(streamed_tool_call('WriteInvestigationFile', {
                'path': 'partial.cpp', 'content': 'must not execute',
            }))
            raise RuntimeError('stream interrupted')

    monkeypatch.setattr(litellm, 'completion', lambda **kwargs: chunks())
    with pytest.raises((ValueError, RuntimeError), match='empty completion stream|stream interrupted'):
        verifier.complete({'stream': True}, 'Audit', {})
    assert env.files == {}
    assert verifier.remaining_actions == 1
    assert not verifier.evidence


def test_audit_sees_actual_failed_probe_and_preserves_files(monkeypatch, tmp_path):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=3)
    responses = iter([
        tool_call("ReadOnlySearch", {"command": "sed -n '1,5p' include/base.hpp"}),
        tool_call("WriteInvestigationFile", {"path": "probe.cpp", "content": "int main() {}"}),
        tool_call("RunInvestigationCommand", {"command": "g++ probe.cpp -o /tmp/probe", "evidence_role": "premise"}),
        tool_call("SubmitHypothesisAudit", {"decision": "REVISE"}),
    ])
    requests = []

    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)

    monkeypatch.setattr(litellm, "completion", complete)
    assert json.loads(verifier.complete({}, "Original audit", {})) == {"decision": "REVISE"}
    assert "[AUDIT-RUN-1] exit_code=1" in requests[-1]["messages"][-1]["content"]
    assert "type constraint is false" in requests[-1]["messages"][-1]["content"]
    assert [x["id"] for x in verifier.evidence] == ["AUDIT-SEARCH-1", "AUDIT-RUN-1"]
    assert verifier.evidence[-1]["exit_code"] == 1
    saved = json.loads((tmp_path / "hypothesis_audit_evidence.json").read_text())
    assert saved["probe_files"] == {"probe.cpp": "int main() {}"}
    assert saved["remaining_actions"] == 0
    assert [t["function"]["name"] for t in requests[-1]["tools"]] == ["SubmitHypothesisAudit"]
    assert env.files == {"/repo/probe.cpp": "int main() {}"}


def test_budget_is_shared_across_audits_and_cannot_execute_extra_call(monkeypatch, tmp_path):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=1)
    responses = iter([
        tool_call("ReadOnlySearch", {"command": "sed -n '1,5p' include/base.hpp"}),
        tool_call("SubmitHypothesisAudit", {"decision": "REVISE"}),
        tool_call("RunInvestigationCommand", {"command": "extra_compile"}),
    ])
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: next(responses))
    verifier.complete({}, "Initial audit", {})
    with pytest.raises(ValueError, match="budget exhausted"):
        verifier.complete({}, "Revised audit", {})
    assert "extra_compile" not in env.commands


def test_no_blind_submission_or_tracked_file_edit(monkeypatch, tmp_path):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path)
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: tool_call("SubmitHypothesisAudit", {}))
    with pytest.raises(ValueError, match="without any repository verification"):
        verifier.complete({}, "Audit", {})
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: tool_call(
        "RunInvestigationCommand", {"command": "change_tracked_file"}))
    with pytest.raises(ValueError, match="must not modify tracked"):
        verifier.complete({}, "Audit", {})


def test_native_probe_arguments_decode_json_exactly_once(monkeypatch, tmp_path):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=2)
    content = '#include <cstdio>\nint main() { puts("\\\"中文\\\"\\n"); }\n'
    command = r"grep '\\n' probe.cpp"
    responses = iter([
        tool_call('WriteInvestigationFile', {'path': 'probe.cpp', 'content': content}),
        tool_call('ReadOnlySearch', {'command': command}),
        tool_call('SubmitHypothesisAudit', {'decision': 'REVISE'}),
    ])
    monkeypatch.setattr(litellm, 'completion', lambda **kwargs: next(responses))
    verifier.complete({}, 'Audit', {})
    assert env.files['/repo/probe.cpp'] == content
    assert command in env.commands


def test_restore_scout_version_preserves_bytes_and_never_overwrites_repository(monkeypatch, tmp_path):
    import hashlib

    class ExistingFileContainer(FakeContainer):
        def run_bash(self, command, timeout):
            if command == 'test -e existing.hpp':
                return '', 0
            return super().run_bash(command, timeout)

    env = ExistingFileContainer()
    source = '#include <cstdio>\nint main() { puts("\\\"中文\\\""); }\n'
    verifier = HypothesisEvidenceTools(env, tmp_path, scout_probe_snapshot={
        'evidence_id': 'RUN-7', 'command': 'g++ probe.cpp -o /tmp/probe',
        'files': {'probe.cpp': source, 'existing.hpp': 'must not overwrite'},
    })
    assert env.files == {'/repo/probe.cpp': source}
    assert verifier.remaining_actions == 8
    assert not verifier.session.evidence
    saved = json.loads((tmp_path / 'hypothesis_audit_evidence.json').read_text())
    assert saved['restored_scout_snapshot']['evidence_id'] == 'RUN-7'
    assert saved['restored_scout_snapshot']['restored_files'] == {
        'probe.cpp': hashlib.sha256(source.encode()).hexdigest()}
    assert 'existing.hpp' in saved['restored_scout_snapshot']['skipped_files']
    monkeypatch.setattr(litellm, 'completion', lambda **kwargs: tool_call('SubmitHypothesisAudit', {}))
    with pytest.raises(ValueError, match='without any repository verification'):
        verifier.complete({}, 'Audit', {})


def batch_calls(*calls):
    response = copy.deepcopy(calls[0])
    response.choices[0].message.tool_calls = [
        copy.deepcopy(item.choices[0].message.tool_calls[0]) for item in calls]
    for index, call in enumerate(response.choices[0].message.tool_calls):
        call.id = f'call_{index}'
    return response


def test_batch_evidence_runs_in_order_and_returns_each_observation(monkeypatch, tmp_path):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=2)
    responses = iter([
        batch_calls(
            tool_call('WriteInvestigationFile', {'path': 'probe.cpp', 'content': 'int main() {}'}),
            tool_call('RunInvestigationCommand', {'command': 'g++ probe.cpp -o /tmp/probe'})),
        tool_call('SubmitHypothesisAudit', {'decision': 'REVISE'}),
    ])
    requests = []

    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)

    monkeypatch.setattr(litellm, 'completion', complete)
    assert json.loads(verifier.complete({}, 'Audit', {})) == {'decision': 'REVISE'}
    assert verifier.remaining_actions == 0
    assert verifier.session.actions[-1]['frozen_probe_files'] == {'probe.cpp': 'int main() {}'}
    messages = requests[-1]['messages'][-2:]
    assert [item['tool_call_id'] for item in messages] == ['call_0', 'call_1']
    assert '[AUDIT-RUN-1] exit_code=1' in messages[-1]['content']
    assert [tool['function']['name'] for tool in requests[-1]['tools']] == ['SubmitHypothesisAudit']


@pytest.mark.parametrize('submission', [False, True])
def test_batch_cannot_exceed_budget_or_submit_with_unseen_evidence(monkeypatch, tmp_path, submission):
    env = FakeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=1)
    second = (tool_call('SubmitHypothesisAudit', {'decision': 'ACCEPT'}) if submission else
              tool_call('RunInvestigationCommand', {'command': 'extra_compile'}))
    response = batch_calls(
        tool_call('WriteInvestigationFile', {'path': 'probe.cpp', 'content': 'int main() {}'}), second)
    monkeypatch.setattr(litellm, 'completion', lambda **kwargs: response)
    with pytest.raises(ValueError, match='separate' if submission else 'budget exhausted'):
        verifier.complete({}, 'Audit', {})
    assert not env.files
    assert not verifier.evidence
    assert verifier.remaining_actions == 1


@pytest.mark.parametrize('defect', ['missing_path', 'old_comment_drift', 'semantic_change', 'timeout'])
def test_light_argument_refine_is_bounded_and_preserves_requested_edit(monkeypatch, tmp_path, defect):
    class ProbeContainer(FakeContainer):
        def run_bash(self, command, timeout):
            if command.startswith('cat -- '):
                return self.files['/repo/probe.cpp'], 0
            return super().run_bash(command, timeout)
    env = ProbeContainer()
    verifier = HypothesisEvidenceTools(env, tmp_path, max_actions=2)
    current = '// actual comment\nint x = 1;\n'
    env.files['/repo/probe.cpp'] = current
    verifier.session.investigation_files['probe.cpp'] = current
    verifier.actions['ReadOnlySearch'].execute(verifier.actions['ReadOnlySearch'].args_schema.model_validate({
        'command':'sed -n 1,5p include/base.hpp'}))
    args = {'old':'// stale comment\nint x = 1;\n', 'new':'int x = 2;\n'}
    if defect != 'missing_path':
        args['path']='probe.cpp'
    responses=iter([tool_call('ReplaceInvestigationText', args),
                    tool_call('SubmitHypothesisAudit', {'decision':'REVISE'})])
    requests=[]
    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        if 'tools' in kwargs:
            return next(responses)
        assert kwargs['max_tokens']==2048 and kwargs['max_retries']==0 and kwargs['timeout']==60
        assert kwargs['extra_body']['reasoning_effort']=='low'
        if defect=='timeout':
            raise TimeoutError('light call timed out')
        old = 'int x = 1;' if defect=='semantic_change' else current
        # semantic_change: attempting to replace a different actual statement.
        if defect=='semantic_change':
            old = '// actual comment'
        return litellm.ModelResponse(choices=[{'message':{'role':'assistant', 'content':json.dumps({
            'path':'probe.cpp', 'old':old})}}])
    monkeypatch.setattr(litellm,'completion',complete)
    assert json.loads(verifier.complete({'model':'openai/glm-5.3-flash'},'Audit',{}))=={'decision':'REVISE'}
    assert verifier.remaining_actions==1
    assert verifier.argument_refine_used
    assert len([r for r in requests if 'tools' not in r])==1
    expected = current if defect in ('semantic_change','timeout') else args['new']
    assert env.files['/repo/probe.cpp']==expected
    if defect in ('semantic_change','timeout'):
        assert 'Evidence actions remaining: 1' in requests[-1]['messages'][-1]['content']
    assert verifier._refine_replace_arguments(args, {}) is None


def test_invalid_general_tool_args_become_observation_without_execution(monkeypatch, tmp_path):
    env=FakeContainer()
    verifier=HypothesisEvidenceTools(env,tmp_path,max_actions=2)
    responses=iter([tool_call('RunInvestigationCommand', {}),
                    tool_call('ReadOnlySearch', {'command':'sed -n 1,5p include/base.hpp'}),
                    tool_call('SubmitHypothesisAudit', {'decision':'REVISE'})])
    requests=[]
    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)
    monkeypatch.setattr(litellm,'completion',complete)
    verifier.complete({},'Audit',{})
    assert 'no action executed' in requests[1]['messages'][-1]['content']
    assert verifier.remaining_actions==0 and not verifier.argument_refine_used
    assert not any(r.get('exit_code') for r in verifier.session.actions if r.get('action')=='RunInvestigationCommand')
