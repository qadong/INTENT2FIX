import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from INTENT2FIX.actions.run_tests import RunTestsArgs
from INTENT2FIX.actions.schema import ActionArguments
from INTENT2FIX.actions.view_code import ViewCodeArgs, CodeSpan
from INTENT2FIX.actions.view_diff import ViewDiffArgs
from INTENT2FIX.completion.schema import (
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
    AllMessageValues,
)
from INTENT2FIX.message_history.message_history import MessageHistoryGenerator
from INTENT2FIX.node import Node
from INTENT2FIX.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class NodeMessage(BaseModel):
    user_message: Optional[str] = Field(default=None, description="The user message")
    assistant_message: Optional[str] = Field(default=None, description="The assistant message")
    action: Optional[ActionArguments] = Field(default=None, description="The action")
    observation: Optional[str] = Field(default=None, description="The observation")


class CompactMessageHistoryGenerator(MessageHistoryGenerator):
    message_cache: bool = Field(default=False, description="Cache the message history if the LLM supports it")

    def generate_messages(self, node: Node) -> List[AllMessageValues]:
        messages = []
        # print(f'!!!!!!!!!!!!length{len(node)}')
        # logging.DEBUG(f'!!!!!!!!!!!!length{len(node)}')
        node_messages = self.get_node_messages(node)

        tool_idx = 0
        tokens = 0

        for action, observation in node_messages:
            tool_calls = []
            tool_idx += 1
            tool_call_id = f"tool_{tool_idx}"

            exclude = None
            content = None

            if not self.thoughts_in_action:
                exclude = {"thoughts"}
                content = action.thoughts

            tool_call = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": action.name,
                    "arguments": action.model_dump_json(exclude=exclude),
                },
            }

            tool_calls.append(tool_call)

            tokens += count_tokens(action.model_dump_json(exclude=exclude))

            messages.append(ChatCompletionAssistantMessage(role="assistant", tool_calls=tool_calls, content=content))
            messages.append(
                ChatCompletionToolMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    content=f"Observation: {observation}",
                )
            )

            tokens += count_tokens(observation)

        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")

        return messages

    def get_node_messages(self, node: "Node") -> List[NodeMessage]:
        """
        Creates a list of (action, observation) tuples from the node's trajectory.
        Respects token limits while preserving ViewCode context.

        Returns:
            List of tuples where each tuple contains:
            - ActionArguments object
            - Observation summary string
        """
        previous_nodes = node.get_trajectory()
        logger.info(f"Previous nodes: {len(previous_nodes)}")
        if not previous_nodes:
            return []

        # Calculate initial token count
        total_tokens = node.file_context.context_size()
        total_tokens += count_tokens(node.get_root().message)

        # Pre-calculate test output tokens if there's a patch
        test_output_tokens = 0
        test_output = None
        run_tests_args = None


        node_messages = []
        shown_files = set()
        shown_diff = False  # Track if we've shown the first diff
        last_test_status = None  # Track the last test status


        for i, previous_node in enumerate(reversed(previous_nodes)):
            current_messages: List[NodeMessage] = []

            user_message = previous_node.user_message
            assistant_message = previous_node.assistant_message

            if previous_node.feedback_data and not user_message:
                user_message = previous_node.feedback_data.feedback

            if user_message or assistant_message:
                current_messages.append(NodeMessage(user_message=user_message, assistant_message=assistant_message))

            if previous_node.action_steps:
                for i, action_step in enumerate(previous_node.action_steps):
                    if action_step.action.name == "ViewCode":
                        # Always include ViewCode actions
                        file_path = action_step.action.files[0].file_path

                        if file_path not in shown_files:
                            context_file = previous_node.file_context.get_context_file(file_path)
                            if context_file and (context_file.span_ids or context_file.show_all_spans):
                                shown_files.add(context_file.file_path)
                                observation = context_file.to_prompt(
                                    show_span_ids=False,
                                    show_line_numbers=True,
                                    exclude_comments=False,
                                    show_outcommented_code=True,
                                    outcomment_code_comment="... rest of the code",
                                )
                            else:
                                observation = action_step.observation.message
                            current_messages.append(NodeMessage(action=action_step.action, observation=observation))
                    else:
                        # Count tokens for non-ViewCode actions
                        observation_str = (
                            action_step.observation.summary
                            if self.include_file_context
                            and hasattr(action_step.observation, "summary")
                            and action_step.observation.summary
                            else action_step.observation.message
                            if action_step.observation
                            else "No output found."
                        )

                        # Calculate tokens for this message pair
                        action_tokens = count_tokens(action_step.action.model_dump_json())
                        observation_tokens = count_tokens(observation_str)
                        message_tokens = action_tokens + observation_tokens

                        # Only add if within token limit
                        if total_tokens + message_tokens <= self.max_tokens:
                            total_tokens += message_tokens
                            current_messages.append(NodeMessage(action=action_step.action, observation=observation_str))
                        else:
                            # Skip remaining non-ViewCode messages if we're over the limit
                            continue

                # Handle file context for non-ViewCode actions
                if self.include_file_context and previous_node.action.name != "ViewCode":
                    files_to_show = set()
                    has_edits = False
                    for context_file in previous_node.file_context.get_context_files():
                        if (
                            context_file.was_edited or context_file.was_viewed
                        ) and context_file.file_path not in shown_files:
                            files_to_show.add(context_file.file_path)
                        if context_file.was_edited:
                            has_edits = True

                    shown_files.update(files_to_show)

                    if files_to_show:
                        # Batch all files into a single ViewCode action
                        code_spans = []
                        observations = []

                        for file_path in files_to_show:
                            context_file = previous_node.file_context.get_context_file(file_path)
                            if context_file.show_all_spans:
                                code_spans.append(CodeSpan(file_path=file_path))
                            elif context_file.span_ids:
                                code_spans.append(
                                    CodeSpan(
                                        file_path=file_path,
                                        span_ids=context_file.span_ids,
                                    )
                                )
                            else:
                                continue

                            observations.append(
                                context_file.to_prompt(
                                    show_span_ids=False,
                                    show_line_numbers=True,
                                    exclude_comments=False,
                                    show_outcommented_code=True,
                                    outcomment_code_comment="... rest of the code",
                                )
                            )

                        if code_spans:
                            thought = f"Let's view the content in the updated files"
                            args = ViewCodeArgs(files=code_spans, thoughts=thought)
                            current_messages.append(NodeMessage(action=args, observation="\n\n".join(observations)))

                    # Show ViewDiff on first edit
                    if has_edits and self.include_git_patch and not shown_diff:
                        patch = node.file_context.generate_git_patch()
                        if patch:
                            view_diff_args = ViewDiffArgs(
                                thoughts="Let's review the changes made to ensure we've properly implemented everything required for the task. I'll check the git diff to verify the modifications."
                            )
                            diff_tokens = count_tokens(patch) + count_tokens(view_diff_args.model_dump_json())
                            if total_tokens + diff_tokens <= self.max_tokens:
                                total_tokens += diff_tokens
                                current_messages.append(
                                    NodeMessage(
                                        action=view_diff_args,
                                        observation=f"Current changes in workspace:\n```diff\n{patch}\n```",
                                    )
                                )
                                shown_diff = True

                    # Add test results only if status changed or first occurrence
                    if (
                        node.file_context.has_runtime
                        and previous_node.observation
                        and previous_node.observation.properties.get("diff")
                    ):
                        # current_test_status = node.file_context.get_test_status()
                        # if last_test_status is None or current_test_status != last_test_status:
                        #     if node.file_context.has_test_patch():
                        #         thoughts = "Run the updated tests to verify the changes."
                        #     else:
                        #         thoughts = (
                        #             "Before adding new test cases I run the existing tests to verify regressions."
                        #         )

                        #     run_tests_args = RunTestsArgs(
                        #         thoughts=thoughts,
                        #         test_files=list(node.file_context._test_files.keys()),
                        #     )

                        #     test_output = ""
                        #     if last_test_status is None:
                        #         # Show full details for first test run
                        #         failure_details = node.file_context.get_test_failure_details()
                        #         if failure_details:
                        #             test_output += failure_details + "\n\n"

                        #     test_output += node.file_context.get_test_summary()

                        #     # Calculate and check token limits
                        #     test_tokens = count_tokens(test_output) + count_tokens(run_tests_args.model_dump_json())
                        #     if total_tokens + test_tokens <= self.max_tokens:
                        #         total_tokens += test_tokens
                        #         current_messages.append(NodeMessage(action=run_tests_args, observation=test_output))
                        #         last_test_status = current_test_status
            
            # 新增编译和测试逻辑
                        #TODO: Need Rewrite
                        copy_status = True
                        compile_message = ""
                        test_summary = ""
                        thoughts = "Run the tests to verify the changes and regressions."
                        run_tests_args = RunTestsArgs(
                                thoughts=thoughts,
                                test_files=list(node.file_context._test_files.keys()),
                            )
                        
                        if last_test_status is None and node.file_context.test_output=='':
                            if node.file_context._runtime:
                                # 获取所有被修改的文件
                                patched_files = [
                                    file_path for file_path, file in node.file_context._files.items() 
                                    if file.patch or file.is_new
                                ]
                                
                                if not patched_files:
                                    logger.warning("No files were modified or created.")
                                    copy_message = False
                                    copy_message = "No files were modified or created."
                                else:
                                    # 1. 创建临时文件并复制到容器
                                    import tempfile
                                    import os
                                    tmp_files = []  # 保存临时文件路径，用于后续清理
                                    
                                    try:
                                        for file_path in patched_files:
                                            # 创建临时文件
                                            tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                                            tmp_files.append(tmp_file.name)
                                            
                                            # 写入修改后的内容
                                            file_content = node.file_context.get_file(file_path).content
                                            tmp_file.write(file_content)
                                            tmp_file.close()
                                            
                                            # 复制到容器
                                            container_path = f"{node.file_context._runtime.work_dir}/{file_path}"
                                            if not node.file_context._runtime.copy_anything_to_container(tmp_file.name, container_path):
                                                copy_status = False
                                                copy_message = f"Failed to copy {file_path} to container"
                                                logging.error(f"{copy_message}")
                                                break
                                        else:  # 所有文件复制成功
                                            # 2. 执行编译（如果是 C/C++/Java）
                                            # if node.file_context._code_index._settings.language in ['c', 'cpp', 'java']:
                                            #TODO: 需要根据语言类型来判断是否需要执行编译
                                            # logger.info("Running Compile...")
                                            # compile_status, compile_message = node.file_context._runtime.run_compile('/app/cxxopts')
                                            # logger.info(f"Compiling status: {compile_status}")
                                            
                                            # # 3. 如果编译成功，执行测试
                                            # if compile_status:
                                            #     test_summary = node.file_context._runtime.run_test(
                                            #         language='cpp'
                                            #     )
                                            

                                            logger.info("Running test")
                                            
                                            test_summary = node.file_context._runtime.run_test(
                                                    test_command = r'''
                                                        chmod 755 /home/run.sh && 
                                                        sed -i "s|ctest|ctest --output-on-failure|g" /home/run.sh && 
                                                        cd /home && 
                                                        ./run.sh | grep -A 10 "FAILED\|error"
                                                    '''
                                            )
                                            print(test_summary)
                                            
                                            
                                            
                                    finally:
                                        # 清理所有临时文件
                                        for tmp_file_path in tmp_files:
                                            try:
                                                os.unlink(tmp_file_path)
                                            except:
                                                logger.warning(f"Failed to delete temporary file: {tmp_file_path}")

                                # 构建测试输出
                                test_output = ""
                                
                                if not copy_status:
                                    test_output += f"\n\nCompiling failure:\n{copy_message}"
                                else:
                                    test_output += "\n\nAutomated test cases have been run."
                                    if node.file_context._runtime.is_test_failed(test_summary):
                                        test_output += f"\n\n**Test Result:**\n Tests failed:\n{test_summary}\n You could use `ViewCode` to see the detail failure test case.\n **Instruction: Do not create new test cases.\n**Explanation: Existing test cases already satisfy coverage for all branches."
                                    else:
                                        test_output += "\n\n**Test Result:**\nAll tests passed!"

                                node.file_context.test_output = test_output
                                last_test_status = test_output

                        else:
                            test_output = previous_node.file_context.test_output
                        #     # Calculate and check token limits
                        test_tokens = count_tokens(test_output) + count_tokens(run_tests_args.model_dump_json())
                        if total_tokens + test_tokens <= self.max_tokens:
                            total_tokens += test_tokens

                            current_messages.append(NodeMessage(action=run_tests_args, observation=test_output))
                        
                        
                        
                        

            node_messages = current_messages + node_messages

        logger.info(f"Generated message history with {total_tokens} tokens")
        return node_messages


        node_messages = list(reversed(temp_collected_messages))

        # --- Start of new pruning logic ---
        if total_tokens > getattr(self, 'max_tokens', 4096):
            max_tokens_limit = getattr(self, 'max_tokens', 4096)
            logger.info(f"Token limit exceeded ({total_tokens}/{max_tokens_limit}). Attempting to prune messages.")
            tokens_to_shed = total_tokens - max_tokens_limit

            # Iterate through messages from oldest to newest for pruning
            for i in range(len(node_messages)):
                if tokens_to_shed <= 0:
                    break

                message_to_prune = node_messages[i]

                if message_to_prune.action and message_to_prune.observation:
                    original_observation_content = message_to_prune.observation
                    # Avoid re-summarizing already summarized content
                    if original_observation_content.startswith("Summary:") or original_observation_content.startswith("RunTests Summary:"):
                        continue

                    original_observation_tokens = count_tokens(original_observation_content)
                    summarized_observation_content = None
                    
                    action_name = message_to_prune.action.name 
                    action_args = message_to_prune.action

                    # Prioritize StringReplace or similar edit actions for summarization
                    # Using a generic "Edit" or "Replace" check; adjust "StringReplace" if it's a very specific action name
                    if "replace" in action_name.lower() or "edit" in action_name.lower() or action_name == "StringReplace":
                        file_path_str = "a file"
                        # Try to get file_path from common attributes
                        if hasattr(action_args, 'file_path') and isinstance(action_args.file_path, str):
                            file_path_str = action_args.file_path
                        elif hasattr(action_args, 'files') and isinstance(action_args.files, list) and len(action_args.files) > 0:
                            first_file_item = action_args.files[0]
                            if hasattr(first_file_item, 'file_path') and isinstance(first_file_item.file_path, str):
                                file_path_str = first_file_item.file_path
                            elif isinstance(first_file_item, str): # If files is a list of path strings
                                file_path_str = first_file_item
                        
                        start_line = getattr(action_args, 'start_line', None)
                        end_line = getattr(action_args, 'end_line', None)

                        if start_line is not None and end_line is not None:
                            summarized_observation_content = f"Summary: Modified {file_path_str} between lines {start_line}-{end_line}."
                        else:
                            summarized_observation_content = f"Summary: Content modified in {file_path_str}."
                    
                    elif action_name == "RunTests":
                        # Check original observation for keywords to make summary more informative
                        if "all tests passed" in original_observation_content.lower():
                            summarized_observation_content = "RunTests Summary: All tests passed."
                        elif "tests failed" in original_observation_content.lower() or "failure" in original_observation_content.lower():
                            summarized_observation_content = "RunTests Summary: Some tests failed."
                        else:
                            summarized_observation_content = "RunTests Summary: Results processed."

                    if summarized_observation_content:
                        summarized_observation_tokens = count_tokens(summarized_observation_content)
                        
                        if summarized_observation_tokens < original_observation_tokens:
                            tokens_saved_this_step = original_observation_tokens - summarized_observation_tokens
                            
                            message_to_prune.observation = summarized_observation_content
                            total_tokens -= tokens_saved_this_step
                            tokens_to_shed -= tokens_saved_this_step
                            logger.info(f"Pruned observation for action '{action_name}'. Saved {tokens_saved_this_step} tokens. Total tokens now: {total_tokens}. Remaining to shed: {max(0, tokens_to_shed)}")
            
            if tokens_to_shed > 0:
                logger.warning(f"Could not shed enough tokens. Still over by {tokens_to_shed} tokens. Final total: {total_tokens}")
            else:
                logger.info(f"Successfully pruned messages. New total tokens: {total_tokens}")
        # --- End of new pruning logic ---

        logger.info(f"Generated message history with {total_tokens} tokens (after potential pruning).")
        return node_messages
