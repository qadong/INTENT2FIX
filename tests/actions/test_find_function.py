from dotenv import load_dotenv
from INTENT2FIX.actions.find_class import FindClass, FindClassArgs
from INTENT2FIX.actions.find_code_snippet import FindCodeSnippet, FindCodeSnippetArgs
from INTENT2FIX.benchmark.swebench import create_repository, create_index
from INTENT2FIX.benchmark.utils import get_INTENT2FIX_instance
from INTENT2FIX.completion import BaseCompletionModel, LLMResponseFormat
from INTENT2FIX.file_context import FileContext

load_dotenv()


def test_find_function_init_method():
    instance_id = "django__django-13658"
    instance = get_INTENT2FIX_instance(instance_id)
    repository = create_repository(instance)
    code_index = create_index(instance, repository)
    file_context = FileContext(repo=repository)
    completion_model = BaseCompletionModel.create(model="test", response_format=LLMResponseFormat.TOOLS)

    action = FindClass(
        repository=repository, code_index=code_index, completion_model=completion_model
    )

    action_args = FindClassArgs(
        scratch_pad="",
        class_name="ManagementUtility",
        function_name="__init__",
    )

    message = action.execute(action_args, file_context)
    print(message)
    assert len(file_context.files) == 1
    assert "ManagementUtility.__init__" in file_context.files[0].span_ids


def test_find_function():
    instance_id = 'nlohmann__json-3543'
    instance = get_INTENT2FIX_instance(instance_id)
    repository = create_repository(instance)
    code_index = create_index(instance, repository)
    file_context = FileContext(repo=repository)
    completion_model = BaseCompletionModel.create(model="openrouter/deepseek/deepseek-v3.2-exp", response_format=LLMResponseFormat.REACT)

    action = FindClass(
        repository=repository, code_index=code_index, completion_model=completion_model
    )

    action_args = FindClassArgs(
        scratch_pad="",
        class_name="binary_reader",
        file_pattern="include/nlohmann/detail/input/*.hpp",
    )

    message = action.execute(action_args, file_context)
    print(message)
    
# test_find_function()



def test_find_CodeSnippet_method():
    instance_id = "nlohmann__json-3543"
    instance = get_INTENT2FIX_instance(instance_id)
    repository = create_repository(instance)
    code_index = create_index(instance, repository)
    file_context = FileContext(repo=repository)
    completion_model = BaseCompletionModel.create(model="test", response_format=LLMResponseFormat.REACT)

    action = FindCodeSnippet(
        repository=repository, code_index=code_index, completion_model=completion_model
    )

    action_args = FindCodeSnippetArgs(
        file_pattern="**/*.hpp",
        code_snippet="class basic_json",
    )

    message = action.execute(action_args, file_context)
    print(message)
    # assert len(file_context.files) == 1
    # assert "ManagementUtility.__init__" in file_context.files[0].span_ids

test_find_function()