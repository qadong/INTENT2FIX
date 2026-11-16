from INTENT2FIX.actions.view_code import (
    ViewCode,
    ViewCodeArgs,
    CodeSpan,
)
from INTENT2FIX.benchmark.swebench import create_repository
from INTENT2FIX.benchmark.utils import get_INTENT2FIX_instance
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.completion.base import BaseCompletionModel, LLMResponseFormat
import dotenv

dotenv.load_dotenv()
def test_request_non_existing_method():
    instance_id = "django__django-11039"
    instance = get_INTENT2FIX_instance(instance_id)
    repository = create_repository(instance)
    file_context = FileContext(repo=repository)

    action = ViewCode(
        repository=repository
    )

    args = ViewCodeArgs(
        scratch_pad="def non_existing_method():",
        files=[
            CodeSpan(
                file_path="tests/migrations/test_commands.py",
                span_ids=["test_sqlmigrate"],
            )
        ],
    )

    output = action.execute(args, file_context)
    print(output.message)


def test_request_many_spans():
    instance_id = "fmtlib__fmt-3569"
    instance = get_INTENT2FIX_instance(instance_id)
    repository = create_repository(instance)
    file_context = FileContext(repo=repository)

    completion_model = BaseCompletionModel.create(
    model="openrouter/deepseek/deepseek-v3.2-exp",
    temperature=0.0,
    response_format=LLMResponseFormat.TOOLS,
    thoughts_in_action=True
    )

    action = ViewCode(
        repository=repository,
        completion_model=completion_model,
    )

    args = ViewCodeArgs(
        scratch_pad="Adding relevant code spans for the sqlmigrate logic to modify the output_transaction assignment.",
        files=[
            CodeSpan(
                file_path="include/fmt/format.h",
                start_line=3000,
                end_line=9999,
                span_ids=[]
            )
        ],
    )

    output = action.execute(args, file_context)

    print(file_context.model_dump())
    print(output.message)


# def test_request_many_spans():
#     instance_id = "django__django-11039"
#     instance = get_INTENT2FIX_instance(instance_id)
#     repository = create_repository(instance)
#     file_context = FileContext(repo=repository)

#     action = ViewCode(
#         repository=repository
#     )

#     args = ViewCodeArgs(
#         scratch_pad="Adding relevant code spans for the sqlmigrate logic to modify the output_transaction assignment.",
#         files=[
#             CodeSpan(
#                 file_path="django/core/management/commands/sqlmigrate.py",
#                 start_line=0,
#                 end_line=50
#             ) 
#         ],
#     )

#     output = action.execute(args, file_context)

#     print(file_context.model_dump())
#     print(output.message)


test_request_many_spans()
