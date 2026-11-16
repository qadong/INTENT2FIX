from INTENT2FIX.file_context import FileContext
from INTENT2FIX.loop import AgenticLoop
from INTENT2FIX.node import Node


def test_append_user_feedback_preserves_trajectory_and_context():
    context = FileContext(repo=None)
    root = Node(node_id=0, user_message="repair", file_context=context)
    finished = Node(node_id=1, parent=root, file_context=context.clone())
    finished.terminal = True
    root.add_child(finished)
    loop = AgenticLoop.model_construct(root=root)

    feedback = loop.append_user_feedback("review report")

    assert feedback.parent is finished
    assert feedback.user_message == "review report"
    assert not feedback.terminal
    assert feedback.file_context is not finished.file_context
    assert feedback.get_trajectory() == [root, finished, feedback]
