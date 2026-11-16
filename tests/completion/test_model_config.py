from INTENT2FIX.completion.base import BaseCompletionModel, LLMResponseFormat
from INTENT2FIX.model_config import get_model_config
from INTENT2FIX.schema import MessageHistoryType


def test_volcengine_deepseek_v4_flash_uses_april_build():
    config = get_model_config("volcengine-deepseek-v4-flash")

    assert config["model"] == "openai/deepseek-v4-flash-260425"
    assert config["model_base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert config["model_api_key_env"] == "ARK_API_KEY"
    assert config["extra_body"] == {"thinking": {"type": "enabled"}}


def test_glm_53_flash_uses_zai_native_tool_api():
    config = get_model_config("glm-5.3-flash")

    assert config["model"] == "openai/glm-5.3-flash"
    assert config["model_base_url"] == "https://api.z.ai/api/paas/v4"
    assert config["model_api_key_env"] == "ZAI_API_KEY"
    assert config["tool_choice"] == "auto"
    assert config["timeout"] == 600.0
    assert config["max_retries"] == 0
    assert config["max_tokens"] == 32768
    assert config["compact_structured_output"] is True
    assert config["extra_body"] == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "high",
    }


def test_qwen_38_flash_uses_dashscope_native_tool_api():
    config = get_model_config("qwen3.8-flash")

    assert config["model"] == "openai/qwen3.8-flash"
    assert config["model_base_url"] == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert config["model_api_key_env"] == "DASHSCOPE_API_KEY"
    assert config["response_format"] == LLMResponseFormat.TOOLS
    assert config["message_history_type"] == MessageHistoryType.MESSAGES
    assert config["tool_choice"] == "auto"
    assert config["max_tokens"] == 131072
    assert config["extra_body"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
    }


def test_qwen_38_flash_reads_dashscope_api_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")
    config = get_model_config("qwen3.8-flash").copy()
    config.pop("compact_native_prompt")
    config.pop("message_history_max_tokens")
    config.pop("few_shot_examples")

    model = BaseCompletionModel.create(**config)

    assert model.model_api_key == "test-dashscope-key"


def test_completion_model_reads_provider_specific_api_key(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-ark-key")
    config = get_model_config("volcengine-deepseek-v4-flash").copy()
    config.pop("compact_native_prompt")
    config.pop("message_history_max_tokens")
    config.pop("few_shot_examples")

    model = BaseCompletionModel.create(**config)

    assert model.model_api_key == "test-ark-key"

    clone = model.clone()

    assert clone.model_api_key_env == "ARK_API_KEY"
    assert clone.model_api_key == "test-ark-key"
