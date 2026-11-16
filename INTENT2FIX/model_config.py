"""Model configuration settings for different LLMs."""

from INTENT2FIX.completion.base import LLMResponseFormat
from INTENT2FIX.schema import MessageHistoryType

# Base configuration that all models inherit from
BASE_MODEL_CONFIG = {
    "temperature": 0.0,
    "thoughts_in_action": False,
    "max_tokens": 4000,
    "few_shot_examples": True,
}

# Claude 3.5 Sonnet configuration
CLAUDE_35_SONNET = {
    **BASE_MODEL_CONFIG,
    "model": "claude-3-5-sonnet-20241022",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
}

# Claude 3.5 Haiku configuration
CLAUDE_35_HAIKU = {
    **BASE_MODEL_CONFIG,
    "model": "claude-3-5-haiku-20241022",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
}

# GPT-4o configuration
GPT4O = {
    **BASE_MODEL_CONFIG,
    "model": "gpt-4o-2024-11-20",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
    "thoughts_in_action": True,
}

AZURE_GPT4O = {**GPT4O, "model": "azure/gpt-4o"}

# GPT-4o Mini configuration
GPT4O_MINI = {
    **BASE_MODEL_CONFIG,
    "model": "gpt-4o-mini-2024-07-18",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
    "thoughts_in_action": True,
}

# O1 Preview configuration
O1_PREVIEW = {
    **BASE_MODEL_CONFIG,
    "model": "o1-preview-2024-09-12",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "max_tokens": 8000,
}

# O1 Mini configuration
O1_MINI = {
    **BASE_MODEL_CONFIG,
    "model": "o1-mini-2024-09-12",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "disable_thoughts": True,
    "max_tokens": 8000,
}

# DeepSeek Chat configuration
DEEPSEEK_CHAT = {
    **BASE_MODEL_CONFIG,
    "model": "deepseek/deepseek-chat",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# DeepSeek Reasoner Chat configuration
DEEPSEEK_REASONER_CHAT = {
    **BASE_MODEL_CONFIG,
    "model": "deepseek/deepseek-reasoner",
    "temperature": None,
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "disable_thoughts": True,
    "few_shot_examples": True,
    "merge_same_role_messages": True,
    "max_tokens": 8000,
}

# DeepSeek V3.1 via OpenRouter (Free)
DEEPSEEK_V31_FREE = {
    **BASE_MODEL_CONFIG,
    "model": "deepseek/deepseek-chat",  # Standard DeepSeek Chat model
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# DeepSeek V3.2 Experimental configuration
DEEPSEEK_V32_EXP = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/deepseek/deepseek-v3.2-exp",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "max_tokens": 128000,  # Large context window
    "model_base_url": "https://openrouter.ai/api/v1",  # OpenRouter endpoint
}

# DeepSeek V3.2 configuration
DEEPSEEK_V32 = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/deepseek/deepseek-v3.2",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "max_tokens": 128000,  # Large context window
    "model_base_url": "https://openrouter.ai/api/v1",  # OpenRouter endpoint
}

# DeepSeek V4 Flash via official DeepSeek API.
DEEPSEEK_V4_FLASH = {
    **BASE_MODEL_CONFIG,
    "model": "deepseek/deepseek-v4-flash",
    "temperature": None,
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
    "disable_thoughts": True,
    "few_shot_examples": False,
    "compact_native_prompt": True,
    # Reserve space in the 1M context window for output, tools, system prompt, and the current task.
    "message_history_max_tokens": 850000,
    "max_tokens": 128000,
    "extra_body": {"thinking": {"type": "enabled"}},
}

# DeepSeek V4 Flash April build via Volcengine Ark's OpenAI-compatible API.
# Keep this separate from DEEPSEEK_V4_FLASH: the unversioned official
# DeepSeek alias can advance to a newer model release.
VOLCENGINE_DEEPSEEK_V4_FLASH = {
    **DEEPSEEK_V4_FLASH,
    "model": "openai/deepseek-v4-flash-260425",
    "model_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "model_api_key_env": "ARK_API_KEY",
}

# GLM-5.3-Flash via Z.AI's OpenAI-compatible general API. The Coding Plan
# endpoint is intentionally not used because this runner is a custom client.
GLM_53_FLASH = {
    **BASE_MODEL_CONFIG,
    "model": "openai/glm-5.3-flash",
    "temperature": 1.0,
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
    "disable_thoughts": True,
    "few_shot_examples": False,
    "compact_native_prompt": True,
    "compact_structured_output": True,
    "message_history_max_tokens": 850000,
    "max_tokens": 32768,
    # Thinking responses regularly exceed the global 120-second default. Wait
    # for the original request instead of submitting duplicate SDK retries.
    "timeout": 600.0,
    "max_retries": 0,
    "model_base_url": "https://api.z.ai/api/paas/v4",
    "model_api_key_env": "ZAI_API_KEY",
    # Z.AI supports only tool_choice=auto. Preserved thinking keeps reasoning
    # coherent across the runner's multi-turn native tool trajectory.
    "tool_choice": "auto",
    "extra_body": {
        "thinking": {
            "type": "enabled",
            "clear_thinking": False,
        },
        "reasoning_effort": "low",
    },
}

# Qwen3.8-Flash via Alibaba Cloud Model Studio's OpenAI-compatible API.
# Thinking-mode tool calls require tool_choice=auto, and Qwen3.8 requires the
# provider reasoning field to be preserved across the native-tool trajectory.
QWEN_38_FLASH = {
    **BASE_MODEL_CONFIG,
    "model": "openai/qwen3.8-flash",
    "temperature": 1.0,
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
    "disable_thoughts": True,
    "few_shot_examples": False,
    "compact_native_prompt": True,
    "message_history_max_tokens": 850000,
    "max_tokens": 131072,
    "timeout": 600.0,
    "max_retries": 0,
    "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model_api_key_env": "DASHSCOPE_API_KEY",
    "tool_choice": "auto",
    "extra_body": {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
    },
}

GPT_5_MINI = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/openai/gpt-5-mini",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "max_tokens": 400000,  # Very large context window
    "model_base_url": "https://openrouter.ai/api/v1",  # OpenRouter endpoint
}

GPT_5 = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/openai/gpt-5",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES_COMPACT,
    "max_tokens": 400000,  # Very large context window
    "model_base_url": "https://openrouter.ai/api/v1",  # OpenRouter endpoint
}

# GPT-5.2 configuration
GPT_5_2 = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/openai/gpt-5.2",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES_COMPACT,
    "max_tokens": 200000,  # Very large context window
    "model_base_url": "https://openrouter.ai/api/v1",  # OpenRouter endpoint
}

# Claude Sonnet 4.5 configuration
CLAUDE_SONNET_45 = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/anthropic/claude-sonnet-4.5",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES_COMPACT,
    "max_tokens": 200000,  # Very large context window
    "model_base_url": "https://openrouter.ai/api/v1",  # OpenRouter endpoint
}

# Claude 3.7 Sonnet configuration (OpenRouter)
CLAUDE_37_SONNET = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/anthropic/claude-3.7-sonnet",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES_COMPACT,
    "max_tokens": 200000,
    "model_base_url": "https://openrouter.ai/api/v1",  # OpenRouter endpoint
}

# Gemini 1206 configuration
GEMINI_1206 = {
    **BASE_MODEL_CONFIG,
    "model": "gemini/gemini-exp-1206",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# Gemini Flash configuration
GEMINI_FLASH = {
    **BASE_MODEL_CONFIG,
    "model": "gemini/gemini-2.0-flash-exp",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# Gemini Flash Think configuration
GEMINI_FLASH_THINK = {
    **BASE_MODEL_CONFIG,
    "model": "gemini/gemini-2.0-flash-thinking-exp",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# Gemini 2.5 Pro configuration
GEMINI_25_PRO = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/google/gemini-2.5-pro",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES_COMPACT,
    "max_tokens": 200000,
    "model_base_url": "https://openrouter.ai/api/v1",
}

# Llama 3.1 405B Instruct configuration
LLAMA_31_405B = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/meta-llama/llama-3.1-405b-instruct",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "model_base_url": "https://openrouter.ai/api/v1",
}

# Llama 3.1 70B Instruct configuration
LLAMA_31_70B = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "model_base_url": "https://openrouter.ai/api/v1",
}

# Qwen 2.5 Coder configuration
QWEN_25_CODER = {
    **BASE_MODEL_CONFIG,
    "model": "qwen/qwen-2.5-coder-32b-instruct",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}



# DeepSeek V3 configuration with specific provider
DEEPSEEK_V3_0324 = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/deepseek/deepseek-chat-v3-0324",
    "model_base_url": "https://openrouter.ai/api/v1",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "max_tokens": 128000,
    "extra_body": {
        "provider": {
            "order": ["AtlasCloud"],
            "allow_fallbacks": False
        }
    }
}

KIMI_K2 = {
    **BASE_MODEL_CONFIG,
    "model": "openai/kimi-k2-0711-preview",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "model_base_url": "https://api.moonshot.cn/v1"
}
 

SUPPORTED_MODELS = [
    # AZURE_GPT4O,
    CLAUDE_35_SONNET,
    CLAUDE_35_HAIKU,
    CLAUDE_SONNET_45,
    CLAUDE_37_SONNET,
    O1_MINI,
    GPT4O,
    GPT4O_MINI,
    GEMINI_1206,
    GEMINI_FLASH,
    GEMINI_25_PRO,
    GPT_5_MINI,
    GPT_5,
    GPT_5_2,
    DEEPSEEK_CHAT,
    DEEPSEEK_REASONER_CHAT,
    DEEPSEEK_V31_FREE,
    DEEPSEEK_V32_EXP,
    DEEPSEEK_V32,
    DEEPSEEK_V4_FLASH,
    VOLCENGINE_DEEPSEEK_V4_FLASH,
    GLM_53_FLASH,
    QWEN_38_FLASH,
    LLAMA_31_70B,
    QWEN_25_CODER,
    DEEPSEEK_V3_0324,
    KIMI_K2,
]

MODEL_CONFIGS = {
    **{config["model"]: config for config in SUPPORTED_MODELS},
    "deepseek-v3-0324": DEEPSEEK_V3_0324,
    "deepseek-v4-flash": DEEPSEEK_V4_FLASH,
    "volcengine-deepseek-v4-flash": VOLCENGINE_DEEPSEEK_V4_FLASH,
    "glm-5.3-flash": GLM_53_FLASH,
    "qwen3.8-flash": QWEN_38_FLASH,
    "gpt5-mini": GPT_5_MINI,
    "gpt5.2": GPT_5_2,
    "gemini-2.5-pro": GEMINI_25_PRO,
    "claude-3.7-sonnet": CLAUDE_37_SONNET,
}


def get_model_config(model_name: str) -> dict:
    """Get the configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found in supported models")
    return MODEL_CONFIGS[model_name]


if __name__ == "__main__":
    # Example usage
    try:
        config = get_model_config("openai/kimi-k2-0711-preview")
        print("Model configuration:", config)
    except ValueError as e:
        print(e)
