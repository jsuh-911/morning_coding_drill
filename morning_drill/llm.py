import os
from pathlib import Path


def project_root(start=None):
    """
    Walk upward until PROJECT.md is found, so the drill works whether it is
    run from the project root or from notebooks/.
    """
    here = Path(start or Path.cwd()).resolve()

    for candidate in [here, *here.parents]:
        if (candidate / "PROJECT.md").exists():
            return candidate

    return here


def load_env():
    """
    Load the project .env, if python-dotenv is available and a file exists.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    env_path = project_root() / ".env"

    if not env_path.exists():
        return False

    load_dotenv(env_path, override=False)
    return True


PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def get_model():
    """
    Build the coach model from .env, or return None.

    None is a fully supported state: the graph then skips every coach node
    and the drill behaves exactly as it did before the trainer existed.
    """
    load_env()

    if os.getenv("TRAINER_ENABLED", "1") != "1":
        return None

    provider = os.getenv("TRAINER_PROVIDER", "anthropic").strip().lower()
    model_name = os.getenv("TRAINER_MODEL", "claude-sonnet-5").strip()

    key_var = PROVIDER_KEYS.get(provider)

    if key_var is None:
        return None

    if not os.getenv(key_var):
        return None

    # `temperature` is rejected with a 400 on current Claude models
    # (Sonnet 5, Opus 4.6+ and later). Set TRAINER_TEMPERATURE only when
    # pointing at an older model that still accepts it.
    extra = {}
    temperature = os.getenv("TRAINER_TEMPERATURE")

    if temperature:
        extra["temperature"] = float(temperature)

    max_tokens = int(os.getenv("TRAINER_MAX_TOKENS", "2048"))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, max_tokens=max_tokens, **extra)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, max_completion_tokens=max_tokens, **extra)


def coach_status():
    """
    Human-readable explanation of whether the coach is available.
    """
    load_env()

    if os.getenv("TRAINER_ENABLED", "1") != "1":
        return "coach disabled (TRAINER_ENABLED is not 1)"

    provider = os.getenv("TRAINER_PROVIDER", "anthropic").strip().lower()
    key_var = PROVIDER_KEYS.get(provider)

    if key_var is None:
        return f"coach disabled (unknown TRAINER_PROVIDER: {provider!r})"

    if not os.getenv(key_var):
        return f"coach disabled ({key_var} not set; copy .env.example to .env)"

    return f"coach active ({provider} / {os.getenv('TRAINER_MODEL')})"


def text_of(message):
    """
    Plain text from a chat message.

    Current Claude models return a list of content blocks (thinking blocks
    included, with empty text under the default display setting), so
    `.content` is not a string.
    """
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))

        return "\n".join(p for p in parts if p).strip()

    return str(content)
