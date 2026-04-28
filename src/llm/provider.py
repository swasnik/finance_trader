import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
    """
    Get configured LLM instance from environment.
    
    Priority: explicit args > env vars > defaults
    Falls back gracefully if keys are missing.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")

    if provider == "anthropic":
        model = model or os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Create a .env file from .env.example")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, max_tokens=4096, **kwargs)

    elif provider == "openai":
        model = model or os.getenv("LLM_MODEL", "gpt-4o")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Create a .env file from .env.example")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, max_tokens=4096, **kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'")


def is_llm_configured() -> bool:
    """Check if any LLM provider is configured."""
    return bool(
        os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
