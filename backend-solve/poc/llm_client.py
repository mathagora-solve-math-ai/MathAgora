"""Thin wrapper around OpenRouter / OpenAI API using the openai SDK."""

import os
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Models that should go through the OpenAI API directly (BYOK on OpenRouter)
OPENAI_DIRECT_PREFIXES = ("openai/",)

# Models that require the Responses API instead of Chat Completions
OPENAI_RESPONSES_API = {"gpt-5-codex"}

# Models that must NOT receive an explicit temperature param
OPENAI_NO_TEMP = {"gpt-5", "gpt-5-codex"}


def _get_openrouter_client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def call_llm(
    prompt: str,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """Call an LLM and return the text response.

    Routes openai/* models directly to the OpenAI API;
    everything else goes through OpenRouter.
    Set json_mode=True to request JSON output (structured output).
    """
    if model.startswith(OPENAI_DIRECT_PREFIXES):
        client = _get_openai_client()
        native_model = model.split("/", 1)[1]

        if native_model in OPENAI_RESPONSES_API:
            # gpt-5-codex requires the Responses API
            kwargs = dict(
                model=native_model,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=max_tokens,
            )
            if json_mode:
                kwargs["text"] = {"format": {"type": "json_object"}}
            resp = client.responses.create(**kwargs)
            return resp.output_text or ""

        # Standard Chat Completions
        kwargs = dict(
            model=native_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        if native_model not in OPENAI_NO_TEMP:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
    else:
        client = _get_openrouter_client()
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
