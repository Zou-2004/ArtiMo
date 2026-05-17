#!/usr/bin/env python3
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import AzureOpenAI, OpenAI


AZURE_GPT54_ENDPOINT = "https://eee-icdevai02-openai-xjq-aue.openai.azure.com/"
AZURE_GPT54_DEPLOYMENT = "gpt-5.4"
AZURE_GPT54_API_VERSION = "2024-12-01-preview"


def _is_azure_gpt54_model(model: str | None, provider: str | None = None) -> bool:
    if (provider or "auto").strip().lower() == "gemini":
        return False
    m = str(model or "").strip().lower()
    return m in {"gpt-5.4", "gpt5.4"}


def _infer_provider(model: str | None, provider: str | None, base_url: str | None) -> str:
    p = (provider or "auto").strip().lower()
    if p in {"openai", "gemini"}:
        return p
    if base_url and "/google" in str(base_url).lower():
        return "gemini"
    m = (model or "").lower()
    if "gemini" in m:
        return "gemini"
    return "openai"


def resolve_api_config(
    model: str | None = None,
    provider: str | None = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    if _is_azure_gpt54_model(model=model, provider=provider):
        resolved_endpoint = (
            base_url
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or AZURE_GPT54_ENDPOINT
        )
        resolved_api_key = (
            api_key
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("AZURE_OPENAI_KEY")
        )
        resolved_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or AZURE_GPT54_API_VERSION
        resolved_deployment = os.getenv("AZURE_OPENAI_GPT54_DEPLOYMENT") or AZURE_GPT54_DEPLOYMENT
        if not resolved_api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY not set for gpt-5.4 (or pass --api_key)")
        return {
            "provider": "azure_openai",
            "azure_endpoint": resolved_endpoint,
            "api_key": resolved_api_key,
            "api_version": resolved_api_version,
            "deployment": resolved_deployment,
            "base_url": resolved_endpoint,
        }

    use_provider = _infer_provider(model=model, provider=provider, base_url=base_url)
    if use_provider == "gemini":
        resolved_base_url = (
            base_url
            or os.getenv("GOOGLE_GEMINI_BASE_URL")
            or "https://api.openai-proxy.org/google"
        )
        resolved_api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not resolved_api_key:
            raise RuntimeError("GEMINI_API_KEY not set (or pass --api_key)")
    else:
        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai-proxy.org/v1"
        )
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY not set (or pass --api_key)")
    return {
        "provider": use_provider,
        "base_url": resolved_base_url,
        "api_key": resolved_api_key,
    }


def make_openai_client(
    model: str | None = None,
    provider: str | None = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
):
    cfg = resolve_api_config(model=model, provider=provider, api_key=api_key, base_url=base_url)
    if cfg["provider"] == "azure_openai":
        client = AzureOpenAI(
            api_version=cfg["api_version"],
            azure_endpoint=cfg["azure_endpoint"],
            api_key=cfg["api_key"],
        )
    else:
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    return client, cfg


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _extract_genai_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    # Fallback for structured candidates
    try:
        cands = getattr(resp, "candidates", None) or []
        for c in cands:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) or []
            chunks = []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t:
                    chunks.append(t)
            if chunks:
                return "\n".join(chunks)
    except Exception:
        pass
    return str(resp)


def generate_content_text(
    *,
    model: str,
    prompt: str,
    image_paths: list[Path] | None = None,
    image_items: list[tuple[str, Path]] | None = None,
    provider: str = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, dict[str, Any]]:
    cfg = resolve_api_config(model=model, provider=provider, api_key=api_key, base_url=base_url)
    items: list[tuple[str, Path]] = []
    if image_items:
        for label, p in image_items:
            pp = Path(p)
            if pp.exists():
                items.append((str(label), pp))
    elif image_paths:
        for i, p in enumerate(image_paths, start=1):
            pp = Path(p)
            if pp.exists():
                items.append((f"IMAGE_{i}:{pp.name}", pp))
    if cfg["provider"] == "gemini":
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError("google-genai not installed; install with `pip install google-genai`") from exc
        # Default to direct Gemini API mode when only an API key is provided.
        # Vertex AI requires additional project/location configuration and should
        # be enabled explicitly via env when desired.
        vertexai = _env_bool("GOOGLE_GENAI_USE_VERTEXAI", False)
        client = genai.Client(
            api_key=cfg["api_key"],
            vertexai=vertexai,
            http_options={"base_url": cfg["base_url"]},
        )
        parts = [types.Part.from_text(text=prompt)]
        for idx, (label, p) in enumerate(items, start=1):
            parts.append(types.Part.from_text(text=f"[ATTACHMENT {idx}] {label}"))
            parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=_guess_mime(p)))
        resp = client.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(),
        )
        return _extract_genai_text(resp), cfg

    if cfg["provider"] == "azure_openai":
        client = AzureOpenAI(
            api_version=cfg["api_version"],
            azure_endpoint=cfg["azure_endpoint"],
            api_key=cfg["api_key"],
        )
    else:
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    if items:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for idx, (label, p) in enumerate(items, start=1):
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            mime = _guess_mime(p)
            content.append({"type": "text", "text": f"[ATTACHMENT {idx}] {label}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    else:
        content = prompt
    req = {
        "messages": [{"role": "user", "content": content}],
        "model": (cfg["deployment"] if cfg["provider"] == "azure_openai" else model),
    }
    if cfg["provider"] == "azure_openai":
        req["max_completion_tokens"] = 16384
    resp = client.chat.completions.create(**req)
    msg = resp.choices[0].message.content or ""
    return msg, cfg
