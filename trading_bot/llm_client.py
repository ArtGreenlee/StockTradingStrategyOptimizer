"""Minimal multi-provider chat client for the Orchestrator tab.

Talks to either:
  * an **Anthropic Messages API** endpoint (``POST {base}/v1/messages``) -- e.g.
    Claude served via a Databricks serving endpoint -- when ``ANTHROPIC_*`` is
    configured (this takes precedence), or
  * any **OpenAI-compatible** ``/chat/completions`` endpoint -- OpenAI, Azure
    OpenAI, GitHub Models, or a corporate Copilot gateway.

It uses only ``requests`` (already a dependency), reuses the app's proxy / TLS
settings so it works behind a corporate firewall, and never logs credentials.

Provider selection + configuration come from
:class:`~trading_bot.config.Settings` (see ``llm_provider``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import requests

from .config import Settings
from .metrics import METRICS


def _record_token_usage(data: dict) -> None:
    """Record LLM token usage from a response into the shared metrics.

    Handles both the OpenAI shape (``usage.total_tokens``) and the Anthropic
    shape (``usage.input_tokens`` + ``usage.output_tokens``). Silently does
    nothing if the endpoint omits a usage block.
    """
    try:
        usage = data.get("usage") or {}
        total = usage.get("total_tokens")
        if total is None:
            inp = usage.get("input_tokens", 0) or 0
            out = usage.get("output_tokens", 0) or 0
            total = inp + out
        if total:
            METRICS.record_tokens(int(total))
    except Exception:
        pass



class LLMError(RuntimeError):
    """Raised when the LLM endpoint is misconfigured or returns an error."""


class LLMClient:
    """Thin wrapper around an Anthropic or OpenAI-compatible chat endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = requests.Session()
        # Reuse the corporate proxy / TLS configuration.
        proxies = settings.proxies()
        if proxies:
            self._session.proxies.update(proxies)
        if settings.proxy_ca_bundle:
            self._session.verify = settings.proxy_ca_bundle
        elif not settings.proxy_verify_ssl:
            self._session.verify = False

    @property
    def configured(self) -> bool:
        return self._settings.llm_configured

    @property
    def provider(self) -> str:
        return self._settings.llm_provider

    @property
    def description(self) -> str:
        return self._settings.llm_description

    def _endpoint_and_headers(self):
        s = self._settings
        if s.llm_api_version:
            # Azure OpenAI shape: /openai/deployments/<deployment>/chat/completions
            url = (
                f"{s.llm_base_url}/openai/deployments/{s.llm_model}"
                f"/chat/completions?api-version={s.llm_api_version}"
            )
            headers = {"api-key": s.llm_api_key, "Content-Type": "application/json"}
        else:
            url = f"{s.llm_base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {s.llm_api_key}",
                "Content-Type": "application/json",
            }
        return url, headers

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 400,
        timeout: float = 60.0,
    ) -> str:
        """Send a chat request and return the assistant's text content.

        Dispatches to the Anthropic or OpenAI path based on configuration.
        Raises :class:`LLMError` on configuration problems or API failures.
        """
        provider = self._settings.llm_provider
        if provider == "anthropic":
            return self._chat_anthropic(messages, temperature, max_tokens, timeout)
        if provider == "openai":
            return self._chat_openai(messages, temperature, max_tokens, timeout)
        raise LLMError(
            "LLM endpoint not configured. Set ANTHROPIC_BASE_URL/"
            "ANTHROPIC_AUTH_TOKEN (or LLM_BASE_URL/LLM_API_KEY) in your .env."
        )

    # ---- OpenAI-compatible ----------------------------------------------

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        url, headers = self._endpoint_and_headers()
        body = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = self._session.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as exc:
            raise LLMError(f"Request failed: {type(exc).__name__}: {exc}") from exc

        if resp.status_code >= 400:
            snippet = (resp.text or "")[:300]
            raise LLMError(f"HTTP {resp.status_code} from endpoint: {snippet}")

        try:
            data = resp.json()
            _record_token_usage(data)
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"Unexpected response shape: {type(exc).__name__}: {exc}"
            ) from exc

    # ---- Anthropic Messages API -----------------------------------------

    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        s = self._settings
        url = f"{s.anthropic_base_url}/v1/messages"
        headers = {
            # ANTHROPIC_AUTH_TOKEN -> bearer auth (Databricks PAT / gateway token).
            "Authorization": f"Bearer {s.anthropic_auth_token}",
            "anthropic-version": s.anthropic_version or "2023-06-01",
            "Content-Type": "application/json",
        }
        headers.update(s.anthropic_headers())  # e.g. x-databricks-* flags

        # Anthropic requires the system prompt as a top-level field; only
        # user/assistant turns belong in "messages".
        system_parts = [
            m.get("content", "") for m in messages if m.get("role") == "system"
        ]
        convo = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        body: Dict[str, object] = {
            "model": s.anthropic_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            body["system"] = "\n\n".join(p for p in system_parts if p)

        try:
            resp = self._session.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as exc:
            raise LLMError(f"Request failed: {type(exc).__name__}: {exc}") from exc

        # Some hosted models (e.g. certain Databricks-served Claude models)
        # reject "temperature" as deprecated/unsupported. If so, drop it and
        # retry once -- this keeps the client portable across endpoints.
        if resp.status_code == 400 and "temperature" in (resp.text or "").lower():
            body.pop("temperature", None)
            try:
                resp = self._session.post(
                    url, headers=headers, json=body, timeout=timeout
                )
            except requests.RequestException as exc:
                raise LLMError(f"Request failed: {type(exc).__name__}: {exc}") from exc

        if resp.status_code >= 400:
            snippet = (resp.text or "")[:300]
            raise LLMError(f"HTTP {resp.status_code} from endpoint: {snippet}")

        try:
            data = resp.json()
            _record_token_usage(data)
            blocks = data.get("content", [])
            text = "".join(
                b.get("text", "")
                for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if not text:
                raise KeyError("no text blocks in response content")
            return text
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(
                f"Unexpected response shape: {type(exc).__name__}: {exc}"
            ) from exc
