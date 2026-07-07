"""Configuration loading and safety guards.

This module is the single source of truth for runtime settings. It loads
values from a local ``.env`` file (never committed) and *hard-codes* the use
of Alpaca's paper-trading endpoint so the bot can never be pointed at the
live, real-money API by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the bot."""

    api_key: str
    secret_key: str
    ticker: str
    max_position_shares: int
    order_qty: int
    poll_interval_seconds: int
    lookback_minutes: int
    # Always True. There is intentionally no way to flip this to live trading.
    paper: bool = True
    # --- Proxy / corporate network settings ---
    # ``proxy_url`` applies to both HTTP and HTTPS. The per-scheme overrides
    # take precedence when set. ``no_proxy`` is a comma-separated bypass list.
    proxy_url: str = ""
    http_proxy: str = ""
    https_proxy: str = ""
    no_proxy: str = ""
    # Path to a corporate root CA bundle (the secure way to handle a
    # TLS-intercepting proxy). Leave blank to use system defaults.
    proxy_ca_bundle: str = ""
    # Disabling TLS verification is insecure; kept as a last resort and
    # surfaced loudly in logs / README.
    proxy_verify_ssl: bool = True
    # Options data feed: "indicative" (free, 15-min delayed) or "opra"
    # (real-time, requires an OPRA subscription). Used for UOA detection.
    options_feed: str = "indicative"
    # --- LLM / Copilot endpoint (Orchestrator tab: sentiment analysis) ---
    # An OpenAI-compatible chat endpoint. Works with OpenAI, Azure OpenAI,
    # GitHub Models, and most corporate Copilot gateways.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # When set, requests use the Azure OpenAI URL shape + "api-key" header
    # instead of the standard "Authorization: Bearer" header.
    llm_api_version: str = ""
    # --- Anthropic (Claude) endpoint, e.g. via a Databricks serving endpoint ---
    # When configured, the Orchestrator uses the Anthropic Messages API shape
    # (POST {base}/v1/messages) and takes precedence over the LLM_* settings.
    anthropic_base_url: str = ""
    anthropic_auth_token: str = ""
    anthropic_model: str = ""
    # Newline-separated "Name: Value" extra headers (e.g. Databricks flags).
    anthropic_custom_headers: str = ""
    anthropic_version: str = "2023-06-01"

    def proxies(self) -> Dict[str, str]:
        """Return a requests-style proxies mapping (may be empty)."""
        http = self.http_proxy or self.proxy_url
        https = self.https_proxy or self.proxy_url
        result: Dict[str, str] = {}
        if http:
            result["http"] = http
        if https:
            result["https"] = https
        return result

    @property
    def proxy_enabled(self) -> bool:
        return bool(self.proxies())

    @property
    def anthropic_configured(self) -> bool:
        """True when the Anthropic endpoint has a base URL + auth token."""
        placeholders = {"", "your_anthropic_token_here"}
        return (
            bool(self.anthropic_base_url)
            and self.anthropic_auth_token not in placeholders
        )

    @property
    def llm_provider(self) -> str:
        """Which Orchestrator backend to use: 'anthropic', 'openai', or 'none'.

        Anthropic takes precedence when both are configured.
        """
        if self.anthropic_configured:
            return "anthropic"
        if bool(self.llm_base_url) and self.llm_api_key not in {
            "",
            "your_llm_api_key_here",
        }:
            return "openai"
        return "none"

    @property
    def llm_configured(self) -> bool:
        """True when any Orchestrator LLM backend is configured."""
        return self.llm_provider != "none"

    def anthropic_headers(self) -> Dict[str, str]:
        """Parse ANTHROPIC_CUSTOM_HEADERS into a header mapping.

        Accepts newline- or literal ``\\n``-separated ``Name: Value`` lines.
        """
        headers: Dict[str, str] = {}
        raw = (self.anthropic_custom_headers or "").replace("\\n", "\n")
        for line in raw.splitlines():
            if ":" in line:
                name, _, value = line.partition(":")
                name = name.strip()
                if name:
                    headers[name] = value.strip()
        return headers

    @property
    def llm_description(self) -> str:
        """Credential-free summary of the active LLM endpoint for the GUI."""
        provider = self.llm_provider
        if provider == "anthropic":
            return (
                f"{self.anthropic_model or 'claude'} via Anthropic @ "
                f"{self.anthropic_base_url}"
            )
        if provider == "openai":
            kind = "Azure OpenAI" if self.llm_api_version else "OpenAI-compatible"
            return f"{self.llm_model} via {kind} @ {self.llm_base_url}"
        return "not configured (set ANTHROPIC_* or LLM_* in .env)"

    @property
    def proxy_description(self) -> str:
        """Human-readable, credential-masked proxy summary for the GUI."""
        if not self.proxy_enabled:
            return "direct (no proxy)"
        mapping = self.proxies()
        target = mapping.get("https") or mapping.get("http") or ""
        masked = _mask_proxy(target)
        bypass = f", bypass={self.no_proxy}" if self.no_proxy else ""
        if not self.proxy_verify_ssl:
            tls = ", TLS verify OFF"
        elif self.proxy_ca_bundle:
            tls = ", custom CA"
        else:
            tls = ""
        return f"{masked}{bypass}{tls}"


def _get_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _get_str(*names: str, default: str = "") -> str:
    """Return the first non-empty environment value among ``names``."""
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip() != "":
            return raw.strip()
    return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _mask_proxy(url: str) -> str:
    """Mask any user:pass credentials embedded in a proxy URL for display."""
    try:
        parts = urlsplit(url)
        if parts.username or parts.password:
            netloc = parts.hostname or ""
            if parts.port:
                netloc += f":{parts.port}"
            netloc = "***@" + netloc
            return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
        return url
    except ValueError:
        return url


def load_settings() -> Settings:
    """Load settings from environment / .env file.

    Raises:
        RuntimeError: if API credentials are missing or still set to the
            placeholder values from ``.env.example``.
    """
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()

    placeholders = {"", "your_paper_api_key_here", "your_paper_secret_key_here"}
    if api_key in placeholders or secret_key in placeholders:
        raise RuntimeError(
            "Missing Alpaca paper API credentials.\n"
            "Copy .env.example to .env and fill in ALPACA_API_KEY / "
            "ALPACA_SECRET_KEY with your PAPER trading keys from "
            "https://app.alpaca.markets/."
        )

    return Settings(
        api_key=api_key,
        secret_key=secret_key,
        ticker=os.getenv("TICKER", "AAPL").strip().upper(),
        max_position_shares=_get_int("MAX_POSITION_SHARES", 10),
        order_qty=_get_int("ORDER_QTY", 1),
        poll_interval_seconds=_get_int("POLL_INTERVAL_SECONDS", 15, minimum=5),
        lookback_minutes=_get_int("LOOKBACK_MINUTES", 390, minimum=30),
        paper=True,
        proxy_url=_get_str("PROXY_URL"),
        http_proxy=_get_str("HTTP_PROXY", "http_proxy"),
        https_proxy=_get_str("HTTPS_PROXY", "https_proxy"),
        no_proxy=_get_str("NO_PROXY", "no_proxy"),
        proxy_ca_bundle=_get_str("PROXY_CA_BUNDLE"),
        proxy_verify_ssl=_get_bool("PROXY_VERIFY_SSL", True),
        options_feed=_get_str("OPTIONS_FEED", default="indicative").lower(),
        llm_base_url=_get_str("LLM_BASE_URL").rstrip("/"),
        llm_api_key=_get_str("LLM_API_KEY"),
        llm_model=_get_str("LLM_MODEL", default="gpt-4o-mini"),
        llm_api_version=_get_str("LLM_API_VERSION"),
        anthropic_base_url=_get_str("ANTHROPIC_BASE_URL").rstrip("/"),
        anthropic_auth_token=_get_str("ANTHROPIC_AUTH_TOKEN"),
        anthropic_model=_get_str(
            "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
        ),
        anthropic_custom_headers=_get_str("ANTHROPIC_CUSTOM_HEADERS"),
        anthropic_version=_get_str("ANTHROPIC_VERSION", default="2023-06-01"),
    )
