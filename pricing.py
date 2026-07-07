"""Per-token prices for the active LLM, resolved by model name.

OpenAI exposes no pricing API, so we key into a community-maintained catalog
(LiteLLM's ``model_prices_and_context_window.json``) by model name. The catalog
is fetched once per process, cached to disk so subsequent/offline runs still have
prices, and finally falls back to a small built-in table. All figures are USD per
single token (multiply by token counts to get a cost).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from config import settings

# USD-per-token fallback for common OpenAI models, used only when the catalog
# can neither be fetched nor read from disk cache. Keep minimal; the catalog is
# the source of truth.
_FALLBACK: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.5e-6, "cached": 1.25e-6, "output": 1e-5},
    "gpt-4o-mini": {"input": 1.5e-7, "cached": 7.5e-8, "output": 6e-7},
    "gpt-4.1": {"input": 2e-6, "cached": 5e-7, "output": 8e-6},
    "gpt-4.1-mini": {"input": 4e-7, "cached": 1e-7, "output": 1.6e-6},
}


@dataclass(frozen=True)
class ModelPrices:
    """USD per single token, split by how the token is billed."""

    input: float  # fresh (uncached) input token
    cached_input: float  # input token served from the prompt cache
    output: float  # generated output token


# Fetched at most once per process.
_catalog: dict | None = None


def _fetch_remote() -> dict | None:
    """Download the price catalog; persist it to disk on success. Best-effort."""
    try:
        req = urllib.request.Request(
            settings.MODEL_PRICES_URL, headers={"User-Agent": "lawyer-chatbot"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - network/parse errors fall back to cache
        return None
    try:
        Path(settings.MODEL_PRICES_CACHE).write_text(
            json.dumps(data), encoding="utf-8"
        )
    except OSError:
        pass
    return data


def _read_disk_cache() -> dict | None:
    try:
        return json.loads(
            Path(settings.MODEL_PRICES_CACHE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def _load_catalog() -> dict:
    global _catalog
    if _catalog is None:
        _catalog = _fetch_remote() or _read_disk_cache() or {}
    return _catalog


def _exact_prices(name: str) -> ModelPrices | None:
    """Prices for an exact model id from the catalog, then the fallback table."""
    entry = _load_catalog().get(name)
    if entry is not None:
        input_cost = entry.get("input_cost_per_token", 0.0)
        return ModelPrices(
            input=input_cost,
            cached_input=entry.get("cache_read_input_token_cost", input_cost),
            output=entry.get("output_cost_per_token", 0.0),
        )
    fb = _FALLBACK.get(name)
    if fb is not None:
        return ModelPrices(
            input=fb["input"], cached_input=fb["cached"], output=fb["output"]
        )
    return None


def _candidates(model: str) -> list[str]:
    """Names to try, most specific first: the model, a ``provider/`` -stripped
    form, then progressively shorter prefixes (drops trailing ``-segment``s so a
    dated snapshot like ``gpt-4.1-2025-04-14`` falls back to ``gpt-4.1``)."""
    names: list[str] = []
    for base in (model, model.split("/", 1)[-1]):
        parts = base.split("-")
        for i in range(len(parts), 0, -1):
            name = "-".join(parts[:i])
            if name not in names:
                names.append(name)
    return names


def get_model_prices(model: str) -> ModelPrices:
    """Resolve per-token prices for ``model`` from the catalog, then fallback.

    Tries the exact model id first, then successively shorter prefixes so dated
    snapshots (e.g. ``gpt-4.1-2025-04-14``) resolve to their base model. The
    catalog stores per-token costs under ``input_cost_per_token`` /
    ``output_cost_per_token`` / ``cache_read_input_token_cost``; a model with no
    cached-input rate bills cached tokens at the full input rate. Fully unknown
    models yield zeros so accounting degrades gracefully.
    """
    for name in _candidates(model):
        prices = _exact_prices(name)
        if prices is not None:
            return prices
    return ModelPrices(input=0.0, cached_input=0.0, output=0.0)
