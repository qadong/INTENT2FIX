from __future__ import annotations

import json
import logging
import hashlib
import difflib
import os
import re
import time
import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, TypeVar

import httpx
import numpy as np

try:
    import voyageai
except ImportError:  # pragma: no cover
    voyageai = None

from .env_utils import load_toenv
from .embedding_manifest import (
    FEATURE_INTENT_BUILDER,
    FUNCTION_INTENT_BUILDER,
    TEST_INTENT_BUILDER,
)

load_toenv()

logger = logging.getLogger(__name__)

DEFAULT_EMBED_DIM = int(os.getenv("CODE_INTENT_EMBED_DIM", "768"))
EMBED_DIM = DEFAULT_EMBED_DIM
INTENT_MAX_RETRIES = max(1, int(os.getenv("INTENT_MAX_RETRIES", "3")))
INTENT_ABORT_ON_FALLBACK = os.getenv("INTENT_ABORT_ON_FALLBACK", "0") == "1"
INTENT_RETRY_BASE_SECONDS = max(
    0.0, float(os.getenv("INTENT_RETRY_BASE_SECONDS", "1"))
)
INTENT_RETRY_MAX_SECONDS = max(
    INTENT_RETRY_BASE_SECONDS,
    float(os.getenv("INTENT_RETRY_MAX_SECONDS", "30")),
)

_T = TypeVar("_T")
FUNCTION_INTENT_VALIDATION_CONTRACT = "qualified-identifiers-v10"
FEATURE_SUMMARY_VALIDATION_CONTRACT = "qualified-identifiers-v11"


def _llm_generation_options(model: Optional[str] = None) -> Dict[str, Any]:
    """Return the request options that affect generated intent content."""
    options: Dict[str, Any] = {
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
        "response_format": {"type": "json_object"},
    }
    thinking_type = os.getenv("LLM_THINKING_TYPE", "").strip().lower()
    if thinking_type:
        if thinking_type not in {"enabled", "disabled"}:
            raise ValueError(
                "LLM_THINKING_TYPE must be 'enabled' or 'disabled'"
            )
        options["thinking"] = {"type": thinking_type}
    reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
    if reasoning_effort and thinking_type != "disabled":
        options["reasoning_effort"] = reasoning_effort
    max_tokens = os.getenv("LLM_MAX_TOKENS", "").strip()
    if max_tokens:
        parsed_max_tokens = int(max_tokens)
        if parsed_max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS must be positive")
        options["max_tokens"] = parsed_max_tokens
    return options


def llm_generation_profile() -> Dict[str, Any]:
    """Stable cache provenance for model and content-affecting request options."""
    return {
        "model": os.getenv("LLM_MODEL") or "heuristic-fallback",
        **_llm_generation_options(os.getenv("LLM_MODEL")),
    }

_VERB_CANDIDATES = [
    "get",
    "set",
    "init",
    "load",
    "save",
    "update",
    "check",
    "validate",
    "create",
    "destroy",
    "format",
    "parse",
    "handle",
    "convert",
    "render",
    "compute",
    "log",
]


def _align_dim(vec: np.ndarray) -> np.ndarray:
    if vec.size == EMBED_DIM:
        return vec.astype("float32")
    if vec.size > EMBED_DIM:
        return vec[:EMBED_DIM].astype("float32")
    padded = np.zeros(EMBED_DIM, dtype="float32")
    padded[: vec.size] = vec.astype("float32")
    return padded


@dataclass
class LLMConfig:
    base_url: Optional[str]
    api_key: Optional[str]
    model: Optional[str]
    timeout: float = 60.0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


class LLMClient:

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = httpx.Client(timeout=config.timeout) if config.enabled else None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_async_client(self) -> httpx.AsyncClient:
        if not self.config.enabled:
            raise RuntimeError("LLM client not configured")
        loop = asyncio.get_running_loop()
        if self._async_client is not None and self._async_loop is not loop:
            # Batch callers close the client before their event loop exits. If a
            # direct caller did not, never reuse an httpx pool bound to that old
            # loop; doing so fails later with the opaque "Event loop is closed".
            self._async_client = None
            self._async_loop = None
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.config.timeout)
            self._async_loop = loop
        return self._async_client

    async def aclose(self) -> None:
        client = self._async_client
        self._async_client = None
        self._async_loop = None
        if client is not None:
            await client.aclose()

    async def _request_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = self._get_async_client()
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        resp = await client.post(endpoint, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            if os.getenv("INTENT_DEBUG") == "1":
                print(f"[LLM-Debug] HTTP Error {resp.status_code}: {resp.text}")
            raise
            
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            if os.getenv("INTENT_DEBUG") == "1":
                print(f"[LLM-Debug] JSON Decode Error. Raw response: {resp.text}")
            raise ValueError(f"LLM API returned invalid JSON: {exc}") from exc

    async def _chat_async(self, system_prompt: str, user_prompt: str) -> str:
        if not self.config.enabled:
            raise RuntimeError("LLM client not configured")
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **_llm_generation_options(self.config.model),
        }
        data = await self._request_async(payload)
        content = data["choices"][0]["message"]["content"]
        if not content:
            if os.getenv("INTENT_DEBUG") == "1":
                print(f"[LLM-Debug] API returned empty content for model {self.config.model}")
            raise ValueError(f"LLM model {self.config.model} returned empty response content")
        return content

    async def generate_function_intent_async(self, func: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        content = _build_function_prompt(func, context)
        response = await self._chat_async(_FUNC_SYSTEM_PROMPT, content)
        return _parse_intent_response(response)

    async def generate_test_intent_async(self, test: Dict[str, Any]) -> Dict[str, Any]:
        content = _build_test_intent_prompt(test)
        response = await self._chat_async(_TEST_INTENT_SYSTEM_PROMPT, content)
        return _parse_test_intent_response(response)

    async def generate_class_intent_async(self, cls: Dict[str, Any]) -> Dict[str, Any]:
        content = _build_class_prompt(cls)
        response = await self._chat_async(_CLASS_SYSTEM_PROMPT, content)
        return _parse_class_response(response)

    async def summarize_cluster_async(
        self,
        members: List[Dict[str, Any]],
        cluster_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = _build_cluster_prompt(members, cluster_context)
        response = await self._chat_async(_CLUSTER_SYSTEM_PROMPT, prompt)
        return _parse_feature_response(response)

    async def refine_identifier_grounding_async(
        self,
        generated: Dict[str, Any],
        issues: List[str],
        evidence_catalog: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = json.dumps(
            {
                "generated_json": generated,
                "rejected_identifiers": issues,
                "grounded_evidence": evidence_catalog,
            },
            ensure_ascii=False,
        )
        response = await self._chat_async(
            _IDENTIFIER_GROUNDING_SYSTEM_PROMPT, prompt
        )
        return _robust_json_loads(response)

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self._client is not None
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        resp = self._client.post(endpoint, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            if os.getenv("INTENT_DEBUG") == "1":
                print(f"[LLM-Debug] HTTP Error {resp.status_code}: {resp.text}")
            raise
            
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            if os.getenv("INTENT_DEBUG") == "1":
                print(f"[LLM-Debug] JSON Decode Error. Raw response: {resp.text}")
            raise ValueError(f"LLM API returned invalid JSON: {exc}") from exc

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self.config.enabled:
            raise RuntimeError("LLM client not configured")
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **_llm_generation_options(self.config.model),
        }
        data = self._request(payload)
        content = data["choices"][0]["message"]["content"]
        if not content:
            if os.getenv("INTENT_DEBUG") == "1":
                print(f"[LLM-Debug] API returned empty content for model {self.config.model}")
            raise ValueError(f"LLM model {self.config.model} returned empty response content")
        return content

    def generate_function_intent(self, func: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        content = _build_function_prompt(func, context)
        response = self._chat(_FUNC_SYSTEM_PROMPT, content)
        return _parse_intent_response(response)

    def generate_class_intent(self, cls: Dict[str, Any]) -> Dict[str, Any]:
        content = _build_class_prompt(cls)
        response = self._chat(_CLASS_SYSTEM_PROMPT, content)
        return _parse_class_response(response)

    def summarize_cluster(
        self,
        members: List[Dict[str, Any]],
        cluster_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = _build_cluster_prompt(members, cluster_context)
        response = self._chat(_CLUSTER_SYSTEM_PROMPT, prompt)
        return _parse_feature_response(response)

    def refine_identifier_grounding(
        self,
        generated: Dict[str, Any],
        issues: List[str],
        evidence_catalog: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = json.dumps(
            {
                "generated_json": generated,
                "rejected_identifiers": issues,
                "grounded_evidence": evidence_catalog,
            },
            ensure_ascii=False,
        )
        response = self._chat(_IDENTIFIER_GROUNDING_SYSTEM_PROMPT, prompt)
        return _robust_json_loads(response)

    def summarize_test_feature(
        self,
        test_intent: Dict[str, Any],
        members: List[Dict[str, Any]],
        member_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = _build_test_feature_prompt(test_intent, members, member_context)
        response = self._chat(_TEST_FEATURE_SYSTEM_PROMPT, prompt)
        return _parse_feature_response(response)


@dataclass
class VoyageEmbeddingConfig:
    api_key: Optional[str]
    model: Optional[str]
    base_url: Optional[str] = None
    timeout: float = 60.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)


class VoyageEmbeddingClient:

    def __init__(self, config: VoyageEmbeddingConfig) -> None:
        self.config = config
        self._client = None
        self._async_client = None
        if config.enabled and voyageai is not None:
            if config.base_url:
                logger.warning(
                    "VoyageAI SDK does not support custom base_url, VOYAGE_API_URL ignored."
                )
            self._client = voyageai.Client(api_key=config.api_key)
            self._async_client = voyageai.AsyncClient(api_key=config.api_key)
        elif config.enabled and voyageai is None:
            logger.warning(
                "voyageai SDK not installed, using stub for embedding."
            )

    async def embed_texts_async(self, texts: List[str]) -> np.ndarray:
        if not self.config.enabled or self._async_client is None:
            raise RuntimeError("Voyage embedding client not configured")
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype="float32")
        result = await self._async_client.embed(
            texts,
            model=self.config.model,
        )
        vectors = []
        embeddings = getattr(result, "embeddings", None) or result.get("embeddings")
        if embeddings is None:
            raise ValueError("Voyage embedding response missing embeddings field")
        for emb in embeddings:
            vec = np.array(emb, dtype="float32")
            if vec.size == 0:
                vec = np.zeros(EMBED_DIM, dtype="float32")
            vectors.append(_align_dim(vec))
        if len(vectors) != len(texts):
            raise ValueError("Voyage API response count does not match request")
        return np.vstack(vectors)

    def embed_texts(
        self, texts: List[str], input_type: Optional[str] = None
    ) -> np.ndarray:
        if not self.config.enabled or self._client is None:
            raise RuntimeError("Voyage embedding client not configured")
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype="float32")
        result = self._client.embed(
            texts,
            model=self.config.model,
            input_type=input_type,
        )
        vectors = []
        embeddings = getattr(result, "embeddings", None) or result.get("embeddings")
        if embeddings is None:
            raise ValueError("Voyage embedding response missing embeddings field")
        for emb in embeddings:
            vec = np.array(emb, dtype="float32")
            if vec.size == 0:
                vec = np.zeros(EMBED_DIM, dtype="float32")
            vectors.append(_align_dim(vec))
        if len(vectors) != len(texts):
            raise ValueError("Voyage API response count does not match request")
        return np.vstack(vectors)

    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.config.enabled or self._client is None:
            raise RuntimeError("Voyage client not configured")
        result = self._client.rerank(
            query,
            documents,
            model="rerank-2",
            top_k=top_k
        )
        
        reranked = []
        for hit in result.results:
            reranked.append({
                "index": hit.index,
                "score": hit.relevance_score,
                "document": hit.document
            })
        return reranked


def _load_llm_client() -> LLMClient:
    config = LLMConfig(
        base_url=os.getenv("LLM_API_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL"),
    )
    if not config.enabled:
        logger.info("LLM client not configured, using heuristic fallback")
    return LLMClient(config)


def _load_embedding_client() -> VoyageEmbeddingClient:
    api_key = os.getenv("VOYAGE_API_KEY")
    model = os.getenv("VOYAGE_EMBED_MODEL")
    base_url = os.getenv("VOYAGE_API_URL")
    
    config = VoyageEmbeddingConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    if not config.enabled:
        logger.warning(
            "Voyage embedding not fully configured (VOYAGE_API_KEY=%s, VOYAGE_EMBED_MODEL=%s), using local stub",
            "set" if api_key else "missing",
            model or "missing"
        )
    return VoyageEmbeddingClient(config)


_LLM_CLIENT = _load_llm_client()
_EMBED_CLIENT = _load_embedding_client()


async def close_llm_async_client() -> None:
    """Close the HTTP pool before an asyncio.run() batch closes its loop."""
    await _LLM_CLIENT.aclose()


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def embed_query(text: str) -> np.ndarray:
    if _EMBED_CLIENT.config.enabled:
        try:
            return _EMBED_CLIENT.embed_texts([text], input_type="query")[0]
        except Exception as exc:  # pragma: no cover
            if os.getenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK") == "1":
                raise RuntimeError("Voyage query embedding call failed") from exc
            logger.warning("Voyage query embedding call failed, using stub: %s", exc)
    elif os.getenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK") == "1":
        raise RuntimeError("Voyage embedding client not configured")
    return _stub_embed_texts([text])[0]


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    if _EMBED_CLIENT.config.enabled:
        try:
            return _EMBED_CLIENT.embed_texts(texts)
        except Exception as exc:  # pragma: no cover
            if os.getenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK") == "1":
                raise RuntimeError("Voyage document embedding call failed") from exc
            logger.warning("Voyage embedding call failed, using stub: %s", exc)
    elif os.getenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK") == "1":
        raise RuntimeError("Voyage embedding client not configured")
    return _stub_embed_texts(texts)


async def embed_texts_async(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    if _EMBED_CLIENT.config.enabled:
        try:
            return await _EMBED_CLIENT.embed_texts_async(texts)
        except Exception as exc:  # pragma: no cover
            if os.getenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK") == "1":
                raise
            logger.warning("Voyage embedding call failed, using stub: %s", exc)
    elif os.getenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK") == "1":
        raise RuntimeError("Voyage embedding client not configured")
    return _stub_embed_texts(texts)


def generate_intent_from_code(func: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if _LLM_CLIENT.config.enabled:
        readable_name = func.get("fq_name") or func.get("short_name") or func.get("id")
        request_context = dict(context or {})
        grounding_retries = 0
        grounding_refinements = 0
        identifier_repairs: Dict[str, str] = {}
        identifier_redactions: List[str] = []

        def generate_grounded_intent() -> Dict[str, Any]:
            nonlocal grounding_retries, grounding_refinements
            nonlocal request_context, identifier_repairs, identifier_redactions
            result = _LLM_CLIENT.generate_function_intent(
                func, request_context or None
            )
            result, repairs = _repair_function_intent_root_names(
                result, func, context
            )
            identifier_repairs.update(repairs)
            issues = function_intent_grounding_issues(result, func, context)
            result, repairs = _repair_close_identifier_names(
                result, issues, _function_identifier_evidence(func, context)
            )
            identifier_repairs.update(repairs)
            issues = function_intent_grounding_issues(result, func, context)
            if issues:
                grounding_refinements += 1
                evidence = _function_identifier_evidence(func, context)
                refined = _LLM_CLIENT.refine_identifier_grounding(
                    result, issues, _identifier_evidence_catalog(evidence, issues)
                )
                result = _merge_identifier_refinement(result, refined)
                result = _parse_intent_response(
                    json.dumps(result, ensure_ascii=False)
                )
                result, repairs = _repair_function_intent_root_names(
                    result, func, context
                )
                identifier_repairs.update(repairs)
                issues = function_intent_grounding_issues(result, func, context)
                result, repairs = _repair_close_identifier_names(
                    result, issues, evidence
                )
                identifier_repairs.update(repairs)
                issues = function_intent_grounding_issues(result, func, context)
                if issues:
                    result, redactions = _redact_ungrounded_qualified_identifiers(
                        result, issues, evidence
                    )
                    identifier_redactions.extend(redactions)
                    issues = function_intent_grounding_issues(
                        result, func, context
                    )
            if issues:
                grounding_retries += 1
                request_context = {
                    **(context or {}),
                    "grounding_correction": {
                        "rejected_identifiers": issues,
                        "instruction": (
                            "Regenerate the whole JSON response. Remove or copy-correct "
                            "every rejected identifier using only the supplied source and "
                            "relationship names."
                        ),
                    },
                }
                raise ValueError(
                    "function intent contains ungrounded identifiers: "
                    + ", ".join(issues)
                )
            return result

        try:
            intent = _call_llm_with_retry(
                generate_grounded_intent,
                f"function_intent:{readable_name}",
            )
            if os.getenv("INTENT_DEBUG") == "1":
                print(
                    f"[LLM-FuncIntent] {readable_name}: "
                    f"{json.dumps(intent, ensure_ascii=False)}"
                )
            intent["_intent_generation"] = "llm"
            intent["_intent_grounding_retries"] = grounding_retries
            intent["_intent_grounding_refinements"] = grounding_refinements
            intent["_intent_identifier_repairs"] = identifier_repairs
            intent["_intent_identifier_redactions"] = sorted(
                set(identifier_redactions)
            )
            return intent
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM function intent generation failed [%s]: %s", readable_name, exc)
            if INTENT_ABORT_ON_FALLBACK:
                raise RuntimeError(
                    f"LLM function intent failed after retries (func={readable_name})"
                ) from exc
    intent = _heuristic_intent(func)
    intent["_intent_generation"] = "heuristic"
    return intent


async def generate_intent_from_code_async(func: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if _LLM_CLIENT.config.enabled:
        readable_name = func.get("fq_name") or func.get("short_name") or func.get("id")
        request_context = dict(context or {})
        grounding_retries = 0
        grounding_refinements = 0
        identifier_repairs: Dict[str, str] = {}
        identifier_redactions: List[str] = []

        async def generate_grounded_intent() -> Dict[str, Any]:
            nonlocal grounding_retries, grounding_refinements
            nonlocal request_context, identifier_repairs, identifier_redactions
            result = await _LLM_CLIENT.generate_function_intent_async(
                func, request_context or None
            )
            result, repairs = _repair_function_intent_root_names(
                result, func, context
            )
            identifier_repairs.update(repairs)
            issues = function_intent_grounding_issues(result, func, context)
            result, repairs = _repair_close_identifier_names(
                result, issues, _function_identifier_evidence(func, context)
            )
            identifier_repairs.update(repairs)
            issues = function_intent_grounding_issues(result, func, context)
            if issues:
                grounding_refinements += 1
                evidence = _function_identifier_evidence(func, context)
                refined = await _LLM_CLIENT.refine_identifier_grounding_async(
                    result, issues, _identifier_evidence_catalog(evidence, issues)
                )
                result = _merge_identifier_refinement(result, refined)
                result = _parse_intent_response(
                    json.dumps(result, ensure_ascii=False)
                )
                result, repairs = _repair_function_intent_root_names(
                    result, func, context
                )
                identifier_repairs.update(repairs)
                issues = function_intent_grounding_issues(result, func, context)
                result, repairs = _repair_close_identifier_names(
                    result, issues, evidence
                )
                identifier_repairs.update(repairs)
                issues = function_intent_grounding_issues(result, func, context)
                if issues:
                    result, redactions = _redact_ungrounded_qualified_identifiers(
                        result, issues, evidence
                    )
                    identifier_redactions.extend(redactions)
                    issues = function_intent_grounding_issues(
                        result, func, context
                    )
            if issues:
                grounding_retries += 1
                request_context = {
                    **(context or {}),
                    "grounding_correction": {
                        "rejected_identifiers": issues,
                        "instruction": (
                            "Regenerate the whole JSON response. Remove or copy-correct "
                            "every rejected identifier using only the supplied source and "
                            "relationship names."
                        ),
                    },
                }
                raise ValueError(
                    "function intent contains ungrounded identifiers: "
                    + ", ".join(issues)
                )
            return result

        try:
            intent = await _call_llm_with_retry_async(
                generate_grounded_intent,
                f"function_intent:{readable_name}",
            )
            if os.getenv("INTENT_DEBUG") == "1":
                print(f"[LLM-FuncIntent] {readable_name}: {json.dumps(intent, ensure_ascii=False)}")
            intent["_intent_generation"] = "llm"
            intent["_intent_grounding_retries"] = grounding_retries
            intent["_intent_grounding_refinements"] = grounding_refinements
            intent["_intent_identifier_repairs"] = identifier_repairs
            intent["_intent_identifier_redactions"] = sorted(
                set(identifier_redactions)
            )
            return intent
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM function intent generation failed [%s]: %s", readable_name, exc)
            if INTENT_ABORT_ON_FALLBACK:
                raise RuntimeError(f"LLM function intent failed after retries (func={readable_name})") from exc
    intent = _heuristic_intent(func)
    intent["_intent_generation"] = "heuristic"
    return intent


async def generate_test_intent_from_code_async(test: Dict[str, Any]) -> Dict[str, Any]:
    if _LLM_CLIENT.config.enabled:
        readable_name = test.get("fq_name") or test.get("short_name") or test.get("id")
        try:
            intent = await _call_llm_with_retry_async(
                lambda: _LLM_CLIENT.generate_test_intent_async(test),
                f"test_intent:{readable_name}",
            )
            intent["_intent_generation"] = "llm"
            return intent
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM test intent generation failed [%s]: %s", readable_name, exc)
            if INTENT_ABORT_ON_FALLBACK:
                raise RuntimeError(
                    f"LLM test intent failed after retries (test={readable_name})"
                ) from exc
    if INTENT_ABORT_ON_FALLBACK:
        raise RuntimeError("LLM test-intent client not configured")
    intent = _heuristic_test_intent(test)
    intent["_intent_generation"] = "heuristic"
    return intent


def generate_class_intent(cls: Dict[str, Any]) -> Dict[str, Any]:
    if _LLM_CLIENT.config.enabled:
        readable_name = cls.get("fq_name") or cls.get("short_name") or cls.get("id")
        try:
            intent = _call_llm_with_retry(
                lambda: _LLM_CLIENT.generate_class_intent(cls),
                f"class_intent:{readable_name}",
            )
            if os.getenv("INTENT_DEBUG") == "1":
                print(
                    f"[LLM-ClassIntent] {readable_name}: "
                    f"{json.dumps(intent, ensure_ascii=False)}"
                )
            return intent
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM class intent generation failed [%s]: %s", readable_name, exc)
    return _heuristic_class_intent(cls)


async def generate_class_intent_async(cls: Dict[str, Any]) -> Dict[str, Any]:
    if _LLM_CLIENT.config.enabled:
        readable_name = cls.get("fq_name") or cls.get("short_name") or cls.get("id")
        try:
            intent = await _call_llm_with_retry_async(
                lambda: _LLM_CLIENT.generate_class_intent_async(cls),
                f"class_intent:{readable_name}",
            )
            if os.getenv("INTENT_DEBUG") == "1":
                print(f"[LLM-ClassIntent] {readable_name}: {json.dumps(intent, ensure_ascii=False)}")
            return intent
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM class intent generation failed [%s]: %s", readable_name, exc)
    return _heuristic_class_intent(cls)


def _cluster_identifier_evidence(
    members: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    authoritative_keys = (
        "signature",
        "class_name",
        "namespaces",
        "classes",
        "grounded_identifiers",
    )
    return [
        {key: member.get(key) for key in authoritative_keys if member.get(key)}
        for member in members
    ]


def _close_evidence_token_candidates(
    component: str, evidence_tokens: Iterable[str]
) -> List[tuple[int, float, str]]:
    normalized_component = component.lstrip("_")
    scored: List[tuple[int, float, str]] = []
    for candidate in evidence_tokens:
        if candidate == normalized_component or len(candidate) < 5:
            continue
        prefix = len(os.path.commonprefix([normalized_component, candidate]))
        suffix = len(
            os.path.commonprefix([normalized_component[::-1], candidate[::-1]])
        )
        direct_prefix = (
            candidate.startswith(normalized_component)
            and len(normalized_component) >= 5
        )
        direct_suffix = (
            candidate.endswith(normalized_component)
            and len(normalized_component) >= 5
        )
        score = difflib.SequenceMatcher(
            None, normalized_component, candidate
        ).ratio()
        if direct_prefix or direct_suffix or (
            score >= 0.82 and prefix >= 4 and suffix >= 4
        ):
            scored.append((prefix + suffix, score, candidate))
    return sorted(scored, key=lambda item: (-item[0], -item[1], item[2]))


def _has_unique_close_evidence_token(
    component: str, evidence_tokens: Iterable[str]
) -> bool:
    candidates = _close_evidence_token_candidates(component, evidence_tokens)
    if not candidates:
        return False
    if len(candidates) == 1:
        return True
    return (
        candidates[0][0] != candidates[1][0]
        or candidates[0][1] - candidates[1][1] >= 0.05
    )


def _identifier_grounding_issues(
    generated: Dict[str, Any], evidence_value: Any
) -> List[str]:
    evidence = json.dumps(evidence_value, ensure_ascii=False)
    output = " ".join(_iter_string_values(generated))
    identifier_pattern = re.compile(
        r"\b[A-Za-z_]\w*(?:::[A-Za-z_~]\w*)+"
    )
    token_pattern = re.compile(r"\b[A-Za-z_]\w*\b")
    evidence_tokens = set(token_pattern.findall(evidence))
    evidence_roots = {
        identifier.split("::", 1)[0]
        for identifier in identifier_pattern.findall(evidence)
    }
    issues: Set[str] = set()
    dangling_scope_pattern = re.compile(
        r"\b[A-Za-z_]\w*(?:::[A-Za-z_~]\w*)*::(?![A-Za-z_~])"
    )
    issues.update(dangling_scope_pattern.findall(output))
    for identifier in identifier_pattern.findall(output):
        components = [component.lstrip("~") for component in identifier.split("::")]
        if components[0] not in evidence_roots or any(
            component not in evidence_tokens for component in components
        ):
            issues.add(identifier)

    qualified_tokens = {
        component.lstrip("~")
        for identifier in identifier_pattern.findall(output)
        for component in identifier.split("::")
    }
    code_like_prefixes = (
        "get_",
        "set_",
        "parse_",
        "find_",
        "start_",
        "resume_",
        "make_",
        "is_",
        "has_",
        "read_",
        "write_",
        "unexpect_",
        "from_",
    )
    for token in token_pattern.findall(output):
        if token in evidence_tokens or token in qualified_tokens:
            continue
        if (
            token.startswith("_")
            or "__" in token
            or token.startswith(code_like_prefixes)
            or ("_" in token and token.upper() == token)
            or (token[0].islower() and re.search(r"[a-z][A-Z]", token))
            or (
                "_" in token
                and _has_unique_close_evidence_token(token, evidence_tokens)
            )
        ):
            issues.add(token)

    output_roots = {
        identifier.split("::", 1)[0]
        for identifier in identifier_pattern.findall(output)
    }
    for root in evidence_roots:
        if len(root) < 5:
            continue
        for token in output_roots:
            if token in evidence_tokens or len(token) < 4:
                continue
            if root.lower().startswith(token.lower()) and token.lower() != root.lower():
                issues.add(f"{token} (ungrounded shortening of {root})")
    return sorted(issues)


def _iter_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_string_values(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            yield from _iter_string_values(item)


def _identifier_evidence_catalog(
    evidence_value: Any, issues: Optional[List[str]] = None
) -> Dict[str, Any]:
    evidence = json.dumps(evidence_value, ensure_ascii=False)
    identifier_pattern = re.compile(
        r"\b[A-Za-z_]\w*(?:::[A-Za-z_~]\w*)+"
    )
    token_pattern = re.compile(r"\b[A-Za-z_]\w*\b")
    tokens = sorted(set(token_pattern.findall(evidence)))
    suggestions = {}
    for issue in issues or []:
        identifier = issue.split(" (", 1)[0]
        component = identifier.rsplit("::", 1)[-1].lstrip("~")
        matches = difflib.get_close_matches(component, tokens, n=5, cutoff=0.55)
        if matches:
            suggestions[issue] = matches
    return {
        "qualified_identifiers": sorted(set(identifier_pattern.findall(evidence)))[
            :200
        ],
        "identifier_tokens": tokens[:400],
        "suggested_replacements": suggestions,
    }


def _redact_ungrounded_qualified_identifiers(
    generated: Dict[str, Any], issues: List[str], evidence_value: Any
) -> tuple[Dict[str, Any], List[str]]:
    evidence = json.dumps(evidence_value, ensure_ascii=False)
    evidence_tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", evidence))
    replacements: Dict[str, str] = {}
    for issue in issues:
        if " (ungrounded shortening of " in issue:
            continue
        identifier = issue.split(" (", 1)[0]
        final_component = identifier.rsplit("::", 1)[-1].lstrip("~")
        replacement = (
            final_component
            if final_component in evidence_tokens
            else "the referenced internal helper"
        )
        replacements[identifier] = replacement
        if "::" in identifier and final_component not in evidence_tokens:
            replacements[final_component] = replacement
    if not replacements:
        return generated, []

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            for source, target in sorted(
                replacements.items(), key=lambda item: -len(item[0])
            ):
                value = re.sub(
                    _identifier_replacement_pattern(source), target, value
                )
            return value
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            redacted = {key: redact(item) for key, item in value.items()}
            tags = redacted.get("tags")
            if isinstance(tags, list):
                redacted["tags"] = [
                    tag
                    for tag in tags
                    if tag not in {
                        "the referenced internal helper",
                        "internal helper",
                    }
                ]
            return redacted
        return value

    return redact(generated), sorted(replacements)


def _repair_identifier_root_names(
    generated: Dict[str, Any], evidence_value: Any
) -> tuple[Dict[str, Any], Dict[str, str]]:
    evidence = json.dumps(evidence_value, ensure_ascii=False)
    output = " ".join(_iter_string_values(generated))
    identifier_pattern = re.compile(
        r"\b[A-Za-z_]\w*(?:::[A-Za-z_~]\w*)+"
    )
    token_pattern = re.compile(r"\b[A-Za-z_]\w*\b")
    evidence_tokens = set(token_pattern.findall(evidence))
    evidence_roots = {
        identifier.split("::", 1)[0]
        for identifier in identifier_pattern.findall(evidence)
    }
    output_roots = {
        identifier.split("::", 1)[0]
        for identifier in identifier_pattern.findall(output)
    }
    replacements: Dict[str, str] = {}
    for token in output_roots:
        if token in evidence_tokens or len(token) < 4:
            continue
        candidates = [
            root
            for root in evidence_roots
            if root.lower() == token.lower()
            or root.lower().startswith(token.lower())
        ]
        if len(candidates) == 1:
            replacements[token] = candidates[0]
    if not replacements:
        return generated, {}

    def repair(value: Any) -> Any:
        if isinstance(value, str):
            for source, target in sorted(
                replacements.items(), key=lambda item: -len(item[0])
            ):
                value = re.sub(rf"\b{re.escape(source)}\b", target, value)
            return value
        if isinstance(value, list):
            return [repair(item) for item in value]
        if isinstance(value, dict):
            return {key: repair(item) for key, item in value.items()}
        return value

    return repair(generated), replacements


def _identifier_replacement_pattern(identifier: str) -> str:
    relative_guard = r"(?<!::)" if "::" in identifier else ""
    end_guard = r"\b" if identifier[-1:].isalnum() or identifier.endswith("_") else r"(?!\w)"
    return rf"{relative_guard}\b{re.escape(identifier)}{end_guard}"


def _scoped_identifier_candidate(
    identifier: str, evidence_identifiers: List[str]
) -> Optional[str]:
    source_parts = [part.lstrip("~") for part in identifier.split("::")]
    candidates: List[tuple[float, str]] = []
    for candidate in evidence_identifiers:
        candidate_parts = [part.lstrip("~") for part in candidate.split("::")]
        if len(candidate_parts) != len(source_parts):
            continue
        mismatches = [
            (source, target)
            for source, target in zip(source_parts, candidate_parts)
            if source != target
        ]
        if len(mismatches) != 1:
            continue
        source, target = mismatches[0]
        normalized_source = source.lstrip("_")
        normalized_target = target.lstrip("_")
        prefix = len(os.path.commonprefix([normalized_source, normalized_target]))
        suffix = len(
            os.path.commonprefix(
                [normalized_source[::-1], normalized_target[::-1]]
            )
        )
        score = difflib.SequenceMatcher(
            None, normalized_source, normalized_target
        ).ratio()
        recognizable = (
            (
                len(normalized_source) >= 2
                and normalized_target.startswith(normalized_source)
            )
            or (
                len(normalized_source) >= 5
                and normalized_target.endswith(normalized_source)
            )
            or (score >= 0.82 and prefix >= 4 and suffix >= 4)
        )
        if recognizable:
            candidates.append((score, candidate))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        return None
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.05:
        return None
    return candidates[0][1]


def _repair_close_identifier_names_once(
    generated: Dict[str, Any], issues: List[str], evidence_value: Any
) -> tuple[Dict[str, Any], Dict[str, str]]:
    catalog = _identifier_evidence_catalog(evidence_value)
    evidence_identifiers = catalog["qualified_identifiers"]
    evidence_tokens = catalog["identifier_tokens"]
    replacements: Dict[str, str] = {}

    for issue in issues:
        if " (ungrounded shortening of " in issue:
            continue
        identifier = issue.split(" (", 1)[0]
        if identifier.endswith("::"):
            base_identifier = identifier[:-2]
            if all(
                component.lstrip("~") in evidence_tokens
                for component in base_identifier.split("::")
            ):
                replacements[identifier] = base_identifier
                continue
        if "::" in identifier:
            suffix_matches = [
                candidate
                for candidate in evidence_identifiers
                if candidate.endswith("::" + identifier)
            ]
            if len(suffix_matches) == 1:
                replacements[identifier] = suffix_matches[0]
                continue
            scoped_candidate = _scoped_identifier_candidate(
                identifier, evidence_identifiers
            )
            if scoped_candidate:
                replacements[identifier] = scoped_candidate
                continue

        component = identifier.rsplit("::", 1)[-1].lstrip("~")
        normalized_component = component.lstrip("_")
        if (
            normalized_component != component
            and normalized_component in evidence_tokens
        ):
            replacements[component] = normalized_component
            continue
        if component in evidence_tokens or len(normalized_component) < 5:
            continue
        scored = _close_evidence_token_candidates(
            normalized_component, evidence_tokens
        )
        if not scored:
            continue
        if (
            len(scored) > 1
            and scored[0][0] == scored[1][0]
            and scored[0][1] - scored[1][1] < 0.05
        ):
            continue
        replacements[component] = scored[0][2]

    if not replacements:
        return generated, {}

    def repair(value: Any) -> Any:
        if isinstance(value, str):
            for source, target in sorted(
                replacements.items(), key=lambda item: -len(item[0])
            ):
                value = re.sub(
                    _identifier_replacement_pattern(source), target, value
                )
            return value
        if isinstance(value, list):
            return [repair(item) for item in value]
        if isinstance(value, dict):
            return {key: repair(item) for key, item in value.items()}
        return value

    return repair(generated), replacements


def _repair_close_identifier_names(
    generated: Dict[str, Any], issues: List[str], evidence_value: Any
) -> tuple[Dict[str, Any], Dict[str, str]]:
    result = generated
    current_issues = issues
    all_replacements: Dict[str, str] = {}
    for _ in range(3):
        result, replacements = _repair_close_identifier_names_once(
            result, current_issues, evidence_value
        )
        if not replacements:
            break
        all_replacements.update(replacements)
        current_issues = _identifier_grounding_issues(result, evidence_value)
        if not current_issues:
            break
    return result, all_replacements


def _merge_identifier_refinement(
    generated: Dict[str, Any], refined: Dict[str, Any]
) -> Dict[str, Any]:
    candidate = refined
    expected_keys = set(generated)
    if not expected_keys.intersection(candidate):
        nested_dicts = [value for value in candidate.values() if isinstance(value, dict)]
        if len(nested_dicts) == 1:
            candidate = nested_dicts[0]
    return {
        key: candidate.get(key, value)
        for key, value in generated.items()
    }


def _cluster_summary_grounding_issues(
    summary: Dict[str, Any], members: List[Dict[str, Any]]
) -> List[str]:
    return _identifier_grounding_issues(
        summary, _cluster_identifier_evidence(members)
    )


def _repair_cluster_summary_root_names(
    summary: Dict[str, Any], members: List[Dict[str, Any]]
) -> tuple[Dict[str, Any], Dict[str, str]]:
    return _repair_identifier_root_names(
        summary, _cluster_identifier_evidence(members)
    )


def _function_identifier_evidence(
    func: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "signature": func.get("fq_name")
        or func.get("signature")
        or func.get("id"),
        "return_type": func.get("return_type"),
        "parameter_types": func.get("parameter_types") or [],
        "namespaces": func.get("namespaces") or [],
        "classes": func.get("classes") or [],
        "source": func.get("code") or "",
        "documentation": func.get("raw_comment") or "",
        "relations": context or {},
    }


def function_intent_grounding_issues(
    intent: Dict[str, Any],
    func: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    return _identifier_grounding_issues(
        intent, _function_identifier_evidence(func, context)
    )


def _repair_function_intent_root_names(
    intent: Dict[str, Any],
    func: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    return _repair_identifier_root_names(
        intent, _function_identifier_evidence(func, context)
    )


def summarize_cluster(
    members: List[Dict[str, Any]],
    cluster_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if _LLM_CLIENT.config.enabled:
        request_context = dict(cluster_context or {})
        grounding_retries = 0
        grounding_refinements = 0
        identifier_repairs: Dict[str, str] = {}
        identifier_redactions: List[str] = []

        def generate_grounded_summary() -> Dict[str, Any]:
            nonlocal grounding_retries, grounding_refinements
            nonlocal request_context, identifier_repairs, identifier_redactions
            result = _LLM_CLIENT.summarize_cluster(members, request_context)
            result, repairs = _repair_cluster_summary_root_names(result, members)
            identifier_repairs.update(repairs)
            issues = _cluster_summary_grounding_issues(result, members)
            evidence = _cluster_identifier_evidence(members)
            result, repairs = _repair_close_identifier_names(
                result, issues, evidence
            )
            identifier_repairs.update(repairs)
            issues = _cluster_summary_grounding_issues(result, members)
            if issues:
                grounding_refinements += 1
                refined = _LLM_CLIENT.refine_identifier_grounding(
                    result, issues, _identifier_evidence_catalog(evidence, issues)
                )
                result = _merge_identifier_refinement(result, refined)
                result = _parse_feature_response(
                    json.dumps(result, ensure_ascii=False)
                )
                result, repairs = _repair_cluster_summary_root_names(
                    result, members
                )
                identifier_repairs.update(repairs)
                issues = _cluster_summary_grounding_issues(result, members)
                result, repairs = _repair_close_identifier_names(
                    result, issues, evidence
                )
                identifier_repairs.update(repairs)
                issues = _cluster_summary_grounding_issues(result, members)
                if issues:
                    result, redactions = _redact_ungrounded_qualified_identifiers(
                        result, issues, evidence
                    )
                    identifier_redactions.extend(redactions)
                    issues = _cluster_summary_grounding_issues(result, members)
            if issues:
                grounding_retries += 1
                request_context = {
                    **(cluster_context or {}),
                    "grounding_correction": {
                        "rejected_identifiers": issues,
                        "instruction": (
                            "Regenerate the whole JSON response. Remove or copy-correct "
                            "every rejected identifier using only spellings present in members."
                        ),
                    },
                }
                raise ValueError(
                    "feature summary contains ungrounded identifiers: "
                    + ", ".join(issues)
                )
            return result

        try:
            summary = _call_llm_with_retry(
                generate_grounded_summary,
                "cluster_summary",
            )
            if os.getenv("INTENT_DEBUG") == "1":
                print(
                    "[LLM-ClusterSummary] "
                    f"{json.dumps(summary, ensure_ascii=False)}"
                )
            summary["_intent_generation"] = "llm"
            summary["_intent_grounding_retries"] = grounding_retries
            summary["_intent_grounding_refinements"] = grounding_refinements
            summary["_intent_identifier_repairs"] = identifier_repairs
            summary["_intent_identifier_redactions"] = sorted(
                set(identifier_redactions)
            )
            return summary
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM summarization failed: %s", exc)
            if INTENT_ABORT_ON_FALLBACK:
                raise RuntimeError("LLM summarization failed after retries") from exc
    summary = _heuristic_cluster_summary(members, cluster_context)
    summary["_intent_generation"] = "heuristic"
    return summary


def rerank_documents(query: str, documents: List[str], top_k: int = 10, *, strict: bool = False) -> List[Dict[str, Any]]:
    if _EMBED_CLIENT.config.enabled:
        try:
            return _EMBED_CLIENT.rerank(query, documents, top_k=top_k)
        except Exception as exc:
            if strict:
                raise
            logger.warning("Voyage rerank failed: %s", exc)
    if strict:
        raise RuntimeError("Voyage rerank unavailable")
    return [{"index": i, "score": 1.0 / (i + 1), "document": doc} for i, doc in enumerate(documents[:top_k])]


# ---------------------------------------------------------------------------
# fallback / prompt helpers
# ---------------------------------------------------------------------------

def _stub_embed_texts(texts: List[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        hash_digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(hash_digest[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=EMBED_DIM).astype("float32")
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        vectors.append(vec)
    return np.vstack(vectors)


def _split_identifier(name: str) -> List[str]:
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", name)
    if not parts:
        return [name.lower()]
    return [p.lower() for p in parts]


def _guess_action_object(short_name: str) -> tuple[str, str]:
    lowered = short_name.lower()
    for verb in _VERB_CANDIDATES:
        if lowered.startswith(verb):
            remainder = short_name[len(verb):].strip("_:")
            noun = remainder or "value"
            noun_parts = _split_identifier(noun)
            noun_text = " ".join(noun_parts).strip()
            action_phrase = f"{verb} {noun_text}".strip()
            return action_phrase, noun_text or "value"
    words = _split_identifier(short_name)
    noun_text = " ".join(words or ["value"])
    return f"handle {noun_text}", noun_text


def _heuristic_intent(func: Dict[str, Any]) -> Dict[str, Any]:
    fq_name = func.get("fq_name") or func.get("signature") or func.get("id", "")
    short_name = func.get("short_name") or fq_name.split("::")[-1]
    doc = (func.get("raw_comment") or "").strip()
    action_phrase, obj_phrase = _guess_action_object(short_name)
    short_intent = doc.splitlines()[0].strip("/* ") if doc else f"{action_phrase}."
    if not short_intent:
        short_intent = f"{action_phrase}."

    conditions: List[str] = []
    guards = func.get("macro_guards") or []
    if guards:
        conditions.append(f"Only when macros {', '.join(guards[:3])} are enabled")
    if func.get("is_method"):
        classes = "::".join(func.get("classes") or [])
        if classes:
            conditions.append(f"Requires {classes} instance")

    side_effects: List[str] = []
    if func.get("log_strings"):
        side_effects.append("Logs or error output")
    if func.get("macros_used"):
        side_effects.append("Depends on compile-time macros")

    raw_summary = doc or (
        f"Implements {fq_name} in {func.get('def_file', 'unknown file')}."
    )

    return {
        "short_intent": short_intent,
        "action": action_phrase,
        "object": obj_phrase,
        "conditions": conditions,
        "side_effects": side_effects,
        "raw_summary": raw_summary,
        "usage_context": "",
    }


def _heuristic_test_intent(test: Dict[str, Any]) -> Dict[str, Any]:
    name = str(test.get("fq_name") or test.get("short_name") or "test case")
    focal_entities = test.get("focal_entities") or []
    focal_names = [str(item.get("name") or "") for item in focal_entities if item.get("name")]
    oracle = str(test.get("oracle_code") or "").strip()
    action = f"exercise {focal_names[0]}" if focal_names else f"execute {name}"
    return {
        "short_intent": f"Validate the observable behavior exercised by {name}.",
        "action": action,
        "object": focal_names[0] if focal_names else name,
        "conditions": [],
        "side_effects": [],
        "raw_summary": oracle or f"Test case {name} in {test.get('def_file', 'unknown file')}.",
        "usage_context": "Repository regression test.",
        "oracle": oracle,
        "fixture": str(test.get("parent_name") or ""),
        "covered_symbols": focal_names[:3],
    }


def _heuristic_class_intent(cls: Dict[str, Any]) -> Dict[str, Any]:
    fq_name = cls.get("fq_name") or cls.get("id", "")
    short_name = cls.get("short_name") or fq_name.split("::")[-1]
    doc = (cls.get("raw_comment") or "").strip()

    short_intent = doc.splitlines()[0].strip("/* ") if doc else f"Class {short_name}."
    if not short_intent:
        short_intent = f"Class {short_name}."

    return {
        "short_intent": short_intent,
        "raw_summary": doc or f"Defines class {fq_name} in {cls.get('def_file', 'unknown file')}.",
        "responsibility": f"Management of {short_name} related logic.",
    }


def _heuristic_cluster_summary(
    members: List[Dict[str, Any]],
    cluster_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from collections import Counter

    if not members:
        return {
            "feature_name": "Empty Feature",
            "feature_description": "No functions assigned.",
            "tags": [],
        }

    context = cluster_context or {}
    if context.get("summary_role") == "coarse_navigation":
        representatives = [
            m.get("signature") or m.get("func_id")
            for m in members
            if m.get("_cluster_representative")
        ]
        representatives = [name for name in representatives if name]
        total = context.get("total_member_count") or len(members)
        centers = ", ".join(representatives[:3]) or "sampled functions"
        return {
            "feature_name": f"Structural neighborhood around {centers}",
            "feature_description": (
                f"Coarse navigation community containing {total} functions and centered "
                f"on {centers}. Use its refined child communities or member functions "
                "to identify a concrete behavior."
            ),
            "tags": ["structural-neighborhood", *representatives[:3]],
        }

    actions = Counter(m.get("action", "") for m in members if m.get("action"))
    objects = Counter(m.get("object", "") for m in members if m.get("object"))
    top_action = actions.most_common(1)[0][0] if actions else "handle request"
    top_object = objects.most_common(1)[0][0] if objects else "data"
    feature_name = f"{top_action.title()} ({top_object})"
    summary_sentences = [
        m.get("raw_summary", "") for m in members if m.get("raw_summary")
    ]
    summary_text = " ".join(summary_sentences[:3]).strip()
    if not summary_text:
        summary_text = (
            f"Describes how functions collaborate to {top_action} for {top_object}."
        )
    tags = sorted({*(top_action.split()), *(top_object.split())})
    return {
        "feature_name": feature_name.strip(),
        "feature_description": summary_text,
        "tags": tags or ["feature"],
    }


def generate_stub_intent(func: Dict[str, Any]) -> Dict[str, Any]:
    return _heuristic_intent(func)


def summarize_test_feature(
    test_intent: Dict[str, Any], members: List[Dict[str, Any]]
) -> Dict[str, Any]:
    member_context = _collect_member_context(members, test_intent)
    if _LLM_CLIENT.config.enabled:
        try:
            summary = _call_llm_with_retry(
                lambda: _LLM_CLIENT.summarize_test_feature(
                    test_intent, members, member_context
                ),
                "test_feature_summary",
            )
            if os.getenv("INTENT_DEBUG") == "1":
                print(
                    "[LLM-TestFeature] "
                    f"{json.dumps(summary, ensure_ascii=False)}"
                )
            summary["_intent_generation"] = "llm"
            return summary
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM test feature summarization failed: %s", exc)
            if INTENT_ABORT_ON_FALLBACK:
                raise RuntimeError(
                    "LLM test feature summarization failed after retries"
                ) from exc
    summary = _summarize_test_feature_stub(test_intent, members, member_context)
    summary["_intent_generation"] = "heuristic"
    return summary


def _summarize_test_feature_stub(
    test_intent: Dict[str, Any],
    members: List[Dict[str, Any]],
    member_context: Dict[str, Any],
) -> Dict[str, Any]:
    test_short = test_intent.get("short_intent") or "Test-driven scenario"
    test_detail = test_intent.get("raw_summary", test_short)

    member_summary = (
        _heuristic_cluster_summary(members)
        if members
        else {
            "feature_name": "Test Verified Feature",
            "feature_description": "Functions covered by the associated tests.",
            "tags": ["test", "feature"],
        }
    )

    member_highlights = [
        m.get("short_intent") for m in members[:5] if m.get("short_intent")
    ]
    if not member_highlights:
        member_highlights = [
            m.get("raw_summary")
            for m in members[:5]
            if m.get("raw_summary")
        ]

    description_parts = [member_summary["feature_description"]]
    if member_highlights:
        description_parts.append(
            "Key covered functions: " + "; ".join(member_highlights)
        )
    description_parts.append(f"Validated by tests such as: {test_detail}")

    module_hint = _select_module_hint(member_context)
    feature_name = (
        f"{module_hint} — {member_summary['feature_name']}"
        if module_hint
        else member_summary["feature_name"]
    )
    if test_short and test_short.lower() not in feature_name.lower():
        feature_name = f"{feature_name} (tested)"

    tags = sorted(
        {
            *(member_summary.get("tags") or []),
            *test_short.lower().split(),
            *member_context.get("keywords", []),
            *member_context.get("top_dirs", []),
            *member_context.get("top_scopes", []),
            *member_context.get("domain_tags", []),
            "test-verified",
        }
    )

    return {
        "feature_name": feature_name,
        "feature_description": " ".join(description_parts),
        "tags": [tag for tag in tags if tag],
    }


def _build_function_prompt(func: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    code_snippet = func.get("code") or ""
    fq_name = func.get("fq_name") or func.get("short_name") or func.get("id")
    ret_type = func.get("return_type") or ""
    params = func.get("parameter_types") or []
    param_str = f"({', '.join(params)})"
    declaration = f"{ret_type} {fq_name}{param_str}".strip()
    
    doc = func.get("raw_comment") or ""
    file_path = func.get("def_file")
    namespaces = "::".join(func.get("namespaces") or [])
    classes = "::".join(func.get("classes") or [])

    payload = {
        "declaration": declaration,
        "file": file_path,
        "namespaces": namespaces,
        "classes": classes,
        "doc_comment": doc,
        "code": code_snippet[:8000],
    }

    if context:
        payload["context"] = context
    
    return json.dumps(payload, ensure_ascii=False)


def _build_test_intent_prompt(
    test: Dict[str, Any], *, include_source_locations: bool = False
) -> str:
    focal_entities = [
        {
            "name": focal.get("name"),
            "file": focal.get("file"),
            "relations": focal.get("relations") or [],
            "short_intent": focal.get("short_intent") or "",
            "raw_summary": focal.get("raw_summary") or "",
            "source": str(focal.get("source") or "")[:1200],
        }
        for focal in (test.get("focal_entities") or [])[:3]
    ]
    test_identity = {
        "name": test.get("fq_name") or test.get("short_name") or test.get("id"),
        "framework": test.get("framework") or "",
        "file": test.get("def_file"),
    }
    if include_source_locations:
        test_identity["start_line"] = test.get("def_start_line")
        test_identity["end_line"] = test.get("def_end_line")
    test_identity["parent"] = test.get("parent_name") or ""
    payload = {
        "test_identity": test_identity,
        "test_code": str(test.get("code") or "")[:8000],
        "inherited_setup": str(test.get("inherited_setup") or "")[:4000],
        "extracted_oracle": str(test.get("oracle_code") or "")[:2400],
        "direct_focal_entities": focal_entities,
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_class_prompt(cls: Dict[str, Any]) -> str:
    fq_name = cls.get("fq_name") or cls.get("short_name") or cls.get("id")
    bases = cls.get("bases") or []
    kind = cls.get("kind") or "class"
    code = cls.get("code") or ""
    base_str = f" : {', '.join(bases)}" if bases else ""
    declaration = f"{kind} {fq_name}{base_str}"
    
    doc = cls.get("raw_comment") or ""
    file_path = cls.get("def_file")
    
    payload = {
        "declaration": declaration,
        "kind": kind,
        "file": file_path,
        "doc_comment": doc,
    }
    if code and len(code) < 16000:
        payload["code"] = code
    else:
        payload["member_summary"] = cls.get("member_summary") or []
        if code:
            payload["note"] = "Code is too long and was replaced by member summary."

    return json.dumps(payload, ensure_ascii=False)


def _build_cluster_prompt(
    members: List[Dict[str, Any]],
    cluster_context: Optional[Dict[str, Any]] = None,
) -> str:
    payload = []
    for m in members[:50]:
        readable_name = m.get("signature") or m.get("func_id")
        payload.append({
            "function": readable_name,
            "file": m.get("file"),
            "class": m.get("class_name"),
            "exact_identifiers": m.get("grounded_identifiers") or [],
            "structural_representative": bool(m.get("_cluster_representative")),
            "sample_role": m.get("_cluster_sample_role") or "coverage_sample",
            "short_intent": m.get("short_intent"),
            "raw_summary": m.get("raw_summary"),
            "action": m.get("action"),
            "object": m.get("object"),
        })
    return json.dumps(
        {"cluster_context": cluster_context or {}, "members": payload},
        ensure_ascii=False,
    )


def _build_test_feature_prompt(
    test_intent: Dict[str, Any],
    members: List[Dict[str, Any]],
    member_context: Dict[str, Any],
) -> str:
    test_readable_name = test_intent.get("signature") or test_intent.get("func_id")
    payload = {
        "test_scenario": {
            "name": test_readable_name,
            "short_intent": test_intent.get("short_intent"),
            "raw_summary": test_intent.get("raw_summary"),
        },
        "member_context": member_context,
        "covered_functions": [
            {
                "name": m.get("signature") or m.get("func_id"),
                "short_intent": m.get("short_intent"),
                "raw_summary": m.get("raw_summary"),
                "action": m.get("action"),
                "object": m.get("object"),
                "file_hint": _shorten_path(m.get("file")),
                "class_name": m.get("class_name"),
            }
            for m in members[:50]
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _feature_summary_input_hash(kind: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "builder": FEATURE_INTENT_BUILDER,
        "kind": kind,
        "generation_profile": llm_generation_profile(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def feature_summary_prompt_contract() -> Dict[str, str]:
    sentinel_member = {
        "func_id": "sentinel-id",
        "signature": "sentinel::function",
        "file": "src/sentinel.cpp",
        "class_name": "sentinel::owner",
        "_cluster_representative": True,
        "_cluster_sample_role": "structural_representative",
        "short_intent": "Sentinel short intent.",
        "raw_summary": "Sentinel raw summary.",
        "action": "inspect sentinel",
        "object": "sentinel",
    }
    cluster_context = {
        "cluster_kind": "graph_community",
        "summary_role": "coarse_navigation",
        "total_member_count": 2,
        "sampled_member_count": 1,
        "has_refined_children": True,
    }
    test_intent = {
        "func_id": "sentinel-test",
        "signature": "sentinel::test",
        "short_intent": "Validate sentinel behavior.",
        "raw_summary": "Sentinel test summary.",
    }
    member_context = _collect_member_context([sentinel_member], test_intent)
    return {
        "cluster": _feature_summary_input_hash(
            "cluster-contract",
            _CLUSTER_SYSTEM_PROMPT,
            _build_cluster_prompt([sentinel_member], cluster_context),
        ),
        "test": _feature_summary_input_hash(
            "test-contract",
            _TEST_FEATURE_SYSTEM_PROMPT,
            _build_test_feature_prompt(
                test_intent, [sentinel_member], member_context
            ),
        ),
    }


def cluster_summary_input_hash(
    members: List[Dict[str, Any]],
    cluster_context: Optional[Dict[str, Any]] = None,
) -> str:
    return _feature_summary_input_hash(
        "cluster",
        _CLUSTER_SYSTEM_PROMPT,
        _build_cluster_prompt(members, cluster_context),
    )


def test_feature_summary_input_hash(
    test_intent: Dict[str, Any], members: List[Dict[str, Any]]
) -> str:
    member_context = _collect_member_context(members, test_intent)
    return _feature_summary_input_hash(
        "test",
        _TEST_FEATURE_SYSTEM_PROMPT,
        _build_test_feature_prompt(test_intent, members, member_context),
    )


_FUNC_SYSTEM_PROMPT = """
You are constructing grounded retrieval metadata for one C++ function. Use only the supplied
declaration, source, documentation, owner, and direct caller/callee/test names. Relationship names
are navigation evidence; they do not authorize invented runtime behavior.

Produce a JSON object with these keys:
- short_intent: one concrete sentence describing what the function computes, validates, converts,
  mutates, dispatches, or exposes. Do not merely restate the identifier or say "handle X".
- action: a precise verb phrase.
- object: the concrete value, state, protocol, or API acted on.
- conditions: only branch conditions or preconditions visible in the supplied evidence; otherwise [].
- side_effects: only observable mutation, I/O, error reporting, allocation, or state advancement
  visible in the supplied evidence; otherwise []. Do not infer invalidation or compatibility rules.
- raw_summary: 1-2 sentences describing the implementation role and its directly evidenced place
  in the call chain.
- usage_context: one sentence grounded in callers/tests, or an empty string when unknown.

Preserve qualified C++ API and type names when they distinguish overloads or subsystems. For a
forwarder, adapter, parser state transition, or template constraint, state that role explicitly.
Copy every identifier, enum value, and format name exactly as supplied; never abbreviate, repair,
or invent a spelling. Omit a name when the supplied evidence does not establish it.
Return strictly valid JSON.
""".strip()


def function_intent_input_hash(
    func: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> str:
    """Fingerprint every deterministic input that can change a function intent."""
    payload = {
        "builder": FUNCTION_INTENT_BUILDER,
        "validation_contract": FUNCTION_INTENT_VALIDATION_CONTRACT,
        "generation_profile": llm_generation_profile(),
        "source_hash": func.get("content_hash") or "",
        "system_prompt": _FUNC_SYSTEM_PROMPT,
        "user_prompt": _build_function_prompt(func, context),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_TEST_INTENT_SYSTEM_PROMPT = """
You are summarizing one repository C++ test case for semantic retrieval.
Use only the supplied test source, extracted assertions, and direct focal-entity evidence.
Do not infer product requirements that are not observable in those inputs.

Return strictly valid JSON with these keys:
- short_intent: one sentence describing the observable behavior validated by the test.
- action: the public operation exercised by the test.
- object: the public API or value under test.
- conditions: a list of concrete setup, input, or state preconditions.
- side_effects: a list of observable state changes, if any.
- raw_summary: 1-2 sentences connecting setup, action, and oracle.
- usage_context: one sentence describing the regression or API scenario.
- oracle: a concise description of the assertion, expected value, exception, or compile property.
- fixture: the fixture or parent test/section, or an empty string.
- covered_symbols: a list containing only directly evidenced production symbols.

When graph/source evidence is absent, summarize the test behavior without inventing a focal API.
""".strip()


def test_intent_input_hash(test: Dict[str, Any]) -> str:
    """Fingerprint semantic test-intent input without source coordinates."""
    payload = {
        "builder": TEST_INTENT_BUILDER,
        "generation_profile": llm_generation_profile(),
        "system_prompt": _TEST_INTENT_SYSTEM_PROMPT,
        "user_prompt": _build_test_intent_prompt(test),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _legacy_test_intent_input_hash_v4(test: Dict[str, Any]) -> str:
    """Reproduce the v4 location-sensitive key for verified cache migration."""
    payload = {
        "builder": "test-intent-v4",
        "generation_profile": llm_generation_profile(),
        "source_hash": test.get("content_hash") or "",
        "system_prompt": _TEST_INTENT_SYSTEM_PROMPT,
        "user_prompt": _build_test_intent_prompt(
            test, include_source_locations=True
        ),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

_CLASS_SYSTEM_PROMPT = """
You are a senior C++ engineer. Given a JSON describing a class's metadata,
produce a JSON object with keys:
short_intent (concise sentence describing the class's purpose), 
raw_summary (1-2 sentences explaining its role and design),
responsibility (concise phrase of what this class manages).
Return strictly valid JSON.
""".strip()

_IDENTIFIER_GROUNDING_SYSTEM_PROMPT = """
You correct identifier spelling in an existing JSON metadata object. Preserve its keys, behavioral
claims, scope, and level of detail. Change only rejected identifiers. Copy the closest exact
qualified identifier from grounded_evidence when it denotes the same entity; consult
suggested_replacements for source spellings close to a rejected token. Otherwise remove the
identifier and rephrase that clause without naming an unsupported entity. Do not introduce any
identifier absent from grounded_evidence. Return the complete corrected JSON object directly,
without wrapping it in generated_json, result, or any other key.
""".strip()

_CLUSTER_SYSTEM_PROMPT = """
You receive grounded metadata for functions selected as one structural-semantic C++ community.
cluster_context states whether this is a functional cluster, class component, or coarse navigation
community, and whether the supplied rows sample a larger membership. Rows marked
structural_representative are graph centers; they are not proof that every member has the same
responsibility. For a functional cluster or class component, infer the narrowest shared capability
supported by both the representatives and the coverage sample. For coarse_navigation, do not turn
one dominant representative family into the intent of the whole community: name it as a structural
neighborhood and describe the distinct grounded roles it connects. Refined children carry the
specific functional intents. Do not describe the entire library, join unrelated member names, or
invent product requirements. If a cluster is an internal mechanism, name that mechanism rather
than forcing a user-facing label. Preserve useful qualified API, class, protocol, and module names.
Copy identifier spellings exactly from the input and omit names that are not established there.
Never shorten or normalize a namespace, library, class, or function name; for example, a supplied
simdjson name must remain simdjson rather than simd.

Return strictly valid JSON with keys: feature_name, feature_description, tags (list of keywords).
For a functional cluster, the description should state the shared responsibility and main
collaborating roles. For coarse_navigation, it should state that this is a navigation community and
list the distinct observed roles without claiming that they implement one behavior. Use 1-3
sentences. Tags must be concise retrieval terms grounded in the input.
""".strip()

_TEST_FEATURE_SYSTEM_PROMPT = """
You receive JSON describing a test intent and the functions covered by it.
Return JSON with keys: feature_name, feature_description, and tags (list of keywords).
Describe the product capability validated by the tests—mention the tests briefly,
but keep the focus on the functionality delivered by the covered code.

Example:
Input JSON =>
{
  "test_intent": {"short_intent": "Ensure uploading a photo sets EXIF data"},
  "member_context": {"top_dirs": ["media/upload"], "keywords": ["EXIF", "jpeg"]},
  "functions": [
    {"short_intent": "Parse EXIF headers", "file_hint": "media/upload/exif.cc"},
    {"short_intent": "Store photo metadata in DB", "file_hint": "media/db/store.cc"}
  ]
}
Expected model output =>
{
  "feature_name": "Media upload — photo metadata preserved",
  "feature_description": "When users upload photos, the service parses EXIF headers and stores the metadata alongside the image so downstream features can display capture time and camera info. Tests verify EXIF parsing and DB persistence.",
  "tags": ["media", "upload", "exif", "metadata"]
}
""".strip()


def _shorten_path(path: Optional[str], max_parts: int = 3) -> Optional[str]:
    if not path:
        return None
    try:
        parts = Path(path).parts
        if len(parts) <= max_parts:
            return "/".join(parts)
        return "/".join(parts[-max_parts:])
    except Exception:
        return path


def _collect_member_context(
    members: List[Dict[str, Any]],
    test_intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scopes: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    dirs: Counter[str] = Counter()
    keywords: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()

    for member in members:
        namespaces = member.get("namespaces") or []
        cls = member.get("classes") or []
        scope = "::".join([*namespaces, *cls])
        if scope:
            scopes[scope] += 1
        class_name = member.get("class_name")
        if class_name:
            classes[class_name] += 1
        file_path = member.get("file")
        hint = _shorten_path(file_path, max_parts=4)
        if hint:
            dirs[hint] += 1
            _add_domain_tokens(domain_tokens, hint)
        short_intent = member.get("short_intent") or ""
        for word in short_intent.lower().split():
            if len(word) > 2:
                keywords[word.strip(",.")] += 1
        _add_domain_tokens(domain_tokens, short_intent)
        _add_domain_tokens(domain_tokens, member.get("raw_summary"))

    if test_intent:
        _add_domain_tokens(domain_tokens, test_intent.get("short_intent"))
        _add_domain_tokens(domain_tokens, test_intent.get("raw_summary"))
        if test_intent.get("file"):
            _add_domain_tokens(domain_tokens, test_intent.get("file"))

    def _top(counter: Counter[str], limit: int = 5) -> List[str]:
        return [item for item, _ in counter.most_common(limit)]

    return {
        "top_scopes": _top(scopes),
        "top_classes": _top(classes),
        "top_dirs": _top(dirs),
        "keywords": _top(keywords),
        "domain_tags": _top(domain_tokens, limit=8),
    }


def _select_module_hint(context: Dict[str, Any]) -> Optional[str]:
    for key in ("top_scopes", "top_classes", "top_dirs", "domain_tags"):
        values = context.get(key)
        if values:
            return values[0]
    return None


_DOMAIN_STOP_WORDS = {
    "the",
    "and",
    "with",
    "this",
    "that",
    "when",
    "test",
    "tests",
    "function",
    "value",
    "file",
    "case",
    "check",
    "ensure",
    "fmt",
    "detail",
    "internal",
    "base",
    "impl",
    "class",
    "object",
    "short",
    "long",
}


def _add_domain_tokens(counter: "Counter[str]", text: Optional[str]) -> None:
    if not text:
        return
    for raw in re.findall(r"[A-Za-z0-9_:./-]+", text):
        segments = re.split(r"[:/._-]", raw)
        for seg in segments:
            if not seg:
                continue
            for token in _split_identifier(seg):
                token = token.lower()
                if len(token) <= 2 or token in _DOMAIN_STOP_WORDS:
                    continue
                counter[token] += 1


def _robust_json_loads(content: str) -> Dict[str, Any]:
    content = content.strip()
    if not content:
        raise ValueError("LLM returned empty content")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass
            
    raise ValueError(f"Could not parse JSON from: {content[:200]}...")


def _parse_intent_response(content: str) -> Dict[str, Any]:
    data = _robust_json_loads(content)
    required = ["short_intent", "action", "object", "conditions", "side_effects", "raw_summary", "usage_context"]
    for key in required:
        data.setdefault(key, "" if key not in ("conditions", "side_effects") else [])
    if not isinstance(data.get("conditions"), list):
        data["conditions"] = [str(data["conditions"])]
    if not isinstance(data.get("side_effects"), list):
        data["side_effects"] = [str(data["side_effects"])]
    return data


def _parse_test_intent_response(content: str) -> Dict[str, Any]:
    data = _parse_intent_response(content)
    data.setdefault("oracle", "")
    data.setdefault("fixture", "")
    covered_symbols = data.get("covered_symbols") or []
    if not isinstance(covered_symbols, list):
        covered_symbols = [str(covered_symbols)]
    data["covered_symbols"] = [
        str(symbol).strip() for symbol in covered_symbols if str(symbol).strip()
    ][:6]
    return data


def _parse_class_response(content: str) -> Dict[str, Any]:
    data = _robust_json_loads(content)
    required = ["short_intent", "raw_summary", "responsibility"]
    for key in required:
        data.setdefault(key, "")
    return data


def _parse_feature_response(content: str) -> Dict[str, Any]:
    data = _robust_json_loads(content)
    data.setdefault("feature_name", "Unnamed Feature")
    data.setdefault("feature_description", "")
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    data["tags"] = [str(t).strip() for t in tags if t]
    return data


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    response = exc.response
    if response is None or response.status_code != 429:
        return None
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    exponential = min(
        INTENT_RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1)),
        INTENT_RETRY_MAX_SECONDS,
    )
    retry_after = _retry_after_seconds(exc)
    return max(exponential, retry_after or 0.0)


def _call_llm_with_retry(fn: Callable[[], _T], label: str) -> _T:
    last_exc: Optional[Exception] = None
    for attempt in range(1, INTENT_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            logger.warning(
                "LLM call failed (%s/%s) [%s]: %s",
                attempt,
                INTENT_MAX_RETRIES,
                label,
                exc,
            )
            if attempt < INTENT_MAX_RETRIES:
                time.sleep(_retry_delay_seconds(exc, attempt))
    assert last_exc is not None
    raise last_exc


async def _call_llm_with_retry_async(
    fn: Callable[[], Awaitable[_T]], label: str
) -> _T:
    last_exc: Optional[Exception] = None
    for attempt in range(1, INTENT_MAX_RETRIES + 1):
        try:
            return await fn()
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            logger.warning(
                "LLM call failed (%s/%s) [%s]: %s",
                attempt,
                INTENT_MAX_RETRIES,
                label,
                exc,
            )
            if attempt < INTENT_MAX_RETRIES:
                await asyncio.sleep(_retry_delay_seconds(exc, attempt))
    assert last_exc is not None
    raise last_exc
