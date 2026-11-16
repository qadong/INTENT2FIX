"""
路径类环境变量在源码中只保留一套名字（勿再增加同义变量）：

- REPO_DIR：各 SweBench 实例源码目录的父路径
- INDEX_STORE_DIR：索引 / intent 产物根目录
- INTENT2FIX_BUILD_DIR：各实例独立 build 目录的父路径
"""
from __future__ import annotations

import os
from pathlib import Path

# 未设置环境变量时的默认根目录（可写、不绑定特定机器）；生产请用 .env 覆盖。
_DEFAULT_REPO_PARENT = "/tmp/repos"
_DEFAULT_INDEX_STORE = "/tmp/INTENT2FIX/index-store"
_DEFAULT_BUILD_PARENT = "/tmp/INTENT2FIX/builds"


def _path_from_env(key: str, default: str) -> Path:
    raw = (os.environ.get(key) or "").strip() or default
    return Path(raw).expanduser().resolve()


def repo_instances_parent() -> Path:
    """各 instance 源码目录（子目录名为 instance_id）的父路径。"""
    return _path_from_env("REPO_DIR", _DEFAULT_REPO_PARENT)


def index_store_parent() -> Path:
    """索引根（内含 <instance>/index_output 等）。"""
    return _path_from_env("INDEX_STORE_DIR", _DEFAULT_INDEX_STORE)


def index_store_parent_optional() -> Path | None:
    """同上；未设置 INDEX_STORE_DIR 时返回 None（用于必须显式配置的场景）。"""
    raw = (os.environ.get("INDEX_STORE_DIR") or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def build_instances_parent() -> Path:
    """各实例独立 build 目录的父路径。"""
    return _path_from_env("INTENT2FIX_BUILD_DIR", _DEFAULT_BUILD_PARENT)


def build_intent_output_default() -> str:
    """build_intent 的 --output-path 默认值，与全局索引根一致。"""
    return (os.environ.get("INDEX_STORE_DIR") or "").strip()


def build_intent_repo_cache_default() -> str:
    """build_intent 内置 demo 的实例根目录，与 REPO_DIR 一致。"""
    return str(repo_instances_parent())
