#!/usr/bin/env bash
# 在仓库根目录批量跑 code_intent/batch_index.py（JSONL 数据集）。
# 路径默认来自仓库根 .env：REPO_DIR / INDEX_STORE_DIR / INTENT2FIX_BUILD_DIR。
# 实时日志：终端输出 + 追加 code_intent/batch_index_summary.log（可用另一终端 tail -f）。
#
# 用法示例：
#   ./scripts/batch_index_dataset.sh
#   ./scripts/batch_index_dataset.sh --limit 5
#   BATCH_INDEX_PARALLEL_INSTANCES=3 ./scripts/batch_index_dataset.sh --jsonl INTENT2FIX/benchmark/foo.jsonl
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 "${ROOT}/code_intent/batch_index.py" "$@"
