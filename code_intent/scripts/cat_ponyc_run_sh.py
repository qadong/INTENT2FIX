#!/usr/bin/env python3
"""
从各 mswebench ponyc 镜像读取 ``/home/run.sh``，写入 JSON：``instance_id`` → 脚本全文（失败为 ``null``）。

默认输出：``INTENT2FIX/benchmark/ponyc_image_run_sh.json``（可用 ``--output`` 覆盖）。

并发执行 ``docker run``（默认 ``-j`` 为 ``min(32, 实例数)``）；运行中在 **stderr** 约每 10% 打印 ``进度 k/n``，避免长时间无输出。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUT = _REPO_ROOT / "INTENT2FIX" / "benchmark" / "ponyc_image_run_sh.json"


def _docker_registry_prefix() -> str:
    return os.environ.get("CODE_INTENT_REGISTRY_PREFIX", "6pqg5gclf6yaz1ch6g.xuanyuan.run").strip().rstrip(
        "/"
    )


def _with_registry(image: str) -> str:
    p = _docker_registry_prefix()
    if not p:
        return image
    img = (image or "").strip()
    if not img or img.startswith(f"{p}/"):
        return img
    return f"{p}/{img}"


def resolve_mswebench_image(instance_id: str) -> str:
    """与 demo._resolve_docker_image 一致（不含 CODE_INTENT_IMAGE 单镜像覆盖）。"""
    match = re.match(r"^([^_]+)__(.+)-(\d+)$", instance_id)
    if not match:
        raise ValueError(f"无法解析 instance_id: {instance_id!r}")
    org, repo, pr_id = match.groups()
    return _with_registry(f"mswebench/{org.lower()}_m_{repo.lower()}:pr-{pr_id}")


def load_ponyc_instance_ids(jsonl: Path) -> list[str]:
    ids: list[str] = []
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            iid = str(r.get("instance_id", "")).strip()
            if iid.startswith("ponylang__ponyc-"):
                ids.append(iid)
    return sorted(set(ids), key=lambda x: (int(x.rsplit("-", 1)[-1]), x))


def cat_run_sh(image: str, *, dry_run: bool) -> tuple[int, str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "cat",
        image,
        "/home/run.sh",
    ]
    if dry_run:
        return 0, " ".join(cmd)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    body = (p.stdout or "").strip()
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        return p.returncode, (body + ("\n" + err if err else "")).strip()
    return p.returncode, body


def fetch_instance(iid: str, *, dry_run: bool) -> dict[str, str | None | bool]:
    """单实例：拉镜像内 run.sh；供线程池调用。"""
    try:
        img = resolve_mswebench_image(iid)
    except ValueError as e:
        return {
            "iid": iid,
            "run_sh": None,
            "image": "",
            "detail": str(e),
            "ok": False,
        }
    if dry_run:
        _, hint = cat_run_sh(img, dry_run=True)
        return {
            "iid": iid,
            "run_sh": None,
            "image": img,
            "detail": hint,
            "ok": True,
        }
    code, text = cat_run_sh(img, dry_run=False)
    if code == 0:
        return {"iid": iid, "run_sh": text, "image": img, "detail": "", "ok": True}
    return {
        "iid": iid,
        "run_sh": None,
        "image": img,
        "detail": f"docker exit {code}",
        "ok": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="在各 ponyc mswebench 镜像内 cat /home/run.sh，写入 JSON")
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=_REPO_ROOT / "INTENT2FIX" / "benchmark" / "multi_swebench_c_change.jsonl",
        help="含 instance_id 的 jsonl",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"输出 JSON 路径（默认 {_DEFAULT_OUT.relative_to(_REPO_ROOT)}）",
    )
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0 表示全部）")
    ap.add_argument("--dry-run", action="store_true", help="只打印 docker 命令，不写 JSON、不拉容器")
    ap.add_argument(
        "--print",
        action="store_true",
        help="同时把每个实例的 image / run.sh 打到 stdout（内容大时慎用）",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="并发 docker 任务数；0 表示自动 min(32, 实例数)，至少为 1",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="不打印运行中进度（默认 stderr 约每 10%% 打一行）",
    )
    args = ap.parse_args()

    if not args.jsonl.is_file():
        print(f"❌ 找不到 jsonl: {args.jsonl}", file=sys.stderr)
        sys.exit(2)

    ids = load_ponyc_instance_ids(args.jsonl)
    if args.limit > 0:
        ids = ids[: args.limit]

    if not ids:
        print("未找到 ponylang__ponyc-* 实例", file=sys.stderr)
        sys.exit(1)

    n = len(ids)
    workers = args.jobs if args.jobs > 0 else min(32, max(1, n))
    print(
        f"共 {n} 个 ponyc 实例；REGISTRY={_docker_registry_prefix() or '(无)'}；并发 -j {workers}",
        file=sys.stderr,
    )

    rows_by_id: dict[str, dict[str, str | None | bool]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_instance, iid, dry_run=args.dry_run): iid for iid in ids}
        done = 0
        step = max(1, n // 10)  # 约 10 次进度行
        for fut in as_completed(futures):
            row = fut.result()
            rows_by_id[str(row["iid"])] = row
            done += 1
            if not args.no_progress and (done == 1 or done == n or done % step == 0):
                print(f"进度 {done}/{n} …", file=sys.stderr, flush=True)

    out_obj: dict[str, str | None] = {}
    failed = 0
    for iid in ids:
        row = rows_by_id[iid]
        if args.dry_run:
            out_obj[iid] = None
        else:
            rs = row["run_sh"]
            out_obj[iid] = rs if isinstance(rs, str) else None
            if not row["ok"]:
                failed += 1

    if args.print:
        for iid in ids:
            row = rows_by_id[iid]
            print(f"=== {iid} ===")
            if row["image"]:
                print(f"image: {row['image']}")
            if row["detail"]:
                print(row["detail"])
            rs = row["run_sh"]
            if isinstance(rs, str) and rs:
                print(rs)
            elif not args.dry_run and row["ok"]:
                print("(empty run.sh)")
            print()

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        ok_n = sum(1 for v in out_obj.values() if v is not None)
        print(
            f"✅ 已写入 {args.output}（{ok_n}/{len(out_obj)} 条有 run.sh 内容）",
            file=sys.stderr,
        )
    else:
        print("ℹ️ dry-run：未写 JSON", file=sys.stderr)

    if failed:
        print(f"⚠️ 解析失败或 docker 非零/无内容: {failed} 个（JSON 中对应键为 null）", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
