"""模式耗时对比评测

对 fast / balanced / deep / auto 四种模式进行对比，输出：
1. JSON 原始数据
2. Markdown 表格
3. 简单的模式表现总结
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from run_full_evaluation import generate_all_documents, TEST_DIR, TEST_QUERIES

BASE = "http://localhost:8400/api/v1"
OUT_DIR = TEST_DIR.parent
MODES = ["fast", "balanced", "deep", "auto"]

TOKEN = None


def login():
    global TOKEN
    r = requests.post(f"{BASE}/login", json={"username": "admin", "password": "admin123"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    TOKEN = data["token"]
    print(f"Login OK: {data['name']}")


def headers():
    return {"X-Auth-Token": TOKEN}


def ensure_docs():
    docs = generate_all_documents()
    print(f"Generated {len(docs)} docs")
    uploaded, skipped = 0, 0
    for i, (name, content) in enumerate(docs, 1):
        path = TEST_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        with open(path, "rb") as f:
            r = requests.post(
                f"{BASE}/document/upload",
                files={"file": (name, f)},
                data={"collection": "shared"},
                headers=headers(),
                timeout=120,
            )
        r.raise_for_status()
        if r.json().get("duplicate"):
            skipped += 1
        else:
            uploaded += 1
        if i % 10 == 0 or i == len(docs):
            print(f"  upload {i}/{len(docs)}")
    print(f"Uploaded {uploaded}, skipped {skipped}")
    time.sleep(4)


def parse_stream(resp):
    full_answer = ""
    sources = []
    trace = {}
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: [SOURCES]"):
            try:
                payload = line.replace("data: [SOURCES]", "").replace("[/SOURCES]", "")
                sources = json.loads(payload)
            except Exception:
                pass
        elif line.startswith("data: [TRACE]"):
            try:
                payload = line.replace("data: [TRACE]", "").replace("[/TRACE]", "")
                trace = json.loads(payload)
            except Exception:
                pass
        elif line.startswith("data: ") and not line.startswith("data: ["):
            full_answer += line[6:]
    return full_answer, sources, trace


def run_mode(mode: str):
    print(f"\n=== MODE: {mode} ===")
    mode_rows = []
    kw_scores = []
    hit_scores = []
    latency_scores = []
    source_counts = []

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected = test.get("keywords", [])
        category = test.get("category", "unknown")
        difficulty = test.get("difficulty", "medium")

        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{BASE}/chat/stream",
                json={"query": query, "top_k": 10, "mode": mode},
                headers=headers(),
                stream=True,
                timeout=180,
            )
            resp.raise_for_status()
            answer, sources, trace = parse_stream(resp)
            wall_ms = (time.perf_counter() - start) * 1000.0
            total_ms = float(trace.get("total_ms", wall_ms))
            phase = trace.get("phase", {})
            actual_mode = trace.get("mode", mode)
            query_variants = trace.get("query_variants_count", 0)
        except Exception as exc:
            answer, sources, trace = "", [], {"error": str(exc)}
            wall_ms = (time.perf_counter() - start) * 1000.0
            total_ms = wall_ms
            phase = {}
            actual_mode = mode
            query_variants = 0

        kw_hits = sum(1 for kw in expected if kw.lower() in answer.lower())
        kw_recall = kw_hits / max(len(expected), 1) if expected else 0.0
        hit = 1.0 if sources else 0.0
        latency_scores.append(total_ms)
        kw_scores.append(kw_recall)
        hit_scores.append(hit)
        source_counts.append(len(sources))

        mode_rows.append({
            "query": query,
            "category": category,
            "difficulty": difficulty,
            "expected_keywords": expected,
            "keyword_recall": kw_recall,
            "hit_rate": hit,
            "source_count": len(sources),
            "wall_ms": round(wall_ms, 1),
            "total_ms": round(total_ms, 1),
            "actual_mode": actual_mode,
            "query_variants_count": query_variants,
            "phase": phase,
            "trace": trace,
        })
        mark = "PASS" if kw_recall > 0.5 else ("PART" if kw_recall > 0 else "FAIL")
        print(f"  [{i:02d}] {mark} {query[:40]:40s} | kw={kw_recall:.2f} hit={hit:.0f} src={len(sources):<2d} total={total_ms:.0f}ms")

    return {
        "mode": mode,
        "count": len(mode_rows),
        "avg_keyword_recall": float(np.mean(kw_scores)) if kw_scores else 0.0,
        "avg_hit_rate": float(np.mean(hit_scores)) if hit_scores else 0.0,
        "avg_total_ms": float(np.mean(latency_scores)) if latency_scores else 0.0,
        "median_total_ms": float(np.median(latency_scores)) if latency_scores else 0.0,
        "avg_source_count": float(np.mean(source_counts)) if source_counts else 0.0,
        "rows": mode_rows,
        "by_category": summarize(rows=mode_rows, key="category"),
        "by_difficulty": summarize(rows=mode_rows, key="difficulty"),
    }


def summarize(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key, "unknown")].append(row)
    out = {}
    for name, items in groups.items():
        out[name] = {
            "count": len(items),
            "avg_keyword_recall": float(np.mean([x["keyword_recall"] for x in items])) if items else 0.0,
            "avg_hit_rate": float(np.mean([x["hit_rate"] for x in items])) if items else 0.0,
            "avg_total_ms": float(np.mean([x["total_ms"] for x in items])) if items else 0.0,
        }
    return out


def write_markdown(results):
    lines = []
    lines.append("# 模式耗时对比")
    lines.append("")
    lines.append("| 模式 | 查询数 | 关键词召回 | 命中率 | 平均耗时 | 中位耗时 | 平均来源数 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode in MODES:
        r = results[mode]
        lines.append(
            f"| {mode} | {r['count']} | {r['avg_keyword_recall']:.1%} | {r['avg_hit_rate']:.1%} | "
            f"{r['avg_total_ms']:.0f} ms | {r['median_total_ms']:.0f} ms | {r['avg_source_count']:.1f} |"
        )
    lines.append("")
    lines.append("## 简要观察")
    base = results["balanced"]
    for mode in MODES:
        if mode == "balanced":
            continue
        r = results[mode]
        delta = r["avg_total_ms"] - base["avg_total_ms"]
        lines.append(f"- {mode}: 相比 balanced，平均耗时 {delta:+.0f} ms")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  Enterprise RAG 模式耗时对比评测")
    print("=" * 60)
    login()
    ensure_docs()

    results = {}
    for mode in MODES:
        results[mode] = run_mode(mode)

    json_path = OUT_DIR / "mode_compare_results.json"
    md_path = OUT_DIR / "mode_compare_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(write_markdown(results))

    print("\n" + "=" * 60)
    print(f"Saved JSON: {json_path}")
    print(f"Saved MD:   {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
