"""快速评估 - 上传文档 + 运行查询 + 输出报告"""
import sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import requests
import numpy as np

BASE = "http://localhost:8400/api/v1"
TOKEN = None

def login():
    global TOKEN
    r = requests.post(f"{BASE}/login", json={"username":"admin","password":"admin123"})
    TOKEN = r.json()["token"]
    print(f"登录成功: {r.json()['name']}")

def H(): return {"X-Auth-Token": TOKEN}

# Run the full doc generator
sys.path.insert(0, os.path.dirname(__file__))
from run_full_evaluation import generate_all_documents, TEST_DIR, TEST_QUERIES

login()

# Generate and upload docs
docs = generate_all_documents()
print(f"\n生成 {len(docs)} 个企业文档")

uploaded, skipped = 0, 0
for i, (name, content) in enumerate(docs):
    path = TEST_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        with open(path, "rb") as f:
            r = requests.post(f"{BASE}/document/upload",
                files={"file": (name, f)},
                data={"collection": "shared"},
                headers=H())
        if r.json().get("duplicate"): skipped += 1
        else: uploaded += 1
    except Exception as e:
        print(f"  失败: {name} - {e}")

print(f"上传: {uploaded} 新增, {skipped} 跳过 (共 {len(docs)} 个)\n")
time.sleep(3)

# Run queries
print(f"运行 {len(TEST_QUERIES)} 个测试查询...\n")
results = {"queries": [], "kw_recall": [], "hit": []}

for i, test in enumerate(TEST_QUERIES):
    query = test["query"]
    expected = test.get("keywords", [])
    category = test.get("category", "未知")
    difficulty = test.get("difficulty", "medium")

    full_answer = ""
    sources = []
    try:
        r = requests.post(f"{BASE}/chat/stream", json={"query": query, "top_k": 10},
                         headers=H(), stream=True, timeout=120)
        for line in r.iter_lines(decode_unicode=True):
            if not line: continue
            if line.startswith("data: [SOURCES]"):
                try:
                    s = line.replace("data: [SOURCES]","").replace("[/SOURCES]","")
                    sources = json.loads(s)
                except: pass
            elif line.startswith("data: ") and not line.startswith("data: ["):
                full_answer += line[6:]
    except Exception as e:
        print(f"  [{i+1:2d}] ERROR: {query[:30]}... - {e}")
        results["kw_recall"].append(0); results["hit"].append(0)
        results["queries"].append({"query": query, "kw_recall": 0, "hit": 0,
                                    "cat": category, "diff": difficulty, "error": str(e)})
        continue

    kw_hits = sum(1 for kw in expected if kw.lower() in full_answer.lower())
    kw_recall = kw_hits / max(len(expected), 1) if expected else 0.0
    hit = 1.0 if sources else 0.0

    results["kw_recall"].append(kw_recall)
    results["hit"].append(hit)
    results["queries"].append({"query": query, "kw_recall": kw_recall, "hit": hit,
                                "cat": category, "diff": difficulty, "sources": len(sources)})
    s = "PASS" if kw_recall > 0.5 else ("PART" if kw_recall > 0 else "FAIL")
    print(f"  [{i+1:2d}] {s} {query[:40]:40s} | KW:{kw_recall:.2f} | SRC:{len(sources)} | {category}")

print(f"\n{'='*60}")
print(f"  评估完成!")
print(f"  总查询: {len(results['kw_recall'])}")
print(f"  平均关键词召回率: {np.mean(results['kw_recall']):.1%}")
print(f"  平均命中率: {np.mean(results['hit']):.1%}")

# By category
cats = {}
for q in results["queries"]:
    c = q["cat"]; cats.setdefault(c, []).append(q["kw_recall"])
print(f"\n  按分类:")
for c, vals in sorted(cats.items()):
    print(f"    {c:<12}: {len(vals):>2}条, 召回率={np.mean(vals):.1%}")

# By difficulty
diffs = {}
for q in results["queries"]:
    d = q["diff"]; diffs.setdefault(d, []).append(q["kw_recall"])
print(f"\n  按难度:")
for d, vals in sorted(diffs.items()):
    print(f"    {d:<8}: {len(vals):>2}条, 召回率={np.mean(vals):.1%}")

print(f"{'='*60}")

# Save results
out = {
    "total_queries": len(results["kw_recall"]),
    "avg_keyword_recall": float(np.mean(results["kw_recall"])),
    "avg_hit_rate": float(np.mean(results["hit"])),
    "by_category": {c: {"count": len(vals), "kw_recall": float(np.mean(vals))}
                    for c, vals in cats.items()},
    "by_difficulty": {d: {"count": len(vals), "kw_recall": float(np.mean(vals))}
                      for d, vals in diffs.items()},
    "queries": results["queries"],
}
with open(TEST_DIR.parent / "eval_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {TEST_DIR.parent / 'eval_result.json'}")
