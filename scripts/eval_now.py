"""Enterprise RAG 全面评估 —— 上传文档 + 运行42个查询 + 生成报告"""
import sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import numpy as np

BASE = "http://localhost:8400/api/v1"
TOKEN = None

def login():
    global TOKEN
    r = requests.post(f"{BASE}/login", json={"username":"admin","password":"admin123"})
    TOKEN = r.json()["token"]
    print(f"Login OK: {r.json()['name']}")

def H():
    return {"X-Auth-Token": TOKEN}

sys.path.insert(0, os.path.dirname(__file__))
from run_full_evaluation import generate_all_documents, TEST_DIR, TEST_QUERIES

login()

# Generate and upload
docs = generate_all_documents()
print(f"Generated {len(docs)} docs")
uploaded, skipped = 0, 0
for i, (name, content) in enumerate(docs):
    path = TEST_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/document/upload",
            files={"file": (name, f)}, data={"collection": "shared"}, headers=H())
    if r.json().get("duplicate"): skipped += 1
    else: uploaded += 1
print(f"Uploaded {uploaded} new, {skipped} skipped")
time.sleep(3)

# Run queries
print(f"\nRunning {len(TEST_QUERIES)} queries...\n")
results = {"kw_recall": [], "hit": [], "queries": []}

for i, test in enumerate(TEST_QUERIES):
    query = test["query"]
    expected = test.get("keywords", [])
    category = test.get("category", "unknown")
    difficulty = test.get("difficulty", "medium")
    full_answer = ""
    sources = []
    try:
        r = requests.post(f"{BASE}/chat/stream", json={"query": query, "top_k": 10},
                         headers=H(), stream=True, timeout=180)
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: [SOURCES]"):
                try:
                    s = line.replace("data: [SOURCES]", "").replace("[/SOURCES]", "")
                    sources = json.loads(s)
                except Exception:
                    pass
            elif line.startswith("data: ") and not line.startswith("data: ["):
                full_answer += line[6:]
    except Exception as e:
        print(f"  [{i+1:2d}] ERR: {query[:35]}... - {e}")
        results["kw_recall"].append(0)
        results["hit"].append(0)
        results["queries"].append({"q": query, "kw": 0, "hit": 0, "cat": category, "diff": difficulty})
        continue

    kw_hits = sum(1 for kw in expected if kw.lower() in full_answer.lower())
    kw_recall = kw_hits / max(len(expected), 1) if expected else 0.0
    hit = 1.0 if sources else 0.0
    results["kw_recall"].append(kw_recall)
    results["hit"].append(hit)
    results["queries"].append({"q": query, "kw": kw_recall, "hit": hit, "cat": category, "diff": difficulty})
    s = "PASS" if kw_recall > 0.5 else ("PART" if kw_recall > 0 else "FAIL")
    print(f"  [{i+1:2d}] {s} {query[:45]:45s} | KW={kw_recall:.2f} HIT={hit} | {category}")

# Report
print(f"\n{'='*60}")
print(f"  Enterprise RAG Evaluation Complete!")
print(f"  Total queries: {len(results['kw_recall'])}")
avg_kw = np.mean(results["kw_recall"])
avg_hit = np.mean(results["hit"])
print(f"  Avg Keyword Recall: {avg_kw:.1%}")
print(f"  Avg Hit Rate: {avg_hit:.1%}")

cats = {}
for q in results["queries"]:
    c = q["cat"]
    cats.setdefault(c, []).append(q["kw"])
print("  By Category:")
for c, vals in sorted(cats.items()):
    print(f"    {c:<12}: {len(vals):>2} queries, recall={np.mean(vals):.1%}")

diffs = {}
for q in results["queries"]:
    d = q["diff"]
    diffs.setdefault(d, []).append(q["kw"])
print("  By Difficulty:")
for d, vals in sorted(diffs.items()):
    print(f"    {d:<8}: {len(vals):>2} queries, recall={np.mean(vals):.1%}")
print(f"{'='*60}")

out = {
    "total": len(results["kw_recall"]),
    "avg_kw_recall": float(avg_kw),
    "avg_hit_rate": float(avg_hit),
    "by_category": {c: {"count": len(v), "kw_recall": float(np.mean(v))} for c, v in cats.items()},
    "by_difficulty": {d: {"count": len(v), "kw_recall": float(np.mean(v))} for d, v in diffs.items()},
}
res_path = TEST_DIR.parent / "eval_result.json"
with open(res_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Saved to {res_path}")
