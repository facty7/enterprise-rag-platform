"""上传所有测试文档"""
import requests, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.run_full_evaluation import generate_all_documents, TEST_DIR

BASE = "http://localhost:8400/api/v1"
r = requests.post(f"{BASE}/login", json={"username":"admin","password":"admin123"})
tok = r.json()["token"]
h = {"X-Auth-Token": tok}
print(f"Login: {r.json()['name']}")

docs = generate_all_documents()
print(f"Generated {len(docs)} docs")
up, skip = 0, 0
for i, (name, content) in enumerate(docs):
    path = TEST_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    if os.path.getsize(path) < 10:
        continue
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/document/upload",
            files={"file": (name, f)}, data={"collection": "shared"}, headers=h)
    d = r.json()
    if d.get("duplicate"): skip += 1
    else: up += 1
    status = "SKIP" if d.get("duplicate") else "OK"
    print(f"  [{i+1:2d}/{len(docs)}] {status:4s} {name[:55]}")
print(f"\nDone: {up} uploaded, {skip} skipped")
