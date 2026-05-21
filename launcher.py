"""Enterprise RAG Knowledge Platform launcher."""
import os, subprocess, sys, webbrowser, time, urllib.request
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    print()
    print("  ==========================================")
    print("    Enterprise RAG Knowledge Platform")
    print("    Hybrid Retrieval + RBAC + Metrics")
    print("  ==========================================")
    print()

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    env = ROOT / ".env"
    if not env.exists():
        example = ROOT / ".env.example"
        if example.exists():
            shutil.copyfile(example, env)
            print("  [Init] Created .env from .env.example")
        else:
            print("  [ERROR] .env.example config template missing!")
            input("  Press Enter to exit..."); sys.exit(1)

    for d in ["files", "data", "data/uploads", "data/qdrant_db",
              "data/reports", "data/logs"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    os.chdir(str(ROOT))

    # 读取端口
    port = os.environ.get("PORT", "").strip()
    if not port:
        port = "8400"
        try:
            with open(env, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("PORT="):
                        port = line.split("=")[1].strip()
        except Exception:
            pass

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "0.0.0.0", "--port", port],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    url = f"http://localhost:{port}/health"
    print("  [Start] Loading retrieval engine (CPU, please wait)...")
    print("  ", end="", flush=True)
    for i in range(120):  # 最多等 2 分钟
        try:
            urllib.request.urlopen(url, timeout=1)
            print(" Ready!")
            break
        except Exception:
            time.sleep(1)
            if i % 10 == 9:
                print(".", end="", flush=True)
    else:
        print()
        print("  [WARN] Startup timeout - check console for errors")

    print()
    print("  ==========================================")
    print(f"    Local:  http://localhost:{port}")
    print(f"    Login:  admin / admin123")
    print(f"    Close this window to stop")
    print("  ==========================================")
    print()
    webbrowser.open(f"http://localhost:{port}")

    try:
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    except KeyboardInterrupt:
        print()
        print("  Server stopped.")
        proc.terminate()
    except Exception:
        proc.terminate()


if __name__ == "__main__":
    main()
