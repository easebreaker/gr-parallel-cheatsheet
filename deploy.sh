#!/usr/bin/env bash
# 把当前目录（机制速查/）同步到 GitHub Pages 仓库。
#
#   ./deploy.sh
#
# 走 GitHub Contents API（经 gh），不依赖 git push 的凭证配置：
# 只要 `gh auth status` 正常就能用。首次会先重跑 build.py。
set -euo pipefail

cd "$(dirname "$0")"

REPO="easebreaker/gr-parallel-cheatsheet"
URL="https://easebreaker.github.io/gr-parallel-cheatsheet/"

command -v gh >/dev/null || { echo "需要 gh：https://cli.github.com/"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "请先 gh auth login"; exit 1; }

echo "==> 重新生成 index.html"
python3 build.py

echo "==> 同步到 $REPO"
REPO="$REPO" python3 - <<'PY'
import base64, json, os, subprocess

REPO = os.environ["REPO"]
ROOT = os.path.dirname(os.path.abspath("build.py"))
SKIP_DIRS = {".git", "__pycache__", ".github"}
SKIP_FILES = {".DS_Store"}

def gh(*args, **kw):
    return subprocess.run(["gh", *args], text=True,
                          capture_output=True, **kw)

# 本地文件（相对路径 -> 绝对路径）
local = {}
for dirpath, dirnames, filenames in os.walk("."):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if fn in SKIP_FILES or fn.endswith(".pyc"):
            continue
        full = os.path.join(dirpath, fn)
        local[os.path.relpath(full, ".")] = full

# 远端文件（path -> sha）
r = gh("api", f"repos/{REPO}/git/trees/main?recursive=1", "-q",
       '.tree[] | select(.type=="blob") | "\\(.path)\\t\\(.sha)"')
remote = {}
for line in (r.stdout or "").splitlines():
    if "\t" in line:
        path, sha = line.split("\t", 1)
        remote[path] = sha

added = updated = removed = 0

for path, full in sorted(local.items()):
    with open(full, "rb") as f:
        raw = f.read()
    content = base64.b64encode(raw).decode()

    # 用 git blob hash 精确比对：内容没变就完全不产生 commit
    h = subprocess.run(["git", "hash-object", full], text=True, capture_output=True)
    local_blob = (h.stdout or "").strip()
    if path in remote and remote[path] == local_blob:
        continue

    old = None
    if path in remote:
        s = gh("api", f"repos/{REPO}/contents/{path}", "-q", ".sha")
        old = (s.stdout or "").strip()

    payload = {"message": f"deploy {path}", "content": content}
    if old:
        payload["sha"] = old
    w = gh("api", "-X", "PUT", f"repos/{REPO}/contents/{path}",
           "--input", "-", input=json.dumps(payload))
    if w.returncode != 0:
        print("  FAIL", path, (w.stderr or "").strip()[:120])
        continue
    if old:
        updated += 1
    else:
        added += 1

# 远端有、本地没有的 -> 删除
for path, sha in sorted(remote.items()):
    if path in local:
        continue
    w = gh("api", "-X", "DELETE", f"repos/{REPO}/contents/{path}", "--input", "-",
           input=json.dumps({"message": f"delete {path}", "sha": sha}))
    if w.returncode == 0:
        removed += 1

print(f"  新增 {added} / 更新 {updated} / 删除 {removed}")
if added == updated == removed == 0:
    print("  没有变化。")
PY

echo "==> 完成。约 30 秒后生效：$URL"
