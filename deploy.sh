#!/usr/bin/env bash
# 把当前目录（机制速查/）同步到 GitHub Pages 仓库并推送。
#
#   ./deploy.sh
#
# 首次使用需要能免密访问 github.com：gh 已登录即可（脚本用 gh 的凭证）。
set -euo pipefail

REPO="easebreaker/gr-parallel-cheatsheet"
URL="https://easebreaker.github.io/gr-parallel-cheatsheet/"
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# 1) 先重新生成 index.html，避免推上去的是旧的
if command -v python3 >/dev/null 2>&1; then
  echo "==> 重新生成 index.html"
  ( cd "$SRC" && python3 build.py )
fi

# 2) 拉下远端
echo "==> 克隆 $REPO"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  GIT_AUTH=(-c "credential.helper=!gh auth git-credential")
else
  GIT_AUTH=()
fi
git "${GIT_AUTH[@]}" clone --depth 1 "https://github.com/${REPO}.git" "$WORK/r"

# 3) 同步（删除远端多余文件；保留 .git / .nojekyll）
echo "==> 同步文件"
rsync -a --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '.DS_Store' \
  "$SRC"/ "$WORK/r"/
touch "$WORK/r/.nojekyll"

# 4) 提交并推送
cd "$WORK/r"
git add -A
if git diff --cached --quiet; then
  echo "没有变化，跳过推送。"
else
  git "${GIT_AUTH[@]}" \
      -c user.name="$(git config --get user.name || echo deploy)" \
      -c user.email="$(git config --get user.email || echo deploy@local)" \
      commit -q -m "deploy: $(date '+%Y-%m-%d %H:%M')"
  git "${GIT_AUTH[@]}" push -q origin HEAD:main
  echo "==> 已推送。约 30 秒后生效：$URL"
fi
