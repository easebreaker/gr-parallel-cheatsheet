#!/usr/bin/env bash
# 把当前目录同步到 GitHub Pages 仓库并推送。
set -euo pipefail
REPO="easebreaker/gr-parallel-cheatsheet"
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone --depth 1 "https://github.com/${REPO}.git" "$WORK/r"

rsync -a --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '.DS_Store' \
  "$SRC"/ "$WORK/r"/

cd "$WORK/r"
git add -A
if git diff --cached --quiet; then
  echo "没有变化，跳过。"
else
  git -c user.name="$(git config --get user.name || echo deploy)" \
      -c user.email="$(git config --get user.email || echo deploy@local)" \
      commit -m "deploy: $(date '+%Y-%m-%d %H:%M')"
  git push
  echo "已推送。页面约 30 秒后更新：https://easebreaker.github.io/gr-parallel-cheatsheet/"
fi
