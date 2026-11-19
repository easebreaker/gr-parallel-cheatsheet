"""把 docs/*.md + models/*.py + parallel/*.py 生成一个自包含的 index.html。

设计要点：
- 零外部依赖（不引 CDN），双击就能开；因为 file:// 下无法 fetch 本地文件，
  所以代码和输出都**内联**进 HTML。
- 每个条目有三个标签页：说明 / 代码 / 运行输出。
- 「运行输出」是 build 时真的跑一遍 demo 抓到的 stdout，不是手抄的。

用法：
    cd 机制速查 && python3 build.py
"""

from __future__ import annotations

import html
import json
import os
import pty
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "models"))
sys.path.insert(0, os.path.join(HERE, "parallel"))

# ----------------------------------------------------------------------
ITEMS = [
    ("map", "总览", "00 · 地图", "docs/00-地图.md", None, None,
     "四个模型 × 三个并行轴的一页总表"),
    ("hstu", "模型", "HSTU", "docs/01-HSTU.md", "models/hstu_min.py",
     "models.hstu_min", "SiLU 代替 softmax ⇒ 纯加归约"),
    ("mtgr", "模型", "MTGR", "docs/02-MTGR.md", "models/mtgr_min.py",
     "models.mtgr_min", "三段掩码 ⇒ 静态段是方阵"),
    ("rankmixer", "模型", "RankMixer", "docs/03-RankMixer.md", "models/rankmixer_min.py",
     "models.rankmixer_min", "转置代替 attention ⇒ 没有 L 维"),
    ("tiger", "模型", "TIGER", "docs/10-TIGER.md", "models/tiger_min.py",
     "models.tiger_min", "SID + 小 beam ⇒ 三个轴都没有对象"),
    ("longer", "模型", "LONGER", "docs/08-LONGER.md", "models/longer_min.py",
     "models.longer_min", "扁矩阵 + softmax ⇒ 唯一需要 LSE"),
    ("hyformer", "模型", "HyFormer", "docs/09-HyFormer.md", "models/hyformer_min.py",
     "models.hyformer_min", "16 个 query 读整段 ⇒ 算便宜、读贵"),
    ("onerec", "模型", "OpenOneRec", "docs/04-OpenOneRec.md", "models/onerec_min.py",
     "models.onerec_min", "3-token 自回归 ⇒ q = B"),
    ("cp", "并行轴", "CP", "docs/05-CP.md", "parallel/cp_min.py",
     "parallel.cp_min", "切 Q / 切 KV / zigzag 三方案"),
    ("ep", "并行轴", "EP", "docs/06-EP.md", "parallel/ep_min.py",
     "parallel.ep_min", "四步 + 三个量 + 路由相关性"),
    ("beam", "并行轴", "beam", "docs/07-beam.md", "parallel/beam_min.py",
     "parallel.beam_min", "KV 冗余 / 重排 / host 同步"),
    ("mixed", "并行轴", "混合并行", "docs/11-混合并行.md", "parallel/mixed_min.py",
     "parallel.mixed_min", "每项只有一个轴能切"),
    ("matrix", "交叉", "模型 × 轴 矩阵", None, None, None,
     "点格子跳到对应条目"),
]


# ----------------------------------------------------------------------
# 极简 markdown -> html（只支持这些文档里实际用到的语法）
# ----------------------------------------------------------------------
def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
               r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = re.sub(r"&lt;(https?://[^&]+)&gt;",
               r'<a href="\1" target="_blank" rel="noopener">\1</a>', s)
    return s


def md_to_html(md: str) -> str:
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    in_list = False
    while i < len(lines):
        line = lines[i]

        # 代码块
        if line.startswith("```"):
            if in_list:
                out.append("</ul>")
                in_list = False
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        # 表格
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            if in_list:
                out.append("</ul>")
                in_list = False
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            t = ["<table><thead><tr>"]
            t += [f"<th>{_inline(h)}</th>" for h in header]
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            if in_list:
                out.append("</ul>")
                in_list = False
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        # 引用
        if line.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>" + md_to_html("\n".join(buf)) + "</blockquote>")
            continue

        # 列表
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        m2 = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m or m2:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline((m or m2).group(1))}</li>")
            i += 1
            continue

        if not line.strip():
            if in_list:
                out.append("</ul>")
                in_list = False
            i += 1
            continue

        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{_inline(line)}</p>")
        i += 1

    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _run_pty(cmd: list[str], env: dict) -> tuple[str, int]:
    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd, stdout=slave, stderr=slave, stdin=subprocess.DEVNULL,
            cwd=HERE, env=env, close_fds=True,
        )
        os.close(slave)
        chunks = []
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)
        proc.wait(timeout=180)
        rc = proc.returncode
    finally:
        try:
            os.close(master)
        except OSError:
            pass
    return b"".join(chunks).decode("utf-8", "replace").replace("\r\n", "\n"), rc


def _run_pipe(cmd: list[str], env: dict) -> tuple[str, int]:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE,
                          env=env, timeout=180)
    return (proc.stdout or "") + (proc.stderr or ""), proc.returncode


def capture(script_rel: str) -> str:
    """在子进程里跑 demo 并抓 stdout。

    两个坑：
    1. macOS 系统 Python 3.9 + numpy 在 stdout 被重定向到管道/文件时会 SIGSEGV，
       所以优先给子进程一个 **pty**（走 tty 路径）。
    2. 不能在当前进程里 import 跑 —— 会把 numpy 拉进 build 进程，同样会崩。

    降级链：pty -> pipe -> 空（HTML 里就不显示「运行输出」标签页）。
    """
    path = os.path.join(HERE, script_rel)
    env = dict(os.environ)
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, path]

    try:
        text, rc = _run_pty(cmd, env)
        if rc == 0 and text.strip():
            return text
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        text, rc = _run_pipe(cmd, env)
        if rc == 0 and text.strip():
            return text
    except (OSError, subprocess.SubprocessError):
        pass

    return ""


# ----------------------------------------------------------------------
def build() -> str:
    data = []
    for ident, group, title, md_path, code_path, mod, blurb in ITEMS:
        body = ""
        if md_path:
            with open(os.path.join(HERE, md_path), encoding="utf-8") as f:
                body = md_to_html(f.read())
        code = ""
        if code_path:
            with open(os.path.join(HERE, code_path), encoding="utf-8") as f:
                code = f.read()
        out = ""
        if code_path:
            try:
                out = capture(code_path)
            except Exception as exc:                    # noqa: BLE001
                out = f"[demo 运行失败] {type(exc).__name__}: {exc}"
        data.append(dict(id=ident, group=group, title=title, blurb=blurb,
                         body=body, code=code, codePath=code_path or "",
                         out=out))

    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0e1116">
<meta name="description" content="七个 GR 模型 × 四个并行轴（CP / EP / TP / beam）的机制速查与可跑的最小实现">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🧩</text></svg>">
<title>GR 模型机制速查 · 模型 × 并行轴</title>
<style>
  :root{
    --bg:#0e1116; --panel:#151a21; --panel2:#1b222b; --fg:#d7dee8; --dim:#8b97a7;
    --line:#242c37; --acc:#5ec8d8; --acc2:#e0b060; --ok:#6fbf73; --bad:#e06c75;
    --mono:"SF Mono",ui-monospace,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--fg);font:14px/1.7 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
  /* ---------- 顶栏（只在窄屏出现） ---------- */
  .topbar{display:none;position:sticky;top:0;z-index:40;align-items:center;gap:10px;
          padding:9px 12px;background:rgba(21,26,33,.96);backdrop-filter:blur(8px);
          border-bottom:1px solid var(--line)}
  .topbar .burger{width:34px;height:34px;flex:0 0 34px;display:flex;align-items:center;
                  justify-content:center;border:1px solid var(--line);border-radius:8px;
                  background:var(--panel2);color:var(--fg);cursor:pointer;font-size:15px}
  .topbar .tt{font-size:13.5px;font-weight:600;white-space:nowrap;overflow:hidden;
              text-overflow:ellipsis}
  .scrim{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:45}
  /* ---------- 侧栏 ---------- */
  aside{width:264px;flex:0 0 264px;background:var(--panel);border-right:1px solid var(--line);
        display:flex;flex-direction:column;height:100vh;position:sticky;top:0;z-index:50}
  .brand{padding:16px 16px 10px}
  .brand h1{margin:0;font-size:15px;letter-spacing:.3px}
  .brand p{margin:4px 0 0;font-size:11.5px;color:var(--dim)}
  .search{padding:0 12px 10px}
  .search input{width:100%;padding:7px 9px;border-radius:6px;border:1px solid var(--line);
                background:var(--panel2);color:var(--fg);font-size:12.5px;outline:none}
  .search input:focus{border-color:var(--acc)}
  nav{overflow-y:auto;flex:1;padding-bottom:24px;-webkit-overflow-scrolling:touch}
  .grp{padding:12px 16px 4px;font-size:10.5px;letter-spacing:1.4px;color:var(--dim);text-transform:uppercase}
  .item{padding:7px 16px 7px 20px;cursor:pointer;border-left:2px solid transparent;font-size:13px}
  .item:hover{background:var(--panel2)}
  .item.on{background:var(--panel2);border-left-color:var(--acc);color:#fff}
  .item small{display:block;color:var(--dim);font-size:10.5px;line-height:1.35}
  /* ---------- 布局 ---------- */
  .shell{display:flex;height:100vh}
  /* ---------- 主区 ---------- */
  main{flex:1;min-width:0;height:100vh;overflow-y:auto;padding:26px 40px 80px;max-width:1080px;
       -webkit-overflow-scrolling:touch}
  .head h2{margin:0 0 4px;font-size:22px}
  .head .blurb{color:var(--dim);font-size:13px;margin:0 0 18px}
  .tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:22px}
  .tab{padding:7px 14px;cursor:pointer;font-size:12.5px;color:var(--dim);border-bottom:2px solid transparent}
  .tab:hover{color:var(--fg)}
  .tab.on{color:var(--acc);border-bottom-color:var(--acc)}
  .pane{display:none}
  .pane.on{display:block}
  /* ---------- 排版 ---------- */
  h1,h2,h3,h4{line-height:1.35}
  .doc h1{font-size:21px;margin:0 0 14px}
  .doc h2{font-size:16.5px;margin:30px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
  .doc h3{font-size:14.5px;margin:22px 0 8px;color:var(--acc)}
  .doc p{margin:9px 0}
  .doc ul{margin:8px 0;padding-left:22px}
  .doc li{margin:4px 0}
  .doc code{background:#0a0d12;border:1px solid var(--line);border-radius:4px;
            padding:1px 5px;font-family:var(--mono);font-size:12px;color:#a8d8e8}
  .doc pre{background:#0a0d12;border:1px solid var(--line);border-radius:8px;
           padding:14px 16px;overflow-x:auto;margin:14px 0}
  .doc pre code{background:none;border:none;padding:0;font-size:12px;line-height:1.55;color:#c3cdd9}
  .doc blockquote{margin:14px 0;padding:10px 16px;border-left:3px solid var(--acc2);
                  background:#171d25;color:#c8d2de;border-radius:0 6px 6px 0}
  .doc blockquote p{margin:5px 0}
  .doc table{border-collapse:collapse;margin:14px 0;font-size:12.5px;width:100%}
  .doc th,.doc td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
  .doc th{background:var(--panel2);color:#cfd8e3;font-weight:600}
  .doc tr:nth-child(even) td{background:#131922}
  .doc a{color:var(--acc)}
  /* ---------- 代码 / 输出 ---------- */
  pre.src{background:#0a0d12;border:1px solid var(--line);border-radius:8px;padding:16px;
          overflow-x:auto;font-family:var(--mono);font-size:12px;line-height:1.55;margin:0}
  .codebar{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;
           font-size:11.5px;color:var(--dim);font-family:var(--mono)}
  .codebar button{background:var(--panel2);border:1px solid var(--line);color:var(--fg);
                  padding:3px 10px;border-radius:5px;cursor:pointer;font-size:11px}
  .codebar button:hover{border-color:var(--acc)}
  /* 简易高亮 */
  .k{color:#c792ea}.s{color:#c3e88d}.c{color:#5f6b7a;font-style:italic}.n{color:#f78c6c}.f{color:#82aaff}
  /* ---------- 矩阵 ---------- */
  .mx{border-collapse:collapse;font-size:13px;margin:18px 0}
  .mx th,.mx td{border:1px solid var(--line);padding:11px 16px}
  .mx th{background:var(--panel2);color:var(--dim);font-weight:600;font-size:12px}
  .mx td.cell{cursor:pointer;text-align:center;min-width:92px;font-weight:600}
  .mx td.cell:hover{outline:2px solid var(--acc)}
  .g0{background:#1d2a20;color:#8fce9a}   /* 无对象 */
  .g1{background:#2a2720;color:#dcc48a}   /* 有条件 */
  .g2{background:#1e2a30;color:#7fd0e0}   /* 主战场 */
  .gn{background:#212630;color:#8b97a7}   /* 无关/无定义 */
  .legend{font-size:12px;color:var(--dim);margin-top:10px}
  .kbd{font-family:var(--mono);background:var(--panel2);border:1px solid var(--line);
       border-radius:4px;padding:0 5px;font-size:11px}
  /* ---------- 窄屏：抽屉式侧栏 ---------- */
  @media (max-width:820px){
    body{font-size:14.5px}
    .topbar{display:flex}
    .shell{height:auto;min-height:100vh;padding-top:0}
    aside{position:fixed;top:0;left:0;bottom:0;height:100%;width:min(84vw,300px);
          flex:none;transform:translateX(-102%);transition:transform .22s ease;
          box-shadow:2px 0 26px rgba(0,0,0,.5)}
    body.drawer aside{transform:translateX(0)}
    body.drawer .scrim{display:block}
    main{height:auto;overflow:visible;padding:14px 15px 72px;max-width:none}
    .brand p{display:none}
    .head h2{font-size:19px}
    .doc h1{font-size:18.5px}
    .doc h2{font-size:15.5px;margin:24px 0 8px}
    .doc h3{font-size:14px}
    /* 宽表格在窄屏横向滚动，不挤成竖排 */
    .doc{overflow-x:auto}
    .doc table{font-size:11.5px;min-width:0}
    .doc table th,.doc table td{padding:5px 7px;white-space:nowrap}
    .doc pre,.doc pre code{font-size:11px}
    .doc pre{padding:11px 12px}
    .mx th,.mx td{padding:8px 9px;font-size:11.5px}
    .mx td.cell{min-width:70px}
    .tabs{gap:0;overflow-x:auto}
    .tab{padding:8px 12px;white-space:nowrap}
  }
</style>
</head>
<body>
<div class="topbar">
  <button class="burger" id="burger" aria-label="目录">☰</button>
  <div class="tt" id="tt">GR 机制速查</div>
</div>
<div class="shell">
<aside id="side">
  <div class="brand">
    <h1>GR 机制速查</h1>
    <p>7 个模型 · 4 个并行轴 · 可跑的最小实现</p>
  </div>
  <div class="search"><input id="q" placeholder="搜索（/ 聚焦）"></div>
  <nav id="nav"></nav>
</aside>
<div class="scrim" id="scrim"></div>
<main id="main"></main>
</div>
<script>
const DATA = __DATA__;

/* ---------- 极简 Python 高亮（字符串 -> 注释 -> 关键字） ---------- */
function hl(src){
  const store = [];
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  let out = esc(src);
  // 1) 字符串与三引号块（先吃掉，避免里面的 # 被当注释）
  out = out.replace(/("{3}[\s\S]*?"{3}|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')/g, m => {
    store.push('<span class="s">' + m + '</span>');
    return '\u0000' + (store.length - 1) + '\u0001';
  });
  // 2) 注释
  out = out.replace(/(#[^\n]*)/g, m => {
    store.push('<span class="c">' + m + '</span>');
    return '\u0000' + (store.length - 1) + '\u0001';
  });
  // 3) 关键字
  out = out.replace(/\b(def|class|return|if|else|elif|for|in|while|with|as|import|from|not|and|or|None|True|False|assert|raise|try|except|finally|lambda|yield|pass|break|continue|global|self)\b/g,
                   '<span class="k">$1</span>');
  // 4) 还原（可能要两轮：注释里嵌了字符串占位符）
  for(let pass = 0; pass < 3; pass++){
    out = out.replace(/\u0000(\d+)\u0001/g, (_, i) => store[+i]);
  }
  return out;
}

const NAV = {
  '总览': ['map'],
  '模型': ['hstu','mtgr','longer','hyformer','tiger','rankmixer','onerec'],
  '并行轴': ['cp','ep','beam','mixed'],
  '交叉': ['matrix']
};

function renderNav(){
  const nav = document.getElementById('nav');
  nav.innerHTML = '';
  for(const [grp, ids] of Object.entries(NAV)){
    const g = document.createElement('div');
    g.className = 'grp'; g.textContent = grp;
    nav.appendChild(g);
    for(const id of ids){
      const it = DATA.find(d => d.id === id);
      if(!it || !match(it)) continue;
      const d = document.createElement('div');
      d.className = 'item' + (id === current ? ' on' : '');
      d.innerHTML = '<span>'+it.title+'</span><small>'+it.blurb+'</small>';
      d.onclick = () => goto(id);
      nav.appendChild(d);
    }
  }
}

function match(it){
  const q = (document.getElementById('q').value || '').trim().toLowerCase();
  if(!q) return true;
  const blob = [it.title, it.blurb, it.body, it.code, it.out].join(' ').toLowerCase();
  return blob.includes(q);
}

let current = null;

function goto(id){
  current = id;
  const it = DATA.find(d => d.id === id);
  const main = document.getElementById('main');
  if(id === 'matrix'){
    main.innerHTML = matrixHTML();
    const tt = document.getElementById('tt');
    if(tt) tt.textContent = '模型 × 并行轴';
    drawer(false);
    renderNav();
    return;
  }
  const tabs = [['doc','说明'],['code','代码'],['out','运行输出']];
  let h = '<div class="head"><h2>'+it.title+'</h2><p class="blurb">'+it.blurb+'</p></div>';
  h += '<div class="tabs">';
  tabs.forEach(([k,label],ix) => {
    if(k === 'code' && !it.code) return;
    if(k === 'out' && !it.out) return;
    h += '<div class="tab'+(ix===0?' on':'')+'" data-k="'+k+'">'+label+'</div>';
  });
  h += '</div>';
  h += '<div class="pane on doc" data-k="doc">'+it.body+'</div>';
  if(it.code){
    h += '<div class="pane" data-k="code">'
       + '<div class="codebar"><span>'+it.codePath+'</span>'
       + '<button onclick="copyCode()">复制</button></div>'
       + '<pre class="src" id="srccode">'+hl(it.code)+'</pre></div>';
  }
  if(it.out){
    h += '<div class="pane" data-k="out">'
       + '<div class="codebar"><span>python3 '+it.codePath.split("/").pop()+' 的真实 stdout</span></div>'
       + '<pre class="src">'+it.out.replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</pre></div>';
  }
  main.innerHTML = h;
  main.querySelectorAll('.tab').forEach(t => {
    t.onclick = () => {
      main.querySelectorAll('.tab').forEach(x => x.classList.remove('on'));
      main.querySelectorAll('.pane').forEach(x => x.classList.remove('on'));
      t.classList.add('on');
      main.querySelector('.pane[data-k="'+t.dataset.k+'"]').classList.add('on');
    };
  });
  main.scrollTop = 0;
  window.scrollTo(0,0);
  const tt = document.getElementById('tt');
  if(tt) tt.textContent = it.title;
  drawer(false);
  renderNav();
}

function copyCode(){
  const txt = document.getElementById('srccode').innerText;
  navigator.clipboard && navigator.clipboard.writeText(txt);
}

const MXDATA = [
  ['TIGER',      ['无对象','g0'], ['不适用','gn'], ['不需要','g0'], ['无对象','g0'], 'S=20 / B=10~20 ⇒ 全是小量'],
  ['HSTU',       ['能切但不需要','g0'], ['仅 MoE 版','gn'], ['可以','g1'], ['无关','gn'], '纯加归约，但 KV 单卡装得下'],
  ['MTGR',       ['有对象','g2'], ['有条件','g1'], ['可以','g1'], ['无关','gn'], '静态段是方阵 ⇒ 失衡 1.00'],
  ['LONGER',     ['对象小','g1'], ['论文未用','gn'], ['可以','g1'], ['无关','gn'], 'q/L≈1e-2，Amdahl 只 1.2~1.3×'],
  ['RankMixer',  ['无定义','gn'], ['主战场','g2'], ['可以','g1'], ['无关','gn'], 'T 是特征子空间数，没有 L 维'],
  ['HyFormer',   ['买延迟','g2'], ['仅 MoE','gn'], ['有用','g1'], ['不适用','gn'], 'q≈16，算便宜读贵 ⇒ CP 切读'],
  ['OpenOneRec', ['有对象','g1'], ['闭源才有','gn'], ['8 卡触顶','g1'], ['主战场','g2'], 'q=B=512，beam 是主维度'],
];
const MXA = ['模型','CP','EP','TP','beam','一句话'];
const MXCODE = ['tiger','hstu','mtgr','longer','rankmixer','hyformer','onerec'];
const MXAXIS = ['cp','ep','ep','beam'];

function matrixHTML(){
  let h = '<div class="head"><h2>模型 × 并行轴</h2>'
        + '<p class="blurb">点任意格子跳到对应条目。绿色=没有对象，黄色=有条件，蓝色=主战场。</p></div>';
  h += '<table class="mx"><thead><tr>';
  MXA.forEach(a => h += '<th>'+a+'</th>');
  h += '</tr></thead><tbody>';
  MXDATA.forEach((row,ri) => {
    h += '<tr><td class="cell" data-go="'+MXCODE[ri]+'" style="text-align:left"><b>'+row[0]+'</b></td>';
    for(let c=0;c<4;c++){
      const [label,cls] = row[c+1];
      h += '<td class="cell '+cls+'" data-go="'+MXAXIS[c]+'">'+label+'</td>';
    }
    h += '<td style="color:#8b97a7">'+row[5]+'</td></tr>';
  });
  h += '</tbody></table>';
  h += '<p class="legend">'
     + '<b>无对象 / 不需要</b>：可行但没意义（容量或数量太小）　'
     + '<b>无定义 / 不适用</b>：根本没有那一维（RankMixer 没有序列维；dense 模型没有专家）<br>'
     + '<b>有条件</b>：需要附加约束（路由相关性 ρ / GLN 归约轴）　'
     + '<b>主战场</b>：这一维就是模型的主要缩放方向　'
     + '<span style="color:#5ec8d8">点模型名 → 模型页；点格子 → 对应的轴</span></p>';
  return h;
}

document.addEventListener('click', e => {
  const c = e.target.closest('[data-go]');
  if(c) goto(c.dataset.go);
});

/* ---------- 窄屏抽屉 ---------- */
function drawer(open){
  document.body.classList.toggle('drawer', open);
}
document.getElementById('burger').addEventListener('click', () => {
  drawer(!document.body.classList.contains('drawer'));
});
document.getElementById('scrim').addEventListener('click', () => drawer(false));

document.getElementById('q').addEventListener('input', renderNav);
document.addEventListener('keydown', e => {
  if(e.key === '/' && document.activeElement.id !== 'q'){
    e.preventDefault(); document.getElementById('q').focus();
  }
  if(e.key === 'Escape'){ drawer(false); document.getElementById('q').blur(); }
});

goto('map');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    html_text = build()
    out_path = os.path.join(HERE, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"wrote {out_path}  ({len(html_text) / 1024:.1f} KiB)")
