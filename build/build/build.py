#!/usr/bin/env python3
"""Static site generator for the Stanford Frontier AI Learning System.

Usage: python3 build.py <content_dir> <output_dir> <site_name>
Reads content/**/*.md (YAML frontmatter + markdown), renders with the shared
template, writes static HTML, search index, and copies assets.
"""
import os, re, sys, json, shutil, html as htmlmod
import yaml, markdown

CONTENT, OUT, SITE_NAME = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
TPL = open(os.path.join(HERE, "templates/base.html")).read()

md = markdown.Markdown(extensions=["fenced_code", "tables", "sane_lists", "toc", "smarty"])

def parse(path):
    raw = open(path, encoding="utf-8").read()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        raise ValueError(f"No frontmatter in {path}")
    fm = yaml.safe_load(m.group(1))
    return fm, m.group(2)

def ts_link(video_id, body):
    # [mm:ss](ts:12:34) -> youtube timestamp link
    def repl(m):
        label, ts = m.group(1), m.group(2)
        parts = [int(x) for x in ts.split(":")]
        s = 0
        for p in parts: s = s * 60 + p
        url = f"https://www.youtube.com/watch?v={video_id}&t={s}s"
        return f'<a href="{htmlmod.escape(url)}" target="_blank" rel="noopener">{label} <span class="ts">⏱{htmlmod.escape(ts)}</span></a>'
    return re.sub(r"\[([^\]]+)\]\(ts:([0-9:]+)\)", repl, body)

CALL = {"NOTE": "Note", "WARN": "Warning", "KEY": "Key idea", "PROF": "Professor note",
        "CAVEAT": "Caveat", "INTERVIEW": "Interview signal", "PAPER": "Paper"}

def callouts(body):
    def repl(m):
        kind, inner = m.group(1), m.group(2)
        label = CALL.get(kind, kind.title())
        inner_html = md.convert(inner.strip()) if False else inner
        return f'<div class="callout"><span class="co-label">{label}</span>{inner_html}</div>'
    # apply to blockquotes starting with [!TYPE]
    lines = body.split("\n")
    out, buf, kind = [], [], None
    def flush():
        nonlocal buf, kind
        if kind:
            inner_md = "\n".join(x[2:] for x in buf)
            inner_html = markdown.markdown(inner_md, extensions=["fenced_code", "tables", "sane_lists", "smarty"])
            label = CALL.get(kind, kind.title())
            out.append(f'<div class="callout"><span class="co-label">{label}</span>{inner_html}</div>')
        elif buf:
            out.extend(buf)
        buf, kind = [], None
    for ln in lines:
        mm = re.match(r"^>\s*\[!(\w+)\]\s*(.*)$", ln)
        if mm:
            flush(); kind = mm.group(1).upper(); rest = mm.group(2)
            buf = ["> " + rest] if rest else []
        elif kind and (ln.startswith(">") or ln.strip() == ""):
            buf.append(ln)
        elif kind:
            flush(); out.append(ln)
        else:
            out.append(ln)
    flush()
    return "\n".join(out)

def render_body(body_md, video_id):
    if video_id:
        body_md = ts_link(video_id, body_md)
    body_md = callouts(body_md)
    h = md.convert(body_md)
    md.reset()
    # mermaid blocks
    h = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
               lambda m: f'<div class="mermaid">{htmlmod.unescape(m.group(1))}</div>',
               h, flags=re.S)
    # figures: images with title -> figure + figcaption
    h = re.sub(r'<p><img alt="([^"]*)" src="([^"]*)" title="([^"]*)" /></p>',
               r'<figure><img alt="\1" src="\2"><figcaption>\3</figcaption></figure>', h)
    return h

def video_embed(fm):
    vid = fm.get("video_id")
    if not vid:
        return ""
    title = htmlmod.escape(fm.get("video_title", fm.get("title", "Lecture video")))
    cap = fm.get("video_caption", "Original Stanford lecture. Timestamps in the text link to the exact moment.")
    return (f'<div class="video-wrap"><iframe src="https://www.youtube-nocookie.com/embed/{vid}" '
            f'title="{title}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
            f'allowfullscreen loading="lazy"></iframe></div><p class="video-cap">{cap}</p>')

def sources_box(fm):
    srcs = fm.get("sources") or []
    if not srcs:
        return ""
    items = []
    for s in srcs:
        tag = s.get("tag", "source").upper()
        label = htmlmod.escape(s.get("label", ""))
        url = s.get("url")
        if url:
            items.append(f'<li><span class="src-tag">{htmlmod.escape(tag)}</span><a href="{htmlmod.escape(url)}" target="_blank" rel="noopener">{label}</a></li>')
        else:
            items.append(f'<li><span class="src-tag">{htmlmod.escape(tag)}</span>{label}</li>')
    return '<section class="sources"><h2>Sources</h2><ul>' + "".join(items) + "</ul></section>"

# ---- collect pages ----
pages = []
for dirpath, _, files in os.walk(CONTENT):
    for f in sorted(files):
        if not f.endswith(".md") or f.startswith("_"):
            continue
        path = os.path.join(dirpath, f)
        rel = os.path.relpath(path, CONTENT)
        fm, body = parse(path)
        url = rel[:-3] + ".html"
        pages.append({"fm": fm, "body": body, "url": url, "rel": rel})

def sort_key(p):
    fm = p["fm"]
    return (fm.get("course_order", 99), fm.get("order", 99), fm.get("title", ""))
pages.sort(key=sort_key)

# prev/next within same course
by_course = {}
for p in pages:
    by_course.setdefault(p["fm"].get("course_slug", ""), []).append(p)
for course, lst in by_course.items():
    for i, p in enumerate(lst):
        if i > 0: p["prev"] = lst[i-1]
        if i < len(lst)-1: p["next"] = lst[i+1]

def sidebar(active_url):
    parts = []
    courses = {}
    for p in pages:
        c = p["fm"].get("course_slug", "")
        courses.setdefault(c, {"name": p["fm"].get("course_name", c), "items": []})
        courses[c]["items"].append(p)
    for slug, c in courses.items():
        parts.append(f"<h3>{htmlmod.escape(c['name'])}</h3><ul>")
        for p in c["items"]:
            act = ' class="active"' if p["url"] == active_url else ""
            pid = p["fm"].get("page_id", p["url"])
            parts.append(f'<li><a href="{p["url"]}" data-page="{htmlmod.escape(pid)}"{act}>{htmlmod.escape(p["fm"].get("nav", p["fm"].get("title","")))}</a></li>')
        parts.append("</ul>")
    return "\n".join(parts)

def breadcrumb(p):
    fm = p["fm"]
    return (f'<nav class="breadcrumb"><a href="index.html">Home</a> / '
            f'<a href="#">{htmlmod.escape(fm.get("course_name",""))}</a> / '
            f'{htmlmod.escape(fm.get("title",""))}</nav>')

def meta_line(p):
    fm = p["fm"]
    bits = []
    if fm.get("date"): bits.append(htmlmod.escape(str(fm["date"])))
    if fm.get("instructor"): bits.append("Instructor: " + htmlmod.escape(fm["instructor"]))
    if fm.get("duration"): bits.append(htmlmod.escape(str(fm["duration"])))
    if fm.get("offering"): bits.append(htmlmod.escape(str(fm["offering"])))
    return f'<p class="meta-line">{" · ".join(bits)}</p>' if bits else ""

def pager(p):
    out = ['<nav class="pager">']
    if p.get("prev"):
        q = p["prev"]
        out.append(f'<a href="{q["url"]}"><span class="lbl">← Previous</span>{htmlmod.escape(q["fm"].get("title",""))}</a>')
    else:
        out.append("<span></span>")
    if p.get("next"):
        q = p["next"]
        out.append(f'<a href="{q["url"]}"><span class="lbl">Next →</span>{htmlmod.escape(q["fm"].get("title",""))}</a>')
    else:
        out.append("<span></span>")
    out.append("</nav>")
    return "".join(out)

def progress_row():
    return ('<div class="progress-row"><button class="mark-done">☐ <span>Mark as complete</span></button>'
            '<div class="progress-bar"><i></i></div><span class="pct"></span></div>')

# ---- write pages ----
os.makedirs(OUT, exist_ok=True)
search_idx = []
for p in pages:
    fm, body_md = p["fm"], p["body"]
    depth = p["url"].count("/") - 1
    root = "../" * max(depth, 0)
    body_html = render_body(body_md, fm.get("video_id"))
    body_html = video_embed(fm) + body_html
    html_page = TPL
    html_page = html_page.replace("{{ root }}", root)
    html_page = html_page.replace("{{ site_name }}", htmlmod.escape(SITE_NAME))
    html_page = html_page.replace("{{ page_title }}", htmlmod.escape(fm.get("title", "")))
    html_page = html_page.replace("{{ meta_desc }}", htmlmod.escape(fm.get("summary", fm.get("title", ""))[:160]))
    html_page = html_page.replace("{{ sidebar }}", sidebar(p["url"]))
    html_page = html_page.replace("{{ breadcrumb }}", breadcrumb(p))
    html_page = html_page.replace("{{ h1 }}", htmlmod.escape(fm.get("title", "")))
    html_page = html_page.replace("{{ meta_line }}", meta_line(p))
    html_page = html_page.replace("{{ body }}", progress_row() + body_html)
    html_page = html_page.replace("{{ sources_box }}", sources_box(fm))
    html_page = html_page.replace("{{ pager }}", pager(p))
    html_page = re.sub(r"<body>", f'<body data-page="{htmlmod.escape(fm.get("page_id", p["url"]))}">', html_page, count=1)
    dest = os.path.join(OUT, p["url"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "w", encoding="utf-8").write(html_page)
    text = re.sub(r"<[^>]+>", " ", body_html)
    text = re.sub(r"\s+", " ", text)[:800]
    search_idx.append({"title": fm.get("title", ""), "url": p["url"],
                       "crumb": fm.get("course_name", ""), "text": text})

open(os.path.join(OUT, "search_index.json"), "w").write(json.dumps(search_idx))

# assets
for d in ["assets", "assets/vendor"]:
    src = os.path.join(HERE, d)
    dst = os.path.join(OUT, d)
    if os.path.exists(dst): shutil.rmtree(dst)
    if os.path.exists(src): shutil.copytree(src, dst)

# figures from content (survive rebuilds)
fig_src = os.path.join(CONTENT, "assets", "figures")
fig_dst = os.path.join(OUT, "assets", "figures")
if os.path.exists(fig_src):
    os.makedirs(fig_dst, exist_ok=True)
    for f in os.listdir(fig_src):
        shutil.copy2(os.path.join(fig_src, f), os.path.join(fig_dst, f))

print(f"Built {len(pages)} pages -> {OUT}")
