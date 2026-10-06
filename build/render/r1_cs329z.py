import sys, json
from playwright.sync_api import sync_playwright

pages = [
    "index.html",
    "l01-intro-agentic-systems.html",
    "l02-compound-ai-systems.html",
    "l03-llms-for-builders.html",
    "l04-reasoning-and-context.html",
    "l05-rag-pipeline.html",
    "l06-retrieval-methods.html",
    "l07-agentic-retrieval-failure-modes.html",
    "l08-evaluating-agents.html",
]
viewports = [375, 768, 1200]
base = "file:///home/hatch/workspace/stanford-frontier-ai/site/v2/cs329z/"
results = {}
with sync_playwright() as p:
    browser = p.chromium.launch()
    for pg in pages:
        results[pg] = {}
        for w in viewports:
            page = browser.new_page(viewport={"width": w, "height": 900})
            page_errors = []
            console_errors = []
            http_broken = []
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.on("console", lambda m: console_errors.append(m.text[:300]) if m.type == "error" else None)
            page.on("response", lambda r: http_broken.append(f"{r.status} {r.url}") if r.status >= 400 else None)
            try:
                page.goto(base + pg, wait_until="networkidle", timeout=45000)
            except Exception as e:
                try:
                    page.goto(base + pg, wait_until="domcontentloaded", timeout=30000)
                except Exception as e2:
                    page_errors.append(f"GOTO-FAIL: {e2}")
            page.wait_for_timeout(1500)
            overflow = page.evaluate("""() => {
                const vw = document.documentElement.clientWidth;
                let max = 0, els = [];
                document.querySelectorAll('body *').forEach(el => {
                    const r = el.getBoundingClientRect();
                    if (r.right > max) { max = r.right; els = [el.tagName + '.' + el.className]; }
                });
                const sw = document.documentElement.scrollWidth;
                return {vw: vw, maxRight: max, over: Math.max(0, max - vw),
                        scrollWidth: sw, swOver: Math.max(0, sw - vw), tag: els[0]};
            }""")
            broken_imgs = page.evaluate("""() => {
                const out = [];
                document.querySelectorAll('img').forEach(img => {
                    if (img.naturalWidth === 0) out.push(img.getAttribute('src'));
                });
                return out;
            }""")
            results[pg][w] = {
                "overflow_px": round(overflow["over"], 1),
                "sw_overflow_px": round(overflow["swOver"], 1),
                "over_tag": overflow["tag"],
                "broken_imgs": broken_imgs,
                "http_broken": http_broken,
                "page_errors": page_errors,
                "console_errors": console_errors,
            }
            ok = (overflow["over"] < 1 and not broken_imgs and not http_broken
                  and not page_errors and not console_errors)
            status = "OK" if ok else "CHECK"
            print(f"{pg} @ {w}: {status} (overflow={round(overflow['over'],1)}px, "
                  f"sw_overflow={round(overflow['swOver'],1)}px, broken_imgs={len(broken_imgs)}, "
                  f"http={len(http_broken)}, page_errors={len(page_errors)}, console={len(console_errors)})",
                  flush=True)
            if status == "CHECK":
                print(f"  detail: {json.dumps(results[pg][w])[:800]}", flush=True)
            page.close()
    browser.close()
json.dump(results, open("/home/hatch/workspace/stanford-frontier-ai/build/render/r1_cs329z.json", "w"), indent=1)
print("wrote build/render/r1_cs329z.json")
