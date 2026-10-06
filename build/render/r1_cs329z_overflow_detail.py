import json
from playwright.sync_api import sync_playwright

pages = {
    "l01-intro-agentic-systems.html": 224.3,
    "l02-compound-ai-systems.html": 163.9,
    "l03-llms-for-builders.html": 37.0,
    "l05-rag-pipeline.html": 81.3,
    "l06-retrieval-methods.html": 151.3,
    "l08-evaluating-agents.html": 34.4,
}
base = "file:///home/hatch/workspace/stanford-frontier-ai/site/v2/cs329z/"
out = {}
with sync_playwright() as p:
    browser = p.chromium.launch()
    for pg in pages:
        page = browser.new_page(viewport={"width": 375, "height": 900})
        page.goto(base + pg, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(1000)
        info = page.evaluate("""() => {
            const vw = document.documentElement.clientWidth;
            const clips = [];
            document.querySelectorAll('body *').forEach(el => {
                const r = el.getBoundingClientRect();
                if (r.right > vw + 1) {
                    // walk ancestors for overflow-x
                    let a = el.parentElement, clipper = null;
                    while (a && a !== document.body) {
                        const ox = getComputedStyle(a).overflowX;
                        if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') { clipper = a.tagName + '.' + a.className; break; }
                        a = a.parentElement;
                    }
                    const bodyOx = getComputedStyle(document.body).overflowX;
                    const htmlOx = getComputedStyle(document.documentElement).overflowX;
                    clips.push({
                        tag: el.tagName + (el.className ? '.' + String(el.className).split(' ').slice(0,2).join('.') : ''),
                        over: Math.round(r.right - vw),
                        clipper: clipper,
                        bodyOx: bodyOx, htmlOx: htmlOx,
                        snippet: (el.textContent || '').trim().slice(0, 60)
                    });
                }
            });
            return {
                swOver: document.documentElement.scrollWidth - vw,
                bodyScrollW: document.body.scrollWidth,
                items: clips.slice(0, 8)
            };
        }""")
        out[pg] = info
        print(f"== {pg}: swOver={info['swOver']}", flush=True)
        for it in info["items"]:
            print(f"   {it['tag']} over={it['over']}px clipper={it['clipper']} bodyOx={it['bodyOx']} htmlOx={it['htmlOx']}", flush=True)
            print(f"      text: {it['snippet'][:60]}", flush=True)
        page.close()
    browser.close()
json.dump(out, open("/home/hatch/workspace/stanford-frontier-ai/build/render/r1_cs329z_overflow_detail.json", "w"), indent=1)
