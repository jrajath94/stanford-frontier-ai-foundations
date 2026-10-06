import sys, json
from playwright.sync_api import sync_playwright

pages = [
    "index.html", "cheatsheet.html",
    "l01-introduction.html", "l02-linear-regression.html", "l03-logistic-regression.html",
    "l04-glms-softmax.html", "l05-gda-naive-bayes.html", "l06-bias-variance.html",
    "l07-neural-networks-1.html", "l08-backpropagation.html", "l09-kmeans-gmm.html",
    "l10-em-pca.html", "l11-diffusion-models.html", "l12-foundation-models.html",
    "l13-contrastive-rag.html", "l14-transformers.html", "l15-efficient-icl-sft.html",
    "l16-reinforcement-learning.html", "l17-rl-for-llms.html",
]
viewports = [375, 768, 1200]
base = "file:///home/hatch/workspace/stanford-frontier-ai/site/v2/cs229/"
results = {}
with sync_playwright() as p:
    browser = p.chromium.launch()
    for pg in pages:
        results[pg] = {}
        for w in viewports:
            errors = []
            page = browser.new_page(viewport={"width": w, "height": 900})
            page_errors = []
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            broken = []
            page.on("response", lambda r: broken.append(r.url) if r.status >= 400 else None)
            page.goto(base + pg, wait_until="networkidle")
            # overflow: any element wider than viewport
            overflow = page.evaluate("""() => {
                const vw = document.documentElement.clientWidth;
                let max = 0, els = [];
                document.querySelectorAll('body *').forEach(el => {
                    const r = el.getBoundingClientRect();
                    if (r.right > max) { max = r.right; }
                });
                return {vw: vw, maxRight: max, over: Math.max(0, max - vw)};
            }""")
            # broken images: naturalWidth == 0
            broken_imgs = page.evaluate("""() => {
                const out = [];
                document.querySelectorAll('img').forEach(img => {
                    if (img.naturalWidth === 0) out.push(img.getAttribute('src'));
                });
                return out;
            }""")
            results[pg][w] = {
                "overflow_px": round(overflow["over"], 1),
                "broken_imgs": broken_imgs,
                "http_broken": broken,
                "page_errors": page_errors,
            }
            status = "OK" if (overflow["over"] < 1 and not broken_imgs and not broken and not page_errors) else "FAIL"
            print(f"{pg} @ {w}: {status} (overflow={round(overflow['over'],1)}px, broken_imgs={len(broken_imgs)}, http={len(broken)}, errors={len(page_errors)})", flush=True)
            if status == "FAIL":
                print(f"  detail: {json.dumps(results[pg][w])[:500]}", flush=True)
            page.close()
    browser.close()
fails = sum(1 for pg in results for w in results[pg]
            if results[pg][w]["overflow_px"] >= 1 or results[pg][w]["broken_imgs"]
            or results[pg][w]["http_broken"] or results[pg][w]["page_errors"])
print(f"\nTOTAL FAILS: {fails}")
json.dump(results, open("/home/hatch/workspace/stanford-frontier-ai/build/render/r1_cs229.json", "w"), indent=1)
