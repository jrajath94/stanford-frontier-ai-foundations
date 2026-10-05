/* Stanford Frontier AI Learning System — client behavior */
(function () {
  "use strict";
  var root = window.SITE_ROOT || "./";

  /* Theme */
  var theme = localStorage.getItem("sfai-theme") || "light";
  document.documentElement.setAttribute("data-theme", theme);
  document.getElementById("themeToggle").addEventListener("click", function () {
    theme = theme === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("sfai-theme", theme);
  });

  /* Mobile nav */
  var sidebar = document.getElementById("sidebar");
  document.getElementById("navToggle").addEventListener("click", function () {
    sidebar.classList.toggle("open");
  });
  sidebar.addEventListener("click", function (e) {
    if (e.target.tagName === "A") sidebar.classList.remove("open");
  });

  /* Progress: mark done + sidebar checks + progress bar */
  var pageId = document.body.getAttribute("data-page");
  function getDone() {
    try { return JSON.parse(localStorage.getItem("sfai-done") || "[]"); } catch (e) { return []; }
  }
  function setDone(list) { localStorage.setItem("sfai-done", JSON.stringify(list)); }
  var done = getDone();
  document.querySelectorAll(".sidebar a[data-page]").forEach(function (a) {
    if (done.indexOf(a.getAttribute("data-page")) !== -1) {
      var s = document.createElement("span");
      s.className = "done-mark"; s.textContent = "✓";
      a.prepend(s);
    }
  });
  var bar = document.querySelector(".progress-bar i");
  var barLabel = document.querySelector(".progress-row .pct");
  if (bar) {
    var links = document.querySelectorAll(".sidebar a[data-page]");
    var pct = links.length ? Math.round(100 * done.length / links.length) : 0;
    bar.style.width = pct + "%";
    if (barLabel) barLabel.textContent = pct + "% complete";
  }
  var markBtn = document.querySelector(".mark-done");
  if (markBtn && pageId) {
    if (done.indexOf(pageId) !== -1) { markBtn.classList.add("done"); markBtn.querySelector("span").textContent = "Completed ✓"; }
    markBtn.addEventListener("click", function () {
      var d = getDone(); var i = d.indexOf(pageId);
      if (i === -1) { d.push(pageId); markBtn.classList.add("done"); markBtn.querySelector("span").textContent = "Completed ✓"; }
      else { d.splice(i, 1); markBtn.classList.remove("done"); markBtn.querySelector("span").textContent = "Mark as complete"; }
      setDone(d);
    });
  }

  /* KaTeX rendering: \(...\) inline, \[...\] display */
  function renderMath() {
    if (!window.katex) return;
    var walker = document.createTreeWalker(document.querySelector("article"), NodeFilter.SHOW_TEXT);
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(function (node) {
      var text = node.nodeValue;
      if (text.indexOf("\\(") === -1 && text.indexOf("\\[") === -1) return;
      var frag = document.createDocumentFragment();
      var re = /(\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\])/g;
      var last = 0, m;
      while ((m = re.exec(text)) !== null) {
        if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        var raw = m[0], display = raw.slice(0, 2) === "\\[";
        var tex = raw.slice(2, -2);
        var span = document.createElement(display ? "div" : "span");
        try { window.katex.render(tex, span, { displayMode: display, throwOnError: false }); }
        catch (e) { span.textContent = raw; }
        frag.appendChild(span);
        last = m.index + raw.length;
      }
      if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    });
  }

  /* Mermaid */
  function renderMermaid() {
    if (!window.mermaid) return;
    window.mermaid.initialize({ startOnLoad: false, theme: document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "default" });
    document.querySelectorAll("pre code.language-mermaid").forEach(function (code, i) {
      var div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = code.textContent;
      code.parentNode.replaceWith(div);
    });
    window.mermaid.run({ querySelector: ".mermaid" });
  }

  /* Search */
  var searchIndex = null, searchBox = document.getElementById("searchBox"), resultsBox = document.getElementById("searchResults");
  function loadIndex(cb) {
    if (searchIndex) return cb(searchIndex);
    fetch(root + "search_index.json").then(function (r) { return r.json(); }).then(function (j) { searchIndex = j; cb(j); }).catch(function () { cb([]); });
  }
  var searchTimer = null;
  searchBox.addEventListener("input", function () {
    clearTimeout(searchTimer);
    var q = searchBox.value.trim().toLowerCase();
    if (q.length < 2) { resultsBox.hidden = true; return; }
    searchTimer = setTimeout(function () {
      loadIndex(function (idx) {
        var words = q.split(/\s+/);
        var hits = idx.map(function (e) {
          var hay = (e.title + " " + e.text).toLowerCase(); var score = 0;
          words.forEach(function (w) { if (hay.indexOf(w) !== -1) score += (e.title.toLowerCase().indexOf(w) !== -1 ? 3 : 1); });
          return { e: e, score: score };
        }).filter(function (h) { return h.score > 0; }).sort(function (a, b) { return b.score - a.score; }).slice(0, 12);
        resultsBox.innerHTML = "";
        if (!hits.length) { resultsBox.innerHTML = '<a>No matches</a>'; }
        hits.forEach(function (h) {
          var a = document.createElement("a");
          a.href = root + h.e.url;
          a.innerHTML = "";
          var t = document.createElement("span"); t.textContent = h.e.title;
          var c = document.createElement("span"); c.className = "crumb"; c.textContent = "  ·  " + h.e.crumb;
          a.appendChild(t); a.appendChild(c);
          resultsBox.appendChild(a);
        });
        resultsBox.hidden = false;
      });
    }, 180);
  });
  document.addEventListener("click", function (e) {
    if (!resultsBox.contains(e.target) && e.target !== searchBox) resultsBox.hidden = true;
  });

  renderMath();
  renderMermaid();
})();
