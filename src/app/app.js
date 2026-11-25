/* ToxScope — every figure is read from DATA, generated from the project's own CSVs. */
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("data").textContent);
  var EP = DATA.endpoints, CO = DATA.compounds, BM = DATA.benchmarks, META = DATA.meta;
  var RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var el = function (t, c, h) { var n = document.createElement(t); if (c) n.className = c;
    if (h != null) n.innerHTML = h; return n; };
  var mean = function (a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s / a.length; };

  /* what each assay actually measures */
  var BIO = {
    "NR-AR": "Activation of the androgen receptor, the principal route of endocrine disruption in male reproductive tissue.",
    "NR-AR-LBD": "The same receptor as a ligand-binding-domain construct, which isolates direct binding from downstream signalling.",
    "NR-AhR": "The xenobiotic sensor that induces CYP1A1 metabolism; activated by planar aromatic compounds.",
    "NR-Aromatase": "CYP19A1, the enzyme converting androgens to estrogens. Inhibition shifts the hormonal balance.",
    "NR-ER": "Activation of the estrogen receptor, the most studied endocrine-disruption endpoint in the panel.",
    "NR-ER-LBD": "Ligand-binding-domain construct of the estrogen receptor, reported separately from the full-length assay.",
    "NR-PPAR-gamma": "A nuclear receptor governing fat-cell differentiation and insulin sensitivity.",
    "SR-ARE": "The Nrf2-driven antioxidant response element, the cell's front line against oxidative stress.",
    "SR-ATAD5": "A genotoxicity reporter that rises when DNA replication stalls or is damaged.",
    "SR-HSE": "The heat-shock response, triggered when proteins begin to misfold.",
    "SR-MMP": "Mitochondrial membrane potential. Collapse here is a direct marker of mitochondrial toxicity.",
    "SR-p53": "The p53 tumour-suppressor response to DNA damage, central to carcinogenicity screening."
  };

  var MODELS = ["Ensemble", "Multi-task DNN", "ExtraTrees", "Random Forest", "LightGBM"];
  var MDESC = {
    "Ensemble": "A rank-average of the four models below. Each model's scores are converted to percentile ranks before averaging, so differences in how each one scales its probabilities cannot distort the combination.",
    "Multi-task DNN": "A shared trunk of 512, 256 and 128 units feeding twelve sigmoid heads, trained with a masked class-weighted cross-entropy and averaged over five random seeds. One forward pass scores all twelve assays.",
    "ExtraTrees": "Five hundred extremely randomised trees per endpoint. Split thresholds are drawn at random rather than optimised, which lowers variance on sparse fingerprint features.",
    "Random Forest": "Five hundred bootstrap-aggregated decision trees per endpoint, each fitted only on the compounds actually tested for that assay.",
    "LightGBM": "Six hundred gradient-boosted trees per endpoint at learning rate 0.03 with 15 leaves — a histogram-based booster tuned for wide, sparse inputs."
  };
  var MTAG = {
    "Ensemble": "Rank-average of four", "Multi-task DNN": "One network, twelve heads",
    "ExtraTrees": "Single-task, 500 trees", "Random Forest": "Single-task, 500 trees",
    "LightGBM": "Single-task, boosted"
  };

  /* ---------- risk ramp: teal (low) -> amber -> rose (high) ---------- */
  function ramp(p) {
    var s = [[0, 94, 234, 212], [0.45, 255, 194, 75], [1, 255, 77, 109]];
    for (var i = 0; i < s.length - 1; i++) {
      var a = s[i], b = s[i + 1];
      if (p <= b[0]) {
        var t = (p - a[0]) / (b[0] - a[0] || 1);
        return "rgb(" + Math.round(a[1] + (b[1] - a[1]) * t) + "," +
          Math.round(a[2] + (b[2] - a[2]) * t) + "," + Math.round(a[3] + (b[3] - a[3]) * t) + ")";
      }
    }
    return "rgb(255,77,109)";
  }

  /* ---------- molecule drawing from precomputed RDKit 2D coordinates ---------- */
  var ECOL = { N: "#5B9DFF", O: "#FF6B8A", S: "#FFC24B", Cl: "#5EEAD4", Br: "#C98B5A",
    F: "#7FE7C4", I: "#B88BFF", P: "#FF9F45", Na: "#A98BFF", K: "#A98BFF", Si: "#9AA6B5",
    Se: "#FFC24B", Sn: "#9AA6B5", As: "#FF9F45", Hg: "#9AA6B5", Zn: "#9AA6B5", B: "#FFB38A" };

  function molSVG(c, opt) {
    opt = opt || {};
    var pad = opt.pad == null ? 62 : opt.pad, lw = opt.lw || 15, fs = opt.fs || 74;
    var xs = c.atoms.map(function (a) { return a[1]; }), ys = c.atoms.map(function (a) { return a[2]; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var w = Math.max(x1 - x0, 1), h = Math.max(y1 - y0, 1), s = Math.max(w, h);
    var ox = x0 - (s - w) / 2, oy = y0 - (s - h) / 2;
    var X = function (v) { return (v - ox) + pad; }, Y = function (v) { return (s - (v - oy)) + pad; };
    var box = s + pad * 2, out = [];
    c.bonds.forEach(function (b) {
      var A = c.atoms[b[0]], B = c.atoms[b[1]], o = b[2];
      var ax = X(A[1]), ay = Y(A[2]), bx = X(B[1]), by = Y(B[2]);
      var dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1;
      var px = -dy / L * (lw * 1.5), py = dx / L * (lw * 1.5);
      function trim(sym, sx, sy, tx, ty) {
        if (sym === "C") return [sx, sy];
        var k = fs * 0.46 / L;
        return [sx + (tx - sx) * k, sy + (ty - sy) * k];
      }
      var p1 = trim(A[0], ax, ay, bx, by), p2 = trim(B[0], bx, by, ax, ay);
      var line = function (xa, ya, xb, yb) {
        out.push('<line x1="' + xa.toFixed(1) + '" y1="' + ya.toFixed(1) + '" x2="' + xb.toFixed(1) +
          '" y2="' + yb.toFixed(1) + '" stroke="#C7D2E0" stroke-width="' + lw + '" stroke-linecap="round"/>');
      };
      if (o === 2 || o === 3) {
        var g = o === 3 ? 1 : 0.62;
        line(p1[0] + px * g, p1[1] + py * g, p2[0] + px * g, p2[1] + py * g);
        line(p1[0] - px * g, p1[1] - py * g, p2[0] - px * g, p2[1] - py * g);
        if (o === 3) line(p1[0], p1[1], p2[0], p2[1]);
      } else { line(p1[0], p1[1], p2[0], p2[1]); }
    });
    c.atoms.forEach(function (a) {
      if (a[0] === "C") return;
      var cx = X(a[1]), cy = Y(a[2]);
      out.push('<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="' + (fs * 0.46) + '" fill="#0A0E14"/>');
      out.push('<text x="' + cx.toFixed(1) + '" y="' + cy.toFixed(1) + '" fill="' + (ECOL[a[0]] || "#E3E9F2") +
        '" font-size="' + fs + '" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-weight="600" ' +
        'text-anchor="middle" dominant-baseline="central">' + a[0] +
        (a[3] > 0 ? "+" : a[3] < 0 ? "−" : "") + "</text>");
    });
    return '<svg viewBox="0 0 ' + box.toFixed(0) + " " + box.toFixed(0) +
      '" role="img" aria-label="Chemical structure of ' + c.id + '">' + out.join("") + "</svg>";
  }

  /* ---------- hero particle field ---------- */
  function hero() {
    var cv = $("#field"), cx = cv.getContext("2d"), W, H, pts = [], raf = null, vis = true, mx = 0, my = 0, tx = 0, ty = 0;
    function size() {
      var d = Math.min(devicePixelRatio || 1, 2);
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * d; cv.height = H * d; cx.setTransform(d, 0, 0, d, 0, 0);
      var n = Math.round(Math.min(72, Math.max(26, W * H / 26000)));
      pts = [];
      for (var i = 0; i < n; i++) pts.push({ x: Math.random() * W, y: Math.random() * H,
        vx: (Math.random() - .5) * .22, vy: (Math.random() - .5) * .22, r: Math.random() * 1.9 + 1.1 });
    }
    function frame() {
      cx.clearRect(0, 0, W, H);
      tx += (mx - tx) * .045; ty += (my - ty) * .045;
      for (var i = 0; i < pts.length; i++) {
        var p = pts[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < -40) p.x = W + 40; if (p.x > W + 40) p.x = -40;
        if (p.y < -40) p.y = H + 40; if (p.y > H + 40) p.y = -40;
        for (var j = i + 1; j < pts.length; j++) {
          var q = pts[j], dx = p.x - q.x, dy = p.y - q.y, d2 = dx * dx + dy * dy;
          if (d2 < 24000) {
            cx.strokeStyle = "rgba(94,234,212," + ((1 - d2 / 24000) * .3).toFixed(3) + ")";
            cx.lineWidth = 1; cx.beginPath();
            cx.moveTo(p.x + tx * .018, p.y + ty * .018); cx.lineTo(q.x + tx * .018, q.y + ty * .018); cx.stroke();
          }
        }
      }
      for (var k = 0; k < pts.length; k++) {
        var a = pts[k];
        cx.fillStyle = "rgba(160,220,255,.6)"; cx.beginPath();
        cx.arc(a.x + tx * .018, a.y + ty * .018, a.r, 0, 6.2832); cx.fill();
      }
      raf = requestAnimationFrame(frame);
    }
    function start() { if (!raf && vis && !RM) frame(); }
    function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }
    size(); addEventListener("resize", size);
    addEventListener("mousemove", function (e) { mx = e.clientX - innerWidth / 2; my = e.clientY - innerHeight / 2; });
    document.addEventListener("visibilitychange", function () { vis = !document.hidden; vis ? start() : stop(); });
    new IntersectionObserver(function (es) { es[0].isIntersecting ? start() : stop(); }, { threshold: 0 }).observe(cv);
    if (RM) { frame(); stop(); }
  }

  /* ---------- reveal + counters + scroll progress ---------- */
  function reveals() {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        e.target.classList.add("in"); io.unobserve(e.target);
        if (e.target.dataset.count) count(e.target);
        if (e.target.classList.contains("mgrid")) matrixIn(e.target);
        if (e.target.classList.contains("bwrap")) grow(e.target);
      });
    }, { threshold: .14, rootMargin: "0px 0px -6% 0px" });
    document.querySelectorAll(".rv,.mgrid,.bwrap").forEach(function (n) { io.observe(n); });
  }
  function count(n) {
    var to = parseFloat(n.dataset.count), dec = parseInt(n.dataset.dec || "0", 10);
    if (RM) { n.textContent = fmt(to, dec); return; }
    var t0 = performance.now();
    (function tick(t) {
      var k = Math.min((t - t0) / 1250, 1), e = 1 - Math.pow(1 - k, 3);
      n.textContent = fmt(to * e, dec);
      if (k < 1) requestAnimationFrame(tick);
    })(t0);
  }
  function fmt(v, d) { return d ? v.toFixed(d) : Math.round(v).toLocaleString("en-US"); }
  function grow(host) {
    requestAnimationFrame(function () {
      [].forEach.call(host.querySelectorAll("[data-w]"), function (b, i) {
        setTimeout(function () { b.style.width = b.dataset.w + "%"; }, RM ? 0 : i * 26);
      });
    });
  }
  function progress() {
    var bar = $("#prog");
    addEventListener("scroll", function () {
      var h = document.documentElement.scrollHeight - innerHeight;
      bar.style.transform = "scaleX(" + (h > 0 ? scrollY / h : 0).toFixed(4) + ")";
    }, { passive: true });
  }

  /* ---------- sparsity matrix ---------- */
  function matrix() {
    var g = $("#mgrid"), cols = 12, rows = 22, seed = 7;
    g.style.gridTemplateColumns = "repeat(" + cols + ",1fr)";
    function rnd() { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; }
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
      var e = EP[c], d = el("i", "mcell"), u = rnd() * 100;
      d.style.background = u < e.missingRate ? "#101722" : (rnd() * 100 < e.activeRate ? "#FF4D6D" : "#24364F");
      d.style.transitionDelay = (RM ? 0 : (r * cols + c) * 3.4) + "ms";
      d.title = e.id;
      g.appendChild(d);
    }
  }
  function matrixIn(g) { [].forEach.call(g.children, function (c) { c.classList.add("on"); }); }

  /* ---------- compound rails ---------- */
  function topEp(c) { var b = 0; for (var i = 1; i < 12; i++) if (c.pred[i] > c.pred[b]) b = i; return { i: b, p: c.pred[b] }; }
  function card(c) {
    var t = topEp(c), n = el("button", "card");
    n.setAttribute("aria-label", "Open compound " + c.id);
    n.innerHTML = '<div class="thumb">' + molSVG(c, { pad: 70, lw: 17, fs: 84 }) + "</div>" +
      '<div class="meta"><div class="cid">' + c.id + '<span>' + (c.props.formula || "") + "</span></div>" +
      '<div class="top">' + EP[t.i].id + " · " + (t.p * 100).toFixed(0) + "%</div>" +
      '<div class="sub2">' + c.nActive + " active / " + c.nTested + " tested</div>" +
      '<div class="strip">' + EP.map(function (e, i) {
        return '<i title="' + e.id + " " + (c.pred[i] * 100).toFixed(0) + '%" style="background:' + ramp(c.pred[i]) +
          ";opacity:" + (0.32 + c.pred[i] * 0.68).toFixed(2) + '"></i>';
      }).join("") + "</div></div>";
    n.addEventListener("click", function () { openSheet(c); });
    return n;
  }
  function rail(title, note, list) {
    if (!list.length) return null;
    var b = el("div", "rowblk rv");
    b.appendChild(el("div", "rowhd", "<h3>" + title + "</h3><span>" + note + "</span>"));
    var r = el("div", "rail");
    list.slice(0, 22).forEach(function (c) { r.appendChild(card(c)); });
    b.appendChild(r);
    return b;
  }
  function rows() {
    var host = $("#rows");
    var sr = EP.map(function (e, i) { return e.family === "SR" ? i : -1; }).filter(function (i) { return i >= 0; });
    var nr = EP.map(function (e, i) { return e.family === "NR" ? i : -1; }).filter(function (i) { return i >= 0; });
    var by = function (f) { return CO.slice().sort(function (a, b) { return f(b) - f(a); }); };
    [rail("Highest predicted risk", "ranked by mean probability across all twelve assays", by(function (c) { return mean(c.pred); })),
     rail("Flagged on the stress-response panel", "ARE, ATAD5, HSE, MMP and p53 — cellular damage pathways",
          by(function (c) { return mean(sr.map(function (i) { return c.pred[i]; })); })),
     rail("Flagged on nuclear receptors", "the seven hormone-signalling assays",
          by(function (c) { return mean(nr.map(function (i) { return c.pred[i]; })); })),
     rail("Measured active on three or more assays", "laboratory ground truth, not model output",
          CO.filter(function (c) { return c.nActive >= 3; }).sort(function (a, b) { return b.nActive - a.nActive; })),
     rail("Clean across the panel", "measured inactive wherever they were tested",
          CO.filter(function (c) { return c.nActive === 0 && c.nTested >= 11; })
            .sort(function (a, b) { return mean(a.pred) - mean(b.pred); })),
     rail("Complete label rows", "tested on all twelve endpoints — no missing entries",
          CO.filter(function (c) { return c.nTested === 12; }))
    ].forEach(function (n) { if (n) host.appendChild(n); });
  }

  /* ---------- compound sheet ---------- */
  var lastFocus = null;
  function openSheet(c) {
    lastFocus = document.activeElement;
    var sheet = $("#sheet"), t = topEp(c), p = c.props;
    var prop = function (k, v, u) {
      return '<div class="pcell"><span>' + k + '</span><b class="mono">' + (v == null ? "—" : v) +
        (u ? '<em>' + u + "</em>" : "") + "</b></div>";
    };
    var head = '<button class="close" id="xbtn" aria-label="Close">✕</button>' +
      '<div class="shtop"><div><div class="molbox">' + molSVG(c, { pad: 58, lw: 13, fs: 70 }) +
      '</div><div class="smi" title="Canonical SMILES">' + c.smiles + "</div></div>" +
      '<div><div class="skick">Compound</div><h2 class="shtitle">' + c.id + "</h2>" +
      '<div class="formula mono">' + (p.formula || "") + "</div>" +
      '<p class="shlead">Tested on ' + c.nTested + " of 12 Tox21 assays and measured active on " + c.nActive +
      ". Each bar is this network's predicted probability for that assay; the chip beside it is the laboratory result " +
      "for the same compound, which the model never saw during training.</p>" +
      '<div class="props">' +
        prop("Molecular weight", p.mw, " g/mol") + prop("cLogP", p.logp) +
        prop("TPSA", p.tpsa, " Å²") + prop("H-bond donors", p.hbd) +
        prop("H-bond acceptors", p.hba) + prop("Rotatable bonds", p.rotb) +
        prop("Rings", p.rings) + prop("Heavy atoms", p.heavy) +
      "</div>" +
      '<div class="peak"><div><div class="skick">Peak signal</div><div class="mono pk" style="color:' + ramp(t.p) +
      '">' + EP[t.i].id + " · " + (t.p * 100).toFixed(1) + "%</div></div>" +
      '<div><div class="skick">Mean across panel</div><div class="mono pk">' + (mean(c.pred) * 100).toFixed(1) +
      "%</div></div></div></div></div>";

    var body = '<div class="ephd"><span>Assay</span><span>Predicted probability</span><span></span>' +
      "<span>Laboratory result</span></div>" +
      EP.map(function (e, i) {
        var pr = c.pred[i], lab = c.labels[i];
        var chip = lab === 1 ? '<span class="chip c-a">Active</span>'
          : lab === 0 ? '<span class="chip c-i">Inactive</span>' : '<span class="chip c-n">Not tested</span>';
        var agree = lab == null ? "" : ((pr >= .5) === (lab === 1)
          ? '<em class="ok">matches</em>' : '<em class="no">differs</em>');
        return '<div class="eprow"><div class="epn"><b>' + e.id + "</b><span>" + e.name + "</span>" +
          '<i class="bio">' + BIO[e.id] + "</i></div>" +
          '<div class="bar"><i data-w="' + (pr * 100).toFixed(1) + '" style="background:' + ramp(pr) + '"></i></div>' +
          '<div class="pv mono" style="color:' + ramp(pr) + '">' + (pr * 100).toFixed(1) + "%</div>" +
          "<div>" + chip + agree + '<div class="auc mono">assay AUC ' + e.heldOutRoc.toFixed(3) +
          " · " + e.activeRate.toFixed(1) + "% base rate</div></div></div>";
      }).join("");

    sheet.innerHTML = head + '<div class="ep">' + body + "</div>" +
      '<div class="shfoot"><b>These are assay-level predictions, not a safety verdict.</b> A Tox21 model predicts the ' +
      "outcome of a specific in-vitro assay; it says nothing about whether a substance is safe or harmful to a person. " +
      "<i>Assay AUC</i> is this network's held-out ROC-AUC for that endpoint — the lower it is, the less weight the " +
      "probability beside it deserves. <i>Base rate</i> is how often that assay returns active across the whole dataset, " +
      "which is the number any probability should be judged against.</div>";

    $("#modal").classList.add("open");
    document.body.classList.add("lock");
    $("#xbtn").addEventListener("click", closeSheet);
    $("#xbtn").focus();
    sheet.scrollTop = 0;
    setTimeout(function () { grow(sheet); }, RM ? 0 : 90);
  }
  function closeSheet() {
    $("#modal").classList.remove("open");
    document.body.classList.remove("lock");
    if (lastFocus) lastFocus.focus();
  }

  /* ---------- benchmarks ---------- */
  var curModel = "Ensemble";
  function rowsFor(m) { return BM.filter(function (r) { return r.model === m; }); }
  function macroOf(m) {
    var r = rowsFor(m);
    return { roc: mean(r.map(function (x) { return x.roc; })), pr: mean(r.map(function (x) { return x.pr; })) };
  }
  function ledBy(m) {
    var n = 0;
    EP.forEach(function (e) {
      var best = null;
      MODELS.forEach(function (mm) {
        var r = BM.filter(function (x) { return x.model === mm && x.endpoint === e.id; })[0];
        if (r && (!best || r.roc > best.roc)) best = { m: mm, roc: r.roc };
      });
      if (best && best.m === m) n++;
    });
    return n;
  }
  var ROC_LO = 0.5, PR_HI = 1.0;
  function scR(v) { return Math.max(0, (v - ROC_LO) / (1 - ROC_LO) * 100); }
  function scP(v) { return Math.max(0, v / PR_HI * 100); }

  function drawBench() {
    var rs = rowsFor(curModel).slice().sort(function (a, b) { return b.roc - a.roc; });
    var mac = macroOf(curModel);
    var ranked = MODELS.slice().sort(function (a, b) { return macroOf(b).roc - macroOf(a).roc; });
    var rank = ranked.indexOf(curModel) + 1;
    var best = rs[0], worst = rs[rs.length - 1];
    var rf = macroOf("Random Forest").roc, dRf = mac.roc - rf;

    $("#profile").innerHTML =
      '<div class="prank">#' + rank + ' of 5 <span>by macro ROC-AUC</span></div>' +
      '<div class="mname">' + curModel + '</div><div class="mtag">' + MTAG[curModel] + "</div>" +
      '<div class="mbig"><div><div class="k">Macro ROC-AUC</div><div class="v" style="color:var(--teal)">' +
      mac.roc.toFixed(3) + '</div></div><div><div class="k">Macro PR-AUC</div><div class="v" style="color:var(--violet)">' +
      mac.pr.toFixed(3) + "</div></div></div>" +
      '<p class="mdesc">' + MDESC[curModel] + "</p>" +
      '<div class="mfacts">' +
        '<div><span>Leads on</span><b>' + ledBy(curModel) + " of 12 endpoints</b></div>" +
        '<div><span>Strongest assay</span><b>' + best.endpoint + " · " + best.roc.toFixed(3) + "</b></div>" +
        '<div><span>Hardest assay</span><b>' + worst.endpoint + " · " + worst.roc.toFixed(3) + "</b></div>" +
        '<div><span>vs Random Forest</span><b class="' + (dRf >= 0 ? "up" : "dn") + '">' +
          (dRf >= 0 ? "+" : "−") + Math.abs(dRf).toFixed(3) + "</b></div>" +
      "</div>";

    $("#bars").innerHTML =
      '<div class="bhd"><span>Endpoint</span><span>ROC-AUC <i>0.50 → 1.00</i></span><span></span>' +
      '<span>PR-AUC <i>0 → 1.00</i></span><span></span><span>Actives</span></div>' +
      rs.map(function (r) {
        var e = EP.filter(function (x) { return x.id === r.endpoint; })[0];
        return '<div class="brow"><div class="n"><b>' + r.endpoint + "</b><span>" + (e ? e.name : "") + "</span></div>" +
          '<div class="track"><div class="g roc" data-w="' + scR(r.roc).toFixed(1) + '"></div>' +
            '<div class="ci" style="left:' + scR(r.rocLo).toFixed(1) + "%;width:" +
            Math.max(scR(r.rocHi) - scR(r.rocLo), .6).toFixed(1) + '%"></div></div>' +
          '<div class="v">' + r.roc.toFixed(3) + "</div>" +
          '<div class="track"><div class="g pr" data-w="' + scP(r.pr).toFixed(1) + '"></div>' +
            '<div class="ci" style="left:' + scP(r.prLo).toFixed(1) + "%;width:" +
            Math.max(scP(r.prHi) - scP(r.prLo), .6).toFixed(1) + '%"></div></div>' +
          '<div class="v pr">' + r.pr.toFixed(3) + "</div>" +
          '<div class="cnt mono">' + r.active + "<span>/" + r.tested + "</span></div></div>";
      }).join("") +
      '<div class="bnote">Whiskers are 95% confidence intervals from 1,000 stratified bootstrap resamples. ' +
      'PR-AUC starts from each assay’s base rate, not from 0.5 — for NR-PPAR-γ that floor is about 0.03, ' +
      "so a PR-AUC of 0.3 there is roughly ten times better than chance.</div>";
    grow($("#bars"));
  }

  function compareTable() {
    var ranked = MODELS.slice().sort(function (a, b) { return macroOf(b).roc - macroOf(a).roc; });
    var topRoc = macroOf(ranked[0]).roc, topPr = Math.max.apply(null, MODELS.map(function (m) { return macroOf(m).pr; }));
    $("#ctable").innerHTML =
      '<div class="chd"><span>Model</span><span>Macro ROC-AUC</span><span>Macro PR-AUC</span>' +
      '<span>Endpoints led</span><span>Configuration</span></div>' +
      ranked.map(function (m) {
        var mc = macroOf(m);
        return '<button class="crow' + (m === curModel ? " on" : "") + '" data-m="' + m + '">' +
          '<span class="cm">' + m + "</span>" +
          '<span class="cv"><b class="mono' + (mc.roc === topRoc ? " hi" : "") + '">' + mc.roc.toFixed(3) + "</b>" +
            '<i style="width:' + Math.max(4, Math.min((mc.roc - 0.75) / 0.1 * 100, 100)).toFixed(1) + '%"></i></span>' +
          '<span class="cv"><b class="mono' + (mc.pr === topPr ? " hi" : "") + '">' + mc.pr.toFixed(3) + "</b>" +
            '<i class="pr" style="width:' + Math.max(4, Math.min(mc.pr / 0.55 * 100, 100)).toFixed(1) + '%"></i></span>' +
          '<span class="cl mono">' + ledBy(m) + " / 12</span>" +
          '<span class="cc">' + MTAG[m] + "</span></button>";
      }).join("");
    [].forEach.call($("#ctable").querySelectorAll(".crow"), function (b) {
      b.addEventListener("click", function () { setModel(b.dataset.m); });
    });
  }

  function setModel(m) {
    curModel = m;
    [].forEach.call($("#mpick").children, function (x) { x.classList.toggle("on", x.dataset.m === m); });
    drawBench(); compareTable();
  }
  function benchInit() {
    var p = $("#mpick");
    MODELS.forEach(function (m) {
      var b = el("button", m === curModel ? "on" : "", m);
      b.dataset.m = m;
      b.addEventListener("click", function () { setModel(m); });
      p.appendChild(b);
    });
    drawBench(); compareTable();
  }

  /* ---------- endpoint reference table ---------- */
  function endpointTable() {
    var bestOf = {};
    EP.forEach(function (e) {
      var b = null;
      MODELS.forEach(function (m) {
        var r = BM.filter(function (x) { return x.model === m && x.endpoint === e.id; })[0];
        if (r && (!b || r.roc > b.roc)) b = { m: m, roc: r.roc };
      });
      bestOf[e.id] = b;
    });
    $("#eptable").innerHTML =
      '<div class="ethd"><span>Endpoint</span><span>What the assay measures</span><span>Tested</span>' +
      '<span>Active rate</span><span>Missing</span><span>Best ROC-AUC</span></div>' +
      EP.map(function (e) {
        var b = bestOf[e.id];
        return '<div class="etrow"><div class="etn"><b>' + e.id + '</b><span class="fam ' + e.family.toLowerCase() + '">' +
          e.family + "</span><i>" + e.name + "</i></div>" +
          '<div class="etb">' + BIO[e.id] + "</div>" +
          '<div class="mono etnum">' + e.tested.toLocaleString("en-US") + '<span>' + e.active + " active</span></div>" +
          '<div class="etbar"><span class="trk"><i data-w="' +
          Math.min(e.activeRate / 17 * 100, 100).toFixed(1) + '"></i></span><b class="mono">' +
          e.activeRate.toFixed(2) + "%</b></div>" +
          '<div class="mono etnum">' + e.missingRate.toFixed(1) + "%</div>" +
          '<div class="etbest mono"><b>' + b.roc.toFixed(3) + "</b><span>" + b.m + "</span></div></div>";
      }).join("");
  }

  /* ---------- nav ---------- */
  function nav() {
    var n = $("#nav"), links = [].slice.call(n.querySelectorAll("nav a"));
    var f = function () { n.classList.toggle("solid", scrollY > 40); };
    f(); addEventListener("scroll", f, { passive: true });
    var secs = links.map(function (a) { return document.querySelector(a.getAttribute("href")); });
    var so = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        var i = secs.indexOf(e.target);
        links.forEach(function (a, k) { a.classList.toggle("cur", k === i); });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    secs.forEach(function (s) { if (s) so.observe(s); });
  }

  /* ---------- boot ---------- */
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && $("#modal").classList.contains("open")) closeSheet();
  });
  $("#modal").addEventListener("click", function (e) {
    if (e.target.id === "modal" || e.target.classList.contains("scrim")) closeSheet();
  });
  $("#surprise").addEventListener("click", function () { openSheet(CO[Math.floor(Math.random() * CO.length)]); });

  nav(); hero(); matrix(); rows(); benchInit(); endpointTable(); reveals(); progress();
})();
