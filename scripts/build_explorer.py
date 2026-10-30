"""Build docs/explorer.html: an interactive, self-contained page (no CDN, no dependencies) to explore
every member's position. Hover a dot for the member, caucus and numbers; pick a caucus to highlight it;
search by name.

- 119th House map (logistic PCA, 2 dimensions) with caucus labels and the bipartisan yes-rate.
- 119th House Bayesian positions with 90% ranges (scripts/fit_irt.py 119H --chains 2 ...).

Needs scripts/run_analysis.py, scripts/fit_irt.py 119H and scripts/dimension2.py first.

    python scripts/build_explorer.py      # -> docs/explorer.html
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "explorer.html"


def data():
    mem = pd.read_csv(RES / "members_119H.csv")
    by = pd.read_csv(RES / "dim2_members_119H.csv")
    mem = mem.merge(by, on="icpsr", how="left")
    house = [{"n": r["name"], "p": r.party, "s": f"{r.state}-{int(r.district)}" if r.district > 0 else r.state,
              "c": "" if r.caucus == "none" else r.caucus, "x": round(r.dim1, 3), "y": round(r.dim2, 3),
              "b": None if pd.isna(r.bipartisan_yes) else round(r.bipartisan_yes, 3),
              "g": int(r.gmm), "gm": round(r.gmm_max_membership, 2)} for _, r in mem.iterrows()]
    irt = pd.read_csv(RES / "irt_119H_positions.csv").sort_values("position")
    ranked = [{"n": r.bioname, "p": r.party, "s": r.state_abbrev, "x": round(float(r.position), 3),
               "lo": round(float(r.lo), 3), "hi": round(float(r.hi), 3), "v": int(r.votes_cast)}
              for _, r in irt.iterrows()]
    d2 = json.loads((RES / "dim2_119H.json").read_text())
    return {"house": house, "ranked": ranked, "d2": d2}


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Congress from votes alone: explorer</title>
<link rel="icon" href="data:,">
<style>
:root{--surface:#fcfcfb;--page:#f4f3ef;--ink:#0b0b0b;--ink2:#52514e;--muted:#6b6a63;--grid:#e6e5df;
  --dem:#2a78d6;--rep:#e34948;--ind:#6b6a63;--ring:#0b0b0b;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#111110;--ink:#f0efec;--ink2:#c3c2b7;
  --muted:#9a998f;--grid:#2e2e2c;--dem:#3987e5;--rep:#e66767;--ind:#9a998f;--ring:#f0efec;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:0 0 2px}
p{margin:6px 0;color:var(--ink2)}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:18px 0}
label{font-size:13px;color:var(--ink2);display:flex;flex-direction:column;gap:3px}
select,input{font:inherit;padding:6px 8px;border:1px solid var(--grid);border-radius:6px;background:var(--surface);color:var(--ink);min-width:200px}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:16px;margin:16px 0}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:6px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block}
.chart{position:relative}
svg{display:block;width:100%;height:auto;touch-action:none}
.axis text{fill:var(--muted);font-size:11px}.axis line,.gridline{stroke:var(--grid)}
.axlab{fill:var(--muted);font-size:12px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--grid);border-radius:8px;
  padding:8px 10px;font-size:13px;box-shadow:0 2px 10px rgba(0,0,0,.12);max-width:280px;display:none;z-index:2}
.tip strong{display:block;font-size:14px;color:var(--ink)}.tip div{color:var(--ink2)}
.note{font-size:13px;color:var(--muted)}
details{margin-top:16px}summary{cursor:pointer;color:var(--ink2)}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--grid)}th{color:var(--ink2);cursor:pointer}
td.num{text-align:right;font-variant-numeric:tabular-nums}
</style>
</head>
<body>
<main>
<h1>Congress from votes alone</h1>
<p>Every member of the House placed on a map using nothing but their recorded votes. Members who vote
alike sit close together. Hover over a dot (or tap it) for the member; pick a caucus to highlight it.</p>

<div class="controls">
  <label>Highlight a caucus
    <select id="caucus">
      <option value="">None</option>
      <option>Freedom Caucus</option><option>Progressive Caucus</option>
      <option>Blue Dog Coalition</option><option>Problem Solvers Caucus</option>
    </select></label>
  <label>Find a member
    <input id="search" type="search" placeholder="Surname, e.g. Pelosi" autocomplete="off"></label>
</div>

<section class="card">
  <h2>119th House (2025&ndash;): two dimensions</h2>
  <p class="note">Left&ndash;right is dimension 1. Dimension 2 tracks how often a member votes yes on bills that
  both parties' majorities support: higher means more often.</p>
  <div class="legend" id="leg1"></div>
  <div class="chart" id="c1"><svg id="s1" role="img" aria-label="Scatter of 119th House members on two dimensions, coloured by party"></svg><div class="tip" id="t1"></div></div>
  <p class="note" id="stat1"></p>
</section>

<section class="card">
  <h2>119th House: one dimension, with uncertainty</h2>
  <p class="note">A Bayesian model gives each member a position and a 90% range. Wide ranges belong to members
  with few votes (late arrivals, early departures) and to members at their party's far edge.</p>
  <div class="legend" id="leg2"></div>
  <div class="chart" id="c2"><svg id="s2" role="img" aria-label="Every 119th House member ordered by position with a 90% range"></svg><div class="tip" id="t2"></div></div>
</section>

<details>
  <summary>Table view: 119th House members</summary>
  <table id="tbl"><thead><tr><th data-k="n">Member</th><th data-k="p">Party</th><th data-k="s">Seat</th>
  <th data-k="c">Caucus</th><th data-k="x">Dim 1</th><th data-k="y">Dim 2</th><th data-k="b">Yes on bipartisan bills</th></tr></thead><tbody></tbody></table>
</details>

<p class="note" style="margin-top:24px">Data: Voteview (Lewis, Poole, Rosenthal, Boche, Rudkin &amp; Sonnet), voteview.com.
Caucus lists: Wikipedia, retrieved 2026-10-17 (the Freedom Caucus doesn't publish its roster; its list is publicly
identified members). Public officials' public votes. Method and code:
<a href="https://github.com/godot107/ideology-clusters">github.com/godot107/ideology-clusters</a>.</p>
</main>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const PARTY = {Democrat:'--dem', Republican:'--rep', Independent:'--ind'};
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const pct = v => v == null ? 'n/a' : Math.round(v * 100) + '%';
let caucus = '', query = '';

function legend(id, parties) {
  const L = document.getElementById(id); L.textContent = '';
  for (const p of parties) { const s = document.createElement('span'); const w = document.createElement('i');
    w.className = 'sw'; w.style.background = css(PARTY[p]); s.append(w, document.createTextNode(p)); L.appendChild(s); }
  if (caucus && id === 'leg1') { const s = document.createElement('span'); const w = document.createElement('i');
    w.className = 'sw'; w.style.background = 'transparent'; w.style.border = '2px solid ' + css('--ring');
    s.append(w, document.createTextNode(caucus + ' (ringed)')); L.appendChild(s); }
}
function matches(d) { return query && d.n.toLowerCase().includes(query); }
function inCaucus(d) { return caucus && d.c.includes(caucus); }
function line(label, value, parent) { const r = document.createElement('div'); r.textContent = label + ': ' + value; parent.appendChild(r); }

function ticks(lo, hi, n) { const step = (hi - lo) / n; const out = []; for (let i = 0; i <= n; i++) out.push(lo + i * step); return out; }

// ---- chart 1: 2-D scatter
let W1 = 900, H1 = 520; const M1 = {l: 44, r: 14, t: 10, b: 38};
function drawMap() {
  const svg = document.getElementById('s1'); svg.textContent = ''; svg.setAttribute('viewBox', `0 0 ${W1} ${H1}`);
  const xs = D.house.map(d => d.x), ys = D.house.map(d => d.y);
  const x0 = Math.floor(Math.min(...xs)), x1 = Math.ceil(Math.max(...xs)), y0 = Math.floor(Math.min(...ys)), y1 = Math.ceil(Math.max(...ys));
  const X = v => M1.l + (v - x0) / (x1 - x0) * (W1 - M1.l - M1.r), Y = v => H1 - M1.b - (v - y0) / (y1 - y0) * (H1 - M1.t - M1.b);
  const ax = el('g', {class: 'axis'}, svg);
  for (let v = x0; v <= x1; v++) { el('line', {class: 'gridline', x1: X(v), x2: X(v), y1: M1.t, y2: H1 - M1.b}, ax);
    el('text', {x: X(v), y: H1 - M1.b + 16, 'text-anchor': 'middle'}, ax).textContent = v; }
  for (let v = y0; v <= y1; v++) { el('line', {class: 'gridline', x1: M1.l, x2: W1 - M1.r, y1: Y(v), y2: Y(v)}, ax);
    el('text', {x: M1.l - 8, y: Y(v) + 4, 'text-anchor': 'end'}, ax).textContent = v; }
  el('text', {class: 'axlab', x: (W1 + M1.l) / 2, y: H1 - 4, 'text-anchor': 'middle'}, svg).textContent = 'dimension 1: left → right';
  el('text', {class: 'axlab', x: 12, y: (H1 - M1.b) / 2, transform: `rotate(-90 12 ${(H1 - M1.b) / 2})`, 'text-anchor': 'middle'}, svg).textContent = 'dimension 2: yes on bipartisan bills →';
  const dim = caucus || query;
  const order = [...D.house].sort((a, b) => (inCaucus(a) || matches(a)) - (inCaucus(b) || matches(b)));
  for (const d of order) {
    const on = inCaucus(d) || matches(d);
    el('circle', {cx: X(d.x), cy: Y(d.y), r: on ? 6 : 4.5, fill: css(PARTY[d.p] || '--ind'),
      'fill-opacity': dim && !on ? 0.18 : 0.85, stroke: on ? css('--ring') : css('--surface'), 'stroke-width': on ? 2 : 1}, svg);
  }
  const pts = D.house.map(d => [X(d.x), Y(d.y), d]);
  hover(svg, 't1', 'c1', (px, py) => { let best = null, bd = 24 * 24;
    for (const [x, y, d] of pts) { const dd = (x - px) ** 2 + (y - py) ** 2; if (dd < bd) { bd = dd; best = [x, y, d]; } } return best; },
    (tip, d) => { const s = document.createElement('strong'); s.textContent = d.n; tip.appendChild(s);
      line('Party', d.p + ' · ' + d.s, tip); line('Caucus', d.c || 'none listed', tip);
      line('Yes on bipartisan bills', pct(d.b), tip); line('Position', `dim 1 ${d.x.toFixed(2)}, dim 2 ${d.y.toFixed(2)}`, tip);
      line('Mixture group', `${d.g} (${pct(d.gm)} membership)`, tip); });
  const st = document.getElementById('stat1');
  if (caucus) { const g = D.house.filter(inCaucus), b = g.map(d => d.b).filter(v => v != null).sort((a, b) => a - b);
    st.textContent = `${caucus}: ${g.length} members found; median yes on bipartisan bills ${pct(b[Math.floor(b.length / 2)])}.`;
  } else st.textContent = `${D.d2.bipartisan_votes_both_majorities_yes} bills where both parties' majorities voted yes. Freedom Caucus median member: yes on ${pct(D.d2.freedom_caucus_median_bipartisan_yes)}; other Republicans ${pct(D.d2.freedom_caucus_rest_of_party_median_bipartisan_yes)}.`;
}

// ---- chart 2: ranked positions with 90% ranges
let W2 = 900, H2 = 300; const M2 = {l: 44, r: 14, t: 10, b: 30};
function drawRanked() {
  const svg = document.getElementById('s2'); svg.textContent = ''; svg.setAttribute('viewBox', `0 0 ${W2} ${H2}`);
  const R = D.ranked, lo = Math.floor(Math.min(...R.map(d => d.lo))), hi = Math.ceil(Math.max(...R.map(d => d.hi)));
  const X = i => M2.l + i / (R.length - 1) * (W2 - M2.l - M2.r), Y = v => H2 - M2.b - (v - lo) / (hi - lo) * (H2 - M2.t - M2.b);
  const ax = el('g', {class: 'axis'}, svg);
  for (let v = lo; v <= hi; v++) { el('line', {class: 'gridline', x1: M2.l, x2: W2 - M2.r, y1: Y(v), y2: Y(v)}, ax);
    el('text', {x: M2.l - 8, y: Y(v) + 4, 'text-anchor': 'end'}, ax).textContent = v; }
  el('text', {class: 'axlab', x: (W2 + M2.l) / 2, y: H2 - 8, 'text-anchor': 'middle'}, svg).textContent = 'members, ordered by position (left → right)';
  R.forEach((d, i) => { const on = matches(d), c = css(PARTY[d.p] || '--ind'), op = query && !on ? 0.18 : 0.9;
    el('line', {x1: X(i), x2: X(i), y1: Y(d.lo), y2: Y(d.hi), stroke: c, 'stroke-opacity': op * 0.6, 'stroke-width': on ? 2.5 : 1}, svg);
    el('circle', {cx: X(i), cy: Y(d.x), r: on ? 5 : 2.2, fill: c, 'fill-opacity': op, stroke: on ? css('--ring') : 'none', 'stroke-width': 2}, svg); });
  hover(svg, 't2', 'c2', px => { const i = Math.max(0, Math.min(R.length - 1, Math.round((px - M2.l) / (W2 - M2.l - M2.r) * (R.length - 1))));
    return [X(i), Y(R[i].x), R[i]]; },
    (tip, d) => { const s = document.createElement('strong'); s.textContent = d.n; tip.appendChild(s);
      line('Party', d.p + ' · ' + d.s, tip); line('Position', `${d.x.toFixed(2)} (90% range ${d.lo.toFixed(2)} to ${d.hi.toFixed(2)})`, tip);
      line('Votes cast', d.v, tip); });
}

function hover(svg, tipId, boxId, find, fill) {
  const tip = document.getElementById(tipId), box = document.getElementById(boxId);
  let mark = null;
  const move = ev => { const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    const px = (ev.clientX - r.left) * vb.width / r.width, py = (ev.clientY - r.top) * vb.height / r.height;
    const hit = find(px, py); if (mark) { mark.remove(); mark = null; }
    if (!hit) { tip.style.display = 'none'; return; }
    const [x, y, d] = hit; mark = el('circle', {cx: x, cy: y, r: 8, fill: 'none', stroke: css('--ring'), 'stroke-width': 2}, svg);
    tip.textContent = ''; fill(tip, d); tip.style.display = 'block';
    const sx = x * r.width / vb.width, sy = y * r.height / vb.height, bw = box.clientWidth;
    tip.style.left = (sx > bw / 2 ? Math.max(4, sx - tip.offsetWidth - 14) : Math.min(sx + 14, bw - tip.offsetWidth - 4)) + 'px'; tip.style.top = Math.max(0, sy - tip.offsetHeight - 10) + 'px'; };
  svg.onpointermove = move; svg.onpointerdown = move;
  svg.onpointerleave = () => { tip.style.display = 'none'; if (mark) { mark.remove(); mark = null; } };
}

// ---- table view
let sortKey = 'x', sortDir = 1;
function drawTable() {
  const tb = document.querySelector('#tbl tbody'); tb.textContent = '';
  const rows = [...D.house].sort((a, b) => { const u = a[sortKey], v = b[sortKey];
    return (u == null) - (v == null) || (u < v ? -1 : u > v ? 1 : 0) * sortDir; });
  for (const d of rows) { if (query && !matches(d)) continue; if (caucus && !inCaucus(d)) continue;
    const tr = document.createElement('tr');
    for (const [v, num] of [[d.n], [d.p], [d.s], [d.c || ''], [d.x.toFixed(2), 1], [d.y.toFixed(2), 1], [pct(d.b), 1]]) {
      const td = document.createElement('td'); td.textContent = v; if (num) td.className = 'num'; tr.appendChild(td); }
    tb.appendChild(tr); }
}
document.querySelectorAll('#tbl th').forEach(th => th.onclick = () => {
  const k = th.dataset.k; sortDir = k === sortKey ? -sortDir : 1; sortKey = k; drawTable(); });

function size() { const w = Math.max(320, Math.min(900, document.getElementById('c1').clientWidth));
  W1 = W2 = w; H1 = Math.max(340, Math.round(w * 0.58)); H2 = Math.max(240, Math.round(w * 0.33)); }
function all() { size(); legend('leg1', ['Democrat', 'Republican', 'Independent']); legend('leg2', ['Democrat', 'Republican', 'Independent']);
  drawMap(); drawRanked(); drawTable(); }
document.getElementById('caucus').onchange = e => { caucus = e.target.value; all(); };
document.getElementById('search').oninput = e => { query = e.target.value.trim().toLowerCase(); all(); };
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', all);
let rz; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(all, 150); });
all();
</script>
</body>
</html>
"""


def main():
    payload = json.dumps(data(), separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text(PAGE.replace("__DATA__", payload))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
