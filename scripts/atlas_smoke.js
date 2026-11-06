// Smoke test for the ClusterPitch atlas page (no browser needed).
//
//   node scripts/atlas_smoke.js                  the built page, analysis/pitch_atlas.html
//   node scripts/atlas_smoke.js --template       the current scripts/atlas_template.html with the built page's data
//   node scripts/atlas_smoke.js <page.html>      any self-contained build
//
// CARC has no node on PATH: use code-server's, /apps/generic/code-server/4.117.0/lib/node (add --max-old-space-size=4096).
// It runs the page's main inline script in a vm context with a permissive fake DOM, a recording fake Plotly and a fake
// three.js (real vectors and buffers, no drawing), then drives it: layout, the cluster key and its subcluster tree, the
// 2D and 3D morphs, the cluster tags on the map and player cards, the Pitch Simulation Lab's MLB average column and the
// colour menu. CSS layout and real WebGL drawing are not checked. Exit code 0 = every check passed.
const fs = require("fs"), path = require("path"), vm = require("vm");
const ROOT = path.resolve(__dirname, ".."), BUILT = path.join(ROOT, "analysis", "pitch_atlas.html");
function loadPage(arg) {
  if (arg !== "--template") return fs.readFileSync(arg || BUILT, "utf8");
  // the template's data placeholder filled with the built page's data blob, so template edits test without a build
  const tpl = fs.readFileSync(path.join(ROOT, "scripts", "atlas_template.html"), "utf8"), built = fs.readFileSync(BUILT, "utf8");
  const i = built.indexOf("const D = ") + "const D = ".length, k = built.indexOf(" || window.__ATLAS_DATA__", i);
  if (i < 10 || k < 0 || tpl.split("/*__ATLAS_DATA__*/null").length !== 2) throw new Error("cannot splice the built data into the template");
  return tpl.replace("/*__ATLAS_DATA__*/null", () => built.slice(i, k));
}
const html = loadPage(process.argv[2]);
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const src = scripts.reduce((a, b) => (b.length > a.length ? b : a), "");
const markup = html.replace(/<script>[\s\S]*?<\/script>/g, "");

let fails = 0, passes = 0;
const ok = (c, msg) => { if (c) passes++; else { fails++; console.log("FAIL:", msg); } };
function report() { console.log(`${passes} passed, ${fails} failed`); process.exit(fails ? 1 : 0); }

// ---- static: every $("#id") / getElementById("id") literal has an element (markup or a JS template)
const BUILT_BY_ARG = new Set(["simVs"]);   // made by vsSel("simVs", ...) and null-checked before use
const ids = new Set([...src.matchAll(/\$\("#([A-Za-z][\w-]*)"\)|getElementById\("([A-Za-z][\w-]*)"\)/g)].map(m => m[1] || m[2]));
const missing = [...ids].filter(id => !BUILT_BY_ARG.has(id) && !markup.includes(`id="${id}"`) && !src.includes(`id="${id}"`));
ok(!missing.length, "ids used in JS but absent: " + missing.join(", "));

// ---- fake DOM
const perm = new Proxy(function () {}, {
  get: (t, k) => k === Symbol.toPrimitive ? () => 0 : k === "then" ? undefined : k === "length" ? 0 : k === Symbol.iterator ? function* () {} : perm,
  apply: () => perm, set: () => true,
});
const reg = new Map();
function mkEl(sel) {
  const cls = new Set();
  const base = {
    _sel: sel, _h: {}, innerHTML: "", textContent: "", hidden: false, value: "", disabled: false, checked: false, title: "",
    style: {}, dataset: {}, _attrs: {}, data: undefined, layout: undefined,
    classList: { add: (...c) => c.forEach(x => cls.add(x)), remove: (...c) => c.forEach(x => cls.delete(x)),
      toggle: (c, f) => { const on = f === undefined ? !cls.has(c) : !!f; on ? cls.add(c) : cls.delete(c); return on; }, contains: c => cls.has(c) },
    setAttribute(k, v) { this._attrs[k] = String(v); }, getAttribute(k) { return this._attrs[k] ?? null; },
    removeAttribute(k) { delete this._attrs[k]; }, hasAttribute(k) { return k in this._attrs; }, toggleAttribute() {},
    addEventListener(t, f) { (this._h[t] ||= []).push(f); }, removeEventListener() {},
    on(t, f) { (this._h["plotly:" + t] ||= []).push(f); }, removeAllListeners() {},
    querySelector(s) { return getEl(sel + " " + s); }, querySelectorAll() { return []; },
    closest() { return null; }, contains() { return false; }, matches() { return false; },
    focus() {}, blur() {}, click() {}, scrollIntoView() {}, scrollTo() {}, select() {},
    getBoundingClientRect: () => ({ left: 16, top: 300, right: 216, bottom: 336, width: 200, height: 36, x: 16, y: 300 }),
    appendChild: c => c, insertAdjacentHTML() {}, remove() {}, append() {}, prepend() {}, replaceChildren() {}, insertBefore: c => c,
    getContext: (k) => (k === "webgl" || k === "experimental-webgl" ? {} : perm),
    offsetWidth: 900, offsetHeight: 700, clientWidth: 900, clientHeight: 700, scrollTop: 0, scrollLeft: 0, scrollHeight: 700, scrollWidth: 900,
    selectedOptions: [], options: [], children: [], childNodes: [], parentElement: null, firstElementChild: null, nextElementSibling: null,
  };
  return new Proxy(base, { get: (t, k) => (k in t ? t[k] : typeof k === "symbol" ? undefined : perm), set: (t, k, v) => { t[k] = v; return true; } });
}
function getEl(sel) { if (!reg.has(sel)) reg.set(sel, mkEl(sel)); return reg.get(sel); }
const document = new Proxy({
  querySelector: s => getEl(s), querySelectorAll: () => [], getElementById: id => getEl("#" + id),
  createElement: t => mkEl("<" + t + ">"), createElementNS: (n, t) => mkEl("<" + t + ">"), createTextNode: () => mkEl("#text"),
  addEventListener() {}, removeEventListener() {}, body: mkEl("body"), documentElement: mkEl("html"), head: mkEl("head"),
  activeElement: null, fonts: { ready: Promise.resolve() }, hidden: false, visibilityState: "visible", title: "",
}, { get: (t, k) => (k in t ? t[k] : perm) });

// ---- fake Plotly: records calls, keeps data and the axis ranges
const calls = [];
const ext = (data, ax) => { let lo = Infinity, hi = -Infinity; data.forEach(t => { const a = t[ax]; if (Array.isArray(a) && t.visible !== false) for (const v of a) if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }); return isFinite(lo) ? [lo, hi] : [0, 1]; };
const setPath = (o, path, v) => { const ks = path.split("."); let x = o; ks.slice(0, -1).forEach(k => { x = (x[k] ||= {}); }); x[ks[ks.length - 1]] = v; };
const Plotly = {
  react(el, data, layout) {
    calls.push({ f: "react", data, layout }); el.data = data; el.layout = layout;
    const xr = layout.xaxis && layout.xaxis.range ? layout.xaxis.range.slice() : ext(data, "x");
    const yr = layout.yaxis && layout.yaxis.range ? layout.yaxis.range.slice() : ext(data, "y");
    el._fullLayout = { xaxis: { range: xr }, yaxis: { range: yr } };
    return Promise.resolve();
  },
  update(el, d, lay, idx) {
    calls.push({ f: "update", d, lay, idx });
    Object.entries(d).forEach(([k, vals]) => idx.forEach((ti, i) => setPath(el.data[ti], k, vals[i])));
    if (lay && lay["xaxis.range"]) el._fullLayout.xaxis.range = lay["xaxis.range"].slice();
    if (lay && lay["yaxis.range"]) el._fullLayout.yaxis.range = lay["yaxis.range"].slice();
    return Promise.resolve();
  },
  restyle(el, d, idx) { calls.push({ f: "restyle", d, idx }); (idx || []).forEach(ti => Object.entries(d).forEach(([k, v]) => setPath(el.data[ti], k, v))); return Promise.resolve(); },
  relayout(el, d) { calls.push({ f: "relayout", d }); return Promise.resolve(); },
  addTraces(el, traces, at) { calls.push({ f: "addTraces", traces, at }); traces.forEach((t, i) => el.data.splice(at[i], 0, t)); return Promise.resolve(); },
  purge(el) { calls.push({ f: "purge" }); el.data = undefined; el._fullLayout = undefined; },
  Plots: { resize: () => Promise.resolve() }, newPlot() { return Plotly.react(...arguments); },
};

// ---- fake three.js (the r128 surface the page uses)
class V3 { constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; } set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  clone() { return new V3(this.x, this.y, this.z); } copy(v) { return this.set(v.x, v.y, v.z); } negate() { return this.set(-this.x, -this.y, -this.z); }
  lerpVectors(a, b, s) { return this.set(a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s, a.z + (b.z - a.z) * s); } project() { return new V3(0, 0, 0); } }
class Obj3 { constructor() { this.position = new V3(); this.children = []; this.visible = true; this.userData = {}; this.renderOrder = 0; this.frustumCulled = true;
  this.quaternion = { setFromRotationMatrix() {} }; this.scale = new V3(1, 1, 1); } add(o) { this.children.push(o); } remove(o) { const i = this.children.indexOf(o); if (i >= 0) this.children.splice(i, 1); } }
class BufAttr { constructor(a, n) { this.array = a; this.itemSize = n; this.needsUpdate = false; } }
class BufGeo { constructor() { this.attributes = {}; } setAttribute(k, a) { this.attributes[k] = a; return this; } dispose() { this.disposed = true; } }
class Mat { constructor(o = {}) { Object.assign(this, o); } clone() { return new this.constructor({ ...this }); } dispose() { this.disposed = true; } }
class Pts extends Obj3 { constructor(g, m) { super(); this.geometry = g; this.material = m; this.isPoints = true; } }
class Ln extends Obj3 { constructor(g, m) { super(); this.geometry = g; this.material = m; this.isLine = true; } }
class Msh extends Obj3 { constructor(g, m) { super(); this.geometry = g; this.material = m; this.isMesh = true; } }
const THREE = {
  WebGLRenderer: class { constructor() { this.domElement = mkEl("<canvas3d>"); } setPixelRatio() {} getPixelRatio() { return 1; } setSize() {} setClearColor() {} render() {} dispose() {} },
  Scene: class extends Obj3 {}, PerspectiveCamera: class extends Obj3 { constructor() { super(); this.aspect = 1; } updateProjectionMatrix() {} },
  AmbientLight: class extends Obj3 {}, DirectionalLight: class extends Obj3 { constructor() { super(); this.target = new Obj3(); } },
  OrbitControls: class { constructor(cam) { this.object = cam; this.target = new V3(); } update() {} addEventListener() {} dispose() {} },
  Raycaster: class { constructor() { this.params = { Points: {} }; } setFromCamera() {} intersectObjects() { return []; } },
  Vector2: class { constructor(x, y) { this.x = x; this.y = y; } }, Vector3: V3, BufferAttribute: BufAttr, BufferGeometry: BufGeo,
  Points: Pts, Line: Ln, LineSegments: Ln, Mesh: Msh, SphereGeometry: class extends BufGeo {}, CanvasTexture: class {},
  PointsMaterial: class extends Mat {}, LineBasicMaterial: class extends Mat {}, MeshLambertMaterial: class extends Mat {}, MeshPhongMaterial: class extends Mat {},
  Color: class { constructor(r, g, b) { this.r = r; this.g = g; this.b = b; } clone() { return new THREE.Color(this.r, this.g, this.b); } multiplyScalar(k) { this.r *= k; this.g *= k; this.b *= k; return this; } },
  Matrix4: class { makeBasis() { return this; } determinant() { return 1; } }, BackSide: 1, FrontSide: 0,
};

let NOW = 1000; const raf = [];
const ctx = {
  document, Plotly, THREE, console, Math, JSON, Date, Promise, Set, Map, WeakMap, Array, Object, Number, String, Boolean, RegExp, Error, TypeError,
  Int16Array, Int32Array, Uint8Array, Uint16Array, Float32Array, Float64Array, ArrayBuffer, DataView, TextDecoder, TextEncoder, URL, URLSearchParams,
  setTimeout, clearTimeout, setInterval, clearInterval, queueMicrotask, structuredClone, AbortController, isFinite, isNaN, parseFloat, parseInt,
  encodeURIComponent, decodeURIComponent, Intl, Symbol, Proxy, Reflect, BigInt,
  atob: s => Buffer.from(s, "base64").toString("binary"), btoa: s => Buffer.from(s, "binary").toString("base64"),
  performance: { now: () => NOW }, requestAnimationFrame: f => { raf.push(f); return raf.length; }, cancelAnimationFrame() {},
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {} }),
  getComputedStyle: () => ({ getPropertyValue: () => " #2a3038" }),
  ResizeObserver: class { observe() {} unobserve() {} disconnect() {} }, IntersectionObserver: class { observe() {} unobserve() {} disconnect() {} },
  MutationObserver: class { observe() {} disconnect() {} },
  fetch: () => Promise.reject(new Error("no network in the smoke test")),
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} }, sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  location: { hash: "", href: "https://example/", search: "", pathname: "/" }, history: { replaceState() {}, pushState() {}, back() {} },
  navigator: { userAgent: "node", language: "en-US", clipboard: { writeText: () => Promise.resolve() } },
  innerWidth: 1600, innerHeight: 900, scrollY: 0, devicePixelRatio: 1, scrollTo() {}, addEventListener() {}, removeEventListener() {},
  CSS: { escape: s => String(s) }, Blob: class {}, Image: class {},
};
ctx.window = ctx; ctx.self = ctx; ctx.globalThis = ctx;
ctx.console = { ...console, info() {} };   // the page logs its failed free-agent refresh with console.info
vm.createContext(ctx);
const run = code => vm.runInContext(code, ctx);
const tick = () => new Promise(r => setTimeout(r, 0));
const pump = async (dt) => { NOW += dt; const fs2 = raf.splice(0); fs2.forEach(f => f(NOW)); await tick(); };
const J = JSON.stringify;

(async () => {
  try { run(src); } catch (e) { ok(false, "boot threw: " + e.stack.split("\n").slice(0, 4).join(" | ")); report(); return; }
  await tick(); await tick();
  ok(calls.some(c => c.f === "react"), "map drew at boot");
  const TT = run("T");
  ok(run("plotReady") === true, "plotReady after first draw");

  // ---- layout
  const hEnd = markup.indexOf("</header>");
  ok(markup.indexOf('id="clusterSeg"') < hEnd && markup.indexOf('id="dimSeg"') < hEnd && markup.indexOf('id="lqual"') < hEnd, "view switches, meter and cluster switch in the header");
  ok(markup.indexOf('class="measseg"') > markup.indexOf('id="typeCol"') && markup.indexOf('class="measseg"') < markup.indexOf('id="lvlSeg"'), "filter at the top of the left column, above MLB | Triple-A");
  ok(!markup.includes('id="mapTool"') && !markup.includes('id="ptype"') && !markup.includes("Find pitchers"), "toolbar row and finder menus removed");
  ok(/<title>ClusterPitch<\/title>/.test(html) && /<h1>ClusterPitch<\/h1>/.test(html), "named ClusterPitch");
  ok(run("!!D.pitch_clusters") === !getEl("#clWrap").hidden, "cluster switch visibility follows the data");

  // ---- cluster key on one type's map
  const pt = run('TYPES.includes("FF") ? "FF" : TYPES[0]'), ptn = run(`PTN[${J(pt)}]`);
  run(`state.color.all = state.color.t = "cat:cluster"; setView(${J(pt)});`); await tick();
  const items0 = getEl("#mkItems").innerHTML, chips0 = [...items0.matchAll(/data-arch="([^"]+)"/g)].map(m => m[1]);
  const nCl = run(`D.pitch_clusters.types[${J(pt)}].clusters.length`);
  ok(chips0.length === nCl, `key has a button per ${pt} cluster (${chips0.length} vs ${nCl})`);
  ok(!items0.includes(`</i>${ptn} `), `one-type key drops "${ptn}" from the names`);
  ok(getEl("#mkPick").hidden === true && getEl("#mkX").hidden === false, "no pick line yet; × shows");
  const keyClick = getEl("#mapKey")._h.click[0];
  const ev = (attr, val) => ({ target: { closest: s => (s === `[${attr}]` ? { dataset: { [attr.replace("data-", "")]: val } } : null) } });
  const pick = chips0[1] || chips0[0];
  keyClick(ev("data-arch", pick)); await tick();
  ok(run("state.archetype") === pick, "clicking a cluster in the key picks it");
  const items1 = getEl("#mkItems").innerHTML;
  ok([...items1.matchAll(/data-arch="/g)].length === nCl && items1.includes(`data-arch="${pick}" aria-pressed="true"`), "every cluster stays listed; the picked chip is pressed");
  ok((items1.match(/mk-pick muted/g) || []).length === nCl - 1, "the other chips are dimmed");
  ok(getEl("#mkPick").innerHTML.startsWith(pick.split("-").slice(1).join("-") + ":") && /data-archclear/.test(getEl("#mkPick").innerHTML), "pick line: short id, counts, Show all");
  const lit = run(`(() => { const h = clusterHits(state.archetype); return visibleRows(state.view).filter(r => h.rows.has(r)).length; })()`);
  const react = calls.filter(c => c.f === "react").pop();
  const nmFull = run(`VMAP.t["cat:cluster"].names[VMAP.t["cat:cluster"].cats.indexOf(${J(pick.slice(3))})]`);
  const colored = react.data.filter(t => t.customdata && t.name && t.name.startsWith(nmFull)).reduce((a, t) => a + t.customdata.length, 0);
  ok(lit > 0 && colored === lit, `lit dots = the cluster's core (${colored} vs ${lit})`);
  keyClick(ev("data-arch", chips0[0])); await tick();
  ok(run("state.archetype") === chips0[0], "another chip switches the pick");
  keyClick(ev("data-arch", chips0[0])); await tick();
  ok(run("state.archetype") === "", "clicking the picked chip again unpicks");
  keyClick(ev("data-arch", pick)); await tick(); keyClick(ev("data-archclear", "1")); await tick();
  ok(run("state.archetype") === "" && getEl("#mkPick").hidden, "Show all unpicks");
  keyClick(ev("data-arch", pick)); await tick();
  keyClick({ target: { closest: s => (s === "#mkX" ? {} : null) } }); await tick();
  ok(run("state.color.t") === "cat:pitch_type" && run("state.archetype") === "", "× turns clusters off and clears the pick");

  // ---- subclusters: each cluster heads its subclusters
  run(`state.color.all = state.color.t = "cat:subcluster"; render();`); await tick();
  const tree = getEl("#mkItems"), seq = [...tree.innerHTML.matchAll(/class="mk-item mk-pick( mk-par| mk-kid)?[^"]*" data-arch="([^"]+)"/g)].map(m => ({ k: (m[1] || "").trim(), id: m[2] }));
  ok(tree.classList.contains("tree") && seq.some(x => x.k === "mk-par") && seq.some(x => x.k === "mk-kid"), `subcluster key is a tree (${seq.filter(x => x.k === "mk-par").length} heads, ${seq.filter(x => x.k === "mk-kid").length} subclusters)`);
  let curPar = null, orderOk = true;
  seq.forEach(x => { const par = run(`(ARCH_BY_ID[${J(x.id)}] || {}).parent || null`);
    if (x.k === "mk-par") { curPar = x.id; if (run(`ARCH_BY_ID[${J(x.id)}].level`) !== 1) orderOk = false; }
    else if (x.k === "mk-kid") { if (par !== curPar) orderOk = false; }
    else { curPar = null; if (par) orderOk = false; } });
  ok(orderOk, "every subcluster sits under its own cluster; clusters without subclusters stand alone");
  ok(!/mk-kid[^>]*>[^<]*<i[^>]*><\/i>\s*\w+-/.test(tree.innerHTML) && !tree.innerHTML.includes(`</i>${ptn} `), "subcluster names carry no type prefix");
  const head = seq.find(x => x.k === "mk-par");
  keyClick(ev("data-arch", head.id)); await tick();
  const t2 = getEl("#mkItems").innerHTML;
  ok(run("state.archetype") === head.id && t2.includes(`data-arch="${head.id}" aria-pressed="true"`), "clicking a cluster head picks the whole cluster");
  const kids = seq.filter(x => x.k === "mk-kid" && run(`ARCH_BY_ID[${J(x.id)}].parent`) === head.id).map(x => x.id);
  ok(kids.length > 0 && kids.every(id => !new RegExp(`mk-kid muted" data-arch="${id}"`).test(t2)), "its subclusters stay lit");
  run(`state.archetype = ""; state.color.all = state.color.t = "cat:pitch_type"; render();`); await tick();

  // ---- morphs, 2D
  run(`setView("all")`); await tick(); await tick();
  const plot = getEl("#plot");
  const rowTraces = () => plot.data.filter(t => Array.isArray(t.customdata) && typeof t.customdata[0] === "number" && t.x && t.x.length === t.customdata.length);
  const b1 = calls.length; run(`goView(${J(pt)})`); await tick();
  ok(calls.slice(b1).some(c => c.f === "addTraces") && run("state.morphing"), "All -> type: ghosts fade, morph running");
  ok(rowTraces().every(t => t.customdata.every((r, i) => Math.abs(t.x[i] - TT.gx[r]) < 1e-9)), "All -> type: starts at the all-pitches positions");
  await pump(400); ok(run("state.morphing"), "still gliding at 400 ms");
  await pump(800); await tick();
  ok(!run("state.morphing") && run("state.view") === pt && calls.filter(c => c.f === "react").pop().data.filter(t => Array.isArray(t.customdata) && typeof t.customdata[0] === "number" && t.x && t.x.length === t.customdata.length).every(t => t.customdata.every((r, i) => Math.abs(t.x[i] - TT.tx[r]) < 1e-9)), "landed: every dot at its type-map position");
  run(`goView("all")`); await tick();
  let bad = 0, sh = 0, fd = 0;
  rowTraces().forEach(t => t.customdata.forEach((r, i) => { const op = Array.isArray(t.marker.opacity) ? t.marker.opacity[i] : t.marker.opacity;
    if (TT.pitch_type[r] === pt) { sh++; if (Math.abs(t.x[i] - TT.tx[r]) > 1e-9) bad++; } else { fd++; if (op !== 0) bad++; } }));
  ok(sh > 0 && fd > 0 && !bad, "type -> All: the type starts on its own map, the others invisible");
  await pump(1300); await tick(); ok(!run("state.morphing") && run("state.view") === "all", "landed on All");
  run(`goView(${J(pt)})`); await tick(); await pump(1300); await tick();
  const other = run(`TYPES.find(t => t !== ${J(pt)})`);
  const b4 = calls.length; run(`goView(${J(other)})`); await tick();
  ok(run("state.view") === "all" && run("state.morphing"), "type -> type: out to All first");
  await pump(500); ok(run("state.view") === "all" && run("state.morphing"), "first leg still gliding at 500 ms (950 ms legs)");
  await pump(500); await tick(); ok(run("state.view") === other && run("state.morphing"), "second leg into the new type");
  ok(calls.slice(b4).some(c => c.f === "addTraces"), "second leg fades the other types");
  await pump(1000); await tick(); ok(run("state.view") === other && !run("state.morphing"), "type -> type lands");
  run(`goView(${J(pt)})`); await tick(); run(`goView("all")`); await tick();
  ok(run("state.view") === "all" && !run("state.morphing"), "a click mid-glide goes straight there");

  // ---- morphs, 3D
  run(`state.dim = 3; state.zoomK = 1; render();`); await tick(); await pump(16);
  const G = run("G3");
  ok(!!G && G.objs.length > 0 && G.view === "all", "3D map on All pitches");
  const isRowPts = o => o.isPoints && Array.isArray(o.userData.rows) && typeof o.userData.rows[0] === "number";
  const near = (o, f) => o.userData.rows.every((r, k) => { const q = f(r), P = o.geometry.attributes.position.array; return [0, 1, 2].every(c => Math.abs(P[3 * k + c] - q[c]) < 1e-3); });
  G.camera.position.set(11, 22, 33); G.controls.target.set(1, 2, 3);
  run(`goView(${J(pt)})`);
  const ghosts3 = G.scene.children.filter(o => o.isPoints && o.userData.pick === false && !o.userData.rows);
  ok(run("state.morphing") && ghosts3.length > 0, "3D All -> type: running, other types as ghosts");
  ok(G.objs.filter(isRowPts).every(o => near(o, r => [TT.gx3[r], TT.gy3[r], TT.gz3[r]])) && G.camera.position.x === 11, "3D: starts at the all-pitches spots, camera where the user had it");
  ok(G.overlay.style.visibility === "hidden", "3D: labels step aside");
  await pump(1300); await tick();
  ok(!run("state.morphing") && G.overlay.style.visibility === "" && !G.scene.children.some(o => ghosts3.includes(o)), "3D: landed, labels back, ghosts gone");
  ok(G.objs.filter(isRowPts).every(o => near(o, r => [TT.tx3[r], TT.ty3[r], TT.tz3[r]])), "3D: every dot at its type-map spot");
  run(`goView(${J(other)})`); ok(run("state.view") === "all" && run("state.morphing"), "3D type -> type: first leg");
  await pump(1000); await tick(); ok(run("state.view") === other && run("state.morphing"), "3D: second leg");
  await pump(1000); await tick(); ok(run("state.view") === other && !run("state.morphing"), "3D type -> type lands");
  run(`state.dim = 2; dispose3D(); setView(${J(pt)});`); await tick(); await tick();

  // ---- the pitcher's cluster: on the map's selected-pitcher card and on his player card
  const j = run(`(() => { for (let r = 0; r < NT; r++) if (T.level[r] === "mlb" && T.cl1 && T.cl1[r] && T.pitch_type[r] === ${J(pt)} && trajOf(r)) return r; return -1; })()`);
  ok(j >= 0, "found an MLB pitch with a cluster and a flight");
  run(`preview(${j})`); await tick();
  ok(getEl("#ctaCl").innerHTML.includes(`data-cltag="${j}"`) && getEl("#ctaCl").innerHTML.includes("cltag"), "map card: his pitch's cluster tag under his name");
  run(`select(${j})`); await tick();
  const card = getEl("#detail").innerHTML + getEl("#dHead").innerHTML;
  ok(new RegExp(`class="cname"[^>]*>[^<]*(<span class="lvlbadge">[^<]*</span>)?<button type="button" class="cltag" data-cltag="${j}"`).test(card), "player card: cluster tag button in the name heading");
  const wantId = run(`"pc:" + (T.cl2 && T.cl2[${j}] && T.cl2[${j}] !== T.cl1[${j}] ? T.cl2[${j}] : T.cl1[${j}])`);
  run(`showClusterOf(${j})`); await tick();
  ok(run("state.view") === pt && run("state.sel") === j && run("state.preview") && !run("state.cardOpen"), "tag click: his type's map, him shown, card closed");
  ok(run("state.archetype") === wantId && run("compsOnMap()") === false && getEl("#mkItems").innerHTML.includes(`data-arch="${wantId}" aria-pressed="true"`), "tag click: that cluster lit and pressed, no lookalike spokes");
  run(`state.archetype = ""; state.color.all = state.color.t = "cat:pitch_type"; select(null)`); await tick();

  // ---- the Pitch Simulation Lab: MLB average as a column beside his pitch
  run(`state.sim = null; state.sel = ${j};`);
  const cards = run(`simScoreCards(${j}, simState(${j}))`);
  const c0 = cards[0];
  ok(c0 && c0.cols.length === 2 && c0.cols[1].avg === true, "solo lab: his pitch + an MLB average column");
  const av = c0 && c0.cols[1].v;
  ok(av && ["mph", "ivb", "hb", "relH", "relS", "ext", "arm", "vaa"].every(k => typeof av[k] === "number" && isFinite(av[k])), "MLB average values filled: " + (av ? Object.entries(av).map(([k, v]) => `${k} ${typeof v === "number" ? v.toFixed(1) : v}`).join(", ") : ""));
  ok(av && Math.abs(av.mph - c0.cols[0].v.mph) < 12 && Math.abs(av.relH - c0.cols[0].v.relH) < 3, "MLB average sits in a sensible range of his pitch");
  const sc = run(`simScoreHTML(${j}, simState(${j}))`);
  ok(sc.includes(">MLB avg</th>") && /class="dlt" style="background:/.test(sc) && /The MLB average is [\d.]+ \S+ (higher|lower) than/.test(sc), "scorecard: MLB avg column, coloured Δ, tooltip names the average");
  ok(/Δ = MLB avg minus /.test(sc), "scorecard key reads Δ = MLB avg minus him");
  run(`state.sim.showAvg = false`);
  ok(run(`simScoreCards(${j}, state.sim)`)[0].cols.length === 1, "MLB average off: no column");
  run(`state.sim.showAvg = true`);
  const look = run(`(() => { const c = compsOf(state.view, ${j}).map(x => x[0]).find(c => trajOf(c)); return c === undefined ? -1 : c; })()`);
  if (look >= 0) {
    run(`state.sim.items.push({ id: "ghost:t", kind: "ghost", t: ${J(pt)}, r: ${look}, on: true })`);
    const cc = run(`simScoreCards(${j}, state.sim)`)[0];
    ok(cc.cols.length === 2 && !cc.cols.some(o => o.avg), "with a lookalike flying, the comparison stays him vs the lookalike");
  }
  run(`state.sim = null; select(null)`); await tick();

  // ---- color menu: fixed beside its button on a wide screen
  getEl("#colorMenu").hidden = true;
  getEl("#colorBtn")._h.click[0]({}); await tick();
  const st = getEl("#colorMenu").style;
  ok(getEl("#colorMenu").hidden === false && st.position === "fixed" && parseFloat(st.left) >= 216, `color menu opens fixed beside the button (left ${st.left})`);
  report();
})().catch(e => { ok(false, "threw: " + e.stack.split("\n").slice(0, 5).join(" | ")); report(); });
