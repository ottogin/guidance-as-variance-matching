#!/usr/bin/env node
// M(2) on the 10-Gaussian ellipse with its noise correlated across x_t (l_x) and log sigma (l_t): the sampler behind the
// animation pages and the seed figures. Writes the final samples as float32 [n, 2], little-endian.
//     node reproduce/toys/corr_flow_sim.js SEED LX LT N n OUT.bin     # LX, LT: a number or inf; 0 is independent
function simulate(c, progress) {
  const K = 10, S = 0.15, S2 = S * S, n = c.n, N = c.N, lx = c.lx, lt = c.lt, TAU = 2 * Math.PI;
  const mux = new Float64Array(K), muy = new Float64Array(K);
  for (let k = 0; k < K; k++) { mux[k] = 2 * Math.cos(TAU * k / K); muy[k] = Math.sin(TAU * k / K); }
  function makeRng(seed) {
    let a = seed >>> 0, spare = null;
    const u = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const g = () => { if (spare !== null) { const s = spare; spare = null; return s; }
      const r = Math.sqrt(-2 * Math.log(1 - u())), th = TAU * u(); spare = r * Math.sin(th); return r * Math.cos(th); };
    return { u, g };
  }
  const ts = new Float64Array(N + 1);
  for (let i = 0; i < N; i++) { const v = N > 1 ? 1 + (1 / N - 1) * i / (N - 1) : 1; ts[i] = 3 * v / (1 + 2 * v); }
  const lsig = t => { const m = Math.min(t, 0.999); return Math.log(m / (1 - m)); };
  const start = makeRng(c.seed), R = makeRng((c.seed * 31 + 1009 * (c.salt + 1)) >>> 0);
  const fr = new Float32Array((N + 1) * n * 2), X = new Float64Array(n), Y = new Float64Array(n);
  for (let j = 0; j < n; j++) { X[j] = start.g(); Y[j] = start.g(); fr[2 * j] = X[j]; fr[2 * j + 1] = Y[j]; }
  const zx = new Float64Array(n), zy = new Float64Array(n);
  const finiteT = lt > 0 && lt < Infinity;
  let fill;
  if (lx === 0 && lt === 0) {
    fill = () => { for (let j = 0; j < n; j++) { zx[j] = R.g(); zy[j] = R.g(); } };
  } else if (lx === 0 && lt === Infinity) {
    for (let j = 0; j < n; j++) { zx[j] = R.g(); zy[j] = R.g(); }
    fill = () => {};
  } else if (lx === 0) {                                  // own noise per point, correlated across log sigma
    const F = 32, v = new Float32Array(n * F), ph = new Float32Array(n * F), a = new Float32Array(n * F), b = new Float32Array(n * F), q = Math.sqrt(2 / F);
    for (let e = 0; e < n * F; e++) { v[e] = R.g(); ph[e] = R.u() * TAU; a[e] = R.g() * q; b[e] = R.g() * q; }
    fill = (i, s) => { const w = s / lt;
      for (let j = 0, e = 0; j < n; j++) { let p = 0, r = 0;
        for (let f = 0; f < F; f++, e++) { const co = Math.cos(v[e] * w + ph[e]); p += a[e] * co; r += b[e] * co; }
        zx[j] = p; zy[j] = r; } };
  } else {                                                // one field shared by all points
    const F = lx >= 0.5 ? 256 : 512, q = Math.sqrt(2 / F), scale = lx === Infinity ? 0 : 1 / lx;
    const w1 = new Float64Array(F), w2 = new Float64Array(F), v = new Float64Array(F), ph = new Float64Array(F), a = new Float64Array(F), b = new Float64Array(F), cur = new Float64Array(F);
    const redraw = () => { for (let f = 0; f < F; f++) { w1[f] = R.g() * scale; w2[f] = R.g() * scale; v[f] = R.g(); ph[f] = R.u() * TAU; a[f] = R.g() * q; b[f] = R.g() * q; } };
    redraw();
    fill = (i, s) => {
      if (lt === 0 && i > 0) redraw();
      const w = finiteT ? s / lt : 0;
      for (let f = 0; f < F; f++) cur[f] = ph[f] + v[f] * w;
      for (let j = 0; j < n; j++) { let p = 0, r = 0; const x = X[j], y = Y[j];
        for (let f = 0; f < F; f++) { const co = Math.cos(w1[f] * x + w2[f] * y + cur[f]); p += a[f] * co; r += b[f] * co; }
        zx[j] = p; zy[j] = r; } };
  }
  const lg = new Float64Array(K), g = new Float64Array(K), mx = new Float64Array(K), my = new Float64Array(K);
  for (let i = 0; i < N; i++) {
    const t = ts[i], t2 = ts[i + 1];
    fill(i, lsig(t));
    const first = t >= 1, sig = first ? 0 : t / (1 - t), vv = S2 + sig * sig, tau2 = first ? S2 : S2 * sig * sig / vv;
    for (let j = 0; j < n; j++) {
      const x = X[j], y = Y[j];
      if (first) { g.fill(1 / K); for (let k = 0; k < K; k++) { mx[k] = mux[k]; my[k] = muy[k]; } }
      else {
        const yx = x / (1 - t), yy = y / (1 - t); let best = -Infinity;
        for (let k = 0; k < K; k++) { const dx = yx - mux[k], dy = yy - muy[k]; lg[k] = -0.5 * (dx * dx + dy * dy) / vv;
          if (lg[k] > best) best = lg[k]; mx[k] = mux[k] + S2 / vv * dx; my[k] = muy[k] + S2 / vv * dy; }
        let sum = 0; for (let k = 0; k < K; k++) { g[k] = Math.exp(lg[k] - best); sum += g[k]; }
        for (let k = 0; k < K; k++) g[k] /= sum;
      }
      let mex = 0, mey = 0; for (let k = 0; k < K; k++) { mex += g[k] * mx[k]; mey += g[k] * my[k]; }
      let A = tau2, B = 0, C = tau2;
      for (let k = 0; k < K; k++) { const dx = mx[k] - mex, dy = my[k] - mey; A += g[k] * dx * dx; B += g[k] * dx * dy; C += g[k] * dy * dy; }
      const sd = Math.sqrt(Math.max(A * C - B * B, 0)), tt = Math.sqrt(Math.max(A + C + 2 * sd, 1e-300));
      const La = (A + sd) / tt, Lb = B / tt, Lc = (C + sd) / tt;
      const hx = mex + La * zx[j] + Lb * zy[j], hy = mey + Lb * zx[j] + Lc * zy[j];
      if (t2 === 0) { X[j] = hx; Y[j] = hy; }
      else { const ex = (x - (1 - t) * hx) / t, ey = (y - (1 - t) * hy) / t; X[j] = (1 - t2) * hx + t2 * ex; Y[j] = (1 - t2) * hy + t2 * ey; }
      const o = 2 * ((i + 1) * n + j); fr[o] = X[j]; fr[o + 1] = Y[j];
    }
    if (progress && (i % 8 === 7 || i === N - 1)) progress((i + 1) / N);
  }
  const final = new Uint8Array(n), count = new Float64Array(K); let dist = 0;
  for (let j = 0; j < n; j++) { let bk = 0, bd = Infinity;
    for (let k = 0; k < K; k++) { const dx = X[j] - mux[k], dy = Y[j] - muy[k], d = dx * dx + dy * dy; if (d < bd) { bd = d; bk = k; } }
    final[j] = bk; count[bk]++; dist += bd; }
  let H = 0, top = 0, thin = 0;
  for (let k = 0; k < K; k++) { const p = count[k] / n; if (p > 0) H -= p * Math.log(p); if (p > top) top = p; if (p < 1 / (4 * K)) thin++; }
  return { frames: fr, final, ts, stats: { modes: Math.exp(H), top, thin, spread: dist / n / (2 * S2) } };
}

if (require.main === module) {
  const [seed, lx, lt, N, n, out] = process.argv.slice(2);
  const num = v => (v === "inf" ? Infinity : +v), r = simulate({ n: +n, N: +N, seed: +seed, salt: 0, lx: num(lx), lt: num(lt) });
  require("fs").writeFileSync(out, Buffer.from(r.frames.buffer, 2 * +N * +n * 4, 2 * +n * 4));
  console.log(JSON.stringify(r.stats));
}
