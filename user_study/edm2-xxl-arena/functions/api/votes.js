// POST /api/votes  {rater, skips, from, votes: ["<pos>.<left><right><l|r|t>.<deciseconds>", ...]}
// Vote k of the batch is the rater's vote number from + k. (rater, seq) is the primary key, so a retried
// upload is ignored rather than counted twice.
import { HIDE, ARMS, SHOWN_PAIRS, N, RATER_RE, VOTE_RE, MAX_BATCH, MAX_PER_RATER, LOCKED, closed, json } from "../_shared.js";

export async function onRequestPost({ request, env }) {
  if (LOCKED) return closed();    // before any write, so a stored name or count never changes either
  let body;
  try { body = await request.json(); } catch { return json({ error: "bad json" }, 400); }
  const { rater, skips = 0, from, votes } = body || {};
  if (typeof rater !== "string" || !RATER_RE.test(rater)) return json({ error: "bad rater" }, 400);
  if (!Number.isInteger(from) || from < 0 || !Array.isArray(votes) || votes.length > MAX_BATCH) return json({ error: "bad batch" }, 400);
  if (from + votes.length > MAX_PER_RATER) return json({ error: "too many votes" }, 429);
  const now = Date.now();
  const rows = [];
  for (let k = 0; k < votes.length; k++) {
    const m = typeof votes[k] === "string" && VOTE_RE.exec(votes[k]);
    if (!m) return json({ error: `bad vote ${from + k}` }, 400);
    const pos = +m[1];
    if (pos >= N || HIDE.has(pos) || !ARMS.has(m[2]) || !ARMS.has(m[3]) || m[2] === m[3] || !SHOWN_PAIRS.has(m[2] + m[3])) return json({ error: `bad vote ${from + k}` }, 400);
    rows.push([rater, from + k, pos, m[2], m[3], m[4], Math.min(+m[5], 99999), now]);
  }
  const sk = Number.isInteger(skips) && skips >= 0 ? Math.min(skips, 100000) : 0;
  // name: a non-empty string sets it; empty or absent keeps the stored one (an emptied name box must not erase it)
  const name = typeof body.name === "string" ? body.name.replace(/[\u0000-\u001f\u007f]/g, "").replace(/\s+/g, " ").trim().slice(0, 60) || null : null;
  const hasName = !!name;
  const stmts = [env.DB.prepare(
    `INSERT INTO raters (id, created_at, last_seen, skips, source, name) VALUES (?1, ?2, ?2, ?3, 'web', ?5)
     ON CONFLICT(id) DO UPDATE SET last_seen = excluded.last_seen, skips = max(raters.skips, excluded.skips),
       name = CASE WHEN ?4 THEN excluded.name ELSE raters.name END`).bind(rater, now, sk, hasName ? 1 : 0, name)];
  const ins = env.DB.prepare(
    `INSERT OR IGNORE INTO votes (rater, seq, pos, left_arm, right_arm, result, rt_ds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`);
  for (const r of rows) stmts.push(ins.bind(...r));
  await env.DB.batch(stmts);           // one transaction
  const { n } = await env.DB.prepare(`SELECT count(*) AS n FROM votes WHERE rater = ?`).bind(rater).first();
  return json({ ok: true, stored: n });
}
