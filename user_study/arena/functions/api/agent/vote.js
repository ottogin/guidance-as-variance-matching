// POST /api/agent/vote  (Authorization: Bearer <api_key>)
// {"votes": [{"task_id": "...", "choice": "A" | "B" | "tie", "reason": "optional, <= 1000 chars"}]}  (or one vote object)
// A = the first image (image_a), B = the second. Each task can be answered once.
import { json, agentFrom, unauthorized, LOCKED, closed } from "./_lib.js";

const CHOICE = { a: "l", left: "l", "1": "l", b: "r", right: "r", "2": "r", tie: "t", t: "t", equal: "t", same: "t" };

export async function onRequestPost({ request, env }) {
  if (LOCKED) return closed();
  const agent = await agentFrom(request, env);
  if (!agent) return unauthorized();
  let body;
  try { body = await request.json(); } catch { return json({ error: "send JSON" }, 400); }
  const list = Array.isArray(body?.votes) ? body.votes : body && body.task_id ? [body] : null;
  if (!list || !list.length || list.length > 50) return json({ error: "send {\"votes\": [{\"task_id\", \"choice\"}]} with 1-50 votes" }, 400);
  let { s } = await env.DB.prepare(`SELECT coalesce(max(seq) + 1, 0) AS s FROM votes WHERE rater = ?`).bind(agent.id).first();
  const now = Date.now(), stmts = [], accepted = [], rejected = [];
  for (const v of list) {
    const r = CHOICE[String(v?.choice ?? "").trim().toLowerCase()];
    const t = typeof v?.task_id === "string" && await env.DB.prepare(
      `SELECT id, pos, left_arm, right_arm, done FROM agent_tasks WHERE id = ? AND rater = ?`).bind(v.task_id, agent.id).first();
    if (!t) { rejected.push({ task_id: v?.task_id, error: "unknown task" }); continue; }
    if (t.done || accepted.includes(t.id)) { rejected.push({ task_id: t.id, error: "already answered" }); continue; }
    if (!r) { rejected.push({ task_id: t.id, error: "choice must be A, B or tie" }); continue; }
    const reason = typeof v.reason === "string" ? v.reason.slice(0, 1000) : null;
    stmts.push(env.DB.prepare(`INSERT INTO votes (rater, seq, pos, left_arm, right_arm, result, rt_ds, created_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)`)
      .bind(agent.id, s++, t.pos, t.left_arm, t.right_arm, r, now));
    stmts.push(env.DB.prepare(`UPDATE agent_tasks SET done = 1, result = ?, reason = ? WHERE id = ?`).bind(r, reason, t.id));
    accepted.push(t.id);
  }
  if (stmts.length) {
    stmts.push(env.DB.prepare(`UPDATE raters SET last_seen = ? WHERE id = ?`).bind(now, agent.id));
    await env.DB.batch(stmts);
  }
  return json({ ok: true, accepted: accepted.length, rejected, total_votes: s });
}
