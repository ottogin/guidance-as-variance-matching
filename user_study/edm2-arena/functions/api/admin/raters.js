// Admin API (header x-owner-key: admin password or owner key).
// GET  /api/admin/raters  -> every browser/agent (rater) with its person, name, source, votes, times; the excluded list.
// POST /api/admin/raters  {action: "rename", id, name} | {action: "delete", id} | {action: "exclude", person, excluded}
import { json, ownerFrom, excludedSet, setExcluded } from "../../_shared.js";

const deny = async () => { await new Promise(r => setTimeout(r, 600)); return json({ error: "wrong password" }, 401); };

export async function onRequestGet({ request, env }) {
  if (!ownerFrom(request, env)) return deny();
  const { results } = await env.DB.prepare(
    `SELECT r.id, r.name, r.source, p.person, r.skips, r.created_at, r.last_seen,
            (SELECT count(*) FROM votes v WHERE v.rater = r.id) AS votes,
            (SELECT count(*) FROM votes v WHERE v.rater = r.id AND v.result = 't') AS ties,
            (SELECT count(*) FROM agent_tasks t WHERE t.rater = r.id AND t.done = 0) AS open_tasks
     FROM raters r JOIN rater_person p ON p.id = r.id ORDER BY r.last_seen DESC`).all();
  return json({ raters: results, excluded: [...await excludedSet(env)] }, 200, { "cache-control": "no-store" });
}

export async function onRequestPost({ request, env }) {
  if (!ownerFrom(request, env)) return deny();
  let b;
  try { b = await request.json(); } catch { return json({ error: "bad json" }, 400); }
  const id = typeof b?.id === "string" ? b.id : "";
  if (b?.action === "rename") {
    const name = typeof b.name === "string" ? b.name.replace(/[\u0000-\u001f\u007f]/g, "").replace(/\s+/g, " ").trim().slice(0, 60) : "";
    const r = await env.DB.prepare(`UPDATE raters SET name = ?1 WHERE id = ?2`).bind(name || null, id).run();
    return json({ ok: r.meta.changes === 1 });
  }
  if (b?.action === "delete") {
    const r = await env.DB.batch([
      env.DB.prepare(`DELETE FROM votes WHERE rater = ?`).bind(id),
      env.DB.prepare(`DELETE FROM agent_tasks WHERE rater = ?`).bind(id),
      env.DB.prepare(`DELETE FROM raters WHERE id = ?`).bind(id)]);
    return json({ ok: r[2].meta.changes === 1, votes_deleted: r[0].meta.changes });
  }
  if (b?.action === "exclude" && typeof b.person === "string") {
    const ex = await excludedSet(env);
    if (b.excluded) ex.add(b.person); else ex.delete(b.person);
    await setExcluded(env, [...ex]);
    return json({ ok: true, excluded: [...ex] });
  }
  return json({ error: "unknown action" }, 400);
}
