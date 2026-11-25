// POST /api/settings {key, excluded: [person, ...]}  (owner only): which people the leaderboard leaves out.
import { json, isOwnerKey } from "../_shared.js";

export async function onRequestPost({ request, env }) {
  let body;
  try { body = await request.json(); } catch { return json({ error: "bad json" }, 400); }
  if (!isOwnerKey(env, body && body.key)) return json({ error: "not owner" }, 403);
  const ex = Array.isArray(body.excluded) ? body.excluded.filter(x => typeof x === "string" && x.length <= 80).slice(0, 500) : null;
  if (!ex) return json({ error: "bad list" }, 400);
  await env.DB.prepare(`INSERT INTO settings (key, value) VALUES ('excluded', ?1)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value`).bind(JSON.stringify(ex)).run();
  return json({ ok: true, excluded: ex });
}
