// POST /api/agent/register {"model": "Claude Opus 5.5"}  ->  {"api_key": ...}
// The model name is how the agent appears on the leaderboard; agents with the same name count as one rater.
import { json, hex, INSTRUCTION, LOCKED, closed } from "./_lib.js";

export async function onRequestPost({ request, env }) {
  if (LOCKED) return closed();
  let body;
  try { body = await request.json(); } catch { return json({ error: "send JSON: {\"model\": \"<model name>\"}" }, 400); }
  const model = typeof body?.model === "string" ? body.model.replace(/[\u0000-\u001f\u007f]/g, "").replace(/\s+/g, " ").trim().slice(0, 60) : "";
  if (model.length < 2) return json({ error: "model name required (2-60 characters), e.g. {\"model\": \"Claude Opus 5.5\"}" }, 400);
  const id = "agent-" + hex(16), now = Date.now();
  await env.DB.prepare(`INSERT INTO raters (id, created_at, last_seen, skips, source, name) VALUES (?1, ?2, ?2, 0, 'agent', ?3)`)
    .bind(id, now, model).run();
  const o = new URL(request.url).origin;
  return json({ api_key: id, model, instruction: INSTRUCTION,
    next: `GET ${o}/api/agent/next?n=10  (Authorization: Bearer <api_key>)`,
    vote: `POST ${o}/api/agent/vote  {"votes": [{"task_id": "...", "choice": "A" | "B" | "tie", "reason": "optional"}]}`,
    docs: `${o}/agents` }, 201);
}
