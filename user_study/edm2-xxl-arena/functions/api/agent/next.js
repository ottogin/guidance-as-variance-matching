// GET /api/agent/next?n=10  (Authorization: Bearer <api_key>)
// Returns up to n tasks: unanswered ones first, then new ones. New tasks use the setting pair this agent has seen
// least, a row it has not judged for that pair, and random sides. At most 20 tasks are open per agent.
import { json, hex, agentFrom, unauthorized, INSTRUCTION, LOCKED, closed } from "./_lib.js";
import { TILES, CAPTIONS } from "../../_tiles.js";
import { PAIRS as ALLOWED } from "../../_arms.js";

const key = (a, b) => (a < b ? a + b : b + a);
const PAIRS = ALLOWED.map(p => key(p[0], p[1]));       // as the votes are counted below: letters in sorted order
const POS = Object.keys(CAPTIONS).map(Number);

export async function onRequestGet({ request, env }) {
  if (LOCKED) return closed();
  const agent = await agentFrom(request, env);
  if (!agent) return unauthorized();
  const n = Math.max(1, Math.min(20, parseInt(new URL(request.url).searchParams.get("n")) || 10));
  const open = (await env.DB.prepare(`SELECT id, pos, left_arm, right_arm FROM agent_tasks WHERE rater = ? AND done = 0 ORDER BY created_at LIMIT ?`)
    .bind(agent.id, n).all()).results;
  const tasks = open.map(t => ({ id: t.id, pos: t.pos, L: t.left_arm, R: t.right_arm }));
  if (tasks.length < n) {
    const seen = (await env.DB.prepare(
      `SELECT pos, left_arm AS L, right_arm AS R FROM votes WHERE rater = ?1
       UNION ALL SELECT pos, left_arm, right_arm FROM agent_tasks WHERE rater = ?1 AND done = 0`).bind(agent.id).all()).results;
    const use = Object.fromEntries(PAIRS.map(p => [p, 0])), done = new Set();
    for (const s of seen) { const k = key(s.L, s.R); use[k] = (use[k] || 0) + 1; done.add(s.pos + ":" + k); }
    const stmts = [], now = Date.now();
    while (tasks.length < n) {
      const min = Math.min(...PAIRS.map(p => use[p]));
      const cand = PAIRS.filter(p => use[p] === min), pair = cand[Math.floor(Math.random() * cand.length)];
      const free = POS.filter(p => !done.has(p + ":" + pair));
      const pos = (free.length ? free : POS)[Math.floor(Math.random() * (free.length || POS.length))];
      const [L, R] = Math.random() < .5 ? [pair[0], pair[1]] : [pair[1], pair[0]];
      const id = hex(12);
      use[pair]++; done.add(pos + ":" + pair);
      tasks.push({ id, pos, L, R });
      stmts.push(env.DB.prepare(`INSERT INTO agent_tasks (id, rater, pos, left_arm, right_arm, created_at) VALUES (?, ?, ?, ?, ?, ?)`)
        .bind(id, agent.id, pos, L, R, now));
    }
    if (stmts.length) await env.DB.batch(stmts);
  }
  const o = new URL(request.url).origin;
  return json({ model: agent.name, instruction: INSTRUCTION, tasks: tasks.map(t => ({
    task_id: t.id, caption: CAPTIONS[t.pos],
    image_a: `${o}/t/${TILES[t.pos + ":" + t.L]}.webp`, image_b: `${o}/t/${TILES[t.pos + ":" + t.R]}.webp` })) },
    200, { "cache-control": "no-store" });
}
