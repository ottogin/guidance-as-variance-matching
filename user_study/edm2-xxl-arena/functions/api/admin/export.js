// GET /api/admin/export (header x-owner-key) -> every vote as CSV, with the rater's name and the settings decoded.
import { ownerFrom, json } from "../../_shared.js";
import { NAMES as NAME } from "../../_arms.js";
export async function onRequestGet({ request, env }) {
  if (!ownerFrom(request, env)) return json({ error: "wrong password" }, 401);
  const { results } = await env.DB.prepare(
    `SELECT v.rater, r.name, r.source, p.person, v.seq, v.pos, v.left_arm, v.right_arm, v.result, v.rt_ds, v.created_at
     FROM votes v JOIN raters r ON r.id = v.rater JOIN rater_person p ON p.id = v.rater ORDER BY v.rater, v.seq`).all();
  const q = s => `"${String(s ?? "").replace(/"/g, '""')}"`;
  const rows = [["rater_id", "name", "source", "person", "seq", "row_pos", "left", "right", "result", "winner", "seconds", "time_utc"].join(",")];
  for (const x of results) {
    const winner = x.result === "l" ? NAME[x.left_arm] : x.result === "r" ? NAME[x.right_arm] : "tie";
    rows.push([q(x.rater), q(x.name), x.source, q(x.person), x.seq, x.pos, q(NAME[x.left_arm]), q(NAME[x.right_arm]), x.result,
      q(winner), x.rt_ds == null ? "" : (x.rt_ds / 10).toFixed(1), new Date(x.created_at).toISOString()].join(","));
  }
  return new Response(rows.join("\n") + "\n", { headers: { "content-type": "text/csv; charset=utf-8",
    "content-disposition": `attachment; filename="edm2-xxl-arena-votes-${new Date().toISOString().slice(0, 10)}.csv"`, "cache-control": "no-store" } });
}
