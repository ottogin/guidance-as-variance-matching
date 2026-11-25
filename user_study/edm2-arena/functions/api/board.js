// GET /api/board?rater=<browser id>[&key=<owner key>]
// Per person (browsers grouped by name, see rater_person), per unordered arm pair: wins / ties / losses of the
// alphabetically first arm. The caller's own browser is left out (the page adds it from its local list) and the
// caller's person is flagged `me`.
// Everyone else: only people the owner has not excluded, anonymous except AI raters (`agent`: model name).
// Owner (valid key): every person, with a display name, vote count, browsers and the excluded flag, so the page can
// toggle them. The display name of a person is a random alias (settings.aliases: person -> alias);
// AI raters keep their model name. Real names stay in raters.name, seen only on /admin.
// `locked`: the study is closed (no vote is stored); the page then stops adding the visitor's new votes to Everyone.
import { RATER_RE, json, isOwnerKey, excludedSet, LOCKED } from "../_shared.js";

export async function onRequestGet({ request, env }) {
  const qs = new URL(request.url).searchParams;
  const q = qs.get("rater") || "";
  const me = RATER_RE.test(q) ? q : "";
  const owner = isOwnerKey(env, qs.get("key"));
  const excluded = await excludedSet(env);
  const mine = me && await env.DB.prepare(`SELECT person FROM rater_person WHERE id = ?`).bind(me).first();
  const mePerson = mine ? mine.person : "id:" + me;
  const { results } = await env.DB.prepare(
    `SELECT p.person, c.arm, c.opp, sum(c.wins) AS wins, sum(c.ties) AS ties, sum(c.games - c.wins - c.ties) AS losses
     FROM pair_counts c JOIN rater_person p ON p.id = c.rater
     WHERE c.arm < c.opp AND c.rater <> ?1
     GROUP BY p.person, c.arm, c.opp ORDER BY p.person, c.arm, c.opp`).bind(me).all();
  const by = new Map();
  for (const r of results) {
    if (!owner && excluded.has(r.person)) continue;
    if (!by.has(r.person)) by.set(r.person, []);
    by.get(r.person).push([r.arm, r.opp, r.wins, r.ties, r.losses]);
  }
  const agents = new Map((await env.DB.prepare(   // AI raters are named publicly; people stay anonymous
    `SELECT person, max(name) AS name FROM rater_person WHERE person LIKE 'agent:%' GROUP BY person`).all()).results.map(x => [x.person, x.name]));
  let info = new Map(), alias = {};
  if (owner) {   // display names and totals over ALL browsers, the caller's included
    const row = await env.DB.prepare(`SELECT value FROM settings WHERE key = 'aliases'`).first();
    try { alias = row ? JSON.parse(row.value) : {}; } catch { alias = {}; }
    const { results: ppl } = await env.DB.prepare(
      `SELECT p.person, max(p.name) AS name, count(DISTINCT p.id) AS browsers, count(v.seq) AS votes, max(r.source) AS source
       FROM rater_person p JOIN raters r ON r.id = p.id LEFT JOIN votes v ON v.rater = p.id GROUP BY p.person`).all();
    info = new Map(ppl.map(x => [x.person, x]));
    for (const person of info.keys()) if (!by.has(person)) by.set(person, []);
  }
  const raters = [...by].map(([person, c]) => {
    const o = { me: person === mePerson, c };
    if (agents.has(person)) o.agent = agents.get(person);
    if (owner) {
      const x = info.get(person) || {};
      o.person = person; o.excluded = excluded.has(person); o.votes = x.votes || 0; o.browsers = x.browsers || 0;
      o.name = agents.get(person) || alias[person] || (person.startsWith("id:") ? `Anonymous (${person.slice(3, 9)}…)` : "Unnamed rater");
      o.source = x.source || "web";
    }
    return o;
  });
  return json({ raters, owner, locked: LOCKED, meExcluded: excluded.has(mePerson) }, 200, { "cache-control": "private, max-age=5" });
}
