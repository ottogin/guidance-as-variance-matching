// Caption positions taken down (NSFW): the page never shows them and the API refuses new votes on them.
// Votes already cast on them stay in the table and still count.
export const HIDE = new Set([191, 197, 241, 304, 492]);
export const ARMS = new Set(["f", "d", "o", "s", "i", "l", "v"]);
// The study is closed. No vote is stored any more (people or agents); the page keeps a visitor's votes
// in their browser ("Just me") and "Everyone" shows only the votes stored up to now. page.template.html has the
// same flag for the page side.
export const LOCKED = true;
export const closed = () => json({ error: "the study is closed: votes are no longer stored" }, 410);
export const N = 500;
export const RATER_RE = /^[A-Za-z0-9_-]{16,64}$/;
export const VOTE_RE = /^(\d{1,3})\.([a-z])([a-z])([lrt])\.(\d{1,5})$/;
export const MAX_BATCH = 200;
export const MAX_PER_RATER = 5000;

export const json = (obj, status = 200, headers = {}) =>
  new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json", ...headers } });

// Owner check: the owner link carries ?key=<OWNER_KEY> (a Pages secret). Constant-time compare.
const same = (a, b) => {
  if (!a || typeof b !== "string" || a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
};
// Owner = the owner link's key (OWNER_KEY) or the admin password (ADMIN_PASSWORD); both are Pages secrets.
export function isOwnerKey(env, key) { return same(env.OWNER_KEY || "", key) || same(env.ADMIN_PASSWORD || "", key); }
export const ownerFrom = (request, env) => isOwnerKey(env, request.headers.get("x-owner-key"));
export async function setExcluded(env, list) {
  await env.DB.prepare(`INSERT INTO settings (key, value) VALUES ('excluded', ?1)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value`).bind(JSON.stringify(list)).run();
}
export async function excludedSet(env) {
  const row = await env.DB.prepare(`SELECT value FROM settings WHERE key = 'excluded'`).first();
  try { return new Set(row ? JSON.parse(row.value) : []); } catch { return new Set(); }
}
