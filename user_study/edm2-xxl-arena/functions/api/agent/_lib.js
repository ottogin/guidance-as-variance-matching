// Shared helpers for the agent API.
import { json, LOCKED, closed } from "../../_shared.js";
export { json, LOCKED, closed };
export const INSTRUCTION =
  "Each task shows one ImageNet class and two images (A and B) generated for it by the same class-conditional diffusion " +
  "model (EDM2) with two different settings, from the same initial noise. Pick the one you prefer overall: image " +
  "quality, how natural it looks, and how well it shows the class. If you genuinely can't choose, answer tie.";
export const hex = n => Array.from(crypto.getRandomValues(new Uint8Array(n)), b => b.toString(16).padStart(2, "0")).join("");
export async function agentFrom(request, env) {   // Authorization: Bearer <api_key>  (the key is the agent's rater id)
  const m = /^Bearer\s+([A-Za-z0-9_-]{16,64})$/.exec(request.headers.get("authorization") || "");
  if (!m) return null;
  return await env.DB.prepare(`SELECT id, name FROM raters WHERE id = ? AND source = 'agent'`).bind(m[1]).first();
}
export const unauthorized = () => json({ error: "missing or unknown API key: send 'Authorization: Bearer <api_key>' (POST /api/agent/register to get one)" }, 401);
