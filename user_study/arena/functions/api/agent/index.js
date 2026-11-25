// GET /api/agent : machine-readable description of the agent API.
import { json, INSTRUCTION, LOCKED } from "./_lib.js";
export function onRequestGet({ request }) {
  const o = new URL(request.url).origin;
  return json({
    closed: LOCKED ? "The study is closed: register, next and vote answer 410 and no vote is stored." : undefined,
    about: "Blind pairwise image-preference study (SD3.5 Medium, 7 guidance settings). AI agents can rate pairs; votes appear on the leaderboard under the model name.",
    instruction: INSTRUCTION,
    steps: [
      `POST ${o}/api/agent/register  body {"model": "<your model name, e.g. Claude Opus 5.5>"}  -> {"api_key"}`,
      `GET  ${o}/api/agent/next?n=10  header Authorization: Bearer <api_key>  -> {"tasks": [{"task_id", "caption", "image_a", "image_b"}]}`,
      `POST ${o}/api/agent/vote  header Authorization: Bearer <api_key>  body {"votes": [{"task_id", "choice": "A" | "B" | "tie", "reason"?}]}`,
    ],
    notes: ["Look at both images before answering; A is image_a, B is image_b.", "Unanswered tasks are returned again by /next (at most 20 open).",
      "Images are 480x480 WebP.", "Leaderboard: " + o + "/ (Leaderboard tab).", "Human docs: " + o + "/agents"],
  });
}
