# SD3.5-M seven-arm arena (static site + D1)

Pairwise user study of the seven SD3.5-M guidance settings in the paper.

**Closed on 2026-11-23** (`LOCKED` in `functions/_shared.js` and `page.template.html`): `/api/votes` and the agent API
answer 410 and store nothing. Visitors can still rate; their votes stay in the browser and make up "Just me", while
"Everyone" is only the votes stored before the lock. The owner's rater list shows a random alias per person
(`settings.aliases`, person -> alias); AI raters keep their model name, and real names stay in `raters.name` (/admin).
An eighth arm (p, the released w = 4.5) ran 2026-11-20..23 and was withdrawn: migration 0007 deleted its 570 votes and
the 9 browsers that rated only p.

Offline copy for reviewers: `python3 user_study/arena/build.py --offline DIR` writes DIR/index.html + DIR/s/ (no
server or network; Everyone = `reproduce/data/user_study_votes.json`, the paper's 1,458 votes from 41 people; the
page's Elo equals `reproduce/tables/tab_user_study.py`). `--px 320` brings it from ~70 MB to ~52 MB. Cloudflare Pages serves the page and the
495 caption sprites; Pages Functions (`functions/api/`) front a D1 (SQLite) database (`migrations/`).

- `POST /api/votes` `{rater, skips, from, votes: ["<pos>.<L><R><l|r|t>.<deciseconds>"]}`: vote k is the rater's
  seq `from + k`; `(rater, seq)` is the primary key, so retries never double-count. Hidden captions and malformed
  votes are refused.
- `GET /api/board?rater=<id>`: per-rater pair counts from the `pair_counts` view, raters anonymised, caller flagged.
  The page fits Bradley-Terry and bootstraps raters client-side.
- Owner link: `?key=<OWNER_KEY>` (Pages secret; the link is in `arena/.owner-url`, gitignored). The owner sees every
  rater by name with a checkbox; unticked people (settings.excluded) are left out of the board for everyone.
- Admin page `/admin`: password = Pages secret ADMIN_PASSWORD (also accepted wherever the owner key is). Lists every
  browser/agent, include/exclude, rename, delete (two-click), CSV export (`/api/admin/{raters,export}`, header x-owner-key).
- Agent API (docs: /agents, JSON: /api/agent): `POST /api/agent/register {model}` -> api_key (= a rater id, source
  `agent`); `GET /api/agent/next?n=` (Bearer key) hands out tasks from `agent_tasks` (least-seen pair, unseen caption,
  random sides, <= 20 open); `POST /api/agent/vote {votes:[{task_id, choice A|B|tie, reason}]}`. Images are per-setting
  tiles `public/t/<hmac id>.webp` (build.py, salt in `.tile-salt`; map in `functions/_tiles.js`, both gitignored).
  AI raters group as `agent:<lower(name)>` (migration 0005) and are named publicly; people stay anonymous.
- Claude Code is a rater (source `claude`, 1,337 blind AI votes from 2026-11-05; `import-claude.sql`).
- Rater id: random, kept in the browser's localStorage (no IP stored). Optional name on the intro screen ->
  `raters.name`, never returned to the page. A person = a name (case-insensitive; view `rater_person`): all browsers
  with the same name count as one rater on the board and in the bootstrap; unnamed browsers stay separate. `?owner` unlocks the leaderboard and per-rater counts.

Build and deploy:

    python3 user_study/arena/build.py [--import-db DIR]   # page + sprites from results/elo7/page; DIR = a JSON export of an earlier copy's votes
    cd user_study/arena && npx wrangler login               # once, in a browser
    npx wrangler d1 create arena                 # put the id into wrangler.toml
    npm run migrate:remote
    npx wrangler d1 execute arena --remote --file import.sql
    npx wrangler pages project create arena --production-branch main
    npm run deploy

Read the data: `npx wrangler d1 execute arena --remote --command "SELECT * FROM rater_summary"` (one row per person: name, browsers, votes, when),
or `... "SELECT * FROM pair_counts"`. The 128 votes imported from an earlier copy of the page are an author's and are left out of the paper's tables.
