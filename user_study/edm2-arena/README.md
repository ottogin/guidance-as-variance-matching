# EDM2 ImageNet-512 arena (static site + D1)

Blind pairwise user study of guidance settings for NVIDIA's EDM2 on ImageNet-512, on the skeleton of the SD3.5-M arena
(`user_study/arena`). Each round shows one ImageNet class and two images of it from the same initial noise, generated
with two settings in random order; the rater picks one or calls a tie.

Arms (`ARMS` and `GROUPS` in `build.py`): EDM2-S guided by the unconditional EDM2-XS (classifier-free guidance, both
at post-hoc EMA 0.025, NVIDIA's `edm2-img512-s-guid-fid` preset), at

- `n`: NVIDIA's FID-tuned scale, w = 1.40;
- `o`: our variance-matched constant, w = 5.6658 (tab:edm2-fail).

Rows are seeds 0-99 of `sample.py --model edm2-512 --n 100` (`--published` and `--w 5.6658`): EDM2's 32-step Heun
sampler, and each seed fixes both the noise and the class, as in `generate_images.py`. Renders and the class of each
seed are fetched into `results/edm2-arena/renders/` (commands in `build.py`'s docstring).

## Pieces

- `build.py` writes `public/index.html` (from `page.template.html`), one sprite per row `public/s/NNN.webp` (every arm
  side by side, slot order shuffled per row and hidden in `KEY`), per-image tiles `public/t/<hmac>.webp` for the agent
  API, `functions/_arms.js` (arm codes, names, allowed pairs, row count) and `functions/_tiles.js` (tile ids and class
  captions; gitignored, never served). Tile ids are an HMAC under `.tile-salt` (gitignored), so they stay stable
  across rebuilds and reveal nothing.
- `POST /api/votes` `{rater, skips, from, votes: ["<pos>.<L><R><l|r|t>.<deciseconds>"], name}`: vote k is the rater's
  seq `from + k`; `(rater, seq)` is the primary key, so retries never double-count.
- `GET /api/board?rater=<id>`: per-person pair counts (the page fits Bradley-Terry and bootstraps raters
  client-side). Owner link `?key=<OWNER_KEY>` shows every rater (under a display name) with a checkbox to leave them out.
- `/admin`: password = Pages secret `ADMIN_PASSWORD` (also accepted wherever the owner key is): list, rename, delete,
  include/exclude raters, CSV export.
- Agent API (docs `/agents`, JSON `/api/agent`): `POST /api/agent/register {model}`, `GET /api/agent/next?n=`,
  `POST /api/agent/vote`. AI raters are named on the board by model; people stay anonymous.
- Closing the study: set `LOCKED = true` in `functions/_shared.js` and in `page.template.html` (with `CLOSED_ON`),
  rebuild and deploy. Votes are then refused with 410 and visitors' votes stay in their browser.

## Adding arms

Sample the new settings for the same seeds (e.g. EDM2-XXL guided by the unconditional EDM2-XS at NVIDIA's w = 1.20 and
at our constant), put them under `results/edm2-arena/renders/<code>/`, add them to `ARMS` and `GROUPS` in `build.py`,
and allow their codes in the `votes` CHECK: edit `migrations/0001_init.sql` before the first deploy, or add a migration
that rebuilds `votes` once the database exists (SQLite cannot change a CHECK in place). `build.py` refuses to build
while the newest migration's CHECK and `ARMS` differ. `CROSS_GROUP = False` restricts raters to pairs within a group.

## Build and deploy

    python3 user_study/edm2-arena/build.py           # needs Pillow and torchvision (for the ImageNet class names)
    cd user_study/edm2-arena && npm ci               # wrangler (node_modules is not committed)
    npx wrangler login                               # once, in a browser
    npx wrangler d1 create edm2-arena                # put the id into wrangler.toml (keep it out of git)
    npm run migrate:remote
    npx wrangler pages project create edm2-arena --production-branch main
    npx wrangler pages secret put OWNER_KEY --project-name edm2-arena
    npx wrangler pages secret put ADMIN_PASSWORD --project-name edm2-arena
    npm run deploy

Local test: `npm run migrate:local`, put `OWNER_KEY=...` and `ADMIN_PASSWORD=...` in `.dev.vars` (gitignored), then
`npm run dev` (`wrangler pages dev`, which takes the D1 binding from `wrangler.toml`; adding `--d1 DB`, as the SD3.5
arena's script does, opens a separate, unmigrated local database).

Read the data: `npx wrangler d1 execute edm2-arena --remote --command "SELECT * FROM rater_summary"`, or
`... "SELECT * FROM pair_counts"`.
