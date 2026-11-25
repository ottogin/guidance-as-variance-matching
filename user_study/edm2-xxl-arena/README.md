# EDM2-XXL ImageNet-512 arena: CFG and autoguidance (static site + D1)

Two independent blind pairwise user studies of guidance for NVIDIA's EDM2-XXL on ImageNet-512, on the skeleton of the
SD3.5-M arena (`user_study/arena`) and of `user_study/edm2-xxl-arena` (EDM2-S). Each round shows one ImageNet class and two
images of it from the same initial noise, from the same study, in random order; the rater picks one or calls a tie.
`CROSS_GROUP = False` in `build.py`, so a round never pairs a CFG image with an autoguidance one, and the leaderboard
fits and centres each study on its own votes.

Arms (`ARMS` and `GROUPS` in `build.py`), each at NVIDIA's FID preset and at our variance-matched constant
(`compute_w.py --global`, n = 1024 ImageNet val images, `runs/<model>/imagenet512-n1024/w.json`):

- Classifier-free guidance, EDM2-XXL <- unconditional EDM2-XS, both at post-hoc EMA 0.015
  (`edm2-img512-xxl-guid-fid`, model key `edm2-512-cfg-xxl-xsu-fid`): `c` NVIDIA w = 1.20, `d` ours w = 5.7267,
  `e` our schedule w(sigma).
- Autoguidance, EDM2-XXL 0.075 <- EDM2-M 0.155 (`edm2-img512-xxl-autog-fid`, model key `edm2-512-ag-xxl-m-fid`):
  `a` NVIDIA w = 2.05, `b` ours w = 9.4460, `f` our schedule w(sigma).

Rows are seeds 0-99 of `sample.py --model <key> --n 100` (`--published`, `--vm global`, `--vm schedule`): EDM2's 32-step Heun
sampler, and each seed fixes both the noise and the class, as in `generate_images.py`, so the two studies (and the
EDM2-S study) show the same class on each row. Renders and the class of each seed are fetched into
`results/edm2-xxl-arena/renders/` (commands in `build.py`'s docstring).

## Pieces

- `build.py` writes `public/index.html` (from `page.template.html`), one sprite per row `public/<SPRITES>/NNN.webp` (every arm
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

Sample the new settings for the same seeds (e.g. another EDM2 preset at NVIDIA's w and
at our constant), put them under `results/edm2-xxl-arena/renders/<code>/`, add them to `ARMS` and `GROUPS` in `build.py`,
and allow their codes in the `votes` CHECK: edit `migrations/0001_init.sql` before the first deploy, or add a migration
that rebuilds `votes` once the database exists (SQLite cannot change a CHECK in place). `build.py` refuses to build
while the newest migration's CHECK and `ARMS` differ. `CROSS_GROUP = False` restricts raters to pairs within a group.

## Build and deploy

    python3 user_study/edm2-xxl-arena/build.py           # needs Pillow and torchvision (for the ImageNet class names)
    cd user_study/edm2-xxl-arena && npm ci               # wrangler (node_modules is not committed)
    npx wrangler login                               # once, in a browser
    npx wrangler d1 create edm2-xxl-arena                # put the id into wrangler.toml (keep it out of git)
    npm run migrate:remote
    npx wrangler pages project create edm2-xxl-arena --production-branch main
    npx wrangler pages secret put OWNER_KEY --project-name edm2-xxl-arena
    npx wrangler pages secret put ADMIN_PASSWORD --project-name edm2-xxl-arena
    npm run deploy

Local test: `npm run migrate:local`, put `OWNER_KEY=...` and `ADMIN_PASSWORD=...` in `.dev.vars` (gitignored), then
`npm run dev` (`wrangler pages dev`, which takes the D1 binding from `wrangler.toml`; adding `--d1 DB`, as the SD3.5
arena's script does, opens a separate, unmigrated local database).

Read the data: `npx wrangler d1 execute edm2-xxl-arena --remote --command "SELECT * FROM rater_summary"`, or
`... "SELECT * FROM pair_counts"`.
