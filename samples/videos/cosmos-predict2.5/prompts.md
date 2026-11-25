# Cosmos-Predict 2.5: Pub. vs Ours

World model: action-conditioned Bridge robot rollouts (320 px). Same seed for both settings of a row.

Pub. w = 1 (released) · Ours w(σ), mean 1.0

Each prompt is a pair: `_pub` (released weight) and `_ours` (variance-matched schedule).

Rows 01–10 are the first 10 evaluation prompts, unselected.

Our schedule stays at w ≈ 1 (1.000–1.008), the released setting, so the pairs are nearly identical. That is the expected outcome here, not a bug.

| # | Prompt | Files |
|---|---|---|
| 01 | Bridge robot episode 40: action-conditioned rollout from the first frame | `01-episode-40_{pub,ours}.mp4` |
| 02 | Bridge robot episode 59: action-conditioned rollout from the first frame | `02-episode-59_{pub,ours}.mp4` |
| 03 | Bridge robot episode 101: action-conditioned rollout from the first frame | `03-episode-101_{pub,ours}.mp4` |
| 04 | Bridge robot episode 137: action-conditioned rollout from the first frame | `04-episode-137_{pub,ours}.mp4` |
| 05 | Bridge robot episode 179: action-conditioned rollout from the first frame | `05-episode-179_{pub,ours}.mp4` |
| 06 | Bridge robot episode 214: action-conditioned rollout from the first frame | `06-episode-214_{pub,ours}.mp4` |
| 07 | Bridge robot episode 261: action-conditioned rollout from the first frame | `07-episode-261_{pub,ours}.mp4` |
| 08 | Bridge robot episode 262: action-conditioned rollout from the first frame | `08-episode-262_{pub,ours}.mp4` |
| 09 | Bridge robot episode 322: action-conditioned rollout from the first frame | `09-episode-322_{pub,ours}.mp4` |
| 10 | Bridge robot episode 427: action-conditioned rollout from the first frame | `10-episode-427_{pub,ours}.mp4` |
