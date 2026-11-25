# Stable Video Diffusion: Pub. vs Ours

Image-to-video from a UCF-101 first frame, 7 fps. Same seed for both settings of a row.

Pub. released frame-wise w from 1 to 3 · Ours w(σ), mean 3.78

Each prompt is a pair: `_pub` (released weight) and `_ours` (variance-matched schedule).

Rows 01–10 are the first 10 evaluation prompts, unselected.

Pub. is SVD's released frame-wise ramp, w from 1 to 3 across frames.

| # | Prompt | Files |
|---|---|---|
| 01 | First frame of a UCF-101 Playing Flute clip | `01-playing-flute_{pub,ours}.mp4` |
| 02 | First frame of a UCF-101 Jumping Jack clip | `02-jumping-jack_{pub,ours}.mp4` |
| 03 | First frame of a UCF-101 Parallel Bars clip | `03-parallel-bars_{pub,ours}.mp4` |
| 04 | First frame of a UCF-101 Javelin Throw clip | `04-javelin-throw_{pub,ours}.mp4` |
| 05 | First frame of a UCF-101 Swing clip | `05-swing_{pub,ours}.mp4` |
| 06 | First frame of a UCF-101 Baseball Pitch clip | `06-baseball-pitch_{pub,ours}.mp4` |
| 07 | First frame of a UCF-101 Fencing clip | `07-fencing_{pub,ours}.mp4` |
| 08 | First frame of a UCF-101 Playing Daf clip | `08-playing-daf_{pub,ours}.mp4` |
| 09 | First frame of a UCF-101 Body Weight Squats clip | `09-body-weight-squats_{pub,ours}.mp4` |
| 10 | First frame of a UCF-101 Head Massage clip | `10-head-massage_{pub,ours}.mp4` |
