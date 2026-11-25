# SD 3.5 Medium: Pub. vs Ours

Text-to-image, COCO 2017 captions, 480 px. Same seed for both settings of a row.

Pub. w = 4.5 (released) · Ours w = 4.14 (constant) and w(σ) (schedule)

Eight settings per prompt, one folder per prompt (the user study's images, plus the released w = 4.5 arm):

- `1-pub-released`: Pub. (released), w = 4.5
- `2-ours-constant`: Ours, w = 4.14
- `3-ours-schedule`: Ours, w(σ)
- `4-fid-tuned`: FID-tuned, w = 1.6
- `5-fd-dinov2-tuned`: FD-DINOv2-tuned, w = 2.4
- `6-interval`: Interval, w = 4.5
- `7-linear`: Linear, w = 4.5
- `8-tv-cfg`: TV-CFG, w = 4.5

Rows 01–10 are the first 10 evaluation prompts, unselected. Rows 11–12 are there because the paper hand-picked them (Fig. 1b). ★ marks every row hand-picked for the paper.

| # | Prompt | Files |
|---|---|---|
| 01 | A clock that is in between two windows on a building. | `01-clock-between-windows-building/` |
| 02 | a person standing at a counter getting some bread ready for grilled cheese sandwiches | `02-person-standing-counter-getting-bread/` |
| 03 ★ | A boy sits on a rock in a hat as he holds a trumpet near an old barn and some farm animals. | `03-boy-sits-rock-hat-holds/` |
| 04 | A large brown bear walking across a lush green field. | `04-large-brown-bear-walking-across/` |
| 05 | A delicious looking donut or cinnamon roll covered in icing | `05-delicious-looking-donut-or-cinnamon/` |
| 06 | Two zebras that are standing near each other in the dirt. | `06-zebras-standing-near-each-other/` |
| 07 | a mug of beef sitting next to a plate with a slice of deep dish pizza on it | `07-mug-beef-sitting-next-plate/` |
| 08 | a sign on a wall with a clock in the background | `08-sign-wall-clock-background/` |
| 09 | A man in a blue shirt catches a Frisbee thrown to him. | `09-man-blue-shirt-catches-frisbee/` |
| 10 | A woman holds a basket with a hot dog while she is surrounded by two other people. | `10-woman-holds-basket-hot-dog/` |
| 11 ★ | An old briefcase that has torn and faded material on it, placed on an upper ledge area. | `11-old-briefcase-torn-faded-material/` |
| 12 ★ | A pizza sits on a white plate near a tray with a salad on it. | `12-pizza-sits-white-plate-near/` |
