# Samples: published guidance vs ours

Every model in the paper's comparison, sampled with the **released guidance weight (Pub.)** and with **our variance-matched weight (Ours)**, using the same seed for both. Open `index.html` in a browser to see them side by side, or browse the folders.

## Layout

```
samples/
├── index.html          viewer for everything below
├── images/<model>/     NN-<prompt>_pub.jpg + NN-<prompt>_ours.jpg
│   └── sd3.5-medium/   NN-<prompt>/ with the seven user-study settings and the released w
├── videos/<model>/     NN-<prompt>_pub.mp4 + NN-<prompt>_ours.mp4
├── audio/<model>/      NN-<prompt>_pub.mp3 + NN-<prompt>_ours.mp3
└── motion/<model>/     NN-<prompt>_pub.mp4 + NN-<prompt>_ours.mp4
```

Every model folder has a `prompts.md` with the full prompt of each row. Rows 01–10 of every model are its first ten evaluation prompts, **unselected**. The image models have a few more rows after those, added because the paper hand-picked them (App. F, Fig. 1b); ★ in `prompts.md` marks every hand-picked row.

## Models

| Modality | Model | Rows | Guidance | Task | Folder |
|---|---|---|---|---|---|
| images | SD 3.5 Medium | 12 | Pub. w = 4.5 (released) · Ours w = 4.14 (constant) and w(σ) (schedule) | Text-to-image, COCO 2017 captions, 480 px | [`images/sd3.5-medium/`](images/sd3.5-medium/prompts.md) |
| images | SD 3.5 Large | 14 | Pub. w = 3.5 (released) · Ours w(σ), mean 4.88 | Text-to-image, COCO 2017 captions, 480 px | [`images/sd3.5-large/`](images/sd3.5-large/prompts.md) |
| images | SD 1.5 | 14 | Pub. w = 7.5 (released) · Ours w(σ), mean 13.3 | Text-to-image, COCO 2017 captions, 480 px | [`images/sd1.5/`](images/sd1.5/prompts.md) |
| images | FLUX.2 [klein] 4B | 14 | Pub. w = 4 (released) · Ours w(σ), mean 8.11 | Text-to-image, MJHQ-30K prompts, 512 px | [`images/flux.2-klein-4b/`](images/flux.2-klein-4b/prompts.md) |
| videos | Wan2.1 1.3B | 10 | Pub. w = 5 (released) · Ours w(σ), mean 10.8 | Text-to-video, OpenVid prompts (re-encoded to 480 px wide) | [`videos/wan2.1-1.3b/`](videos/wan2.1-1.3b/prompts.md) |
| videos | CogVideoX-2B | 10 | Pub. w = 6 (released) · Ours w(σ), mean 14.9 | Text-to-video, OpenVid prompts (re-encoded to 480 px wide) | [`videos/cogvideox-2b/`](videos/cogvideox-2b/prompts.md) |
| videos | Stable Video Diffusion | 10 | Pub. released frame-wise w from 1 to 3 · Ours w(σ), mean 3.78 | Image-to-video from a UCF-101 first frame, 7 fps | [`videos/stable-video-diffusion/`](videos/stable-video-diffusion/prompts.md) |
| videos | Cosmos-Predict 2.5 | 10 | Pub. w = 1 (released) · Ours w(σ), mean 1.0 | World model: action-conditioned Bridge robot rollouts (320 px) | [`videos/cosmos-predict2.5/`](videos/cosmos-predict2.5/prompts.md) |
| audio | AudioLDM 2 | 10 | Pub. w = 3.5 (released) · Ours w(σ), mean 4.11 | Text-to-audio, AudioCaps test captions, 10.24 s, MP3 | [`audio/audioldm2/`](audio/audioldm2/prompts.md) |
| audio | Stable Audio Open | 10 | Pub. w = 7 (released) · Ours w(σ), mean 18.8 | Text-to-music, Song Describer captions, 47 s, MP3 | [`audio/stable-audio-open/`](audio/stable-audio-open/prompts.md) |
| motion | MDM | 10 | Pub. w = 2.5 (released) · Ours w(σ) | Text-to-motion, KIT-ML, stick-figure renders | [`motion/mdm/`](motion/mdm/prompts.md) |

## Notes

- These are the web encodes: images are JPEG; videos are H.264 re-encoded at 320–512 px; audio is MP3 (AudioLDM 2 mono 96 kb/s, Stable Audio Open 128 kb/s). They are not the full-resolution originals.
- Cosmos: our schedule is w ≈ 1, the released value, so its pairs look the same.
