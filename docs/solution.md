# Chunking router

The corpus mixes gameplay, first-person footage, podcasts, wildlife, sports, and film. The index needs a different window for each. This program picks the chunker. It does not name the topic.

## Classes

| Class | What the index should do |
|---|---|
| `gameplay` | Kuration gameplay chain. Windows of about 4 seconds. `pixel_shift`, action-state scoring, and collision run only when a key or mouse log is present. Video-MME has no such log, so the windows are kept and those operators are skipped. |
| `egocentric` | Kuration SLAM chain: `slam_reconstruct`, `slam_quality`, `slam_motion_diversity`. Drop broken tracks. |
| `dialogue` | Speech turns, one anchor frame each. Podcasts and talking-head explainers. |
| `shot_scene` | Hard cuts, then group shots into scenes. Film, wildlife packages, broadcast sports. |

If the top probability is below 0.45 the route is `uncertain`: cuts plus speech, and no Kuration operator. Kurious stores the windows (`search_mode="video"`). It is not the classifier, and the benchmark does not call it.

Starter labels come from the Video-MME subcategory (`src/labels.py`). Esports becomes `gameplay`. Daily life, travel, handicraft, and food become `egocentric`. Explainers and talking-head categories become `dialogue`. Edited coverage, including athletics, football, wildlife, and film, becomes `shot_scene`. On this 90-video sample that map yields 50 `shot_scene`, 22 `dialogue`, 15 `egocentric`, and 3 `gameplay` videos. The holdout is 12, 6, 4, and 1 of those. A class with a single video would stay in training; every class here has a holdout video.

A subcategory is only a prior. Geography can be a lecture or a landscape, so the `label` column in `data/labels.csv` should be skimmed before the score is treated as final.

## Features

Every file pays the same decode cost. `probe.py` seeks to 8 frames spread across the file, 8 frames in a burst that starts at 20% of the duration, and one frame per second for the first 30 seconds. Frames are resized to a 96-pixel short side. The one-per-second scan is capped, so an hour-long file does not cost more than a short clip.

`features.py` computes nine probe clues. The router uses four of them, the ones a still-image network cannot see:

1. `global_flow` — how many pixels the whole frame slid. A pan moves every pixel together.
2. `residual_flow` — pixel change left after that slide is undone.
3. `speech_frac` — fraction of the first 15 seconds above room tone.
4. `cut_rate` — share of the one-per-second pairs whose hue histogram jumps.

Beside those, `embed.py` runs **MobileNetV3-Small** with frozen ImageNet weights on the 8 spread frames, averages the 576-number vectors, and `model.py` shrinks that average to 16 numbers with PCA fit on the training videos only. Those 16 numbers are scaled down so they count as one extra clue, not sixteen votes. Logistic regression is the head. The network is not trained.

`detect.py` can also run frozen **YOLO11n** on the same eight frames, kept at a 320-pixel short side so the detector can see people and objects. It adds six counts: people per frame, objects per frame, and whether a frame contains a screen, an animal, or sports gear. The same command scores three heads on one probe: MobileNet, YOLO, and both. YOLO did not beat MobileNet, so the submitted router stays the MobileNet head.

## Metrics

On the holdout, `evaluate` prints:

- Accuracy and macro-F1. Macro-F1 is the headline, because the four classes are not equally common in the 90-video sample.
- A 95% bootstrap interval for macro-F1, from 1000 resamples of the holdout videos. With a few dozen holdout videos the interval is wide, and that is the honest uncertainty.
- Mean routing cost, with the same interval. Cost is 0 for the right chunker and up to 1 for a damaging swap. Predicting `dialogue` for gameplay stores talk and drops the control-conditioned moments. Predicting `egocentric` for an edited film makes SLAM treat every cut as impossible motion.
- The same cost after the 0.45 deferral. An unsure video takes the hybrid chunker (cuts plus speech). That route is cheap for dialogue and edited video, and only a partial miss for gameplay and egocentric footage.
- Precision and recall per class, and the confusion matrix.
- Probe-plus-predict latency at p50 and p95, and the model size in bytes.

The unit tests check the same metrics on four synthetic patterns (a HUD with a noisy center, a full-frame pan, a still frame, and a hard cut). Those tests do not use Video-MME.

On the 90-video sample the starter labels give 67 training videos and 23 holdout videos. The frozen MobileNet clue, kept beside the four structural clues, scores macro-F1 0.501 (95% bootstrap interval 0.315 to 0.633) and mean routing cost 0.209 (0.087 to 0.330). Accuracy is 0.652. The same split with structural clues only, including cut rate, scored macro-F1 0.388 and routing cost 0.400. Dialogue recall is now 1.00 (6 of 6). Shot-scene recall is 0.50 (6 of 12), up from 0.25. Egocentric recall stays 0.75 (3 of 4). The one gameplay holdout is still called `shot_scene`. Deferral at 0.45 sent 8 videos to the hybrid chunker and the deferred cost was 0.200, a small improvement on the raw guess. Probe plus prediction is 6.4 seconds at the median and 8.8 seconds at the 95th percentile. The head and the PCA are 82 KB; MobileNetV3-Small stays frozen and is not stored in that pickle.

The same probe with YOLO11n counts instead of MobileNet scores macro-F1 0.263 and routing cost 0.530. Accuracy is 0.304. Dialogue recall falls to 0.33, and four of the six dialogue videos are called gameplay. Putting the six YOLO counts beside MobileNet also loses ground: macro-F1 0.464 and routing cost 0.222, with dialogue recall 0.83 instead of 1.00. Person counts do load on dialogue, but they do not separate a talking head from a crowd in a match or a street. The submitted model therefore stays structural clues plus MobileNet. The largest weights are appearance directions, with `cut_rate` still negative for gameplay and positive for egocentric, which is what an edited travel vlog looks like under the starter map. This score still measures agreement with that map. A skim of `data/labels.csv` is still the highest-leverage next step.

## Deployment

```
video
  → probe (8 spread frames + 8-frame burst + 30 one-per-second frames + 15 s audio)
  → 4 structural clues + 16 MobileNet numbers
  → logistic regression
  → class and confidence
  → chunker named in chunkers.py
  → windows with video id, start, end, and class
```

One CPU process handles one file. The pickle is loaded once. The classifier runs once per file, not once per chunk. A decode error raises `ProbeError` so the caller can send that file down the `uncertain` hybrid path and keep indexing. No GPU, no Kafka, and no video transformer: the decision is structural, and the 90-video sample cannot train a 3D network.

## Commands

`run.yaml` is what the evaluator runs: install, download `aintropy-ai/video-mme-sample10` (about 11 GB), write labels, train, evaluate, and predict the first video. CPU only. The download is the long step. Probing is a few seconds per file. The YOLO comparison is `python src/main.py compare --videos data/videomme/videos --labels data/labels.csv`. It needs the `ultralytics` package, which is not in `requirements.txt` because that install pulls OpenCV 5 and drops the face detector the probe still uses.


