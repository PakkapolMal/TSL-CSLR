# PHASE 1 — Pretrained Zero-Shot Segmentation

Read `CLAUDE.md` first. This phase inherits every non-goal in §1 of that file.

---

## Objective

Take a `process_video_*.mp4`, extract MediaPipe Holistic pose, run the pretrained
segmenter zero-shot, and write an ELAN file a human can open and correct.

**This phase produces no trained model and no metrics.** There is no ground truth
yet, so nothing can be scored. Success = a working pipeline plus an informed
human judgement about whether the pretrained boundaries transfer to TSL at all.

The point is to **bootstrap Phase 2**: a human corrects machine output instead of
annotating a blank tier.

---

## Out of scope for Phase 1

- Training, fine-tuning, loss functions.
- Any metric computation, and any attempt to manufacture ground truth.
- Cropping, letterbox trimming, bounding-box work of any kind (`CLAUDE.md` §5.1).
- Reading `detection_results.csv` or `transcript_window_*.csv` (`CLAUDE.md` §1).
- Pose normalization (the segmenter does its own).
- Classical baselines (`ruptures`, velocity thresholding) — Phase 3.
- Threshold tuning on `--min_frames` / `--merge_gap`. Run stock.

---

## Step 1 — Environment

```bash
uv venv --python 3.12
uv pip install "mediapipe<0.10.30" pose-format pyyaml
uv pip install "git+https://github.com/sign-language-processing/segmentation"
uv pip freeze > requirements.lock
```

Set `requires-python = ">=3.12,<3.13"` in `pyproject.toml` (it currently says
`>=3.14`, which cannot work — see `CLAUDE.md` §5.2).

**Smoke test before writing any pipeline code.** Cut a 5-second clip to an
ASCII path and run the whole chain on it:

```bash
ffmpeg -y -ss 20 -t 5 -i <any video> -c:v libx264 -an clip5s.mp4
video_to_pose -i clip5s.mp4 -o clip5s.pose --format mediapipe
pose_to_segments --pose clip5s.pose --elan clip5s.eaf --video clip5s.mp4
```

Expected: `Found N signs, M sentences` and a ~4 KB `.eaf`. On the reference clip
this was 9 signs / 1 sentence in ~9 seconds.

**Gate (machine):** the three commands succeed. If `video_to_pose` raises
`ModuleNotFoundError: No module named 'mediapipe.python'`, the mediapipe pin
did not take — fix that before continuing. Do not start Step 2 until this passes.

---

## Step 2 — `config.yaml`

Write it by hand. This is the only place a dataset path appears.

```yaml
data_root: sample/ThaiSignVis/process_videos/process_videos
output_root: outputs

# Default run set: the three shortest videos (~17.2k frames, ~15 min of pose
# extraction). Three distinct signers, which satisfies the review requirement.
# To run everything, pass --all or extend this list.
run: [house_1_w0, senate_15_w1, house_29_w25]

videos:
  house_1_w0:
    path: "LIVE_การประชุมสภาผู้แทนราษฎร_ครั้งที่_1_(สมัยวิสามัญ)_เป็นพิเศษวันที่_18_มิถุนายน_พ.ศ._2567/process_video_0.mp4"
  senate_15_w1:
    path: "Live_การประชุมวุฒิสภา_ครั้งที่_15_(สมัยสามัญประจำปีครั้งที่สอง)/process_video_1.mp4"
  house_29_w25:
    path: "[LIVE]_การประชุมสภาผู้แทนราษฎร_ครั้งที่_29_(สมัยสามัญประจำปีครั้งที่สอง)_เป็นพิเศษ_22_มี.ค._67/process_video_25.mp4"
  house2_190766_w1:
    path: "[LIVE]_ครั้งที่_2_(สมัยสามัญประจำปีครั้งที่หนึ่ง)_190766/process_video_1.mp4"
  house_24_w1:
    path: "(Live)_การประชุมสภาผู้แทนราษฎรครั้งที่_24_(สมัยสามัญประจำปีครั้งที่หนึ่ง)/process_video_1.mp4"
  senate_181266_w0:
    path: "(LIVE)_การประชุมวุฒิสภา_ครั้งที่_2_(สมัยสามัญประจำปีครั้งที่สอง)_181266/process_video_0.mp4"
```

Read it with `yaml.safe_load`, open it as UTF-8, and join paths with `pathlib`.
There is no `signer` field: one video = one signer, so `video_id` is the signer
identity (`CLAUDE.md` §1).

---

## Step 3 — `src/pipeline.py`

**One file.** Five functions and an argparse `__main__`. Do not split it into
modules; there is no second implementation of any stage.

```
python -m src.pipeline --video <video_id>     # one video
python -m src.pipeline --all                  # every entry in videos:
python -m src.pipeline --video <id> --force   # ignore the pose cache
```

### 3a. `probe(video_path) -> (fps, width, height, n_frames)`

`ffprobe -v error -select_streams v:0 -show_entries stream=... -of json`.
Assert `fps == 30.0` and `width == height == 512`; raise with the actual values
if not. Never assume fps anywhere else — carry the probed value through.

### 3b. `extract_pose(video_path, out_pose, force) -> Path`

Shell out to `video_to_pose -i <video> -o <out> --format mediapipe`.

Skip and log `cached` if `out_pose` exists and `force` is False. This is the only
expensive stage (~20 fps; ~3 min for the shortest video, ~24 min for the longest).

### 3c. `landmark_report(pose) -> dict`

Load with `pose_format.Pose.read`. Using the verified slices in `CLAUDE.md` §5.3,
compute the fraction of frames with any signal for `LEFT_HAND_LANDMARKS`,
`RIGHT_HAND_LANDMARKS`, and `POSE_LANDMARKS`. Log all three.

**Gate (machine, soft):** log the rates; **raise only if either hand is below
60%**. Reference measurement on raw 512×512 frames was 98.7% / 98.0%, so a
figure in the 90s is normal and needs no action. Do not "fix" a low rate by
adding a crop stage — report it and stop.

### 3d. `excluded_spans(pose) -> list[(start_frame, end_frame)]`

Contiguous runs of frames where `confidence[f, 0, 0:33].sum() == 0`
(`POSE_LANDMARKS` absent). Write `excluded_spans.json` with the spans and `fps`.
Expect this to be empty on most videos — that is a valid result, not a bug.

### 3e. `segment(pose_path, eaf_path, video_path) -> list[Segment]`

Run `pose_to_segments --pose ... --elan ... --video ...`, then parse the `.eaf`
back with `pympi` per `CLAUDE.md` §5.5. Rules:

- Read tiers `SIGN` and `SENTENCE` only. **Ignore the `default` tier** — it is
  empty and present in every file.
- `.eaf` times are **milliseconds**. Convert with the `fps` read from the
  `.pose`, not from the config or a constant.
- Lowercase the tier name into `Segment.tier`.
- Subtract `excluded_spans`: drop segments fully inside one, truncate segments
  that straddle an edge.
- Write `segments.json` with `video_id`, `fps`, and the segments.

Pass `env` with `PYTHONUTF8=1` to both subprocess calls (`CLAUDE.md` §4).

---

## Step 4 — `reports/phase1_review.md` (generated skeleton)

After a run, the pipeline writes this file with **every computable number
filled in** and the qualitative answers left blank for a human.

Pre-fill, per video: frames, duration, hand-landmark rates, excluded-span count
and total excluded frames, sign count, sentence count, **implied signs/second**,
mean and median sign duration, and the **coefficient of variation of sign
duration**.

That last number is the constant-rate check: a segmenter that is not tracking
content behaves like a metronome, and a very low CV is the signature. Compute it,
don't ask a human to eyeball it.

Then these headers, left empty:

| Question | Why it matters |
|---|---|
| Do sentence boundaries land on visible pauses / hand drops / body shifts? | If yes, Phase 2 has a viable fallback tier. |
| Are sign boundaries plausible, or is it splitting at a near-constant rate? | Cross-check against the CV computed above. |
| Does behaviour differ noticeably between videos (= between signers)? | Predicts how hard signer-independence will be. |
| Where does it fail — fingerspelling, classifiers, fast sequences? | Directly informs the Phase 2 annotation convention. |

Reference band for signs/second is roughly **1.5–2.5**. The 5-second smoke test
came in at 1.8. Far outside that band signals systematic over/under-segmentation.

---

## Step 5 — Hand off to a human

The pipeline **cannot** finish Phase 1 by itself. End the run by printing:

```
READY FOR HUMAN REVIEW
  Open in ELAN:  outputs/<video_id>/<video_id>.eaf   (video auto-links)
  Then fill in:  reports/phase1_review.md
  Videos to inspect: <the three run ids>, ~2 minutes each
```

Then **stop**. Do not self-assess boundary quality from extracted stills, and do
not write answers into the qualitative sections. An agent cannot open ELAN; a
confident "the boundaries look good" from one is worse than a blank.

---

## Deliverables

```
config.yaml
pyproject.toml                          # requires-python pinned to 3.12
requirements.lock
src/pipeline.py
reports/phase1_review.md                # skeleton + computed numbers
outputs/{video_id}/{video_id}.pose
outputs/{video_id}/{video_id}.eaf
outputs/{video_id}/segments.json
outputs/{video_id}/excluded_spans.json
```

Plus `CLAUDE.md` §2 updated if the structure moved.

---

## Acceptance criteria

### Machine gates — the agent must pass all of these

1. Step 1's smoke test passes on a 5-second clip.
2. `python -m src.pipeline --video <id>` runs end-to-end, mp4 → `.eaf`, with no
   manual intervention, for all three default videos.
3. `probe()` asserts pass: 30 fps, 512×512.
4. Hand-landmark rates are logged per video and both hands are ≥60%.
5. `segments.json` round-trips from the `.eaf`: both tiers present, non-empty,
   frame indices within `[0, n_frames]`, `end_frame > start_frame`.
6. No segment overlaps an excluded span.
7. No hardcoded video filename or dataset path anywhere in `src/`.
8. Re-running without `--force` logs `cached` and does not re-extract pose.
9. `reports/phase1_review.md` exists with all numeric fields populated.

### Human gates — the agent produces the artifact and stops

10. The `.eaf` opens in ELAN with `SIGN` and `SENTENCE` populated and the video
    linked.
11. A human answers the four qualitative questions in `reports/phase1_review.md`.
12. A human judges whether sign-level output is usable, sentence-level only, or
    neither.

**"Sign-level is unusable, sentence-level is decent" is a valid and useful Phase 1
result.** It reshapes Phase 2 rather than blocking it. Say so plainly if that is
what the output shows.

---

## Known pitfalls

- **Installing current mediapipe.** `>=1.0` removed `mp.solutions` and breaks
  `pose-format`. The `<0.10.30` pin is mandatory and forces Python 3.12.
- **Forgetting `PYTHONUTF8=1`.** `pose_to_segments` crashes on a `≤` in its own
  help text under the Windows cp1252 console.
- **Using the Bash tool on the dataset paths.** It cannot open the Thai-named
  folders even when quoted. Use Python or PowerShell.
- **Treating `.eaf` times as frames.** They are milliseconds.
- **Reading the `default` tier.** It is always empty; parsing it yields nothing
  and may look like a failure.
- **Assuming 30 fps instead of probing.** All six happen to be 30, but a wrong
  fps silently corrupts every frame↔time conversion in the project.
- **Adding a crop stage** because hand detection looks low. Cropping is out of
  scope; report the number instead (`CLAUDE.md` §5.1).
- **Tuning `--min_frames` or `--merge_gap`.** There is no ground truth. Any
  tuning now is fitting to your own eyeballs.
- **Scope creep into naming signs.** If a task starts to require a gloss, it
  belongs to a phase that does not exist. Stop and flag it.
