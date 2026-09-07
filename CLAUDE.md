# CLAUDE.md

Project instructions for Claude Code. Read this file fully before making changes.

Everything in §1, §4 and §5 has been **verified against the actual repo and a
working install**. Do not "improve" it from memory or from a library's README —
if reality disagrees with this file, report the disagreement before changing code.

---

## 1. What this project is

**Goal: temporal segmentation of continuous Thai Sign Language (TSL) video.**

Given a video of a TSL interpreter, produce a list of frame spans:

```
[{start_frame, end_frame, tier}, ...]
```

where `tier` is `sign` or `sentence`. That is the entire deliverable. The output
is **span boundaries, not labels**. Nothing in this project ever names a sign.

### Non-goals — do not implement these

If a task seems to require one of these, **stop and flag it** instead of building it.

| Non-goal | Why it is excluded |
|---|---|
| Sign language **translation** (video → Thai text) | Different problem, different architecture. |
| **Gloss recognition** (video → gloss sequence) | We have zero gloss labels and cannot obtain them. |
| **Gloss mapping / lexicon building** | Requires a TSL gloss inventory that does not exist. |
| **CTC-based sequence models** | CTC needs gloss targets. We have none. |
| Anything using the dataset's transcripts | See below. |
| **Interpreter detection / cropping** | Already done by the dataset. See §5.1. |

### The transcripts are not used, at all

The dataset ships `transcript_window_*.csv` and `detection_results.csv`. **We do
not download them and we do not read them.** Two reasons, both settled:

1. The `text` is a YouTube CC transcript of the **spoken Thai audio**, not of the
   signing. Wrong label space, and the interpreter lags the audio by a
   non-deterministic amount, so it cannot be aligned by subtracting a constant.
2. The timing fields (`start`, `duration`) are known-bad in this dataset.

Do not add code that reads them. Do not propose using them as weak supervision,
as a sanity check, or as a segment-count prior. This is closed.

### The data, as it actually is

Verified by `ffprobe` on 2026-09-06 over all six local videos:

- **6 videos, 6 different signers** — one signer per video, 1:1. There is no
  signer metadata anywhere; `video_id` *is* the signer identity. Do not build a
  separate signer field.
- **All 512×512, all exactly 30/1 fps CFR.** Assert this; do not hardcode it.
- **Already cropped to the interpreter**, letterboxed with black bars and a thin
  white border. `cropdetect` over 2 minutes returns `crop=412:388:50:62` on
  3552/3598 frames — the framing is *static*. There is no box jitter to smooth.
- Framing tightness varies between videos (some clip the head, some are loose).
  This is expected and is left as-is; the segmenter normalizes internally.
- Total: 88,557 frames / ~49 minutes.

| slug | frames | seconds |
|---|---|---|
| `house_1_w0` | 3,852 | 128.4 |
| `senate_15_w1` | 5,856 | 195.2 |
| `house_29_w25` | 7,503 | 250.1 |
| `house2_190766_w1` | 17,181 | 572.7 |
| `house_24_w1` | 25,449 | 848.3 |
| `senate_181266_w0` | 28,716 | 957.2 |

---

## 2. Project tree

> Rewrite this block whenever you change the repo structure. Source and data
> directories only — no generated artifacts, `__pycache__`, or venvs.

```
CLAUDE.md
PHASE1.md
config.yaml                  # video map + run settings (§4)
pyproject.toml               # requires-python = ">=3.12,<3.13"
requirements.lock            # verified version set (§5.2)
thaisignvis_kaggle.py        # dataset download; mp4 only
src/
  pipeline.py                # the whole Phase 1 pipeline (§5)
reports/
  phase1_review.md           # generated skeleton + human answers
outputs/{video_id}/
  {video_id}.pose
  {video_id}.eaf
  segments.json
  excluded_spans.json
sample/ThaiSignVis/process_videos/process_videos/
  {session_folder}/process_video_{N}.mp4      # 6 files, 6 sessions
```

**Note the doubled `process_videos/process_videos/` path segment.** That is real,
not a typo.

---

## 3. Phases

Work strictly in order.

| Phase | Name | Output | Status |
|---|---|---|---|
| **1** | Pretrained zero-shot segmentation | pose → segmenter → ELAN pipeline; human review on 3 videos | See `PHASE1.md` |
| **2** | Manual annotation | 8–10 min human-corrected boundaries in ELAN, spread across videos (= across signers); 2 min double-annotated | Not started |
| **3** | Evaluation harness + classical baselines | Frame accuracy, F1@{0.10,0.25,0.50}; `ruptures`/velocity baselines scored against Phase 2 | Not started |
| **4** | Adaptation *(optional)* | Fine-tune or post-process; only if Phase 3 shows a clear gap | Not started |
| **5** | Packaging | Reproducible CLI: video in → spans + ELAN out | Not started |

Phase 2 is the bottleneck, not Phase 1. Phase 1 exists to bootstrap Phase 2 so a
human *corrects* machine output rather than annotating a blank tier.

---

## 4. Repo conventions

- **Python 3.12.** Not 3.13, not 3.14 — see §5.2, this is a hard dependency
  constraint, not a preference.
- **No hardcoded paths or video filenames.** Everything through `config.yaml` or
  argparse. Session folder names contain Thai script and brackets; they appear
  exactly once, in `config.yaml`.
- **Frames are the unit.** Every span-producing function returns frames. Convert
  to seconds/milliseconds only at an I/O boundary, and always carry `fps` with it.
- **Cache the expensive step.** Pose extraction is the only slow stage
  (~20 fps ⇒ ~75 min for all six videos). If `outputs/{video_id}/{video_id}.pose`
  exists, skip it unless `--force` is passed. Say "cached" in the log.
- **Fail loud.** Assert fps and resolution. Raise on a missing video. Do not
  silently produce an empty `.eaf`.
- Single runnable script. No notebook-only logic.
- No seeding ceremony in Phase 1 — there is no training and inference is
  deterministic. Seeds become relevant in Phase 4.

### Windows constraints (verified the hard way)

Both of these have already bitten and will bite again:

1. **The Bash tool cannot open the dataset's Thai-named paths, even quoted.**
   PowerShell and Python both handle them fine. Use Python for all file access;
   if you need a shell for these paths, use PowerShell.
2. **Set `PYTHONUTF8=1`.** Without it, `pose_to_segments` crashes with
   `UnicodeEncodeError: 'charmap' codec can't encode character '≤'` when it
   prints. Set it in-process at the top of `src/pipeline.py`:
   ```python
   import os, sys
   os.environ.setdefault("PYTHONUTF8", "1")
   ```
   and pass `env` with it set to every `subprocess` call.

---

## 5. Technical spec

### 5.1 Interpreter crop — none

**There is no crop stage.** The dataset's `process_video_*.mp4` files are already
interpreter crops (§1). Feed the 512×512 frames to the pose estimator unchanged.

The letterbox costs nothing: hand-landmark presence measured on raw frames is
**98.7% left / 98.0% right**. Do not add cropping, letterbox trimming, box
interpolation, smoothing, or padding. Do not read `detection_results.csv` — its
coordinates are in original-broadcast space and applying them here would be
actively wrong.

### 5.2 Environment — verified working set

`mediapipe >= 1.0` **removed `mp.solutions` entirely**, which breaks
`pose-format`'s Holistic loader with
`ModuleNotFoundError: No module named 'mediapipe.python'`. The pin is mandatory,
and it is why the project is on Python 3.12 (mediapipe 0.10.x ships no 3.13+
wheels).

```bash
uv venv --python 3.12
uv pip install "mediapipe<0.10.30" pose-format pyyaml
uv pip install "git+https://github.com/sign-language-processing/segmentation"
```

Verified resolved versions, all confirmed working end-to-end on 2026-09-06:

```
mediapipe==0.10.21          # hard ceiling: <0.10.30
numpy==1.26.4               # pulled down by mediapipe; <2
pose-format==0.14.1
pympi-ling==1.71
torch==2.14.0
sign-language-segmentation @ git+https://github.com/sign-language-processing/segmentation@22ca3a6f
```

### 5.3 Pose

- Estimator: **MediaPipe Holistic**, via `pose-format`'s CLI. Required — the
  segmenter expects this exact keypoint layout.
- Command (verified):
  ```bash
  video_to_pose -i <video>.mp4 -o <out>.pose --format mediapipe
  ```
- Do **not** pre-normalize. The segmenter runs its own preprocessing.
- Throughput: ~20 fps on CPU.

**Verified component layout** — 576 points per frame. Use these slices; do not
recompute them:

| component | points | slice |
|---|---|---|
| `POSE_LANDMARKS` | 33 | `[0:33]` |
| `FACE_LANDMARKS` | 468 | `[33:501]` |
| `LEFT_HAND_LANDMARKS` | 21 | `[501:522]` |
| `RIGHT_HAND_LANDMARKS` | 21 | `[522:543]` |
| `POSE_WORLD_LANDMARKS` | 33 | `[543:576]` |

A frame counts as "landmark present" for a component when
`confidence[frame, 0, slice].sum() > 0`.

### 5.4 Segmentation

Verified CLI:

```bash
pose_to_segments --pose <in>.pose --elan <out>.eaf --video <video>.mp4
```

Other real flags: `--model`, `--subtitles`, `--no-pose-link`, `--device`,
`--min_frames` (default 3), `--merge_gap` (default 0), `--save-segments`.
**Run stock in Phase 1** — no threshold tuning, there is no ground truth to tune
against. The model checkpoint ships inside the package; no download step.

The `.eaf` it writes contains three tiers: `default` (empty — ignore it), `SIGN`,
and `SENTENCE`. **Times in the `.eaf` are milliseconds, not frames.**

### 5.5 Segment output format

```python
@dataclass
class Segment:
    start_frame: int
    end_frame: int      # exclusive
    tier: str           # "sign" | "sentence"  (lowercase)
```

There is no `confidence` field. The segmenter writes empty `ANNOTATION_VALUE`
elements — it emits no score. Add the field in Phase 3 if a baseline produces one.

The `.eaf` is the segmenter's output; `segments.json` is **ours**, produced by
parsing that `.eaf` back. The round-trip is deliberate: it proves the `.eaf` is
well-formed, which is a Phase 1 acceptance criterion anyway.

```python
import pympi
eaf = pympi.Elan.Eaf(eaf_path)
for tier in ("SIGN", "SENTENCE"):
    for start_ms, end_ms, _value in eaf.get_annotation_data_for_tier(tier):
        Segment(round(start_ms * fps / 1000),
                round(end_ms   * fps / 1000),
                tier.lower())
```

`fps` comes from the `.pose` file (`Pose.read(...).body.fps`), never from an
assumption.

`segments.json` carries `fps`, `video_id`, and the segment list.

### 5.6 Excluded spans

Frames where MediaPipe found no usable interpreter. Since there is no detection
CSV, this is derived from the pose itself: a frame is **excluded** when
`POSE_LANDMARKS` has no signal (`confidence[f, 0, 0:33].sum() == 0`).

Contiguous excluded frames become spans in `excluded_spans.json`, and
`src/pipeline.py` removes any overlap with them from `segments.json` — a segment
fully inside an excluded span is dropped; one that straddles the edge is
truncated. Downstream consumers must never see a confident boundary over frames
where nothing was detected.

### 5.7 Metrics — Phase 3 only

Frame-level accuracy; segmental F1@{0.10, 0.25, 0.50} IoU. Report Phase 2's
inter-annotator agreement alongside model scores: a model near the human ceiling
is solved, not failing. **Do not compute any metric in Phase 1** — there is no
ground truth, and inventing one is a non-goal.

---

## 6. References

**What Phase 1 runs:**
- `sign-language-processing/segmentation` — pose → ELAN sign & sentence
  segmentation. https://github.com/sign-language-processing/segmentation
- `pose-format` — the `.pose` container and `video_to_pose` CLI.
  https://github.com/sign-language-processing/pose-format

**Why the transcripts are unusable (read once, do not implement):**
- Bull et al., *Aligning Subtitles in Sign Language Videos*, ICCV 2021.
  https://www.robots.ox.ac.uk/~vgg/research/bslalign/

**Phase 3 only:**
- `ruptures` — PELT changepoint detection on wrist velocity.
  https://github.com/deepcharles/ruptures

**Deliberately excluded:** CorrNet+, VAC_CSLR, Uni-Sign, SignCLIP, TwoStream-SLR
— gloss-recognition/translation systems, out of scope per §1.
