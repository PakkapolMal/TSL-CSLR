# TSL Segmentation Viewer

Temporal segmentation of continuous **Thai Sign Language** video: given a video of a
TSL interpreter, produce a list of frame spans marking where each *sign* and each
*sentence* begins and ends.

The output is **boundaries only**. Nothing here names or translates a sign.

This repo ships the segmenter's results for four videos plus a browser viewer that
plays each video with a pose overlay and a scrubable sign/sentence timeline, so you
can judge the boundaries by eye.

---

## Quickstart

```bash
docker compose up
```

Then open <http://localhost:8765>.

Everything the viewer needs is already in the clone — the videos included. No dataset
download, no Kaggle account, no credentials.

Stop with `Ctrl+C`, or `docker compose down`.

## Running without Docker

Needs **Python 3.12**.

```bash
pip install -r requirements.txt
python -m src.viewer
```

It serves on `http://127.0.0.1:8765` and opens your browser. Run it from the repo
root. Useful flags: `--port N`, `--video <video_id>` to preselect one, `--no-open` to
skip the browser launch.

## Using the viewer

| Control | What it does |
|---|---|
| video picker (top left) | switch between the four videos |
| **merged** checkbox | show `segments_merged.json` (short gaps between signs closed up) instead of the raw `segments.json` |
| **Continuous** / **Per-sign** | per-sign mode auto-pauses at the end of each sign |
| **prev / next sign** | jump to the neighbouring sign and play it |
| **-1 / +1 frame**, arrow keys | step one frame |
| click a track | seek there |
| click a segment | show its exact frame bounds |

The overlay draws the MediaPipe skeleton: blue = body pose, yellow = left hand,
red = right hand. On the timeline, blue = sign, orange = sentence, and hatched =
excluded (frames where no interpreter was detected at all).

## What's in the repo

Four videos live under `sample/ThaiSignVis/process_videos/process_videos/`. The
doubled `process_videos/process_videos/` segment is real, not a typo, and the folder
names are Thai — leave both exactly as they are or `config.yaml` will stop resolving.

Each video is 512×512, 30 fps, already cropped to the interpreter by the dataset.

`config.yaml` maps a short `video_id` to each path. Per video, `outputs/{video_id}/`
holds:

| file | what it is |
|---|---|
| `segments.json` | the deliverable — see below |
| `segments_merged.json` | same, with sign segments closer than 5 frames merged (review aid only) |
| `excluded_spans.json` | frame runs where no body pose was detected |
| `{video_id}.eaf` | the same annotations as an [ELAN](https://archive.mpi.nl/tla/elan) file |
| `viewer_pose.bin` | precomputed keypoints the viewer's overlay draws |

`segments.json`:

```json
{
  "video_id": "house_29_w25",
  "fps": 30.0,
  "segments": [
    {"start_frame": 72, "end_frame": 76, "tier": "sign"},
    {"start_frame": 81, "end_frame": 86, "tier": "sign"},
    {"start_frame": 72, "end_frame": 326, "tier": "sentence"}
  ]
}
```

`end_frame` is **exclusive**. `tier` is `"sign"` or `"sentence"`. Frames are the unit
throughout; `fps` is carried alongside so you can convert.

The intermediate `.pose` files are not shipped — they are ~780 MB and only needed to
re-run the pose extraction and segmentation pipeline, which this repo does not
package.

## Troubleshooting

**Port 8765 already in use** — `python -m src.viewer --port 9000`, or change the left
half of `"8765:8765"` in `docker-compose.yaml`.

**`No videos have a segments.json yet`** — you are not in the repo root. `cd` there
and rerun.

**Video area is black but the timeline has segments** — the mp4 under `sample/` is
missing. `find sample -name "*.mp4" | wc -l` should print 4, totalling ~116 MB.
