"""Phase 1 review viewer: browser UI to scrub video + pose overlay + segment timeline.

Local alternative/supplement to opening the .eaf in ELAN.

python -m src.viewer                  # serve every video with a segments.json
python -m src.viewer --video <id>     # preselect one video
python -m src.viewer --port 8765
python -m src.viewer --force          # recompute the cached pose_viewer.bin
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import http.server
import json
import re
import struct
import webbrowser
from pathlib import Path

import numpy as np

from src.pipeline import REPO_ROOT, load_config

# Only what the viewer draws: POSE (33) + LEFT_HAND (21) + RIGHT_HAND (21), per CLAUDE.md §5.3 slices.
POSE_SLICE = slice(0, 33)
LEFT_HAND_SLICE = slice(501, 522)
RIGHT_HAND_SLICE = slice(522, 543)
VIEWER_POINTS = 33 + 21 + 21


def build_pose_binary(pose_path: Path, out_path: Path, force: bool) -> Path:
    """Cache a compact [n_frames, VIEWER_POINTS, 2] float32 blob (normalized xy) for the browser."""
    if out_path.exists() and not force:
        return out_path
    from pose_format import Pose  # deferred: only the viewer needs this, and it's slow to import

    pose = Pose.read(pose_path.read_bytes())
    data = pose.body.data  # masked array [frames, people, points, xyz]
    width = pose.header.dimensions.width
    height = pose.header.dimensions.height
    xy = np.ma.concatenate(
        [data[:, 0, POSE_SLICE, :2], data[:, 0, LEFT_HAND_SLICE, :2], data[:, 0, RIGHT_HAND_SLICE, :2]],
        axis=1,
    ).filled(0).astype(np.float32)
    xy[:, :, 0] /= width
    xy[:, :, 1] /= height
    header = struct.pack("<II", xy.shape[0], VIEWER_POINTS)
    out_path.write_bytes(header + xy.tobytes())
    print(f"  [viewer] cached pose binary: {out_path}")
    return out_path


def reviewable_videos(config: dict) -> dict:
    """video_id -> fps, for every video that has a segments.json (i.e. passed the pipeline)."""
    output_root = REPO_ROOT / config["output_root"]
    out = {}
    for video_id in config["videos"]:
        seg_path = output_root / video_id / "segments.json"
        if seg_path.exists():
            out[video_id] = json.loads(seg_path.read_text(encoding="utf-8"))["fps"]
    return out


def make_handler(config: dict, force: bool):
    output_root = REPO_ROOT / config["output_root"]
    data_root = REPO_ROOT / config["data_root"]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # quiet; this is a local review tool, not a real server

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str):
            if not path.exists():
                self.send_error(404)
                return
            file_size = path.stat().st_size
            range_header = self.headers.get("Range")
            start, end = 0, file_size - 1
            status = 200
            if range_header:
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    status = 206
            length = end - start + 1
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass  # client seeked / navigated away mid-stream

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path == "/":
                self._send_file(Path(__file__).with_name("viewer_index.html"), "text/html; charset=utf-8")
                return

            if path == "/api/videos":
                self._send_json(reviewable_videos(config))
                return

            m = re.match(r"^/video/([^/]+)$", path)
            if m:
                video_id = m.group(1)
                video_cfg = config["videos"].get(video_id)
                if not video_cfg:
                    self.send_error(404)
                    return
                self._send_file(data_root / video_cfg["path"], "video/mp4")
                return

            m = re.match(r"^/data/([^/]+)/segments\.json$", path)
            if m:
                self._send_file(output_root / m.group(1) / "segments.json", "application/json")
                return

            m = re.match(r"^/data/([^/]+)/segments_merged\.json$", path)
            if m:
                self._send_file(output_root / m.group(1) / "segments_merged.json", "application/json")
                return

            m = re.match(r"^/data/([^/]+)/excluded_spans\.json$", path)
            if m:
                self._send_file(output_root / m.group(1) / "excluded_spans.json", "application/json")
                return

            m = re.match(r"^/data/([^/]+)/pose\.bin$", path)
            if m:
                video_id = m.group(1)
                pose_path = output_root / video_id / f"{video_id}.pose"
                bin_path = output_root / video_id / "viewer_pose.bin"
                if not pose_path.exists():
                    self.send_error(404)
                    return
                build_pose_binary(pose_path, bin_path, force)
                self._send_file(bin_path, "application/octet-stream")
                return

            self.send_error(404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 review viewer")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--video", help="preselect a video_id")
    parser.add_argument("--force", action="store_true", help="recompute cached pose_viewer.bin")
    parser.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    args = parser.parse_args()

    config = load_config(Path(args.config))
    videos = reviewable_videos(config)
    if not videos:
        raise SystemExit("No videos have a segments.json yet -- run `python -m src.pipeline` first.")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(config, args.force))
    url = f"http://127.0.0.1:{args.port}/"
    if args.video:
        url += f"?video={args.video}"
    print(f"Serving {len(videos)} video(s) at {url}")
    print(f"  {', '.join(videos)}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
