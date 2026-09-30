from pathlib import Path

import kagglehub

# kagglehub's `path=` must point at a single file, not a directory (that's why the
# original directory path 404'd). Download every known file in that video's folder.
folder = "ThaiSignVis/process_videos/process_videos"
subfolder = "[LIVE]_ครั้งที่_2_(สมัยสามัญประจำปีครั้งที่หนึ่ง)_190766"
files = "process_video_1.mp4"

for name in files:
    downloaded = kagglehub.dataset_download(
        "thanawuttimpitak/thaisignvis/versions/2",
        path=f"{folder}/{subfolder}/{name}",
        output_dir=str(Path(__file__).resolve().parent / "sample"),  # repo-relative: no hardcoded paths (CLAUDE.md 4)
        force_download=True,
    )
    print("Downloaded:", downloaded)