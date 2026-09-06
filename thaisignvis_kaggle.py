import kagglehub

# kagglehub's `path=` must point at a single file, not a directory (that's why the
# original directory path 404'd). Download every known file in that video's folder.
folder = "ThaiSignVis/process_videos/process_videos/(LIVE)_การประชุมวุฒิสภา_ครั้งที่_2_(สมัยสามัญประจำปีครั้งที่สอง)_181266"
files = [
    "detection_results.csv",
    "transcript.json",
    "process_video_0.mp4",
    "transcript_window_0.csv",
]

for name in files:
    downloaded = kagglehub.dataset_download(
        "thanawuttimpitak/thaisignvis/versions/2",
        path=f"{folder}/{name}",
        output_dir=r"D:\TSL\sample",
        force_download=True,
    )
    print("Downloaded:", downloaded)