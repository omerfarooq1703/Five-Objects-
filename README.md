# Video indexing with YOLO11l

For every second of every video in a folder of FineVideo Parquet files, this
script records which of 5 objects (person, car, dog, laptop, cell phone) are
present and how many people are visible.

## Install

System packages (Ubuntu), needed by OpenCV:

```
sudo apt-get install -y libgl1 libglib2.0-0
```

Python packages:

```
pip install -r requirements.txt
```

## Preparation command (run once, needs internet)

Run it from the repository folder. It downloads the official `yolo11l.pt`
weights into the current directory:

```
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

## Run

Run from the same folder, so the script finds `yolo11l.pt`:

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

Optional: `--k` (frames sampled per second, default 4) and `--batch`
(default 32).

## How it works

- Reads each Parquet file one video at a time and decodes it with PyAV.
- Takes the first frame of every quarter-second (4 frames per second) instead of every frame.
- Runs YOLO11l at size 640, confidence 0.4, in FP16, on the five classes.
- Per second: an object is present if it is in at least half of the sampled
  frames; people = median count, rounded down.
- A damaged video gets one `error` line; the run continues.

## Notes and limits

- Sampling 4 frames per second is a speed choice. On 4 test videos it matched
  the full-frame reference with F1 0.989 and exact people count 98.6%.
- `run.py` uses one GPU. My results for all 50 files were made by running two
  copies of it in parallel (one per GPU) on Kaggle, on different files.
- Results were produced on Kaggle with Python 3.13 and `av` 19.0.2. `requirements.txt` pins `av` 18.1.1, because
  19.0.2 needs Python 3.12+. I installed it on Python 3.11 and re-ran 3 videos: 1081 of 1082 lines matched
  `results.jsonl.gz` (one `people` value differed by 1).
- `requirements.txt` pins the CUDA 12.6 build of PyTorch (via the extra index URL on its first line).
- Rotation tags are handled only for the old-style `rotate` tag and are untested.
- `results.jsonl.gz` holds the results for the first 50 files.
