# Hardware

All 50 files were processed on Kaggle notebooks (free tier), in three
"Save & Run All" runs: files 0-9, files 10-29 and files 30-49.

- GPU: 2x NVIDIA Tesla T4, 15 GB memory each (one worker process per GPU)
- CPU: Intel Xeon @ 2.20 GHz, 4 cores
- RAM: 31 GB
- Disk: SSD (reported as non-rotational by the system), downloaded data kept on local disk
- Operating system: Ubuntu 24.04.5 LTS, Python 3.13

## Timing

- Inference time (two workers in parallel): 2,484 s + 4,913 s + 4,622 s = 12,019 s (about 3.3 hours)
- Full run from start to finish, including package install, downloading the
  Parquet files and notebook start-up: about 12,280 s (about 3.4 hours)
- Settings: 4 frames sampled per second, batch size 32, FP16
- Output: 461,757 lines, 1,650 videos, 0 error lines
