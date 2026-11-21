import argparse, glob, io, json, os, sys, time
import av
import numpy as np
import pyarrow.parquet as pq
from ultralytics import YOLO

CLASSES = [0, 2, 16, 63, 67]  # COCO ids: person, car, dog, laptop, cell phone
NAMES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}


def get_rotation_cw(stream):
    """Clockwise degrees needed to make the video upright (0 if no rotation tag)."""
    try:
        return int(float(stream.metadata.get("rotate", 0))) % 360
    except Exception:
        return 0


def rotate_cw(img, deg):
    if deg == 90:
        return np.ascontiguousarray(np.rot90(img, k=-1))
    if deg == 180:
        return np.ascontiguousarray(np.rot90(img, k=2))
    if deg == 270:
        return np.ascontiguousarray(np.rot90(img, k=1))
    return img


def run_batch(model, imgs):
    rs = model.predict(imgs, imgsz=640, conf=0.4, classes=CLASSES,
                       quantize=16, verbose=False)
    return [r.boxes.cls.int().tolist() for r in rs]


def process_video(model, data, k, batch):
    """Returns (dets, error). dets = [(time_sec, [class ids in that frame]), ...]."""
    dets, err = [], None
    ts, imgs = [], []
    try:
        c = av.open(io.BytesIO(data))
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        rot = get_rotation_cw(s)
        t0, last_bucket = None, -1
        for f in c.decode(s):
            t = f.time
            if t is None:
                continue
            if t0 is None:
                t0 = t
            t -= t0
            b = int(t * k)              # which 1/k-second slot this frame is in
            if b == last_bucket:        # already took a frame from this slot
                continue
            last_bucket = b
            ts.append(t)
            imgs.append(rotate_cw(f.to_ndarray(format="bgr24"), rot))
            if len(imgs) == batch:
                dets += list(zip(ts, run_batch(model, imgs)))
                ts, imgs = [], []
        if imgs:
            dets += list(zip(ts, run_batch(model, imgs)))
        c.close()
        if not dets:
            err = "no video frames"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:200]
    return dets, err


def aggregate(dets):
    """Per-second results using the brief's rules."""
    by = {}
    for t, cls in dets:
        by.setdefault(int(t), []).append(cls)
    out = []
    for sec in range(max(by) + 1):
        fr = by.get(sec, [])
        if not fr:
            out.append((sec, [], 0))
            continue
        objs = [NAMES[c] for c in CLASSES if sum(c in f for f in fr) * 2 >= len(fr)]
        people = int(np.floor(np.median([f.count(0) for f in fr])))
        out.append((sec, objs, people))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=4, help="frames sampled per second")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    model = YOLO("yolo11l.pt")
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    start = time.time()
    with open(args.output, "w") as out:
        for path in files:
            name = os.path.basename(path)
            tfile = time.time()
            idx = 0
            try:
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                    vid = f"{name}:{idx}"
                    try:
                        data = batch.column("mp4")[0].as_py()
                        dets, err = process_video(model, data, args.k, args.batch)
                    except Exception as e:
                        dets, err = [], f"{type(e).__name__}: {e}"[:200]
                    if dets:
                        for sec, objs, people in aggregate(dets):
                            out.write(json.dumps({"video": vid, "second": sec,
                                                  "objects": objs, "people": people}) + "\n")
                    if err:
                        out.write(json.dumps({"video": vid, "error": err}) + "\n")
                    out.flush()
                    idx += 1
            except Exception as e:  # damaged parquet file
                out.write(json.dumps({"video": f"{name}:{idx}",
                                      "error": f"{type(e).__name__}: {e}"[:200]}) + "\n")
                out.flush()
            print(f"{name}: {idx} videos in {time.time()-tfile:.0f}s", file=sys.stderr)
    print(f"TOTAL {time.time()-start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
