"""RHD step A4: does the *train* split offer any Singapore increment?

The handoff lists "train has no increment" as a confirmed fact, on the grounds that all
1,085 train logs with sensors are already inside the mined OpenScene trainval. Since we
are now being asked to use train data, this checks the claim rather than assuming it —
and it is nearly free, because train IS city-split: the central directory of
nuplan-v1.1_train_singapore.zip enumerates exactly the Singapore train logs.

Writes only under /mnt/skylabNAS/ and results/rhd_*.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhd_a1_zipdir import S3, content_length, find_central_directory, get_range, parse_central_directory

META = "/mnt/skylabNAS/nuplan_sg/meta"
ZIP = "nuplan-v1.1_train_singapore.zip"


def main():
    url = S3 + ZIP
    size = content_length(url)
    print(f"{ZIP}  {size/1e9:.2f} GB", flush=True)
    off, csize, n = find_central_directory(url, size)
    print(f"central dir at {off} ({csize/1e6:.1f} MB, {n} entries)", flush=True)
    ents = parse_central_directory(get_range(url, off, off + csize - 1))
    dbs = [e for e in ents if e["name"].endswith(".db")]
    sg_logs = sorted(e["name"].split("/")[-1][:-3] for e in dbs)
    print(f"Singapore TRAIN logs (all, sensors or not): {len(sg_logs)}")

    sensor = set(l.strip() for l in open(f"{META}/train_sensor_logs.txt") if l.strip())
    print(f"train logs with sensors published (all cities): {len(sensor)}")

    sg_with_sensor = sorted(set(sg_logs) & sensor)
    print(f"Singapore train logs WITH sensors: {len(sg_with_sensor)}")

    by = {e["name"].split("/")[-1][:-3]: e for e in dbs}
    need = sum(by[l]["csize"] for l in sg_with_sensor)
    print(f"bytes to fetch for those dbs: {need/1e9:.2f} GB")

    out = dict(zip_name=ZIP, n_singapore_train_logs=len(sg_logs),
               n_train_logs_with_sensor_all_cities=len(sensor),
               n_singapore_train_logs_with_sensor=len(sg_with_sensor),
               fetch_bytes=need,
               singapore_train_logs_with_sensor=sg_with_sensor,
               all_singapore_train_logs=sg_logs)
    with open(f"{META}/rhd_train_sg_index.json", "w") as f:
        json.dump(dict(entries={k: v for k, v in by.items()}, summary=out), f)
    p = "/data/ruolin/uwm/sim2real_demo_ttc/results/rhd_train_sg_summary.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
