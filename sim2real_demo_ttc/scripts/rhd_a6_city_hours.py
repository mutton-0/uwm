"""RHD A6：nuPlan 官方**从未公布**的按城市小时数，从 zip 中央目录直接算出来。

依据：nuPlan 的 log 文件名尾部两个数字就是该 log 的起止秒偏移，
`end − start` **等于该 log 的精确时长（秒）**。实测 4 个 db，与
`max(lidar_pc.timestamp) − min(...)` 逐秒吻合（见 report §2.6）。

所以只要拿到各分包的**中央目录**（每包几十 MB，不是 1.2 TB 的包体），
就能把每个 log 的时长加起来 —— 全库精确小时数，无需下载任何 db。

train 是按城市分包的，城市直接由包名给出。
val/test 不按城市分包，城市来自 rhd_a5_raw_city.py 的前缀解压扫描。
"""
import json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhd_a1_zipdir import (S3, content_length, find_central_directory,   # noqa: E402
                           get_range, parse_central_directory)

META = "/mnt/skylabNAS/nuplan_sg/meta"
RES = "/data/ruolin/uwm/sim2real_demo_ttc/results"
CACHE = f"{META}/rhd_train_city_index.json"

TRAIN_ZIPS = {
    "sg-one-north": ["nuplan-v1.1_train_singapore.zip"],
    "us-ma-boston": ["nuplan-v1.1_train_boston.zip"],
    "us-pa-pittsburgh-hazelwood": ["nuplan-v1.1_train_pittsburgh.zip"],
    "las_vegas": [f"nuplan-v1.1_train_vegas_{i}.zip" for i in range(1, 7)],
}


def dur_s(stem):
    """log 时长（秒）= 文件名尾部 end − start。"""
    try:
        a, b = stem.split("_")[-2:]
        d = int(b) - int(a)
        return d if 0 < d < 100000 else None
    except Exception:                                       # noqa: BLE001
        return None


def train_index():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out = {}
    for city, zips in TRAIN_ZIPS.items():
        stems, nbytes = [], 0
        for z in zips:
            url = S3 + z
            size = content_length(url)
            off, csize, n = find_central_directory(url, size)
            ents = parse_central_directory(get_range(url, off, off + csize - 1))
            dbs = [e for e in ents if e["name"].endswith(".db")]
            stems += [e["name"].split("/")[-1][:-3] for e in dbs]
            nbytes += size
            print(f"  {z}: {len(dbs)} 个 log，包体 {size/1e9:.1f} GB", flush=True)
        out[city] = {"stems": sorted(set(stems)), "zip_bytes": nbytes}
        print(f"[{city}] train {len(out[city]['stems'])} 个 log", flush=True)
    json.dump(out, open(CACHE, "w"))
    return out


def main():
    rep = {"basis": "log 时长 = 文件名尾部 end−start（秒），实测与 db 内 lidar_pc "
                    "时间跨度逐秒吻合", "splits": {}}
    tr = train_index()
    sens_tr = set(l.strip() for l in open(f"{META}/train_sensor_logs.txt") if l.strip())

    rows = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0, 0.0]))   # [n,h,n_sens,h_sens]
    for city, d in tr.items():
        for s in d["stems"]:
            x = dur_s(s)
            if x is None:
                continue
            r = rows["train"][city]
            r[0] += 1; r[1] += x / 3600
            if s in sens_tr:
                r[2] += 1; r[3] += x / 3600
    rep["train_zip_bytes"] = {c: d["zip_bytes"] for c, d in tr.items()}

    vt = json.load(open(f"{RES}/rhd_raw_city_valtest.json"))
    idx = json.load(open(f"{META}/rhd_zip_index.json"))
    for split in ("val", "test"):
        city_of = vt[split].get("city_by_log", {})
        sens = set(l.strip() for l in open(f"{META}/public_set_{split}_sensor.txt") if l.strip())
        for e in idx[split]["dbs"]:
            s = e["name"].split("/")[-1][:-3]
            c = city_of.get(s)
            x = dur_s(s)
            if c is None or x is None:
                continue
            r = rows[split][c]
            r[0] += 1; r[1] += x / 3600
            if s in sens:
                r[2] += 1; r[3] += x / 3600

    print(f"\n{'split':<7}{'city':<28}{'log 数':>8}{'小时':>10}"
          f"{'其中有传感器 log':>17}{'小时':>9}{'传感器占比':>10}")
    print("-" * 90)
    tot = defaultdict(lambda: [0, 0.0, 0, 0.0])
    for split in ("train", "val", "test"):
        for c, r in sorted(rows[split].items(), key=lambda t: -t[1][1]):
            print(f"{split:<7}{c:<28}{r[0]:>8}{r[1]:>10.1f}{r[2]:>17}{r[3]:>9.1f}"
                  f"{(r[3]/r[1]*100 if r[1] else 0):>9.1f}%")
            for i in range(4):
                tot[c][i] += r[i]
    print("-" * 90)
    for c, r in sorted(tot.items(), key=lambda t: -t[1][1]):
        print(f"{'全库':<7}{c:<28}{r[0]:>8}{r[1]:>10.1f}{r[2]:>17}{r[3]:>9.1f}"
              f"{(r[3]/r[1]*100 if r[1] else 0):>9.1f}%")
    g = [sum(x[i] for x in tot.values()) for i in range(4)]
    print(f"{'全库':<7}{'合计':<28}{g[0]:>8}{g[1]:>10.1f}{g[2]:>17}{g[3]:>9.1f}"
          f"{(g[3]/g[1]*100 if g[1] else 0):>9.1f}%")

    rep["splits"] = {s: {c: {"n_logs": r[0], "hours": r[1],
                             "n_logs_with_sensor": r[2], "hours_with_sensor": r[3]}
                         for c, r in v.items()} for s, v in rows.items()}
    rep["total_by_city"] = {c: {"n_logs": r[0], "hours": r[1],
                                "n_logs_with_sensor": r[2], "hours_with_sensor": r[3]}
                            for c, r in tot.items()}
    json.dump(rep, open(f"{RES}/rhd_city_hours.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {RES}/rhd_city_hours.json")


if __name__ == "__main__":
    main()
