"""RHD A5：数清 val/test 两个 split 里**原始**（不管有没有发布传感器）的各城市 log 数。

A3 只扫了「已发布传感器」的 372 个 db（因为只有那些被下载了）。本脚本补上分母。

做法：不下整包。zip 中央目录已在 rhd_zip_index.json 里，每个 .db 条目是 deflate 流；
`location` 字段在解压后的第 ~16 KB 处（实测），所以对每个条目只 range-取压缩流的
**前 64 KB**，用 zlib 增量解压能解出的部分，在其中找城市字符串即可。
2,730 个 log × 64 KB ≈ 175 MB，而不是 190 GB。
"""
import json, sys, time, urllib.request, zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

THREADS = 16

META = "/mnt/skylabNAS/nuplan_sg/meta"
RES = "/data/ruolin/uwm/sim2real_demo_ttc/results"
CITIES = ["sg-one-north", "us-ma-boston", "las_vegas", "us-pa-pittsburgh-hazelwood"]
PREFIX = 96 * 1024          # 压缩流前缀字节数


def get_range(url, a, b, tries=4):
    for k in range(tries):
        try:
            rq = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            return urllib.request.urlopen(rq, timeout=60).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def local_header_size(url, lho):
    """本地文件头长度 = 30 + 文件名长 + extra 长（中央目录里的 extra 可能与本地不同）。"""
    h = get_range(url, lho, lho + 29)
    n = int.from_bytes(h[26:28], "little"); e = int.from_bytes(h[28:30], "little")
    return 30 + n + e


def city_of(url, ent):
    off = ent["lho"] + local_header_size(url, ent["lho"])
    raw = get_range(url, off, off + PREFIX - 1)
    d = zlib.decompressobj(-15)                     # raw deflate
    try:
        buf = d.decompress(raw)
    except zlib.error:
        buf = b""
    for c in CITIES:
        if c.encode() in buf:
            return c
    return None


def main():
    idx = json.load(open(f"{META}/rhd_zip_index.json"))
    out = {}
    for split in ("val", "test"):
        z = idx[split]
        url = z["url"]
        sens = set(l.strip() for l in open(f"{META}/public_set_{split}_sensor.txt") if l.strip())
        cnt, cnt_sens, unknown = Counter(), Counter(), []
        dbs = z["dbs"]
        by_log = {}
        print(f"[{split}] {len(dbs)} 个 log，{THREADS} 线程，各取前 {PREFIX//1024} KB", flush=True)

        def one(e):
            stem = e["name"].split("/")[-1][:-3]
            try:
                return stem, city_of(url, e)
            except Exception as ex:                                   # noqa: BLE001
                return stem, f"!{type(ex).__name__}"

        done = 0
        with ThreadPoolExecutor(THREADS) as ex_:
            for stem, c in ex_.map(one, dbs):
                done += 1
                if c is None or str(c).startswith("!"):
                    unknown.append(stem)
                else:
                    by_log[stem] = c
                    cnt[c] += 1
                    if stem in sens:
                        cnt_sens[c] += 1
                if done % 200 == 0:
                    print(f"   {done}/{len(dbs)}  {dict(cnt)}  未知 {len(unknown)}", flush=True)
        out[split] = {"n_logs_total": len(dbs), "by_city_raw": dict(cnt),
                      "by_city_with_sensor": dict(cnt_sens),
                      "n_unknown": len(unknown), "unknown": unknown[:20],
                      "city_by_log": by_log}
        print(f"[{split}] 原始城市分布 {dict(cnt)}  未识别 {len(unknown)}", flush=True)
    json.dump(out, open(f"{RES}/rhd_raw_city_valtest.json", "w"), ensure_ascii=False, indent=1)
    print(f"-> {RES}/rhd_raw_city_valtest.json")


if __name__ == "__main__":
    main()
