"""RHD step A2: range-fetch only the log .db files that actually have sensors published.

val+test hold 2,730 logs (193 GB of zip), but only 372 of them have camera blobs
released, and those are the only logs that can ever yield a usable event. Pulling just
those entries out of the two zips by byte range costs ~17.8 GB instead of 193 GB.

Writes only under /mnt/skylabNAS/ per the handoff's conflict boundary.
"""
import concurrent.futures as cf
import json, os, struct, sys, time, urllib.request, zlib

META = "/mnt/skylabNAS/nuplan_sg/meta"
DEST = "/mnt/skylabNAS/nuplan_sg/dbs"


def get_range(url, start, end, tries=5):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for a in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(2 * (a + 1))


def fetch_one(args):
    url, ent, out_path = args
    if os.path.exists(out_path) and os.path.getsize(out_path) == ent["usize"]:
        return 0, out_path            # already complete
    # the local file header repeats the name/extra with its own lengths
    hdr = get_range(url, ent["lho"], ent["lho"] + 29)
    if hdr[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {ent['name']}")
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    data_off = ent["lho"] + 30 + nlen + elen
    raw = get_range(url, data_off, data_off + ent["csize"] - 1)
    blob = zlib.decompress(raw, -15) if ent["method"] == 8 else raw
    if len(blob) != ent["usize"]:
        raise RuntimeError(f"size mismatch {ent['name']}: {len(blob)} != {ent['usize']}")
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, out_path)
    return ent["csize"], out_path


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    idx = json.load(open(f"{META}/rhd_zip_index.json"))
    jobs, total = [], 0
    for split in ("val", "test"):
        os.makedirs(f"{DEST}/{split}", exist_ok=True)
        lines = [l.strip() for l in open(f"{META}/public_set_{split}_sensor.txt") if l.strip()]
        want = {l for l in lines if not l.lower().startswith("file group")}
        by = {e["name"].split("/")[-1][:-3]: e for e in idx[split]["dbs"]}
        miss = want - set(by)
        if miss:
            print(f"[{split}] WARNING {len(miss)} manifest logs absent from zip", flush=True)
        for name in sorted(want & set(by)):
            e = by[name]
            jobs.append((idx[split]["url"], e, f"{DEST}/{split}/{name}.db"))
            total += e["csize"]
    print(f"{len(jobs)} logs to fetch, {total/1e9:.2f} GB compressed", flush=True)

    done_bytes, t0, n = 0, time.time(), 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for got, path in ex.map(fetch_one, jobs):
            done_bytes += got; n += 1
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {done_bytes/1e9:.2f}/{total/1e9:.2f} GB  "
                      f"{done_bytes/1e6/max(el,1):.1f} MB/s  ({el:.0f}s)", flush=True)
    print("DONE fetching db files")


if __name__ == "__main__":
    main()
