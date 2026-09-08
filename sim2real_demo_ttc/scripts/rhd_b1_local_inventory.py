"""RHD: inventory of the *local* NavSim/OpenScene scene library, by city and split.

Answers "how much is actually in the dataset we already have on disk", which is the
number that matters for any further mining: nothing needs downloading if the answer is
already local. Counts distinct scene_token (NavSim's scene unit) and frames, grouped by
map_location, from the per-log .pkl metadata.

Read-only on /data/dataset/navsim/** (that path belongs to the other agent).
Writes only results/rhd_*.
"""
import collections, glob, json, os, pickle, sys
import multiprocessing as mp

NAV = "/data/dataset/navsim/dataset/navsim_logs"
RES = "/data/ruolin/uwm/sim2real_demo_ttc/results"


def scan(path):
    try:
        with open(path, "rb") as f:
            frames = pickle.load(f)
        if not frames:
            return dict(log=os.path.basename(path)[:-4], ok=True, city=None,
                        scenes=0, frames=0)
        cities = collections.Counter(fr.get("map_location") for fr in frames)
        scenes = {fr.get("scene_token") for fr in frames}
        return dict(log=os.path.basename(path)[:-4], ok=True,
                    city=cities.most_common(1)[0][0],
                    n_city=len(cities), scenes=len(scenes), frames=len(frames))
    except Exception as e:
        return dict(log=os.path.basename(path)[:-4], ok=False,
                    err=f"{type(e).__name__}: {e}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    out, per_log = {}, []
    for split in ("trainval", "test"):
        files = sorted(glob.glob(f"{NAV}/{split}/*.pkl"))
        print(f"[{split}] {len(files)} logs", flush=True)
        with mp.Pool(workers) as p:
            recs = p.map(scan, files, chunksize=4)
        for r in recs:
            r["split"] = split
        per_log += recs
        ok = [r for r in recs if r["ok"]]
        bad = [r for r in recs if not r["ok"]]
        by = collections.defaultdict(lambda: dict(logs=0, scenes=0, frames=0))
        for r in ok:
            b = by[r["city"]]
            b["logs"] += 1; b["scenes"] += r["scenes"]; b["frames"] += r["frames"]
        out[split] = dict(n_logs=len(files), n_ok=len(ok), n_failed=len(bad),
                          by_city={k: v for k, v in by.items()})
        print(f"[{split}] ok={len(ok)} failed={len(bad)}", flush=True)
        for r in bad[:3]:
            print("    FAILED", r["log"], r.get("err"), flush=True)

    print(f"\n{'split':<10}{'city':<30}{'logs':>7}{'scenes':>10}{'frames':>12}")
    tot_s = tot_f = tot_l = 0
    for split in ("trainval", "test"):
        for c, v in sorted(out[split]["by_city"].items(), key=lambda kv: -kv[1]["scenes"]):
            print(f"{split:<10}{str(c):<30}{v['logs']:>7}{v['scenes']:>10}{v['frames']:>12}")
            tot_s += v["scenes"]; tot_f += v["frames"]; tot_l += v["logs"]
    print(f"{'TOTAL':<40}{tot_l:>7}{tot_s:>10}{tot_f:>12}")

    merged = collections.defaultdict(lambda: dict(logs=0, scenes=0, frames=0))
    for split in ("trainval", "test"):
        for c, v in out[split]["by_city"].items():
            m = merged[c]
            m["logs"] += v["logs"]; m["scenes"] += v["scenes"]; m["frames"] += v["frames"]
    print(f"\n{'city (both splits)':<30}{'logs':>7}{'scenes':>10}{'frames':>12}")
    for c, v in sorted(merged.items(), key=lambda kv: -kv[1]["scenes"]):
        print(f"{str(c):<30}{v['logs']:>7}{v['scenes']:>10}{v['frames']:>12}")

    sg = {c: v for c, v in merged.items() if c and ("sg" in c.lower() or "singapore" in c.lower())}
    sg_scenes = sum(v["scenes"] for v in sg.values())
    print(f"\nSingapore (right-hand drive): {sg_scenes} scenes across "
          f"{sum(v['logs'] for v in sg.values())} logs -> {sg}")
    print(f"Right-hand-drive share of local library: {sg_scenes/max(tot_s,1)*100:.2f}%")

    res = dict(source=NAV, by_split=out,
               merged_by_city={str(k): v for k, v in merged.items()},
               total_logs=tot_l, total_scenes=tot_s, total_frames=tot_f,
               singapore_scenes=sg_scenes, per_log=per_log)
    p = f"{RES}/rhd_local_navsim_inventory.json"
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
