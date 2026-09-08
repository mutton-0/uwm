"""RHD: inventory of the mined event pools already on disk.

Reports, per pool file: scenario/corpus tags, number of events, distinct scenes/logs,
and the city and split breakdown. Read-only over results/*pool*.json (those belong to
the other agent); writes only results/rhd_*.
"""
import collections, glob, json, os

RES = "/data/ruolin/uwm/sim2real_demo_ttc/results"


def items_of(d):
    """Pools come in a few shapes; pull out the event list whatever the wrapper."""
    if isinstance(d, list):
        return d, {}
    if not isinstance(d, dict):
        return [], {}
    meta = {k: v for k, v in d.items()
            if k in ("scenario", "tag", "corpus", "sources", "n_events", "n_scenes", "n_logs")}
    for key in ("candidates", "events", "items", "pool", "records"):
        if isinstance(d.get(key), list):
            return d[key], meta
    # otherwise: the first list-of-dicts value
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v, meta
    return [], meta


def summarize(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        return dict(file=os.path.basename(path), error=f"{type(e).__name__}: {e}")
    items, meta = items_of(d)
    cities = collections.Counter(
        i.get("city") or i.get("map_location") for i in items if isinstance(i, dict))
    splits = collections.Counter(i.get("split") for i in items if isinstance(i, dict))
    scenes = {i.get("scene") or i.get("scene_name") or i.get("scene_token")
              for i in items if isinstance(i, dict)}
    scenes.discard(None)
    logs = {i.get("log_name") for i in items if isinstance(i, dict)}
    logs.discard(None)
    return dict(file=os.path.basename(path), n_events=len(items),
                n_scenes=len(scenes), n_logs=len(logs),
                cities={str(k): v for k, v in cities.items()},
                splits={str(k): v for k, v in splits.items()},
                meta=meta)


def main():
    rows = [summarize(p) for p in sorted(glob.glob(f"{RES}/*pool*.json"))
            if "rhd_" not in os.path.basename(p)]
    rows = [r for r in rows if not r.get("error")]

    print(f"{'pool file':<46}{'events':>8}{'scenes':>8}{'logs':>7}  cities")
    print("-" * 110)
    for r in sorted(rows, key=lambda x: -x["n_events"]):
        cs = ", ".join(f"{k.split('-')[-1] if k!='None' else '?'}:{v}"
                       for k, v in sorted(r["cities"].items(), key=lambda kv: -kv[1])[:4])
        print(f"{r['file']:<46}{r['n_events']:>8}{r['n_scenes']:>8}{r['n_logs']:>7}  {cs}")

    # right-hand-drive (Singapore) events anywhere in the mined pools
    print("\n=== right-hand-drive (sg-one-north / singapore) events per pool ===")
    tot = 0
    for r in sorted(rows, key=lambda x: x["file"]):
        sg = sum(v for k, v in r["cities"].items()
                 if k and ("sg-" in k.lower() or "singapore" in k.lower()))
        if sg:
            print(f"  {r['file']:<46}{sg:>6}")
            tot += sg
    print(f"  {'TOTAL (may double-count across pool versions)':<46}{tot:>6}")

    p = f"{RES}/rhd_mined_pool_inventory.json"
    with open(p, "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
