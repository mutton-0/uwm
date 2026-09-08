"""RHD: attach a city to every mined event, by joining pool `scene` ids to the local
NavSim metadata (scene_token / scene_name -> map_location).

Motive: the raw trainval pools carry no city field, so the right-hand-drive yield can
only be read off the already-filtered deploy pools. This recovers the *pre-filter*
Singapore yield, which is the number my feasibility arithmetic depends on.

Read-only on /data/dataset/navsim/**; writes only results/rhd_*.
"""
import collections, glob, json, os, pickle, sys
import multiprocessing as mp

NAV = "/data/dataset/navsim/dataset/navsim_logs"
RES = "/data/ruolin/uwm/sim2real_demo_ttc/results"


def scene_map(path):
    """(split, scene_name) -> city for one log.

    NOTE: `scene_name` is of the form log-XXXX-scene-YYYY and is numbered *within a
    split*, so it is NOT unique across splits: 1,334 of 20,068 names (6.6%) occur in
    both trainval and test, and 835 names map to more than one city. Keying on the bare
    name silently mis-attributes those. `scene_token` is unique (0 collisions) but the
    pools do not store it, so the composite key is the only correct join.
    """
    split = path.split("/")[-2]
    try:
        with open(path, "rb") as f:
            frames = pickle.load(f)
    except Exception:
        return {}
    out = {}
    for fr in frames:
        n = fr.get("scene_name")
        if n:
            out[(split, n)] = fr.get("map_location")
    return out


def build_map(workers=16):
    files = []
    for split in ("trainval", "test"):
        files += sorted(glob.glob(f"{NAV}/{split}/*.pkl"))
    print(f"building scene->city map from {len(files)} logs", flush=True)
    m = {}
    with mp.Pool(workers) as p:
        for d in p.imap_unordered(scene_map, files, chunksize=4):
            m.update(d)
    print(f"map size: {len(m)} scene keys", flush=True)
    return m


def split_of(item, pool_meta, fname):
    """Resolve the split for one event: per-item field, else pool metadata, else filename."""
    if isinstance(item, dict) and item.get("split"):
        return item["split"]
    if pool_meta.get("split"):
        return pool_meta["split"]
    if "trainval" in fname:
        return "trainval"
    if "navsim_fx" in fname or "test" in fname:
        return "test"
    return None


def items_of(d):
    if isinstance(d, list):
        return d
    for key in ("candidates", "events", "items", "pool", "records"):
        if isinstance(d.get(key), list):
            return d[key]
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


def main():
    m = build_map(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
    pools = [p for p in sorted(glob.glob(f"{RES}/*pool*.json"))
             if "rhd_" not in os.path.basename(p)]
    rows = []
    print(f"\n{'pool':<46}{'events':>7}{'matched':>9}{'SG':>5}  other cities")
    print("-" * 108)
    for path in pools:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        items = items_of(d)
        if not items:
            continue
        meta = d if isinstance(d, dict) else {}
        fname = os.path.basename(path)
        cities = collections.Counter()
        unmatched = 0
        for i in items:
            if not isinstance(i, dict):
                continue
            sp = split_of(i, meta, fname)
            key = i.get("scene") or i.get("scene_name")
            c = m.get((sp, key)) if sp else None
            if c is None:
                unmatched += 1
            else:
                cities[c] += 1
        sg = sum(v for k, v in cities.items() if "sg-" in str(k).lower())
        matched = sum(cities.values())
        oth = ", ".join(f"{k.split('-')[-1]}:{v}" for k, v in cities.most_common(3)
                        if "sg-" not in k)
        print(f"{os.path.basename(path):<46}{len(items):>7}{matched:>9}{sg:>5}  {oth}")
        rows.append(dict(file=os.path.basename(path), n_events=len(items),
                         matched=matched, unmatched=unmatched, singapore=sg,
                         cities={str(k): v for k, v in cities.items()}))

    print("\n=== pre-filter Singapore (right-hand-drive) yield, raw mined pools ===")
    for r in rows:
        if r["singapore"] and ("trainval" in r["file"] or "navsim_fx" in r["file"]):
            print(f"  {r['file']:<46}{r['singapore']:>4}")

    p = f"{RES}/rhd_pool_city_join.json"
    with open(p, "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
