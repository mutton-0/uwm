"""RHD: attach city + drive side to every mined event, across BOTH corpora.

nuScenes pools key on `scene-NNNN` (nuScenes scene names); NavSim pools key on
`log-XXXX-scene-YYYY`, which is numbered per split and therefore needs a
(split, scene) composite key. This joins each corpus with its own metadata and
reports left- vs right-hand-drive totals.

Drive side:
  right-hand drive (RHD, wheel on the right, traffic keeps left) = Singapore
  left-hand  drive (LHD, wheel on the left,  traffic keeps right) = Boston / Las Vegas / Pittsburgh

Read-only on /data/dataset/**; writes only results/rhd_*.
"""
import collections, glob, json, os, pickle, sys
import multiprocessing as mp

NUSC = "/data/dataset/nuscenes/v1.0-trainval/v1.0-trainval"
NAV = "/data/dataset/navsim/dataset/navsim_logs"
RES = "/data/ruolin/uwm/sim2real_demo_ttc/results"

RHD_KEYS = ("singapore", "sg-")           # right-hand drive


def drive_side(city):
    if not city:
        return None
    c = str(city).lower()
    return "RHD" if any(k in c for k in RHD_KEYS) else "LHD"


def nuscenes_map():
    """nuScenes scene name -> location."""
    scenes = json.load(open(f"{NUSC}/scene.json"))
    logs = {l["token"]: l["location"] for l in json.load(open(f"{NUSC}/log.json"))}
    return {s["name"]: logs.get(s["log_token"]) for s in scenes}


def _nav_one(path):
    split = path.split("/")[-2]
    try:
        with open(path, "rb") as f:
            frames = pickle.load(f)
    except Exception:
        return {}
    return {(split, fr["scene_name"]): fr.get("map_location")
            for fr in frames if fr.get("scene_name")}


def navsim_map(workers=16):
    files = []
    for s in ("trainval", "test"):
        files += sorted(glob.glob(f"{NAV}/{s}/*.pkl"))
    m = {}
    with mp.Pool(workers) as p:
        for d in p.imap_unordered(_nav_one, files, chunksize=4):
            m.update(d)
    return m


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


def split_of(item, meta, fname):
    if isinstance(item, dict) and item.get("split"):
        return item["split"]
    if meta.get("split"):
        return meta["split"]
    if "trainval" in fname:
        return "trainval"
    if "navsim_fx" in fname or "test" in fname:
        return "test"
    return None


def main():
    nm, vm = nuscenes_map(), navsim_map()
    print(f"nuScenes scenes: {len(nm)}   NavSim (split,scene) keys: {len(vm)}", flush=True)
    nusc_cities = collections.Counter(v for v in nm.values())
    print("nuScenes locations:", dict(nusc_cities), flush=True)

    rows = []
    for path in sorted(glob.glob(f"{RES}/*pool*.json")):
        if "rhd_" in os.path.basename(path):
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        items = items_of(d)
        if not items:
            continue
        meta = d if isinstance(d, dict) else {}
        fname = os.path.basename(path)
        corpus = str(meta.get("corpus") or "")
        is_nusc = "nuscenes" in corpus.lower()
        cities, unmatched = collections.Counter(), 0
        for i in items:
            if not isinstance(i, dict):
                continue
            key = i.get("scene") or i.get("scene_name")
            if is_nusc:
                c = nm.get(key)
            else:
                sp = split_of(i, meta, fname)
                c = vm.get((sp, key)) if sp else None
            if c is None:
                # some early pools carry no corpus tag; try both namespaces
                c = nm.get(key) or vm.get(("trainval", key)) or vm.get(("test", key))
            if c is None:
                unmatched += 1
            else:
                cities[c] += 1
        sides = collections.Counter(drive_side(c) for c in cities.elements())
        rows.append(dict(file=fname, corpus=corpus or None,
                         scenario=meta.get("scenario"), n_events=len(items),
                         matched=sum(cities.values()), unmatched=unmatched,
                         cities={str(k): v for k, v in cities.items()},
                         RHD=sides.get("RHD", 0), LHD=sides.get("LHD", 0)))

    print(f"\n{'pool':<48}{'corpus':<10}{'ev':>5}{'match':>7}{'LHD':>6}{'RHD':>5}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: (str(x["corpus"]), -x["n_events"])):
        cor = "nusc" if r["corpus"] and "nuscenes" in r["corpus"].lower() else \
              ("navsim" if r["corpus"] else "?")
        print(f"{r['file']:<48}{cor:<10}{r['n_events']:>5}{r['matched']:>7}"
              f"{r['LHD']:>6}{r['RHD']:>5}")

    p = f"{RES}/rhd_driveside_join.json"
    with open(p, "w") as f:
        json.dump(dict(nuscenes_locations={str(k): v for k, v in nusc_cities.items()},
                       pools=rows), f, indent=2)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
