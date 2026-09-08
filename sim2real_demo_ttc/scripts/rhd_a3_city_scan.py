"""RHD step A3: answer the go/no-go question from the handoff.

  "How many Singapore scenes in nuPlan val/test have sensors published?"
  Decision rule (handoff §3A): if that number < 8,000, the target of ~110 lead events
  is unreachable at the established yield rates, and we report infeasible rather than
  downloading 4 TB of sensor blobs.

Reads only the .db files fetched under /mnt/skylabNAS/; writes results/rhd_*.
"""
import collections, glob, json, os, sqlite3, sys

DBS = "/mnt/skylabNAS/nuplan_sg/dbs"
RES = "/data/ruolin/uwm/sim2real_demo_ttc/results"
META = "/mnt/skylabNAS/nuplan_sg/meta"

# Established yields from the existing trainval mining, quoted in the handoff.
YIELD = {"ghost": 0.0015, "lead": 0.0059}
TARGET_SCENES = 8000          # handoff's stated go/no-go threshold
TARGET_EVENTS = 110


def scan_one(path):
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cur = c.cursor()
        cur.execute("select location, map_version, logfile from log")
        row = cur.fetchone()
        cur.execute("select count(*) from scene")
        n_scene = cur.fetchone()[0]
        cur.execute("select count(*) from lidar_pc")
        n_frame = cur.fetchone()[0]
        c.close()
        return dict(logfile=row[2], location=row[0], map_version=row[1],
                    n_scene=n_scene, n_frame=n_frame, ok=True)
    except Exception as e:
        return dict(logfile=os.path.basename(path)[:-3], ok=False,
                    err=f"{type(e).__name__}: {e}")


def main():
    recs = []
    for split in ("val", "test"):
        files = sorted(glob.glob(f"{DBS}/{split}/*.db"))
        print(f"[{split}] scanning {len(files)} db files", flush=True)
        for i, p in enumerate(files):
            r = scan_one(p)
            r["split"] = split
            recs.append(r)
            if (i + 1) % 50 == 0:
                print(f"   {i+1}/{len(files)}", flush=True)

    bad = [r for r in recs if not r.get("ok")]
    good = [r for r in recs if r.get("ok")]
    print(f"\nscanned ok={len(good)} failed={len(bad)}")
    for r in bad[:5]:
        print("   FAILED", r["logfile"], r.get("err"))

    by_city = collections.defaultdict(lambda: dict(logs=0, scenes=0, frames=0))
    by_split_city = collections.defaultdict(lambda: dict(logs=0, scenes=0))
    for r in good:
        c = r["location"]
        by_city[c]["logs"] += 1
        by_city[c]["scenes"] += r["n_scene"]
        by_city[c]["frames"] += r["n_frame"]
        by_split_city[(r["split"], c)]["logs"] += 1
        by_split_city[(r["split"], c)]["scenes"] += r["n_scene"]

    print("\n=== logs/scenes WITH SENSORS, by city (val+test) ===")
    print(f"{'city':<22}{'logs':>7}{'scenes':>10}{'frames':>12}")
    for c, v in sorted(by_city.items(), key=lambda kv: -kv[1]["scenes"]):
        print(f"{c:<22}{v['logs']:>7}{v['scenes']:>10}{v['frames']:>12}")
    tot_scenes = sum(v["scenes"] for v in by_city.values())
    print(f"{'TOTAL':<22}{len(good):>7}{tot_scenes:>10}")

    sg_keys = [c for c in by_city if "singapore" in c.lower() or c.lower().startswith("sg")]
    sg_scenes = sum(by_city[c]["scenes"] for c in sg_keys)
    sg_logs = sum(by_city[c]["logs"] for c in sg_keys)

    print("\n=== DECISION (handoff §3A) ===")
    print(f"Singapore city keys found: {sg_keys}")
    print(f"Singapore logs with sensors  : {sg_logs}")
    print(f"Singapore scenes with sensors: {sg_scenes}")
    print(f"Threshold                    : {TARGET_SCENES}")
    exp = {k: sg_scenes * v for k, v in YIELD.items()}
    print(f"Expected events at established yields: "
          f"ghost {exp['ghost']:.1f}, lead {exp['lead']:.1f} (target {TARGET_EVENTS} each)")
    feasible = sg_scenes >= TARGET_SCENES
    print(f"VERDICT: {'FEASIBLE — proceed to step B' if feasible else 'NOT FEASIBLE — do not download sensor blobs'}")

    out = dict(
        question="how many Singapore scenes in nuPlan val/test have sensors published",
        n_logs_scanned=len(good), n_logs_failed=len(bad),
        by_city={c: v for c, v in by_city.items()},
        by_split_city={f"{s}|{c}": v for (s, c), v in by_split_city.items()},
        singapore=dict(city_keys=sg_keys, logs=sg_logs, scenes=sg_scenes),
        total_scenes_with_sensors=tot_scenes,
        threshold_scenes=TARGET_SCENES,
        established_yield=YIELD,
        expected_events=exp,
        target_events=TARGET_EVENTS,
        feasible=feasible,
        per_log=good,
    )
    os.makedirs(RES, exist_ok=True)
    p = f"{RES}/rhd_stepA_feasibility.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
