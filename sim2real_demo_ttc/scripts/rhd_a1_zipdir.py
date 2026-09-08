"""RHD step A1: read the nuPlan val/test log-zip central directories over HTTP byte ranges.

Why: the question in the handoff is "how many Singapore scenes have sensors published",
and answering it needs only the per-log sqlite metadata, not the 193 GB of log zips (and
certainly not the 4 TB of sensor blobs). S3 serves byte ranges, so we read each zip's
central directory and record where every `.db` entry lives. A later step range-fetches
only the entries we actually need.

Writes only under /mnt/skylabNAS/ per the handoff's conflict boundary.
"""
import io, json, os, re, struct, sys, urllib.request

S3 = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/"
OUT = "/mnt/skylabNAS/nuplan_sg/meta"
ZIPS = {"val": "nuplan-v1.1_val.zip", "test": "nuplan-v1.1_test.zip"}


def get_range(url, start, end):
    """Fetch bytes [start, end] inclusive."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:
            if attempt == 4:
                raise
            print(f"    retry {attempt+1}: {type(e).__name__}", flush=True)
    raise RuntimeError("unreachable")


def content_length(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])


def find_central_directory(url, size):
    """Locate the central directory, handling zip64 (these archives are > 4 GiB)."""
    tail_len = min(size, 1 << 16)
    tail = get_range(url, size - tail_len, size - 1)

    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("no EOCD found in tail")
    cd_size = struct.unpack("<I", tail[i + 12:i + 16])[0]
    cd_off = struct.unpack("<I", tail[i + 16:i + 20])[0]
    n_ent = struct.unpack("<H", tail[i + 10:i + 12])[0]

    # zip64: the 32-bit fields are saturated and the real values live in the zip64 EOCD
    j = tail.rfind(b"PK\x06\x07")          # zip64 end-of-central-dir locator
    if j >= 0 and (cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_ent == 0xFFFF):
        z64_eocd_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        z = get_range(url, z64_eocd_off, z64_eocd_off + 55)
        if z[:4] != b"PK\x06\x06":
            raise RuntimeError("zip64 EOCD signature missing")
        n_ent = struct.unpack("<Q", z[32:40])[0]
        cd_size = struct.unpack("<Q", z[40:48])[0]
        cd_off = struct.unpack("<Q", z[48:56])[0]
    return cd_off, cd_size, n_ent


def parse_central_directory(buf):
    """Yield (name, compress_type, comp_size, uncomp_size, local_header_offset)."""
    p, out = 0, []
    while p + 46 <= len(buf):
        if buf[p:p + 4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", buf[p + 10:p + 12])[0]
        csize = struct.unpack("<I", buf[p + 20:p + 24])[0]
        usize = struct.unpack("<I", buf[p + 24:p + 28])[0]
        nlen = struct.unpack("<H", buf[p + 28:p + 30])[0]
        elen = struct.unpack("<H", buf[p + 30:p + 32])[0]
        clen = struct.unpack("<H", buf[p + 32:p + 34])[0]
        lho = struct.unpack("<I", buf[p + 42:p + 46])[0]
        name = buf[p + 46:p + 46 + nlen].decode("utf-8", "replace")

        # zip64 extra field carries the real sizes/offset when the 32-bit slots saturate
        extra = buf[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                blk, r = extra[q + 4:q + 4 + hsz], 0
                if usize == 0xFFFFFFFF and r + 8 <= len(blk):
                    usize = struct.unpack("<Q", blk[r:r + 8])[0]; r += 8
                if csize == 0xFFFFFFFF and r + 8 <= len(blk):
                    csize = struct.unpack("<Q", blk[r:r + 8])[0]; r += 8
                if lho == 0xFFFFFFFF and r + 8 <= len(blk):
                    lho = struct.unpack("<Q", blk[r:r + 8])[0]; r += 8
                break
            q += 4 + hsz
        out.append(dict(name=name, method=method, csize=csize, usize=usize, lho=lho))
        p += 46 + nlen + elen + clen
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    index = {}
    for split, zname in ZIPS.items():
        url = S3 + zname
        size = content_length(url)
        print(f"[{split}] {zname}  {size/1e9:.2f} GB", flush=True)
        cd_off, cd_size, n_ent = find_central_directory(url, size)
        print(f"[{split}] central dir at {cd_off} ({cd_size/1e6:.1f} MB, {n_ent} entries)", flush=True)
        cd = get_range(url, cd_off, cd_off + cd_size - 1)
        ents = parse_central_directory(cd)
        dbs = [e for e in ents if e["name"].endswith(".db")]
        tot = sum(e["csize"] for e in dbs)
        print(f"[{split}] entries parsed={len(ents)}  .db files={len(dbs)}  "
              f"compressed total={tot/1e9:.2f} GB", flush=True)
        index[split] = dict(zip_name=zname, url=url, zip_size=size,
                            n_entries=len(ents), dbs=dbs)
    p = f"{OUT}/rhd_zip_index.json"
    with open(p, "w") as f:
        json.dump(index, f)
    print("wrote", p, f"({os.path.getsize(p)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
