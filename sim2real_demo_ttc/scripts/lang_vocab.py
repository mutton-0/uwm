"""语言读数的封闭词表 —— 单一真源，无重依赖。

T2.5（SimLingo）与 P1/P2（阳性对照）必须用**字面相同**的词表，否则"同口径"是空话。
抽成独立模块的直接原因：t25_language 会连带 import g3_metrics→h5py，
而阳性对照跑在另一个 python 环境里（复用 /data/Zhengyang 的 site-packages，无 h5py）。
"""
import re

VRU_RE = re.compile(r"\b(pedestrian|pedestrians|person|people|walker|cyclist|bicycle|bike|"
                    r"motorcycle|scooter|rider|crossing|cross(?:es|ing)?\s+the\s+road|jaywalk\w*)\b", re.I)
SLOW_RE = re.compile(r"\b(decelerat\w*|brak\w*|slow\w*|stop\w*|halt\w*|yield\w*|"
                     r"stay\s+behind|wait\w*|careful\w*|caution\w*)\b", re.I)
FAST_RE = re.compile(r"\b(accelerat\w*|speed\s+up|keep\s+driving|drive\s+through)\b", re.I)


def encode(text: str):
    t = text or ""
    return {"mentions_vru": bool(VRU_RE.search(t)),
            "says_slow": bool(SLOW_RE.search(t)),
            "says_fast": bool(FAST_RE.search(t))}
