#!/usr/bin/env python
"""
Ghosthead sim-vs-real 对照分析 -> **纯静态自包含 HTML**（无 Bokeh / 无 canvas / 无外部依赖）。

历史问题：Bokeh 版在 Chrome(Apple M4) 崩溃于 GPU/canvas 光栅化线程(EXC_BREAKPOINT)。
本版彻底去掉 Bokeh 与 canvas，只用 CSS 标签页 + 原生 <img>(JPEG 压缩, lazy-load)，
图表用已生成的 matplotlib PNG（含全部曲线/散点/柱状），静态页面不会崩。

只消费 outputs/ghosthead_infer/ 下已产出的 PNG。
"""
import os, glob, base64
import cv2

BASE = "/data/ruolin/uwm/outputs/ghosthead_infer"
HTML_OUT = f"{BASE}/bokeh_ghosthead_sim2real.html"


def img_tag(path, w=900, q=72, title=""):
    img = cv2.imread(path)
    if img is None:
        return f'<p class="miss">(缺 {os.path.basename(path)})</p>'
    h0, w0 = img.shape[:2]
    if w0 > w:
        img = cv2.resize(img, (w, int(round(h0 * w / w0))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    cap = f'<div class="cap">{title}</div>' if title else ""
    return f'<figure>{cap}<img loading="lazy" src="{uri}"/></figure>'


def imgs(glob_pat, prefix, w=900):
    parts = [img_tag(p, w, title=os.path.basename(p)[len(prefix):-4]) for p in sorted(glob.glob(glob_pat))]
    return "".join(parts) or '<p class="miss">(无图)</p>'


def scroll(inner, h=640):
    return f'<div class="scroll">{inner}</div>'


def overlay_blocks(w=940):
    """每个场景一块：原始输入(sim+real, t=1s) + 该场景注意力叠加图，方便对照。"""
    blocks = []
    for p in sorted(glob.glob(f"{BASE}/attn_diff/overlay_*.png")):
        sc = os.path.basename(p)[len("overlay_"):-4]
        sim_in = f"{BASE}/{sc}/inputs/transfered_t1.png"
        real_in = f"{BASE}/{sc}/inputs/origin_t1.png"
        blocks.append(
            f'<div class="ovlblock"><div class="scname">{sc}</div>'
            f'<div class="imgrow">{img_tag(sim_in, 460, title="sim input (transfered, t=1s)")}'
            f'{img_tag(real_in, 460, title="real input (origin, t=1s)")}</div>'
            f'{img_tag(p, w, title="attention overlay — sim(top)/real(bottom) × L0..L7")}</div>')
    return "".join(blocks) or '<p class="miss">(无图)</p>'


SECTIONS = [
    ("结论", """
      <p><b>输入</b>：transfered=CARLA(sim)、origin=世界模型真实感(real)，同场景同 GT，唯一变量是输入图。
      <b>⚠️前提</b>：sim/real 谁 in-domain 依赖"ckpt 主要 sim 训练"这一未验证前提；内部差异/因果结论不依赖它。</p>
      <ul>
        <li><b>症状≠病因</b>：JS 散度峰 L7/L0，因果(激活修补)峰 <b>L6</b>（深层 L4–6 主导 9/12）。patch-ALL=1.0。</li>
        <li><b>1-CKA 比 JS 更贴因果</b>(Spearman +0.64 vs +0.24)，但 argmax 仍误落 L7。</li>
        <li><b>内建头</b>：BEV 语义 IoU 低(centerline 0.12)、vehicle 像素 161→100（漏看它车，机制假说）。</li>
        <li><b>因果层强度不预测 PDM 退化幅度</b>(相关≈0)：位点稳定≠后果可预测。</li>
        <li style="color:#a00">LoRA 应放 <b>L6/L5</b>（因果层），不是散度大的 L7/L0。</li>
      </ul>
      <p class="hint">说明：本页为纯静态版（Bokeh/canvas 交互版在部分 Chrome 上会崩，故改静态）。图为 matplotlib 生成，含全部曲线/散点。</p>"""),
]


def main():
    # 各 tab 的 HTML body
    t_intro = SECTIONS[0][1]
    t_triad = (f'<p>逐层均值(n=12)。argmax：JS=L7、1-CKA=L7、<b>recovery=L6</b>；相关性 1-CKA↔recovery +0.64 &gt; JS +0.24。</p>'
               f'<div class="imgrow">{img_tag(f"{BASE}/cka_recovery/cka_vs_recovery.png",560,title="JS vs 1-CKA vs recovery")}'
               f'{img_tag(f"{BASE}/patching/recovery_by_layer.png",560,title="因果 recovery 逐场景")}</div>')
    t_rec = (f'<p>激活修补 recovery(L)：high=该层因果驱动 real 退化。多数在 L5–L6 抬升、L4 有负 dip。</p>'
             f'{img_tag(f"{BASE}/patching/recovery_by_layer.png",860)}')
    t_ovl = (f'<p>每层图像注意力(8×32)叠到输入 RGB。每个场景先给<b>原始输入</b>(sim+real, t=1s)再给叠加图，'
             f'叠加图：上排=sim、下排=real，列 L0..L7；看 <b>L7 列</b> 差异最大。可滚动看全部场景。</p>'
             f'{scroll(overlay_blocks(940))}'
             f'<div class="imgrow">{img_tag(f"{BASE}/attn_diff/attn_layer_js.png",520,title="逐层 JS")}'
             f'{img_tag(f"{BASE}/attn_diff/attn_share_entropy.png",520,title="占比&熵")}</div>')
    t_bev = (f'<p>内建 BEV 语义头(免训)。图例：<b>灰=road</b>、<b>绿=walkways(人行道)</b>、<b>黄=centerline</b>、'
             f'<b>紫=static objects</b>、<b>红=vehicles</b>、<b>青=pedestrians</b>、深底=背景。每张左 sim / 右 real。可滚动看全部场景。</p>'
             f'{scroll(imgs(f"{BASE}/head_probe/bev_sem_*.png","bev_sem_",900))}'
             f'{img_tag(f"{BASE}/head_probe/head_probe_bars.png",900)}')
    t_head = (f'<p>免训、无外部 GT，测 sim-vs-real 一致性。BEV 语义 IoU 均值 road 0.43/centerline 0.12/vehicle 0.48；'
              f'vehicle 像素 161→100（机制假说）；agent-head 置信度 inconclusive。</p>'
              f'{img_tag(f"{BASE}/head_probe/head_probe_bars.png",900)}')
    t_pdm = (f'<p>全部 72 场景 t=1s。<b>因果层强度基本不预测 PDM 退化幅度</b>（散点无结构、相关≈0）。</p>'
             f'{img_tag(f"{BASE}/causal_pdm/causal_pdm_scatter.png",1000,title="三预测量 vs PDM Δtotal")}')

    tabs = [("结论", t_intro), ("逐层症状vs病因", t_triad), ("逐场景recovery", t_rec),
            ("注意力叠加RGB", t_ovl), ("BEV语义", t_bev), ("内建头探针", t_head), ("因果↔PDM", t_pdm)]

    buttons = "".join(f'<button onclick="show({i})"{" class=act" if i==0 else ""}>{n}</button>'
                      for i, (n, _) in enumerate(tabs))
    sections = "".join(f'<section class="tabsec{" show" if i==0 else ""}"><h2>{n}</h2>{b}</section>'
                       for i, (n, b) in enumerate(tabs))

    css = """
    body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;color:#222;background:#fff}
    header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 14px;z-index:10}
    h1{font-size:17px;margin:0 0 8px}
    nav button{margin:2px;padding:6px 12px;border:1px solid #ccc;background:#f4f4f4;border-radius:6px;cursor:pointer;font-size:13px}
    nav button.act{background:#2b6;color:#fff;border-color:#2b6}
    .tabsec{display:none;padding:14px 18px;max-width:1160px}
    .tabsec.show{display:block}
    h2{font-size:16px;border-left:4px solid #2b6;padding-left:8px}
    figure{margin:8px 0}
    figure img{max-width:100%;height:auto;border:1px solid #ccc;display:block}
    .cap{font-size:12px;color:#555;margin-bottom:3px}
    .scroll{max-height:640px;overflow-y:auto;border:1px solid #ddd;padding:6px;background:#fafafa}
    .imgrow{display:flex;flex-wrap:wrap;gap:12px}
    .imgrow figure{flex:1 1 440px}
    .ovlblock{border:1px solid #e0e0e0;border-radius:6px;padding:8px 10px;margin:10px 0;background:#fff}
    .scname{font-weight:600;color:#2b6;font-size:14px;margin-bottom:4px}
    .hint{color:#888;font-size:12px}.miss{color:#a00}
    """
    js = """
    function show(i){
      var s=document.querySelectorAll('.tabsec'),b=document.querySelectorAll('nav button');
      for(var k=0;k<s.length;k++){s[k].classList.toggle('show',k===i);b[k].classList.toggle('act',k===i);}
      window.scrollTo(0,0);
    }
    """
    html = (f'<!doctype html><html lang="zh"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Ghosthead sim2real 分析</title><style>{css}</style></head><body>'
            f'<header><h1>Ghosthead × DiffusionDrive — sim vs real 对照分析</h1><nav>{buttons}</nav></header>'
            f'{sections}<script>{js}</script></body></html>')

    with open(HTML_OUT, "w") as f:
        f.write(html)
    print(f"Saved (static): {HTML_OUT}  ({os.path.getsize(HTML_OUT)//1024} KB)")


if __name__ == "__main__":
    main()
