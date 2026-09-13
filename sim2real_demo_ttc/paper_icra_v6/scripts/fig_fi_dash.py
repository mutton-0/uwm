"""F–I 仪表盘（独立图）：看完轨迹之后医生读哪两个数。
(a) 一个场景问诊三次，逐时刻连线的长度均值即 F 与 I；(b) F–I 平面读数盘。
两个面板的实现在 fi_panels.py，与 Fig. frame（方法总览）同源，改一处两张图一起变。
输出 figures/fi_dash.pdf / .png。"""
import sys,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
sys.path.insert(0,f"{V5}/scripts")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.linewidth":0.6,"axes.edgecolor":"#52514e",
                     "xtick.color":"#52514e","ytick.color":"#52514e"})
from fi_panels import panel_extract,panel_readings,HAS4
fig=plt.figure(figsize=(3.45,1.85))
gs=fig.add_gridspec(1,2,width_ratios=[1.0,1.22],wspace=0.50)
panel_extract(fig.add_subplot(gs[0,0]))
panel_readings(fig.add_subplot(gs[0,1]))
fig.savefig(f"{V5}/figures/fi_dash.pdf",bbox_inches="tight")
fig.savefig(f"{V5}/figures/fi_dash.png",dpi=240,bbox_inches="tight")
print("ok  dusk =",HAS4)
