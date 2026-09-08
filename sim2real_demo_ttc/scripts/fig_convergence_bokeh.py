"""收敛曲线的交互版（Bokeh）：两个视图并列，方便横向对比。

## 为什么是两个视图而不是"全部叠一张"
6 个候选 × 4 根轴 = 18 条线。6 色在 all-pairs 下不合格
（validate_palette: 最差 CVD ΔE 3.2、常视 7.1），按调色规范"超过 3 个就分面
或复合编码"。故：

  视图 A  按**候选**分面（6 格），每格 4 根轴 —— 轴色用槽 1/2/3/7，all-pairs 全过
  视图 B  按**轴**分面（4 格），每格 6 个候选 —— **族色（3 色，all-pairs 过）
          + 线型**复合编码；族的划分本身有语义：TransFuser 系 / VLM 规划器 / VLA

两图共享纵轴含义：单次抽 n 个 scene 的估计离散度 ÷ 全数据精度 h。
降到 1.0 ⇒ 只收 n 个 scene 已和用满全部一样稳。
"""
from __future__ import annotations
import json
from pathlib import Path

from bokeh.io import output_file, save
from bokeh.models import CustomJS, Select
from bokeh.resources import INLINE

from bokeh.layouts import column, gridplot, row
from bokeh.models import ColumnDataSource, HoverTool, Span, Div
from bokeh.plotting import figure

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
OUT = Path("/data/ruolin/uwm/sim2real_demo_ttc/paper_v2.3_icra/figures")
NAME = {"simlingo": "SimLingo", "dd": "DiffusionDrive", "ltf": "LTF",
        "ddv2": "DiffusionDriveV2", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}
ORDER = ["simlingo", "dd", "ltf", "ddv2", "alpa", "autovla"]
# 轴色 = 调色板槽 1/2/3/7，validate_palette --pairs all 全过
AXCOL = {"F-3  $r$": "#2a78d6", "G-VS  selectivity": "#eb6834",
         "C  $C_m$": "#1baf7a", "I  $I_m$": "#4a3aa7"}
AXSHORT = {"F-3  $r$": "F-3", "G-VS  selectivity": "G-VS",
           "C  $C_m$": "C", "I  $I_m$": "I"}
# 族色（3 色，all-pairs 过）+ 线型作二次编码
FAM = {"dd": ("TransFuser", "#2a78d6", "solid"),
       "ltf": ("TransFuser", "#2a78d6", "dashed"),
       "ddv2": ("TransFuser", "#2a78d6", "dotted"),
       "simlingo": ("VLM planner", "#eb6834", "solid"),
       "alpa": ("VLA", "#1baf7a", "solid"),
       "autovla": ("VLA", "#1baf7a", "dashed")}
INK2, GRID = "#52514e", "#d8d7d2"
TOL = 1.0


def style_mc(p):
    """MC 收敛图用：不画 1h 参考线（纵轴是轴读数本身，不是归一化误差）。"""
    p.background_fill_color = "#fcfcfb"; p.border_fill_color = "#fcfcfb"
    p.outline_line_color = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = GRID; p.ygrid.grid_line_width = 0.6
    p.axis.axis_line_color = GRID
    p.axis.major_tick_line_color = GRID; p.axis.minor_tick_line_color = None
    p.axis.major_label_text_color = INK2; p.axis.axis_label_text_color = INK2
    p.title.text_color = "#0b0b0b"; p.title.text_font_size = "9pt"
    return p


def style(p):
    p.background_fill_color = "#fcfcfb"; p.border_fill_color = "#fcfcfb"
    p.outline_line_color = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = GRID; p.ygrid.grid_line_width = 0.6
    p.axis.axis_line_color = GRID
    p.axis.major_tick_line_color = GRID; p.axis.minor_tick_line_color = None
    p.axis.major_label_text_color = INK2; p.axis.axis_label_text_color = INK2
    p.title.text_color = "#0b0b0b"; p.title.text_font_size = "10pt"
    p.add_layout(Span(location=TOL, dimension="width", line_color=GRID,
                      line_dash="dashed", line_width=1.2))
    return p


def mk_hover():
    return HoverTool(tooltips=[("", "@label"), ("scenes", "@n"),
                               ("|err| / h", "@y{0.00}"),
                               ("|err| (原生单位)", "@abs{0.0000}"),
                               ("value", "@median{0.000}")],
                     mode="mouse")


def main():
    D = json.load(open(RES / "convergence_curves.json"))
    C = D["curves"]
    SRCS = []

    # ---------- 视图 0：经典蒙特卡洛收敛图 ----------
    # 每格一个 (候选, 轴)：均值线 + 85/90/95 嵌套置信带 + 全量终值虚线。
    # 各轴原生量纲相差 20 倍以上，不能共享纵轴，故必须一格一个 cell。
    by = {(c["model"], c["axis"]): c for c in C}
    pm = []
    for m in ORDER:
        rowp = []
        for ax, col in AXCOL.items():
            c = by.get((m, ax))
            if c is None:
                rowp.append(None)
                continue
            p = figure(width=290, height=175,
                       title=f"{NAME[m]} · {AXSHORT[ax]}",
                       x_axis_label="scenes",
                       tools="pan,wheel_zoom,box_zoom,reset,save")
            src = ColumnDataSource(dict(
                n=c["n"], mean=c["mean"],
                lo85=c["ci85_lo"], hi85=c["ci85_hi"],
                lo90=c["ci90_lo"], hi90=c["ci90_hi"],
                lo95=c["ci95_lo"], hi95=c["ci95_hi"],
                label=[f"{NAME[m]} · {AXSHORT[ax]}"] * len(c["n"])))
            for q, al in ((95, 0.12), (90, 0.16), (85, 0.22)):
                p.varea(x="n", y1=f"lo{q}", y2=f"hi{q}", source=src,
                        fill_color=col, fill_alpha=al)
            p.line("n", "mean", source=src, color=col, line_width=2.2)
            p.add_layout(Span(location=c["final"], dimension="width",
                              line_color=INK2, line_dash="dashed",
                              line_width=1.2))
            p.add_tools(HoverTool(tooltips=[
                ("", "@label"), ("scenes", "@n"), ("mean", "@mean{0.0000}"),
                ("85% CI", "@lo85{0.000} – @hi85{0.000}"),
                ("90% CI", "@lo90{0.000} – @hi90{0.000}"),
                ("95% CI", "@lo95{0.000} – @hi95{0.000}")], mode="vline"))
            rowp.append(style_mc(p))
        pm.append(rowp)

    # ---------- 视图 A：按候选分面 ----------
    pa = []
    for m in ORDER:
        sub = [c for c in C if c["model"] == m]
        if not sub:
            continue
        p = figure(width=380, height=210, title=NAME[m],
                   x_axis_label="scenes", y_axis_label="|err| p90 / h",
                   y_range=(0, 6), tools="pan,wheel_zoom,box_zoom,reset,save")
        for c in sorted(sub, key=lambda z: z["axis"]):
            src = ColumnDataSource(dict(
                n=c["n"], y=c["err90_h"], abs=c["err90_abs"],
                median=c["median"], y85=c["err85_h"], y90=c["err90_h"], y95=c["err95_h"],
                a85=c["err85_abs"], a90=c["err90_abs"], a95=c["err95_abs"],
                label=[f"{NAME[m]} · {AXSHORT[c['axis']]}"] * len(c["n"])))
            SRCS.append(src)
            p.line("n", "y", source=src, line_width=2.2,
                   color=AXCOL[c["axis"]], legend_label=AXSHORT[c["axis"]])
            p.circle("n", "y", source=src, size=4, color=AXCOL[c["axis"]],
                     alpha=0.7)
        p.add_tools(mk_hover())
        p.legend.location = "top_right"; p.legend.label_text_font_size = "8pt"
        p.legend.background_fill_alpha = 0.75; p.legend.click_policy = "hide"
        pa.append(style(p))

    # ---------- 视图 B：按轴分面 ----------
    pb = []
    for ax in AXCOL:
        sub = [c for c in C if c["axis"] == ax]
        if not sub:
            continue
        p = figure(width=380, height=210, title=AXSHORT[ax],
                   x_axis_label="scenes", y_axis_label="|err| p90 / h",
                   y_range=(0, 6), tools="pan,wheel_zoom,box_zoom,reset,save")
        for c in sorted(sub, key=lambda z: ORDER.index(z["model"])):
            fam, col, dash = FAM[c["model"]]
            src = ColumnDataSource(dict(
                n=c["n"], y=c["err90_h"], abs=c["err90_abs"],
                median=c["median"], y85=c["err85_h"], y90=c["err90_h"], y95=c["err95_h"],
                a85=c["err85_abs"], a90=c["err90_abs"], a95=c["err95_abs"],
                label=[f"{NAME[c['model']]} · {AXSHORT[ax]}"] * len(c["n"])))
            SRCS.append(src)
            p.line("n", "y", source=src, line_width=2.2, color=col,
                   line_dash=dash, legend_label=NAME[c["model"]])
        p.add_tools(mk_hover())
        p.legend.location = "top_right"; p.legend.label_text_font_size = "7.5pt"
        p.legend.background_fill_alpha = 0.75; p.legend.click_policy = "hide"
        pb.append(style(p))

    sel = Select(title="confidence", value="90",
                 options=[("85", "85%"), ("90", "90%"), ("95", "95%")], width=110)
    sel.js_on_change("value", CustomJS(args=dict(srcs=SRCS), code="""
        const q = cb_obj.value;
        for (const s of srcs) {
            // 整体替换 data（Bokeh 2.x 下比就地改字段更可靠地触发重绘）
            const d = Object.assign({}, s.data);
            d['y'] = d['y' + q];
            d['abs'] = d['a' + q];
            s.data = d;
        }
    """))

    head = Div(text="""
<div style="font-family:system-ui;max-width:1180px;color:#0b0b0b">
<h2 style="margin:6px 0">四轴收敛：需要多少数据</h2>
<p style="color:#52514e;line-height:1.5;margin:4px 0 2px">
本页两种读法。<b>第一组</b>是经典蒙特卡洛收敛图：画估计值本身随 n 的走向，
看均值线拉平、置信带收窄；一格一个 (候选, 轴)，因为四轴原生量纲相差 20 倍以上。
<b>第二组</b>把它压成一个可跨轴比较的标量：
纵轴 = <b>单次抽 n 个 scene 时 |估计值 − 终值| 的分位 ÷ 全数据精度 h</b>，
分位由上方下拉框在 <b>85 / 90 / 95%</b> 间切换（悬停另给原生单位的绝对误差）
（h 为该轴在这批材料上的 scene-bootstrap 半宽）。虚线 1.0 表示
<b>九成情况下，只用 n 个 scene 的答案与用满全部的差距不超过后者自身的精度</b>。
用「单次抽样的误差」而非「多次子采样中位数的偏移」——后者近似无偏，
任何 n 下都很小，量的是偏倚不是精度。
点击图例可隐藏曲线；悬停看具体数值。</p>
<p style="color:#52514e;line-height:1.5;margin:2px 0 10px">
第二组内部再分两排：上排按<b>候选</b>分面（4 个轴色）；下排按<b>轴</b>分面
（<b>族色 + 线型</b>复合编码 —— 6 色在 all-pairs 色觉检验下不合格，
故用 3 个族色配线型）。TransFuser 系 = DD/LTF/DDv2，实线/虚线/点线。</p>
</div>""", width=1180)

    # INLINE：把 BokehJS 打进 HTML，离线也能直接打开（默认走 CDN，断网就白屏）
    output_file(str(OUT / "convergence_interactive.html"),
                title="GFIC 四轴收敛（交互版）", mode="inline")
    save(column(head,
                Div(text="<h3 style='font-family:system-ui;margin:8px 0 2px'>"
                         "收敛（蒙特卡洛形式）：估计值 + 嵌套置信带</h3>"
                         "<p style='font-family:system-ui;color:#52514e;"
                         "font-size:13px;margin:0 0 6px'>纵轴 = <b>轴读数本身</b>"
                         "（不是误差，故可以为负 —— F-3 的 r&lt;0 表示恢复危险物后"
                         "反而开得更快，是结论不是画错）。均值线随 n 拉平、置信带随 n "
                         "收窄并裹住虚线（全量终值）即为收敛。各格纵轴独立。</p>",
                    width=1180),
                gridplot(pm, toolbar_location="right"),
                Div(text="<h3 style='font-family:system-ui;margin:16px 0 2px'>"
                         "归一化误差：|估计 − 终值| ÷ h</h3>", width=1180),
                row(sel),
                Div(text="<h3 style='font-family:system-ui;margin:8px 0 2px'>按候选分面</h3>",
                    width=1180),
                gridplot([pa[i:i + 3] for i in range(0, len(pa), 3)],
                         toolbar_location="right"),
                Div(text="<h3 style='font-family:system-ui;margin:14px 0 2px'>按轴分面</h3>",
                    width=1180),
                gridplot([pb[i:i + 2] for i in range(0, len(pb), 2)],
                         toolbar_location="right")),
         resources=INLINE)
    # 说明：HTML 里出现的 mathjax URL 是 BokehJS **内部的懒加载分支**，
    # 只在标签含 LaTeX 时才发请求。本图标题与轴标签全是纯文本，故不会触发，
    # 断网可正常打开。
    print(f"[BOKEH] -> {OUT/'convergence_interactive.html'}")


if __name__ == "__main__":
    main()
