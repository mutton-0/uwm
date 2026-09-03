# DONE — LTF 盲动结论实例核查

产出：
- `results/ltf_sanity_check_report_zh.md` / `_en.md`
- `results/figures/ltf_sanity_check/`（30 张 JPG + `meta_export_v2.json`）
- `results/ltf_baseline_probe.json`（30 帧健康度探针）
- `scripts/ltf_sanity_export.py`、`scripts/ltf_baseline_probe.py`
- amendments §FF/A70

**一句话结论**：不支持"LTF 对危险实体盲视"。checkpoint 健康、遮挡管线正确、
LTF 对遮挡有显著且方向正确的响应并随距离分级；但 (a) 该速度读数
99.91% 由自车速度决定、图像可调动幅度仅 0.0855 m/s，(b) 语料里的"危险"多为路侧行人。
机械判定未变，未改论文。

**待用户决定**：论文该格措辞是否由"盲视/盲目泛化"收窄。
