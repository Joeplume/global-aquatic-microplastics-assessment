# 全球水体微塑料评估：数据与代码

本仓库提供论文分析框架的公开实现，包括 QA/QC 匹配、单位与体积统一、共同 α 粒径校正、五类水体 BNN 与 MEM、严格迁移验证、AOA/C-AOA、全球预测以及 MDC 与监测分析。

## 目录

- `code/`：八个分析模块。
- `configs/pipeline.json`：统一配置。
- `configs/qaqc_prompt_cn.txt`：A–G 证据型 QA/QC 提示词。
- `data/public_case_data.csv`：公开案例数据。
- `data/figure1a_map_points.csv`：Figure 1a 的 4,620 个展示点及其代表的分析记录数，合计 34,853 条。
- `run_full.py`：完整流程入口。

## 安装

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## 运行

```bash
python run_full.py --data data/public_case_data.csv --output outputs
python run_full.py --data <analysis_data.csv> --grid <prediction_grid.csv> --output <output_dir>
```

单位处理保留计数体积、质量体积、面积密度、通量和其他非体积单位的分流；质量浓度只有在具有单颗粒质量或粒径、形态及聚合物信息时才进入质量—数量换算。主要输出为统一后的分析数据、严格迁移验证指标、带 AOA/C-AOA 标记的预测结果和 MDC 监测结果。完整分析数据通过论文 Data Availability 所列 DOI 申请；公开派生数据 DOI 将在正式发布后补充。

## 引用

请引用论文、代码软件 DOI 和公开数据 DOI。正式 DOI 与许可证将在发布前由作者确认。
