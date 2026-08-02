# Part I：PLV 脑网络与控制节点选择

本目录只保留与论文 Fig. 1 对应的最终模型和绘图代码。

## 文件

- `part1_model.py`：唯一的 Part I 模型脚本。读取冻结的 HUP060 多频带 PLV 矩阵，构建“最大生成树骨架 + 最强剩余边”的连通图，计算加权中心性并按固定阈值选择控制节点；仅生成 Source Data，不绘图。
- `figure_01_plv_network_selection.py`：Fig. 1 独立绘图脚本。仅读取模型生成的 Source Data，导出 PNG、TIFF、PDF 和 SVG，不重新计算网络或节点选择。
- `config.yaml`：论文最终参数及输入、输出路径。
- `hup060_clinical_labels.csv`：冻结的 SOZ/切除区标签；标签仅在节点选择完成后用于回顾性着色，不参与中心性评分和阈值选择。

## 复现顺序

从项目根目录执行：

```powershell
python part1_network/part1_model.py
python part1_network/figure_01_plv_network_selection.py
```

模型输出位于 `output/part1/source_data/figure_01/`；论文图位于 `output/part1/figure_01/`。

## 冻结结果

- 36 个节点、63 条边、图密度 0.10、1 个连通分量。
- 中心性权重：degree 0.30、betweenness 0.60、eigenvector 0.10。
- 严格选择规则：分数大于训练期第 65 百分位阈值。
- 共选择 13 个控制节点，其中 5 个与 SOZ/切除区并集重合。

拆分后的绘图结果已经过回归验证：PNG 和 TIFF 与论文现用 Fig. 1 逐像素一致。
