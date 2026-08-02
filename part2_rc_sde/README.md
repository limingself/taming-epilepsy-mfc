# Part II：状态依赖扩散 Graph-RC-SDE

本目录只保留论文 Part II 的一个模型入口和四个逐图绘制入口。

## 唯一模型入口

`part2_model.py` 对应论文中的最终状态依赖扩散 Graph-RC-SDE，而不是旧的常扩散版本。冻结模型为 36 通道、96 个 reservoir 单元、13 维潜状态、延迟 `(1, 8, 32)`、随机种子 11、扩散 ridge 系数 0.1。

```powershell
# 检查冻结模型、参数和文件哈希
python part2_rc_sde/part2_model.py inspect

# 从冻结的匹配滚动结果重新生成 Figs. 2--5 的全部 Source Data
python part2_rc_sde/part2_model.py source-data

# 从原始数据重新执行验证集选模和最终训练（耗时）
python part2_rc_sde/part2_model.py train
```

`source-data` 使用由最终模型生成并冻结的同一组 run-02、context-3、1 s、32 粒子匹配滚动结果，因此不会重新挑选时间窗、粒子或噪声路径。

## 每张论文图的独立代码

- `figure_02_rc_sde_prediction.py`：三类代表节点的轨迹、预测分布和轨迹误差。
- `figure_03_all36_distributions.py`：36 电极预测分布矩阵。
- `figure_04_distribution_errors.py`：36 电极的 W1、均值误差和标准差误差。
- `figure_05_input_ablation.py`：状态、状态+延迟、状态+延迟+PLV 图耦合消融。

每个脚本只生成其文件名所对应的一张图，并同时导出 SVG、PDF、PNG 和 600-dpi TIFF。Source Data 位于 `../output/part2/source_data/`，图片位于 `../output/part2/figure_02/` 至 `figure_05/`。

## 共享依赖说明

为避免 Part III 重复实现相同随机动力学，`part2_model.py` 仍调用项目级共享包 `mfc_pipeline`。其中 Part-II 专用的内部实现为：

- `mfc_pipeline/part2_state_dependent_rc_sde.py`：RC 漂移与状态依赖扩散的数值类；
- `mfc_pipeline/part2_data_pipeline.py`：原始 BIDS-ZIP 数据划分和预处理；
- `mfc_pipeline/part2_training.py`：验证集选择与最终训练过程。

这些是当前模型的内部组成，而不是其他模型版本；用户侧只需运行本目录中的 `part2_model.py`。
