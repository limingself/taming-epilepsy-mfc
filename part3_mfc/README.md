# Part III：Actor–WGAN 均值场分布控制

本目录只保留当前论文 Part III 对应的最终代码：一个模型入口和三张论文图的独立绘图脚本。

## 文件对应关系

| 文件 | 唯一职责 | 论文输出 |
|---|---|---|
| `part3_model.py` | Actor–WGAN 训练、冻结后评价及 matched-ablation 评价 | 冻结 checkpoint 与评价 Source Data |
| `figure_06_actor_wgan_control.py` | 三类代表节点的轨迹、占据分布和控制输入 | Fig. 6 |
| `figure_07_mfc_ablation.py` | 同训练预算的 WGAN、图传播和偏差反馈消融 | Fig. 7 |
| `figure_08_all36_controlled_distributions.py` | 36 通道受控分布总览 | Fig. 8 |

图像均以 PDF、SVG、PNG 和 600-dpi TIFF 输出到 `../output/part3/`，数值绘图数据位于 `../output/part3/source_data/`。

## 模型入口

```powershell
python part3_mfc/part3_model.py train --help
python part3_mfc/part3_model.py evaluate --help
python part3_mfc/part3_model.py evaluate-ablations --help
python part3_mfc/part3_model.py source-data
```

- `train`：只读取 run-01 发作间期参考池、发作前缀及冻结的 Part-II Graph-RC-SDE；不能读取 run-02 未来。
- `evaluate`：策略冻结后才打开 run-02 开发窗口，使用同一组重建布朗增量评价无控制、原始 Actor 和 Actor–WGAN。
- `evaluate-ablations`：在相同 Part-II 模型、初始状态、13 节点掩膜、布朗路径和训练预算下比较完整模型及三个消融模型。
- `source-data`：不训练、不重新选结果，只把冻结的主评价和 matched-ablation 评价同步到三个独立绘图脚本读取的公开 Source Data 目录。

## 三张图分别复现

```powershell
python part3_mfc/figure_06_actor_wgan_control.py
python part3_mfc/figure_07_mfc_ablation.py
python part3_mfc/figure_08_all36_controlled_distributions.py
```

每个绘图脚本只生成文件名中标明的那一张论文图，不会额外生成历史图或中间图。
若从空的 `output/part3/source_data/` 开始，应先运行 `part3_model.py source-data`，再依次运行三个绘图脚本。

## checkpoint 说明

- Fig. 6 和 Fig. 8 使用主结果训练运行 `seed20261011_adv050_anchor020` 的冻结评价。
- Fig. 7 使用单独重训练的 matched-ablation 完整模型 `ablation_full_seed20261011_v1`，并与三个同预算消融 checkpoint 比较。
- 两个“完整模型”具有相同架构，但属于不同训练运行，权重并不相同；代码和图注不得把它们表述为同一个 checkpoint。

## 共享数值依赖

`part3_model.py` 仍显式依赖根目录的 `mfc_pipeline/`，其中包含 Graph-RC-SDE 粒子推进、图控制映射、结构化 Actor、WGAN-GP critic 和经验 Fokker–Planck 粒子传播等底层实现。它不依赖已删除的 `part3_mfc/code/` 历史脚本。

严格地说，当前实现是“结构化状态反馈 Actor + 时间条件 WGAN-GP critic + 经验粒子 Fokker–Planck 演化”。没有单独训练的 HJB 值函数网络，因此结果不应被宣称为一个显式求解 HJB PDE 的双网络方法。
