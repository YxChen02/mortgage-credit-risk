# 房贷贷后违约预警与模型监控

基于美国住宅抵押贷款面板数据，在每个观测期为仍在贷的借款人估计未来 3 期的违约风险。项目用 SQL 构建时点样本和历史特征，对比逻辑回归与直方图梯度提升树，并通过时间外回测和逐期评分 PSI 评估模型。

## 数据与样本

使用 [Credit Risk Analytics 发布的 Mortgage 数据集](https://www.creditriskanalytics.net/datasets-private2.html)。原始 CSV 含 622,489 条观测记录，覆盖 60 个匿名化观测期。删除 305 条完全重复记录后，按借款人 ID 和放款期识别贷款。若同一组合在同一期仍有冲突记录，则剔除整个组合，共剔除 10 个组合。

- **特征**：使用评分时点及此前可得的信息，包括当前余额、贷款价值比（LTV）、放款时 FICO 评分，以及最近三次观测的余额和 LTV 均值。
- **标签**：未来 3 期内首次违约记为正类。窗口内先提前结清，或到窗口末仍可追踪且未违约，记为负类。其余样本的标签记为缺失。
- **时间切分**：训练样本的标签窗口最晚在第 40 期结束，测试样本的评分时点从第 41 期开始。两个集合有 12,735 位重叠借款人，回测衡量的是模型在存量房贷组合上的跨期表现。

## 回测结果

训练集包含 344,541 条时点记录。时间外测试集包含 168,084 条标签已知记录，违约率为 5.14%，另有 451 条记录标签缺失。

- **逻辑回归**：AUC 0.686，KS 0.282，平均精确率 0.094。
- **直方图梯度提升树**：AUC 0.705，KS 0.298，平均精确率 0.107。

模型区分度使用标签已知的记录计算。评分 PSI 沿用训练期分箱，监控各测试期的评分分布，纳入标签缺失记录。

- [模型指标与样本统计](results/metrics.json)
- [逐期评分 PSI 与样本统计](results/monitoring.csv)

## 复现

从数据发布页下载 Mortgage CSV 压缩包，解压至 `data/mortgage.csv`，然后在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python run.py data/mortgage.csv --cutoff 40 --horizon 3 --output results/local
```

命令将生成 `metrics.json` 和 `monitoring.csv`。如需用 LightGBM 对比逻辑回归，可额外安装 `lightgbm` 并添加 `--booster lightgbm`。

SQL 特征与标签定义见 [features.sql](features.sql)，训练和评估入口见 [run.py](run.py)。
