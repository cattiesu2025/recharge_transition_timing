# Grid checkpoint selection — v0.7.2

用户已确认实现。此版是development pilot，不是正式冻结或held-out结果。
保持v0.7.0环境、奖励、网络、训练分布、600000步、RES/BAL/PROD和4100/4101/4102。

每50000步保存并验证，共12个候选；保存时已完成该步所在rollout的梯度更新。
验证使用generated_tasks的aligned/detour×4/6任务×9201/9202，共8布局，电量24–60，共296场。
启动前按任务坐标检查验证布局内部唯一且与原9布局不重合；失败即停止，不按结果筛场景。
随机训练可能偶然生成相同布局，不宣称验证场景严格未见。

选择指标是每个episode未折扣sum(reward)的验证集平均值，各条件/seed独立选择。
严格同分选择较早步数；失败与空返全部计入，不读取onset。折扣回报仅作诊断。
验证环境独立，保存并恢复Python/NumPy/PyTorch随机状态，不改变训练环境、replay或探索进度。
保留全部checkpoint、逐场验证回报/轨迹、best.zip、final.zip、选择记录及来源hash。

best和final各评估原9×37=333开发场景，记录任务量、终局、累计和折扣回报、无任务闭环移动数。
主事件沿用v0.7.1的4×4最终进入且正电量净推进2格；保存4/5区域×1/2/3推进敏感性及旧检测器。
已有开发场景不能称独立正式测试集；不按onset或闭环率选模型，保留所有seed和不改善结果。
闭环移动数为相邻成功WORK之间逐次删除返回同位置的移动闭环之总长度，不等于所有非最短路步数。

## Run

同步本版本源码后，在Katana项目根目录执行（旧v0.7 PBS不变）：

```bash
qsub scripts/katana_recharge_grid_v0.7.2.pbs
```

输出：`outputs/recharge_return_v0.7.2_grid_selection/<condition>/<seed>/`。
包含`checkpoints/step_*.zip`、`validation/step_*/`、`best.zip`、`final.zip`、
`selection.json`、`specification.json`、`metadata.json`、`training_metrics.jsonl`和`eval_best/`、`eval_final/`。
已有目录拒绝覆盖。PBS九任务，8小时墙钟沿用原请求，新增验证开销尚待集群实测。
模型不保存replay，失败后不是精确续训快照；不自动覆盖或恢复中断任务。

本地smoke（6000步、3000步保存间隔，仅验证链路）：

```bash
.venv/bin/python -m experiments.recharge_return.grid_selection --condition RES --seed 4100 \
  --smoke --steps 6000 --interval 3000 \
  --output outputs/recharge_return_v0.7.2_smoke/RES/4100
```

smoke不是学习质量证据。正式pilot入口拒绝非600k预算或非50k保存间隔。
