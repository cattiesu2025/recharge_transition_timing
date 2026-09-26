# v0.7.0 网格扫描结果与 pilot 运行

## 已完成的计算检查

24–60全部37电量×9布局=333组合，743219次路线/电量计算；每个路线类包含所有任务有序子集和两种轴顺序最短路线，不代表全局MDP最优。全部评分矩阵、生成规则和逐组合结果见 `outputs/recharge_return_v0.7.0_sweep/`。路线重复或电量相邻不构成独立统计样本。

主ONSET使用4×4无任务区域。严格正分离表示PROD全部并列最佳路线的最早ONSET仍晚于RES全部并列最佳的最晚ONSET；重叠不证明等价。

|布局族|组合数|PROD严格晚于RES|并列最佳时间集合重叠|反向/缺事件|
|---|---:|---:|---:|---:|
|原布局|37|20|17|0|
|顺路任务：4/6任务，各2布局|148|17|131|0|
|两臂绕行任务：4/6任务，各2布局|148|85|63|0|
|合计|333|122|211|0|

原布局严格分离电量：27、35–36、44–60。44–53时RES最佳做4任务、BAL/PROD做5任务；54–60时后两组做6任务，RES仍4任务。32–34及37–43等区间仍重叠，完整保留。

5×5敏感性为121/333严格正分离；只有原布局电量27从正分离变成重叠。BAL与PROD只有2/333组合严格分离，其余重叠，不声称三组均可区分。所有组合全体最佳（包括失败、空返）均为成功且有工作方案。

RES严格次优回报差范围0.001715–0.195745，中位0.023823；这不是学习收敛保证。原布局60的两方案储备项仍均为0，支持任务产出与时间的权衡，不声称纯储备机制。绕行族比顺路族更常分离是当前预定小型设计集的描述性结果，不是布局总体统计推断。

## 新学习环境

10×10内部可通行MiniGrid，外围另加墙，共12×12渲染格。左上出发，右下正电到站终止；耗尽优先、300步超时独立、完成任务不立即终止。五动作上/下/左/右/WORK；观察位置、电量、剩余时间、全部剩余任务、合法动作mask，禁止按电量强制返航。最后WORK为辅助，主ONSET保持进入充电专用区并持续向站推进的定义。

训练按独立生成seed采样顺路/绕行、4/6任务、24–60电量，每条件seed内episode场景序列匹配。每个固定最终模型评估全部9×37场景。该manifest是开发集，未声称正式held-out。固定600k、RES/BAL/PROD×4100/4101/4102，保留全部模型，禁止挑checkpoint或按效果删场景。

## Commands

本地重新计算扫描（须新输出目录，拒绝覆盖）：

```bash
.venv/bin/python -m scripts.sweep_grid_routes --output outputs/recharge_return_v0.7.0_sweep_repeat
```

本机已有九组6000步smoke，输出位于 `outputs/recharge_return_v0.7.0_grid_smoke/`，仅验证学习更新、保存重载与全场景评估链路，不能当作完整pilot或证明学会任务。完整训练沿用项目Katana执行流程；没有自动提交集群作业。

将本轮新增/修改源文件同步到Katana现有项目根目录（在本地项目根执行）：

```bash
rsync -avhR \
  experiments/recharge_return/agent.py \
  experiments/recharge_return/grid_env.py \
  experiments/recharge_return/grid_run.py \
  experiments/recharge_return/grid_sweep_plan.md \
  experiments/recharge_return/configs/pilot_line_masked_600k.yaml \
  scripts/audit_grid_routes.py \
  scripts/sweep_grid_routes.py \
  scripts/katana_recharge_grid_v0.7.pbs \
  z5535967@kdm.restech.unsw.edu.au:/home/z5535967/projects/recharge_transition_timing/
```

随后在Katana项目根目录提交：

```bash
qsub scripts/katana_recharge_grid_v0.7.pbs
```

每任务写入 `outputs/recharge_return_v0.7.0_grid_pilot/<condition>/<seed>/`，训练失败则不执行后续评估；已有目标目录拒绝覆盖。8小时资源请求仅沿用旧pilot模板，未实测新版集群墙钟。

九组同步回本地后聚合（要求9个600k最终模型、统一设定hash及2997个唯一评估）：

```bash
.venv/bin/python -m experiments.recharge_return.grid_run \
  --aggregate outputs/recharge_return_v0.7.0_grid_pilot
```

输出 `pilot_summary.json`，保留每个seed/场景的成功、空返、耗尽、工作量和ONSET；不产生formal p值。smoke角色和非600k预算被拒绝。新观测/动作与旧一维checkpoint不兼容，不加载旧模型继续训练。
