# Expanded validation — v0.7.3

2026-09-28。用户要求扩大并固定验证布局，再检验checkpoint选择，保留final对照。
此版复用v0.7.2已保存的12×9模型，只重评估/选择，不训练、不改奖励、模型或ONSET。
由于方案是在观察旧开发结果后提出，所有结果仍是探索性pilot，不是确认性证据。

## Fixed layout coverage

验证集：aligned、detour、scattered三族 × 4/6任务 × 每格10布局 = 60布局。
aligned/detour使用既有生成器；scattered从10×10内部除起点和右下5×5区域之外的74个合法格无放回均匀抽取。
scattered扩大到原训练生成器未覆盖的中部组合，是明确的验证分布扩展，不声称已有模型在此分布训练过。
三族与任务数均等权，每布局配24–60所有整数电量，共2220场。不得按回报或onset筛布局。

每个族/任务数组从生成seed9300开始递增，跳过与原9开发布局、旧8验证布局及已选新布局坐标完全相同的候选，取前10个。
仅几何去重，记录实际接受seed与完整坐标。不是挑选最有差异或最易成功的地图。

另固定同规则的60布局审计集，从seed9400开始递增，排除全部验证/历史布局，亦为2220场。
此新审计集不参与checkpoint选择，只在选择锁定后评估best/final，以免继续用旧开发集判断选择是否可靠。
它不是正式held-out：实验设计已受到旧结果影响，且训练随机生成可能偶然重合。
保留原333场开发评估供历史比较；不将原失败地图单独加入验证集。

## Selection and provenance

九组条件/seed独立，在全部12个50k间隔checkpoint中，按新验证集平均未折扣累计奖励最大值选取；严格并列选更早步数。
失败、空返照常计入，选择不读取事件。保存每候选逐场回报、工作/空转/终局和轨迹，全部候选分数；不混用新旧验证回报。
复用v0.7.2 evaluate与select_best，源文件保持不变。输入须完整v0.7.2模型、selection/metadata/specification及匹配hash。
运行前校验12候选、固定预算、condition/seed、原选择正确、best/final哈希、训练源版本与当前环境相关文件一致；拒绝输入/输出重合和覆盖。
新输出保存manifest、输入hash、来源说明、selection.json、best.zip、final.zip、各自新审计和原开发评估、完成metadata。
原模型、v0.7.2选择记录与所有旧评估只读。新选择不改称旧实验的主结果。

## Run and validation

同步源码后，在Katana项目根目录运行：

```bash
qsub scripts/katana_recharge_grid_v0.7.3.pbs
```

输入默认`outputs/recharge_return_v0.7.2_grid_selection/<condition>/<seed>/`，
输出`outputs/recharge_return_v0.7.3_expanded_validation/<condition>/<seed>/`。
九任务数组不再训练；每组12×2220验证加2×(2220+333)评估，共31746场。
8小时墙钟为待实测预算。中断不自动恢复，已有输出拒绝覆盖。

测试包括生成重现/数量/合法坐标/三套布局不重合、仅回报选择、输入hash与预算守卫、防覆盖与完整写入链路。
`--smoke`仅使用每族/任务数1布局、每集222场、50k和600k两个候选，使用真实保存模型验证加载链路；输出必须含smoke。
smoke可含已训练模型，但缩减候选/布局的选择结果不能当成本版全量结果或成功证据。
完整结果出来后比较best/final在新审计集的回报、空转、任务量、终局、按族分层和ONSET；无改善也完整报告。
