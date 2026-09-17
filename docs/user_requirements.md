# User Requirements

## Explicit instructions

- 参考相邻的 `highway_transition_timing` 与 `lunarlander_transition_timing` 实现本项目。
- 在可安装时使用计划指定的 MiniGrid；已安装并锁定 `minigrid==2.5.0`。
- 使用 ResearchPilot 的代码迭代技能记录**每一个版本**。版本记录只追加，保留失败尝试、验证结果和证据边界。

## Research constraints inherited from the project plan

- Pilot、正式冻结、正式训练和 held-out 评估必须区分；pilot 或参考控制器结果不能作为 R1 确认性证据。
- 三奖励条件只改变权重；保留匹配场景、全部 seed、无返航与耗尽等终局。不得用结果挑选模型或事件定义。
- 输出保留配置、模型和逐步轨迹的可核对信息；高电量干预需固定策略和几何。

## Document Preferences

用户没有指定固定模板。版本记录采用中文正文、英文文件名与可核查的命令和路径；不补造缺失的历史结果或精确时间。
