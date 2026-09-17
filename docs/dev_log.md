# Development Log — Recharge Return ONSET

> ResearchPilot 阶段 F 的追加式版本日志。建立于 2026-09-17。下列 v0.0.0–v0.2.0 为根据现存文件和本次会话追溯整理的**开发版本编号**，并非 Git tag 或可下载的旧版快照。此前没有本项目的 `docs/dev_log.md`、`results/` 或 Git 历史，因此旧版逐文件哈希不可恢复。新版本从 v0.2.1 起保存文件哈希清单。此文件以后只追加，不改写历史条目。

## Project Overview

| 项目 | 内容 |
| --- | --- |
| 研究目标 | 电池储备与任务产出权衡下的首次确认返航 onset |
| 参考项目 | `../highway_transition_timing`、`../lunarlander_transition_timing` |
| 当前环境 | MiniGrid 2.5.0 的 17×9 自定义任务，六个原子动作，完整状态观测 |
| 训练与分析 | Stable-Baselines3 Double DQN；匹配场景、seed 级统计及完整性门控 |
| 运行策略 | 本地单元/短程 smoke；完整 pilot、正式冻结及正式训练尚未进行 |

## Implementation Progress

| 模块 | 路径 | 状态 | 证据 |
| --- | --- | --- | --- |
| 任务计划 | `recharge_return_plan.md` | 已写，正式参数待 pilot 校准 | 计划明确 R1–R4 与证据边界 |
| 环境、事件 | `experiments/recharge_return/env.py`, `events.py` | pilot 实现 | MiniGrid 地图/渲染与环境测试 |
| DDQN、运行入口 | `agent.py`, `run.py` | pilot 实现 | 40 步 smoke 训练及 checkpoint 评估 |
| 配置、场景 | `configs/pilot.yaml`, `grids/*.json` | pilot 候选 | 开发/保留网格各 27 场景 |
| 汇总、集群 | `scripts/aggregate_recharge_return.py`, `katana_recharge_pilot_*.pbs` | 代码完成，集群未运行 | 汇总门控单元测试 |
| 正式实验 | formal config、20 seed、held-out 模型评估 | 未开始 | 不存在正式结果 |

## Version Ledger

| 版本 | 日期 | 变更性质 | 可复原快照 | 状态 |
| --- | --- | --- | --- | --- |
| v0.0.0 | 2026-09-10 | 实验计划 | 仅现存计划文档 | 计划 |
| v0.1.0 | 2026-09-17 | Gymnasium 原型 pilot | 否；后被 v0.2.0 覆盖 | 历史开发版 |
| v0.2.0 | 2026-09-17 | MiniGrid 2.5.0 后端 pilot | 否；在 v0.2.1 建立快照前未单独封存 | 当前代码基线 |
| v0.2.1 | 2026-09-17 | 追加式版本治理与当前文件快照 | `docs/versions/v0.2.1.json` | 文档版 |

## Development Log Entries

### 2026-09-10 — v0.0.0：实验计划

**改动原因**：将 ONSET 从驾驶和着陆扩展到作业到补能的阶段切换。

**记录内容**：`recharge_return_plan.md` 定义 R1–R4、需充电/充足电量场景、RES/BAL/PROD 奖励、物理确认返航、20 seed 正式分析与所有终局保留原则。

**验证与结果**：此版本只有计划；没有实现、模型或实测结果。

**证据边界**：计划日期来自文档；无该日期的 Git tag 或文件哈希。

### 2026-09-17 — v0.1.0：可运行的 Gymnasium pilot 原型（追溯）

**改动原因**：在初次安装 MiniGrid 时，沙箱内 `pip` 无法解析包索引域名，需要先验证自定义任务和实验接口。

**改动内容**：实现离散走廊环境、能耗/作业/充电与奖励、返航候选和确认检测；加入 Double DQN、确定性训练场景采样、开发和 held-out 各 27 场景、训练/评估/冻结 CLI、逐步轨迹、汇总和 Katana pilot 脚本。

**验证与结果**：最终原型检查为 12 项测试通过；40 步训练与固定模型评估 smoke 跑通。参考阈值控制器在开发及 held-out 原始场景均为 27/27 完成并确认返航；充足电量组均为 27/27 完成且无返航。

**局限**：40 步模型仅验证流水线，不能推断学习效果。此版源码未单独归档，不提供虚构的文件哈希或可重现旧快照。

### 2026-09-17 — v0.2.0：切换到 MiniGrid 2.5.0（追溯）

**改动原因**：网络访问恢复；计划原定使用 MiniGrid 后端。

**改动内容**：在项目 `.venv` 中安装 `minigrid==2.5.0`；环境改为继承 `MiniGridEnv`，由 MiniGrid `Grid`、`Wall`、`Floor` 提供地图与 RGB 渲染。项目继续定义完整状态、六动作、电池、工作、充电和事件语义；`requirements.txt` 固定版本并更新 README/协议。

**验证与结果**：13 项测试通过；MiniGrid 地图和 RGB 渲染检查通过；40 步训练、checkpoint 保存及评估 smoke 跑通。held-out 参考控制器原始组 27/27 完成且确认返航，高电量组 27/27 完成且无返航。

**局限**：参考控制器只检查任务可行性；完整 3-seed pilot、20-seed formal、冻结和策略效果分析均未进行。此版源码在版本治理建立前未单独封存。

### 2026-09-17 — v0.2.1：建立 ResearchPilot 版本记录

**改动原因**：用户要求记录每一个版本；项目此前没有阶段 F 的追加式日志或版本快照规则。

**改动内容**：新增 `docs/user_requirements.md` 保存版本记录与证据边界；新增本日志追溯 v0.0.0–v0.2.0；新增 `AGENTS.md` 要求后续每次代码、配置、协议或实验结果变化都追加日志并编号；README 添加版本入口；`docs/versions/v0.2.1.json` 保存当前文件 SHA-256 与环境摘要。

**文档同步**：本次为记录体系变更，没有修改任务、奖励、事件或训练配置。

**验证与结果**：待完成当前文件哈希及文档链接校验后补录。此前 v0.2.0 的 13 项测试结果不冒充 v0.2.1 的独立测试。

## Known Issues and Next Run

- 尚无本项目的完整 pilot 学习策略结果；正式冻结和正式 held-out 评估均未开始。
- 当前 `.venv` 的 MiniGrid 安装可验证本地后端；Katana 依赖锁和墙钟预算仍需在实际运行环境确认。
- 下一次改动须先读本日志和 `docs/user_requirements.md`，沿用新的唯一版本号，在每个被改文件完成后追加相应记录，并在验证后追加结果。实验结果即使不符合假设也必须保留。

### 2026-09-17 — v0.2.1 验证结果（追加）

- `docs/versions/v0.2.1.json` 收录 21 个当前文件的 SHA-256；逐文件重算全部匹配，文档链接目标存在。`docs/dev_log.md` 因只追加而明确排除在固定快照之外。
- `.venv/bin/python -m pytest -q`：13 passed；两条来自 pygame/pkg_resources 的弃用警告，未影响测试。
- `python -m compileall -q experiments scripts tests`：通过。
- `bash -n scripts/katana_recharge_pilot_train.pbs scripts/katana_recharge_pilot_eval.pbs`：通过。
- 本次仅建立版本记录，没有运行完整 pilot 或正式实验；v0.2.1 的哈希清单不代表旧版 v0.0.0–v0.2.0 的源码快照。

### 2026-09-17 — v0.2.2：Git 发布准备与计划路径同步

**改动原因**：用户指定 `https://github.com/cattiesu2025/recharge_transition_timing.git` 并要求推送。提交前发现计划文档现位于 `docs/`，而 README、AGENTS 和 v0.2.1 清单仍保留原根目录路径。

**改动内容**：
- `README.md`：修正计划链接，并将当前版本清单指向 v0.2.2。
- `AGENTS.md`：修正下一轮改动必须阅读的计划路径。
- `docs/dev_log.md`：追加本条发布准备记录，保留 v0.2.1 作为历史清单。
- `docs/versions/v0.2.2.json`：保存当前路径和文件内容的 SHA-256。
- Git：初始化 `main`，配置用户指定的空远端，排除 `.venv`、输出和缓存。

**协议影响**：无。任务、奖励、模型和评估规则未变。

**验证与发布结果**：待清单校验、提交和远端推送后追加；若认证失败，将记录实际状态，不称已发布。

### 2026-09-17 — v0.2.2 发布验证（追加）

- `docs/versions/v0.2.2.json` 的 21 个文件路径均存在且 SHA-256 全部匹配；`v0.2.1.json` 保持不变，仍记录移动前的历史路径。
- `git diff --cached --check` 通过；初始提交为 `ad90e05`（`Initialize recharge return pilot v0.2.2`）。
- 已将 `main` 推送到用户指定的 `https://github.com/cattiesu2025/recharge_transition_timing.git`，Git 回报 `main -> main` 并建立 `origin/main` 跟踪关系。
- 本次发布只包含 pilot 代码和文档，不包含 `.venv`、生成输出或完整 pilot/formal 结果。
