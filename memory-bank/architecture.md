# 架构与目录说明 (Architecture & File Manifest)

## 项目规则与版本控制

- `AGENTS.md`: AI 开发规则、Memory Bank 更新要求、AutoDL 远程连接说明。
- `.gitignore`: 排除 Python 缓存、虚拟环境、Notebook 检查点、模型权重、数据集、运行产物和本地临时文件。

## 记忆库

- `memory-bank/architecture.md`: 本文件，记录项目文件和目录职责。
- `memory-bank/implementation-plan.md`: Step 0–4 的实现任务和验收标准。
- `memory-bank/progress.md`: 当前阶段、已完成事项、下一步与阻碍。
- `memory-bank/tech-stack.md`: 目标模型、技术栈、性能目标和硬件约束。

## 源码与验证目录

- `compressor/`: Token 压缩核心算法源码；`.gitkeep` 仅用于保留当前空目录。
- `scripts/`: 环境检查、模型下载和压测工具；`.gitkeep` 仅用于保留当前空目录。
- `tests/`: Pytest 单元测试和正确性回归；`.gitkeep` 仅用于保留当前空目录。
