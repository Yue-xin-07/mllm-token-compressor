# AGENTS.md - 核心开发铁律

## ⚠️ 绝对红线（Mandatory Rules）
1. **Memory Bank 优先原则**：任何会话开始的第一步，必须完整读取 `memory-bank/` 下的所有文档。未经允许不得私自修改技术栈。
2. **拒绝巨型单文件（Anti-Monolith）**：保持高模块化设计。每个脚本/类职责单一，严禁单文件超过 300 行。
3. **测试驱动/闭环原则**：任何算法模块编写完毕后，**必须**在 `tests/` 目录下提供对应的单元测试（`pytest`），并在终端运行通过后，方可进入下一步。
4. **进度实时同步**：每完成一个子步骤，必须立即更新 `memory-bank/progress.md`和`memory-bank/architecture.md` 并提交 Git。

## 🧠 科学家思维与算法设计规范（Research & Algorithm Principles）
面向长上下文 Agent 场景的多模态 Token 动态压缩与推理加速，追求**高准确率保留、真实显存/KV Cache 节省、端到端推理提速**三者统一，拒绝无依据的过度复杂设计：
1. **第一性原理与问题对齐**：紧扣长上下文 Agent 中视觉 Token 数量大、冗余高、长序列延迟与显存压力大的核心痛点设计算法，先明确假设与取舍（Trade-off），再编写实现。
2. **跨领域规律借鉴与融合**：
   - 借鉴**信息论与显著性筛选**：区分高信息/强任务相关 Token 与低信息/背景冗余 Token，避免一刀切误伤关键细节（如小目标、文字、关键状态）。
   - 借鉴**操作系统缓存/工作集思想**：结合上下文演变与任务需求进行动态预算分配与冗余抑制，优先保住当前最关键的“活跃工作集”。
   - 借鉴**算法-系统协同设计（Co-Design）**：兼顾后续注意力计算、位置编码（如 M-RoPE）对齐、KV Cache 占用与算子执行友好性，使压缩能转化为真实系统收益。
3. **可解释、可消融、可验证**：算法核心逻辑必须拆分为清晰的独立函数/类，支持不同策略开关与参数配置，便于做消融实验对比和单元测试验证；所有关键张量维度变化与边界条件必须显式断言或处理。

## AutoDL 远程连接（Codex / VS Code）

- 项目目录：`/root/autodl-tmp/mllm-token-compressor`
- SSH 目标：用户 `root`，主机 `connect.nmb1.seetacloud.com`，端口 `18801`。
- 当前 Codex Windows 执行环境已配置专用别名 `autodl-nmb1-direct`，且本机 SSH 公钥已加入 AutoDL 的 `/root/.ssh/authorized_keys`。同一环境可直接运行 `ssh autodl-nmb1-direct`，无须密码提示；可用 `ssh -o BatchMode=yes autodl-nmb1-direct 'echo KEY_AUTH_OK'` 验证。不要向用户索要聊天密码，不要复制或记录私钥。若 BatchMode 失败，说明当前执行环境未加载对应私钥，应安全配置 key/agent。
- 若该别名在其他机器不可用，或连接在 SSH banner 前报 `Connection timed out during banner exchange`，可能是本机 DNS 将域名解析到了内部地址。用 Cloudflare DoH 查询当前公网 IPv4，再以该 IP 连接：

```powershell
$dns = Invoke-RestMethod -Uri 'https://cloudflare-dns.com/dns-query?name=connect.nmb1.seetacloud.com&type=A' -Headers @{ Accept = 'application/dns-json' }
$ip = ($dns.Answer | Where-Object { $_.type -eq 1 } | Select-Object -First 1).data
ssh -p 18801 -o "HostKeyAlias=[connect.nmb1.seetacloud.com]:18801" root@$ip
```

- 保持 SSH 默认主机密钥校验开启。上面的 `HostKeyAlias` 用已信任的 `[connect.nmb1.seetacloud.com]:18801` 主机记录校验服务器；若校验失败或密钥变化，停止连接并先向用户核实，不要关闭校验。
- 当前别名使用的公网 IP 可能变化；别名失效时通过 DoH 重新查询并更新 SSH 配置中的 `HostName`。不要改动 VS Code 原有的 `seetacloud` 配置。
