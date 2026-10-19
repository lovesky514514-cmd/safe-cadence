# 自检与故障处理

所有命令在仓库根目录执行；不会默认联网。诊断自动写入 UTF-8 的 `outputs/doctor.json`。

```bash
# Linux：完整离线检查
bash scripts/server_run.sh doctor
# 指定真实数据，检查五阶段协议
bash scripts/server_run.sh doctor --data data/WN_CKGE
# 快速检查（跳过较慢的训练/测试）
bash scripts/server_run.sh doctor --quick
```

Windows 将 `bash scripts/server_run.sh` 替换成 `start.bat`。

## 诊断状态

| 状态 | 含义 | 操作 |
|---|---|---|
| `PASS` | 该检查完成且符合当前条件 | 继续下一步 |
| `WARN/SKIP` | 检查未执行或资料尚缺 | 参照报告补齐；不能当成功 |
| `FAIL` | 检测到阻断问题 | 按错误码修复，重跑 `doctor` |

## 常见问题

<details>
<summary>PYTHON / DEPENDENCIES：找不到解释器或依赖</summary>

- 执行 `python --version`、`python -m pip --version`，确认使用当前虚拟环境。
- 执行 `python -m pip install -r requirements.txt`。
- GPU 版 PyTorch 应根据驱动和 CUDA 版本从官方渠道选择；CPU 检查不需要 GPU。
</details>

<details>
<summary>ENCODING：中文乱码或文件编码错误</summary>

- 用 UTF-8 打开 `.md`、`.py`、`.json`；Windows 建议使用 Windows Terminal。
- `start.bat` 是 **ASCII + CRLF**，中文只由 UTF-8 Python 程序输出。
- 执行 `start.bat encoding` 或 `bash scripts/server_run.sh encoding`；不要直接修改脚本为 ANSI/GBK。
</details>

<details>
<summary>INTEGRITY：缺失文件或 SHA-256 不匹配</summary>

- 先确认是否手工编辑了工作区文件。开发修改后，旧清单不再匹配是预期行为。
- 移植验证应使用未修改的 ZIP 或固定提交；若文件丢失，重新解压/克隆，不要禁用校验。
- 清单不证明代码来源可信，也不能代替仓库提交签名。
</details>

<details>
<summary>DATA：快照目录或映射不合法</summary>

- 核对 `data/WN_CKGE/0..4` 每阶段都含 `entity2id.txt`、`relation2id.txt`、`train_id.txt`、`valid.txt`、`test.txt`。
- 当前阶段的 `train_id.txt` 是**增量**数据，不能错误地视为累计训练集。
- 运行 `bash scripts/server_run.sh audit --data data/WN_CKGE`，对照具体行号修复映射。
- 不要为了通过审计删除冲突行，必须先确认官方数据与当前适配协议是否一致。
</details>

<details>
<summary>SELF_CHECK / TESTS：训练或单测失败</summary>

```bash
bash scripts/server_run.sh selfcheck
bash scripts/server_run.sh test
```

- 查看 `logs/last_error.log`，保留首次完整错误堆栈及执行命令。
- 若测试超时，记录 CPU/GPU/线程数、内存、Python/Torch/NumPy 版本；不要直接调大超时掩盖死锁。
- 稀疏/冻结认证的适用条件不满足时，拒绝认证是正常结果；不要修改容差来强行通过。
</details>

## 提交问题时附上

`outputs/doctor.json`、运行命令和退出码、最短的可复现代码、对应 commit、数据文件 hash、关键错误堆栈。敏感凭证、服务器私有地址及大体量数据不要上传。

**重要**：当前离线测试不证明未知服务器上能无故障运行，也不证明任何范围之外的排名都安全。`pilot` 为小预算烟测，数据测试需在正式服务器另行完成。
