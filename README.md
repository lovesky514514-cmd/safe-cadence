# Safe-Cadence

知识图谱增量更新与排名保护的可运行研究工程。支持离线自检、增量快照审计、小规模训练试跑及服务器实验交接。

> **范围**：当前认证工具只对指定查询及声明的参数冻结条件提供保护判定。试跑不代表完整基准，真实数据和 GPU 结果尚待目标服务器验证。

## 开始

Python 3.10+，在项目根目录运行：

```bash
python -m venv .venv
# Linux
.venv/bin/python -m pip install -r requirements.txt
bash scripts/server_run.sh doctor
```

Windows 使用 `.venv\Scripts\python.exe -m pip install -r requirements.txt` 安装，随后双击 `start.bat`，或运行 `start.bat doctor`。

`doctor` 默认离线运行环境检查、中文编码检查、文件校验、合成训练和全部测试，并生成 `outputs/doctor.json`。首次不需要下载数据。

## 服务器数据

```bash
bash scripts/server_run.sh download --dataset WN_CKGE
bash scripts/server_run.sh doctor --data data/WN_CKGE
bash scripts/server_run.sh pilot --data data/WN_CKGE --snapshots 2 --steps 12
```

数据来源固定于 FastKGE 指定提交。`pilot` 只验证训练流水线能否执行，不是正式性能比较。

## 工作交接

- [服务器任务清单](docs/SERVER_TASKS_CN.md)：数据、实验、结果回传与验收。
- [故障自检手册](docs/BUG_SELF_CHECK_CN.md)：诊断命令、失败类别及处理方法。
- [开发约定](docs/MAINTENANCE.md)：模块职责、测试和安全限制。
- [GitHub 协作](docs/GITHUB.md)：首次发布、fork 和提交。

核心模块在 `safecadence/` 和 `safecadence_preflight/`；`research/` 保存可复跑的探索脚本。默认不联网、不安装软件，也不修改服务器上的已有数据。
