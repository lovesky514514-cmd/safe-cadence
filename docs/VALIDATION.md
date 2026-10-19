# 交付验证记录

当前本地检查环境：Linux、Python 3.13.5、PyTorch 2.10.0+cpu、NumPy 2.3.5、SciPy 1.17.0。

| 检查项 | 结果 | 说明 |
|---|---|---|
| 静态 Python 编译 | 通过 | `compileall` |
| 独立 ZIP 解压回归（3 份目录） | 77/77，三次通过 | 包括中文目录路径 |
| 离线数据/训练烟测 | 三次通过 | 自生成小数据；不下载数据 |
| UTF-8、Windows BAT 静态检查 | 三次通过 | BAT 使用 ASCII/CRLF |
| 文件 SHA-256 校验 | 三次通过 | 对照随包清单 |
| 实际 Windows BAT 执行 | 待验证 | GitHub Actions 已安排 Windows 任务 |
| 正式 CKGE 数据/服务器 CUDA | 待验证 | 需下载真实数据并在目标服务器执行 |

最初将三次完整检查串在一个长命令中时出现了外层运行超时；拆成独立进程后全部完成，尚未查明该外层超时的根因。出现类似问题时，逐条运行 `doctor` 并保留日志，不能直接宣称运行环境绝无问题。

复现方式：解压本 ZIP 后执行 `bash scripts/server_run.sh doctor`，检查 `outputs/doctor.json` 中各项状态。`DATA: SKIP` 表示未提供服务器数据，**不是**真实数据验证通过。
