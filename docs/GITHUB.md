# GitHub 协作

仓库只包含源码、测试、脚本和文档；数据、模型权重、日志、虚拟环境均已加入 `.gitignore`。

## 首次发布

先在 GitHub 创建空仓库（建议名称 `safe-cadence`），不要同时生成 README、许可证或 `.gitignore`，避免首次推送冲突。

```bash
git init
git add .
git commit -m "Prepare reproducible server workflow"
git branch -M main
git remote add origin https://github.com/<账号>/safe-cadence.git
git push -u origin main
```

仓库访问权限由拥有者决定。私有仓库 fork 的可用性取决于 GitHub 账户/组织设置；若不允许 fork，可改为授权协作者在分支提交。

## 接手工作

```bash
git clone https://github.com/<账号>/safe-cadence.git
cd safe-cadence
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash scripts/server_run.sh doctor
```

然后按照 [服务器任务清单](SERVER_TASKS_CN.md) 完成数据接入与实验。建议每个独立任务使用分支、Pull Request，并附运行命令、结果和 `outputs/doctor.json`（确认不含敏感路径后再分享）。

代码变更后运行 `python -m scripts.build_manifest`，并将更新后的 `PACKAGE_SHA256.json` 一起提交。GitHub Actions 会检查回归与诊断。许可证请仓库拥有者明确选择，不自动替其授权。
