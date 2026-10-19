# 开发与维护约定

## 模块职责

| 目录 | 职责 |
|---|---|
| `launcher.py` | 命令派发、退出码与异常日志 |
| `safecadence_preflight/` | 增量快照协议、受限安全认证、两阶段烟测 |
| `safecadence/` | TransE、历史排名认证、实验优化器 |
| `research/` | 历史探索运行脚本；**不是**稳定服务入口 |
| `scripts/` | 离线自检、SHA-256、编码诊断、系统入口 |
| `tests_preflight/`, `tests_research/` | 回归、数值性质与协议测试 |

## 改动原则

- 优先复用小而明确的函数。避免为简单逻辑加多层包装，不增加无收益的新语言运行时。
- 每个公开函数需说明输入、输出及安全前提；难理解的分支用一两行注释解释**为什么**，不写装饰性分隔线。
- 新增安全路径必须有反例测试（可接受与必须拒绝），不能仅用随机正例测试。
- 对未知数据或不满足冻结条件的认证请求应拒绝，而不是猜测结果。
- 绘图和特定国家数据演示需另装 `requirements-optional.txt`；默认离线；联网下载必须显式执行 `download`；日志文件 UTF-8；输出路径默认放 `outputs/`。
- 数据、密钥、权重不提交；补丁通过 CI 和 `doctor` 后再合并。

## 验证命令

```bash
bash scripts/server_run.sh doctor
bash scripts/server_run.sh test
bash scripts/server_run.sh encoding
bash scripts/server_run.sh verify
```

修改源码后运行 `python -m scripts.build_manifest` 更新清单，再执行 `verify`；PR 必须包含对应清单更新。全新解压/克隆后的第一次运行应是 `doctor`，不是正式训练。
