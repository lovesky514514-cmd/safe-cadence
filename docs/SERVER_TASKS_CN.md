# 服务器任务清单

将仓库 fork 后，在自己的分支完成以下工作；保留命令、日志和提交哈希，通过 Pull Request 回传。**所有尚未执行的实验均视为待办。**

## 1. 环境验收

```bash
git clone <fork-url>
cd safe-cadence
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash scripts/server_run.sh doctor
```

**交付**：`outputs/doctor.json`、OS/Python/Torch/CUDA/驱动版本、CPU/GPU/内存、安装命令与退出码。不要上传 API key、主机凭证或隐私路径。

> GPU 不可用并不妨碍 CPU 自检，但 GPU 实验必须单独验收驱动和适配的 PyTorch 构建。Windows 上执行 `start.bat doctor`。

## 2. 数据获取与审计

```bash
bash scripts/server_run.sh download --dataset WN_CKGE
bash scripts/server_run.sh doctor --data data/WN_CKGE
bash scripts/server_run.sh download --dataset FB_CKGE
bash scripts/server_run.sh audit --data data/FB_CKGE
```

**必须核对**：五阶段 `train_id.txt` 是**阶段增量**；实体 ID 追加且不重排；关系映射一致；`valid/test` 不进入训练和负采样；过滤排名使用当前阶段可见的事实，不得提前读取未来阶段。保存数据来源 URL、固定 commit、逐文件 SHA-256、审计 JSON。核实数据的使用条款。

## 3. 完成正式实验实现

先运行接入烟测（**仅证明入口通畅**）：

```bash
bash scripts/server_run.sh pilot --data data/WN_CKGE --snapshots 2 --steps 12
```

随后实现与验收完整五阶段训练，不得把 `pilot` 的输出当正式结果。需要补齐的工作：

- Train / Replay / 冻结旧实体 / 正则化基线，以及 LKGE、IncDE、FastKGE、TADG 官方实现的可复现对照。
- 统一双向 filtered MRR、Hits@1/3/10、旧查询持续排名保护、未保护查询损害、更新接受/拒绝率。
- 固定至少五个随机种子，保留超参搜索范围、设备信息、等总时间/等计算预算与日志。
- 报告索引建立、训练、认证、回退、评估的独立时间和峰值内存；分别测试不同保护查询比例。
- 对关键认证决策做独立穷举核对和高精度数值检查；说明设备上的浮点条件。不能将一次 float64 一致性测试写成形式化保证。

## 4. 结果回传与失败标准

建议把 `outputs/` 中的报告、实验脚本提交到新分支；较大的权重和数据使用外部存储并附校验值，**不要直接提交到 GitHub**。

每次提交应附：命令、代码 commit、数据 hash、随机种子、依赖锁定清单、实验结果及失败日志。若简单冻结或事务式发布在相同成本下追平当前方法，应保留负结果并明确说明。

Pull Request 必须通过测试；未通过的数据审计、误放行、未来信息泄漏或错误退出码是合并阻断项。更多处理步骤见 [故障手册](BUG_SELF_CHECK_CN.md)。
