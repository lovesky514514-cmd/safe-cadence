"""Offline server diagnostics with actionable machine-readable results."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

# Failure messages are kept beside their checks so they remain actionable.
REMEDY = {
    'PYTHON': '安装 Python 3.10+，并用虚拟环境重试。',
    'DEPENDENCIES': '用当前 Python 执行 pip install -r requirements.txt。',
    'ENCODING': '恢复原始 UTF-8 文件；Windows 启动脚本必须是 ASCII/CRLF。',
    'INTEGRITY': '文件可能已修改或缺失；重新解压完整 ZIP，开发修改后需重新生成清单。',
    'DATA': '核对 0～4 快照及映射文件；先运行 audit 查看详细错误。',
    'SELF_CHECK': '查看 logs/last_error.log 并复现 selfcheck；避免使用真实数据。',
    'TESTS': '运行 python -m pytest -q tests_preflight tests_research 查看失败断言。',
}


def diagnose(root: Path, *, data: str | None, report: str, quick: bool) -> int:
    """Run offline checks and save a JSON report; 0 means no FAIL items."""
    checks: list[dict] = []

    def record(code: str, status: str, detail: str) -> None:
        checks.append({'code': code, 'status': status, 'detail': detail,
                       'remedy': REMEDY.get(code, '') if status == 'FAIL' else ''})
        print(f'[{status}] {code}: {detail}', flush=True)

    record('PYTHON', 'PASS' if sys.version_info >= (3, 10) else 'FAIL',
           f'Python {platform.python_version()}')
    versions = {}
    missing = []
    for module in ('numpy', 'scipy', 'torch', 'pytest'):
        try:
            obj = importlib.import_module(module)
            versions[module] = str(getattr(obj, '__version__', 'installed'))
        except (ImportError, OSError, RuntimeError) as exc:
            missing.append(f'{module}: {exc}')
    record('DEPENDENCIES', 'FAIL' if missing else 'PASS',
           '; '.join(missing) if missing else ', '.join(f'{k}={v}' for k, v in versions.items()))

    from scripts.check_encoding import check
    from scripts.verify_package import verify
    record('ENCODING', 'PASS' if check() == 0 else 'FAIL', 'UTF-8 与批处理编码')
    record('INTEGRITY', 'PASS' if verify() == 0 else 'FAIL', '仓库文件 SHA-256')

    if data:
        try:
            from safecadence_preflight.ckge_protocol import audit_ckge
            stages = audit_ckge(data, snapshot_count=5)['stage_statistics']
            record('DATA', 'PASS', f'{len(stages)} 个增量快照通过审计')
        except (OSError, ValueError, KeyError) as exc:
            record('DATA', 'FAIL', f'{type(exc).__name__}: {exc}')
    else:
        record('DATA', 'SKIP', '未提供 --data；数据验证将在服务器上执行')

    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(root), env.get('PYTHONPATH', '')]))

    if not quick and not missing:
        for name, command in (
            ('SELF_CHECK', [sys.executable, 'launcher.py', 'selfcheck']),
            ('TESTS', [sys.executable, 'launcher.py', 'test']),
        ):
            try:
                result = subprocess.run(command, cwd=root, env=env,
                                        capture_output=True, text=True,
                                        encoding='utf-8', errors='replace', timeout=180)
                detail = (result.stdout + '\n' + result.stderr).strip().splitlines()
                record(name, 'PASS' if result.returncode == 0 else 'FAIL',
                       f'exit={result.returncode}; ' + ' | '.join(detail[-6:]))
            except subprocess.TimeoutExpired:
                record(name, 'FAIL', '180 秒超时；检查死锁、性能和系统负载')
    else:
        for name in ('SELF_CHECK', 'TESTS'):
            record(name, 'SKIP', '快速模式或依赖缺失')

    torch_info = {}
    if 'torch' in versions:
        import torch
        torch_info = {'cuda_available': bool(torch.cuda.is_available()),
                      'torch_cuda_version': torch.version.cuda,
                      'devices': [torch.cuda.get_device_name(i)
                                  for i in range(torch.cuda.device_count())]}
    report_file = Path(report)
    if not report_file.is_absolute():
        report_file = root / report_file
    report_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {'time_utc': datetime.now(timezone.utc).isoformat(),
               'platform': platform.platform(), 'python': platform.python_version(),
               'dependencies': versions, 'gpu': torch_info, 'checks': checks,
               'ok': all(c['status'] != 'FAIL' for c in checks)}
    report_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    print(f'诊断报告：{report_file}', flush=True)
    return 0 if payload['ok'] else 1
