"""CLI for offline checks, CKGE data preparation, and reproducible smoke runs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def show(message: str) -> None:
    print(message, flush=True)


def setup_help() -> None:
    show('创建环境：python -m venv .venv')
    show('Windows：.venv\\Scripts\\python.exe -m pip install -r requirements.txt')
    show('Linux：.venv/bin/python -m pip install -r requirements.txt')
    show('安装 PyTorch CUDA 版本前，请核实服务器驱动和官方兼容说明。')


def check_environment() -> int:
    deps = {name: importlib.util.find_spec(name) is not None
            for name in ('numpy', 'scipy', 'torch', 'pytest')}
    result = {
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'platform': platform.platform(),
        'dependencies': deps,
        'utf8_mode': sys.flags.utf8_mode,
    }
    show(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if sys.version_info >= (3, 10) and all(deps.values()) else 2


def run_tests() -> int:
    # Subprocess isolation keeps pytest side effects out of other CLI commands.
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    pythonpath = os.pathsep.join(filter(None, (str(ROOT), env.get('PYTHONPATH', ''))))
    env['PYTHONPATH'] = pythonpath
    return subprocess.call([sys.executable, '-m', 'pytest', '-q',
                            'tests_preflight', 'tests_research'], cwd=ROOT, env=env)


def menu() -> int:
    options = {
        '1': 'doctor', '2': 'selfcheck', '3': 'test',
        '4': 'server-tasks', '5': 'setup-help',
    }
    while True:
        show('\nSafe-Cadence：1 环境诊断  2 离线自检  3 回归测试')
        show('              4 服务器任务  5 安装说明  0 退出')
        try:
            choice = input('选择：').strip()
        except EOFError:
            return 0
        if choice == '0':
            return 0
        if choice not in options:
            show('未知选项')
            continue
        result = dispatch(argparse.Namespace(cmd=options[choice]))
        show(f'退出码：{result}')


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Safe-Cadence tools')
    sub = ap.add_subparsers(dest='cmd')
    for cmd in ('check', 'selfcheck', 'test', 'verify', 'encoding',
                'server-tasks', 'setup-help', 'menu'):
        sub.add_parser(cmd)
    doc = sub.add_parser('doctor', help='诊断环境、文件、数据和离线测试')
    doc.add_argument('--quick', action='store_true', help='跳过较慢的自检和测试')
    doc.add_argument('--data', help='可选：五阶段 CKGE 数据目录')
    doc.add_argument('--report', default='outputs/doctor.json', help='UTF-8 JSON 报告位置')
    dl = sub.add_parser('download', help='联网下载固定版本的公开数据')
    dl.add_argument('--dataset', choices=('WN_CKGE', 'FB_CKGE'), required=True)
    dl.add_argument('--output', help='下载目录，默认 data/数据集名')
    dl.add_argument('--snapshots', type=int, default=5)
    audit = sub.add_parser('audit', help='审计增量快照与数据划分')
    audit.add_argument('--data', required=True)
    audit.add_argument('--snapshots', type=int, default=5)
    audit.add_argument('--output', default='outputs/data_audit.json')
    pilot = sub.add_parser('pilot', help='小预算流水线自检，不是正式性能基准')
    pilot.add_argument('--data', required=True)
    pilot.add_argument('--snapshots', type=int, default=2)
    pilot.add_argument('--steps', type=int, default=12)
    pilot.add_argument('--seed', type=int, default=42)
    pilot.add_argument('--dim', type=int, default=24)
    pilot.add_argument('--max-eval', type=int, default=24)
    pilot.add_argument('--output', default='outputs/pilot.json')
    return ap


def dispatch(args: argparse.Namespace) -> int:
    cmd = args.cmd
    if cmd in (None, 'menu'):
        return menu()
    if cmd == 'check':
        return check_environment()
    if cmd == 'setup-help':
        setup_help()
    elif cmd == 'server-tasks':
        show((ROOT / 'docs' / 'SERVER_TASKS_CN.md').read_text(encoding='utf-8'))
    elif cmd == 'test':
        return run_tests()
    elif cmd == 'selfcheck':
        from scripts.selfcheck import run_selfcheck
        return run_selfcheck()
    elif cmd == 'verify':
        from scripts.verify_package import verify
        return verify()
    elif cmd == 'encoding':
        from scripts.check_encoding import check
        return check()
    elif cmd == 'doctor':
        from scripts.doctor import diagnose
        return diagnose(ROOT, data=getattr(args, 'data', None),
                        report=getattr(args, 'report', 'outputs/doctor.json'),
                        quick=getattr(args, 'quick', False))
    elif cmd == 'download':
        from safecadence_preflight.download_ckge import download
        target = args.output or str(ROOT / 'data' / args.dataset)
        download(args.dataset, target, count=args.snapshots)
        show('下载完成。请使用 audit 核验数据。')
    elif cmd == 'audit':
        from safecadence_preflight.ckge_protocol import audit_ckge, save_report
        report = audit_ckge(args.data, snapshot_count=args.snapshots)
        save_report(report, args.output)
        show(json.dumps(report['stage_statistics'], ensure_ascii=False, indent=2))
    elif cmd == 'pilot':
        from safecadence_preflight.pilot import run_pilot
        run_pilot(args.data, args.output, max_snapshots=args.snapshots,
                  steps=args.steps, seed=args.seed, dim=args.dim,
                  max_eval=args.max_eval)
    else:
        return 2
    return 0


def main() -> int:
    try:
        return dispatch(parser().parse_args())
    except (OSError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        # A traceback is necessary when reproducing failures on remote machines.
        log = ROOT / 'logs' / 'last_error.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding='utf-8')
        show(f'错误：{type(exc).__name__}：{exc}')
        show(f'详细日志：{log}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
