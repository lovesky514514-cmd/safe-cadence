from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]


def test_ascii_bat_and_lf_encoding():
    data=(ROOT/'start.bat').read_bytes()
    assert b'\r\n' in data
    assert all(x<128 for x in data)
    assert b'PYTHONUTF8=1' in data
    assert b'%~dp0' in data


def test_launcher_environment_no_network():
    p=subprocess.run([sys.executable,str(ROOT/'launcher.py'),'check'],
                     capture_output=True,text=True,encoding='utf-8')
    assert p.returncode==0,p.stderr
    assert 'dependencies' in p.stdout


def test_docs_utf8_and_server_handoff():
    tasks = (ROOT / 'docs' / 'SERVER_TASKS_CN.md').read_text(encoding='utf-8')
    assert '服务器' in tasks and 'WN_CKGE' in tasks
    help_text = (ROOT / 'docs' / 'BUG_SELF_CHECK_CN.md').read_text(encoding='utf-8')
    assert 'doctor' in help_text and 'FAIL' in help_text
