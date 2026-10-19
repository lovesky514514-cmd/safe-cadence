"""Run every included synthetic experiment and regenerate tables/figures.
Run from the repository root: python reproduce.py
No remote data is downloaded or any external experiment claimed.
"""
import argparse,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip-streams',action='store_true')
    p.add_argument('--skip-tests',action='store_true')
    args=p.parse_args()
    commands=[]
    if not args.skip_tests:
        commands.append([sys.executable,'-m','pytest','-q','tests'])
    commands += [[sys.executable,'benchmark_v2.py','--sizes','120','1000','5000','--seeds','1','2','3','--queries','60','--out','outputs/v2_benchmark.json'],
                 [sys.executable,'evaluate_saved_runs.py'],
                 [sys.executable,'optimize_safe_update.py'],
                 [sys.executable,'test_online_optimizer.py']]
    if not args.skip_streams:
        for seed in (7,11,19):
            commands.append([sys.executable,'run_stream_benchmark.py',
                             '--seed',str(seed),'--base-steps','180','--update-steps','20',
                             '--out',f'outputs/v2_stream_{seed}.json'])
    commands.append([sys.executable,'analyze_results.py'])
    for cmd in commands:
        print('\nRUN:', ' '.join(cmd),flush=True)
        subprocess.run(cmd,check=True)
    print('\nALL REQUESTED REPRODUCTION TASKS COMPLETED',flush=True)
if __name__=='__main__': main()
