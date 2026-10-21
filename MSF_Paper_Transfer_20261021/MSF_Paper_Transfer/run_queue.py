"""Finite performance-only queue. Leave all earlier experiments untouched."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / 'env/bin/python')
OUT = ROOT / 'performance_runs'
OUT.mkdir(exist_ok=True)
lock = (ROOT / 'queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def status(state, **values):
    record = {'state': state, 'pid': os.getpid(), 'time': time.time(), **values}
    (ROOT / 'queue_status.tmp.json').write_text(json.dumps(record, indent=2))
    os.replace(str(ROOT / 'queue_status.tmp.json'), str(ROOT / 'queue_status.json'))
    print(json.dumps(record), flush=True)

def idle():
    # User reprioritized this project. Permit training beside unrelated small jobs
    # when enough memory remains; do not stop projects outside the legacy scope.
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], universal_newlines=True).strip()
    return int(free.splitlines()[0]) >= 25000

def await_gpu(arm):
    quiet = 0
    while quiet < 3:
        if shutil.disk_usage(str(ROOT)).free < 45 * 1024**3:
            status('waiting_disk', arm=arm)
            quiet = 0
        elif idle():
            quiet += 1
            status('waiting_gpu_stable_memory', arm=arm, ready_checks=quiet)
        else:
            quiet = 0
            status('waiting_gpu_existing_experiments', arm=arm)
        time.sleep(30)

for dataset, data in [('gen1', '/root/autodl-tmp/datasets/gen1_standard'),
                      ('coco', '/root/autodl-tmp/datasets/coco2017_sota_transfer')]:
    if dataset == 'coco':
        while not (Path(data) / 'ready.json').exists():
            status('waiting_full_coco2017')
            time.sleep(60)
    for variant in ['baseline', 'paper_binary', 'paper']:
        # Smoke at the actual intended input shape before any 20-epoch allocation.
        for stage, extra in [('gpu_preflight', ['--epochs','1','--max-batches','2']),
                             ('screen20', ['--epochs','20','--eval-every','5'])]:
            name = dataset + '_' + variant + '_' + stage + '_seed1'
            output = OUT / name
            if (output / 'completed.json').exists():
                continue
            await_gpu(name)
            cmd = [PYTHON, str(ROOT/'performance.py'), '--dataset',dataset,'--data',data,'--variant',variant,
                   '--out',str(output),'--batch','2','--accumulate','4','--size','320','--steps','5', '--seed','1'] + extra
            with (OUT / (name+'.log')).open('a') as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env={**os.environ, 'OMP_NUM_THREADS':'2', 'MSF_CUDA_MEMORY_FRACTION':'.72'})
                status('running', arm=name, child=child.pid, command=cmd)
                code = child.wait()
            if code:
                status('failed_needs_diagnosis', arm=name, returncode=code, log=str(OUT/(name+'.log')))
                sys.exit(code)
status('performance_screen_complete_no_other_experiments_started')
