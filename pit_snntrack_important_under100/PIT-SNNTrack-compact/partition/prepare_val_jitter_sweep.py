#!/usr/bin/env python3
"""Generate held-out ESOT500 validation jitter realizations only."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("/root/autodl-tmp/pit_snntrack"))
    ap.add_argument("--workers",type=int,default=8)
    args=ap.parse_args()
    root=args.root
    repo=root/"pit-sdtrack"
    manifest=json.loads((root/"esot500_training_view/manifest.json").read_text())
    sequences=manifest["val"]
    sweep=root/"esot500_jitter_sweep"
    if sweep.exists():
        raise FileExistsError(f"Refusing to overwrite {sweep}")
    jobs=[]
    conditions=[]
    for strength in (0.25,0.5):
        for seed in (11,22,33):
            cond=f"strength_{int(strength*100):02d}_seed_{seed}"
            conditions.append(cond)
            view=sweep/cond/"processed"
            for name in sequences:
                seq=view/name
                seq.mkdir(parents=True,exist_ok=True)
                baseline=root/"esot500_processed"/name
                for policy in ("uniform","fixed_count"):
                    (seq/policy).symlink_to(baseline/policy,target_is_directory=True)
                out=seq/"jitter"/"inter4_stack_3008"
                aedat=root/"datasets/ESOT500-L/aedat4"/f"{name}.aedat4"
                cmd=[str(root/"venv/bin/python"),"-m","partition.generate_partitions",
                     "--aedat",str(aedat),"--output",str(out),"--policy","jitter",
                     "--amount","30","--hidden-amount","30","--hidden-decay","0.8",
                     "--jitter-strength",str(strength),"--seed",str(seed)]
                jobs.append((cond,name,cmd))
    env=os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"]="1"
    completed=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs={pool.submit(subprocess.run,cmd,cwd=str(repo),env=env,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,check=True): (cond,name)
              for cond,name,cmd in jobs}
        for fut in as_completed(futs):
            cond,name=futs[fut]
            res=fut.result()
            completed.append((cond,name,res.stdout.strip()))
            print(f"[{len(completed)}/{len(jobs)}] {cond}/{name}",flush=True)
    (sweep/"manifest.json").write_text(json.dumps({
        "sequences":sequences,"conditions":conditions,
        "strength_definition":"uniform-boundary maximum displacement / uniform bin width",
        "seeds":[11,22,33]},indent=2))
    print("completed jobs",len(completed),"conditions",conditions)
if __name__=="__main__":
    main()
