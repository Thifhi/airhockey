import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import yaml
import argparse
from ppo_airhockey_benchmark.train import start_training
from ppo_airhockey_benchmark.test import start_testing
import pathlib
import shutil
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    test = parser.add_mutually_exclusive_group()
    test.add_argument("--test", action="store_true")
    test.add_argument("--test_dir")
    maybe_load = parser.add_mutually_exclusive_group()
    maybe_load.add_argument("--load", default=None)
    train = maybe_load.add_mutually_exclusive_group()
    train.add_argument("context", choices=["local", "slurm", "_from_slurm"], nargs="?", default="local")
    from_slurm = maybe_load.add_mutually_exclusive_group()
    from_slurm.add_argument("--from_slurm", action="store_true")
    from_slurm.add_argument("--train_dir")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if args.test:
        start_testing(pathlib.Path(args.test_dir))
    
    if args.from_slurm:
        start_training(pathlib.Path(args.train_dir), args.load)
    
    # We will train so set up the folders
    log_dir = pathlib.Path("logs")
    train_dir = log_dir / config["group"] / config["name"]
    os.makedirs(train_dir)
    shutil.copy("config.yaml", train_dir)

    if args.context == "local":
        start_training(train_dir, args.load)
    elif args.context == "slurm":
        with open("slurm.sh", "r+") as f:
            script = f.read()
            script.replace("%%name%%", config["name"])
            for k,v in config["slurm"]:
                script.replace(f"%%{k}%%", v)
            f.write(script)
        cmd = f"slurm.sh --from_slurm --train_dir {train_dir.resolve()}"
        cmd += f"--load {args.load}" if args.load else ""
        subprocess.call(cmd)
