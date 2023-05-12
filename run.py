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
    train.add_argument("context", choices=["local", "slurm"], nargs="?", default="local")
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
        with open("base_slurm.sh", "r") as f:
            script = f.read()
        script = script.replace("%%name%%", config["name"])
        script = script.replace("%%train_dir%%", train_dir.resolve())
        for k,v in config["slurm"].items():
            script = script.replace(f"%%{k}%%", str(v))
        with open("modified_slurm.sh", "w") as f:
            f.write(script)
        st = os.stat("modified_slurm.sh")
        os.chmod("modified_slurm.sh", st.st_mode | 0o111)
        cmd = "./modified_slurm.sh --from_slurm"
        cmd += f" --load {args.load}" if args.load else ""
        subprocess.call(cmd, shell=True)
