import os
import sys
import time
import subprocess

try:
    import ray
    from ray import tune, train
    from ray.tune.search.basic_variant import BasicVariantGenerator
except Exception as e:
    raise RuntimeError(
        "Ray Tune is required. Install with: pip install 'ray[tune,train]' or conda -c conda-forge ray-tune ray-train"
    ) from e

# Lightning CLI arguments for your training
BASE_CLI_ARGS = [
    "-m", "src.models.train", "fit",
    "--config", "configs/tasks/HomekitPredictFluPos.yaml",
    "--config", "configs/models/InceptionTime.yaml",
    "--data.train_path", "/coc/pcba1/Datasets/HomeKit2020/data/processed/split_2020_02_10/train_7_day",
    "--data.val_path", "/coc/pcba1/Datasets/HomeKit2020/data/processed/split_2020_02_10/eval_7_day",
    "--data.batch_size", "256",
    "--model.val_bootstraps", "0",
    "--model.learning_rate", "0.003",
    "--trainer.accelerator", "cuda",
    "--trainer.gpus", "2",
    "--trainer.strategy", "ddp",
    "--trainer.check_val_every_n_epoch", "1",
    "--trainer.max_epochs", "50",
    "--trainer.log_every_n_steps", "50",
    "--early_stopping_patience", "5",
    "--checkpoint_metric", "val/roc_auc",
    "--no_wandb",
]

STORAGE_PATH = "/coc/pcba1/sdhekane3/tdost_revision/orange/HAR/ray_results"
EXPERIMENT_NAME = "tdost_har_2gpu_single"


def run_training_single_2gpu(config: dict):
    """Run one Lightning CLI training using 2 GPUs within a Ray trial."""
    env = os.environ.copy()

    # NCCL safety knobs that often help avoid multi-GPU hangs on single nodes
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    # Ray assigns two GPUs to this trial and sets CUDA_VISIBLE_DEVICES accordingly
    cmd = [sys.executable] + BASE_CLI_ARGS[:]

    # Optional seed passthrough if provided by caller
    if "pl_seed" in config:
        cmd += ["--pl_seed", str(config["pl_seed"])]

    # Create a per-run log file next to where you launched the script
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.abspath(f"training_2gpu_{ts}.log")

    # Stream subprocess output both to file and to this process stdout
    with open(log_path, "w", buffering=1) as log_file:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()

    tune.report(exit_code=process.returncode, log_path=log_path)


def main():
    # Initialize Ray locally; adjust if you use a Ray cluster
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Single trial, no grid search
    param_space = {}

    tuner = tune.Tuner(
        tune.with_resources(
            run_training_single_2gpu,
            resources={"cpu": 2, "gpu": 2},  # request 2 GPUs for this single trial
        ),
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(constant_grid_search=True),
            num_samples=1,
        ),
        run_config=train.RunConfig(
            storage_path=STORAGE_PATH,
            name=EXPERIMENT_NAME,
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    # Propagate failure if the trial failed
    failed = [r for r in results if r.error is not None]
    if failed:
        first = failed[0]
        print(f"Training failed: {first.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
