import os
import sys
import subprocess
from functools import partial

try:
    import ray
    from ray import tune, train
    from ray.tune.search.basic_variant import BasicVariantGenerator
except Exception as e:
    raise RuntimeError(
        "Ray Tune is required to run this wrapper. Please install ray[train,tune]."
    ) from e


# Base CLI args matching the user's single-GPU training command.
# Adjust paths or args if needed.
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
    "--trainer.gpus", "1",
    "--trainer.check_val_every_n_epoch", "1",
    "--trainer.max_epochs", "50",
    "--trainer.log_every_n_steps", "50",
    "--early_stopping_patience", "5",
    "--checkpoint_metric", "val/roc_auc",
    "--no_wandb",
]


def run_lightning_cli(config: dict):
    """Run the Lightning CLI training for a single Ray Tune trial on 1 GPU.

    Ray assigns a single GPU to this trial and sets CUDA_VISIBLE_DEVICES. We pass
    through that environment to the subprocess so Lightning sees only one GPU.
    """
    env = os.environ.copy()

    # Optional: tune pl_seed or any other CLI-exposed arg via config
    cmd = [sys.executable] + BASE_CLI_ARGS[:]

    if "pl_seed" in config:
        cmd += ["--pl_seed", str(config["pl_seed"])]

    if "model.learning_rate" in config:
        cmd += ["--model.learning_rate", str(config["model.learning_rate"])]

    if "data.batch_size" in config:
        cmd += ["--data.batch_size", str(config["data.batch_size"])]

    # Let Ray manage logs per-trial; surface subprocess output to Ray logging
    completed = subprocess.run(cmd, env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Report a simple metric so Tune marks success; Lightning also writes its own logs
    tune.report(exit_code=completed.returncode)


def main():
    # Init Ray locally (auto-detects available GPUs). If running on a cluster, remove address=None.
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Minimal param space to run two concurrent trials using 1 GPU each.
    # Adjust or extend as desired.
    param_space = {
        "pl_seed": tune.grid_search([2494, 2495]),
        # Example tunables:
        # "model.learning_rate": tune.grid_search([0.003, 0.001]),
        # "data.batch_size": tune.grid_search([256]),
    }

    # Configure where Ray stores results (matches the path you were shown)
    storage_path = "/coc/pcba1/sdhekane3/tdost_revision/orange/HAR/ray_results"

    tuner = tune.Tuner(
        tune.with_resources(
            run_lightning_cli,
            resources={"cpu": 1, "gpu": 1},  # 1 GPU per trial
        ),
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(constant_grid_search=True),
            num_samples=1,
            # max_concurrent_trials can be set, but GPU resources typically constrain this automatically
            # max_concurrent_trials=2,
        ),
        run_config=train.RunConfig(
            storage_path=storage_path,
            name="lightning_cli_multi_gpu_wrapper",
        ),
        param_space=param_space,
    )

    results = tuner.fit()
    # Exit with non-zero if any trial failed
    failed = [r for r in results if r.error is not None]
    if failed:
        # Print first failure for convenience
        first = failed[0]
        print(f"A trial failed: {first.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
