import os
import sys
import subprocess

try:
    import ray
    from ray import tune, train
    from ray.tune.search.basic_variant import BasicVariantGenerator
except Exception as e:
    raise RuntimeError(
        "Ray Tune is required. Please install with: pip install 'ray[tune,train]'"
    ) from e

# This script schedules two concurrent 1-GPU trials to utilize 2 GPUs.
# Each trial calls the existing Lightning CLI training you already run by hand.
# Adjust BASE_CLI_ARGS below if you need to change defaults.

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
    "--trainer.gpus", "1",  # 1 GPU per trial; Ray assigns device via CUDA_VISIBLE_DEVICES
    "--trainer.check_val_every_n_epoch", "1",
    "--trainer.max_epochs", "50",
    "--trainer.log_every_n_steps", "50",
    "--early_stopping_patience", "5",
    "--checkpoint_metric", "val/roc_auc",
    "--no_wandb",
]


def run_lightning_cli(config: dict):
    """Train one trial using the Lightning CLI on a single GPU.

    Ray injects CUDA_VISIBLE_DEVICES for isolation; we just invoke the CLI.
    """
    env = os.environ.copy()

    cmd = [sys.executable] + BASE_CLI_ARGS[:]

    # Map a few tunables through to CLI flags if present in config
    if "pl_seed" in config:
        cmd += ["--pl_seed", str(config["pl_seed"])]
    if "model.learning_rate" in config:
        cmd += ["--model.learning_rate", str(config["model.learning_rate"])]
    if "data.batch_size" in config:
        cmd += ["--data.batch_size", str(config["data.batch_size"])]

    # Stream output so Ray captures logs; do not raise on nonzero to report exit_code
    process = subprocess.Popen(cmd, env=env)
    process.wait()

    # Report minimal metric so Tune marks completion
    tune.report(exit_code=process.returncode)


def main():
    # Initialize Ray (local). If on a cluster, configure per your environment.
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Minimal grid to spawn two concurrent trials on two GPUs
    param_space = {
        "pl_seed": tune.grid_search([2494, 2495]),
        # Add more tunables as needed, e.g.:
        # "model.learning_rate": tune.grid_search([0.003, 0.001]),
        # "data.batch_size": tune.grid_search([256]),
    }

    storage_path = "/coc/pcba1/sdhekane3/tdost_revision/orange/HAR/ray_results"

    tuner = tune.Tuner(
        tune.with_resources(
            run_lightning_cli,
            resources={"cpu": 1, "gpu": 1},  # 1 GPU per trial; two trials -> 2 GPUs total
        ),
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(constant_grid_search=True),
            num_samples=1,
            # Let GPU resource limits control parallelism; optionally set max_concurrent_trials=2
        ),
        run_config=train.RunConfig(
            storage_path=storage_path,
            name="tdost_har",  # mirrors your template's exp_name
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    # Exit nonzero if any trial failed
    failed = [r for r in results if r.error is not None]
    if failed:
        first = failed[0]
        print(f"A trial failed: {first.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
