import os
import sys
from datetime import datetime

from train_crnn import build_arg_parser, train_crnn_experiment


EXPERIMENT_QUEUE = [
    {
        "experiment_name": "crnn_rotation5_cpu_h128_b8",
        "epochs": 200,
        "batch_size": 8,
        "hidden_size": 128,
        "num_lstm_layers": 2,
        "rotation_degree": 5.0,
        "cpu_threads": 4,
        "train_dir": "data/train",
        "val_dir": "data/val",
        "test_dir": "data/test",
    },

    # 如果你真的想讓它自動跑第二個訓練，就保留下面這個。
    # 如果只想跑一個正式版，就把下面這段刪掉。
    # {
    #     "experiment_name": "crnn_rotation5_cpu_h64_b4",
    #     "epochs": 200,
    #     "batch_size": 4,
    #     "hidden_size": 64,
    #     "num_lstm_layers": 2,
    #     "rotation_degree": 5.0,
    #     "cpu_threads": 4,
    #     "train_dir": "data/train",
    #     "val_dir": "data/val",
    #     "test_dir": "data/test",
    # },
]


def write_run_log(message):
    os.makedirs("experiments", exist_ok=True)

    log_path = os.path.join("experiments", "run_all_log.txt")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def make_args(config):
    parser = build_arg_parser()
    args = parser.parse_args([])

    for key, value in config.items():
        setattr(args, key, value)

    return args


def main():
    start_time = datetime.now().isoformat()

    write_run_log("=" * 80)
    write_run_log(f"Run all experiments started at: {start_time}")
    write_run_log(f"Command: python {' '.join(sys.argv)}")
    write_run_log(f"Total experiments: {len(EXPERIMENT_QUEUE)}")

    completed = []
    failed = []

    for idx, config in enumerate(EXPERIMENT_QUEUE, start=1):
        exp_name = config["experiment_name"]

        print("\n" + "=" * 80)
        print(f"Starting experiment {idx}/{len(EXPERIMENT_QUEUE)}: {exp_name}")
        print("=" * 80)

        write_run_log("-" * 80)
        write_run_log(f"Starting experiment {idx}/{len(EXPERIMENT_QUEUE)}: {exp_name}")
        write_run_log(f"Started at: {datetime.now().isoformat()}")

        try:
            args = make_args(config)
            exp_dir = train_crnn_experiment(args)

            completed.append({
                "experiment_name": exp_name,
                "experiment_dir": exp_dir
            })

            write_run_log(f"Completed: {exp_name}")
            write_run_log(f"Experiment dir: {exp_dir}")
            write_run_log(f"Finished at: {datetime.now().isoformat()}")

        except Exception as e:
            failed.append({
                "experiment_name": exp_name,
                "error": str(e)
            })

            write_run_log(f"FAILED: {exp_name}")
            write_run_log(f"Error: {str(e)}")
            write_run_log(f"Failed at: {datetime.now().isoformat()}")

            print(f"\nExperiment failed: {exp_name}")
            print(f"Error: {e}")
            print("Continue to next experiment...")

            continue

    write_run_log("=" * 80)
    write_run_log(f"Run all experiments finished at: {datetime.now().isoformat()}")
    write_run_log(f"Completed: {len(completed)}")
    write_run_log(f"Failed: {len(failed)}")

    print("\n" + "=" * 80)
    print("All queued experiments finished.")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print("Log saved to: experiments/run_all_log.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()