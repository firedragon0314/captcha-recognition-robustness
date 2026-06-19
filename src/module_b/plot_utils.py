import os

import pandas as pd
import matplotlib.pyplot as plt


def save_single_plot(history_csv, output_dir, y_columns, title, ylabel, filename):
    if not os.path.exists(history_csv):
        return

    df = pd.read_csv(history_csv)

    if df.empty:
        return

    plt.figure()

    has_any_column = False

    for col in y_columns:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)
            has_any_column = True

    if not has_any_column:
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    png_path = os.path.join(output_dir, f"{filename}.png")
    svg_path = os.path.join(output_dir, f"{filename}.svg")

    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()


def update_all_plots(exp_dir):
    history_csv = os.path.join(exp_dir, "history.csv")
    plot_dir = os.path.join(exp_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)

    save_single_plot(
        history_csv,
        plot_dir,
        ["train_loss", "val_loss", "test_loss"],
        "Loss Curve",
        "Loss",
        "loss"
    )

    save_single_plot(
        history_csv,
        plot_dir,
        ["val_seq_acc", "test_seq_acc"],
        "Sequence Accuracy",
        "Accuracy",
        "seq_acc"
    )

    save_single_plot(
        history_csv,
        plot_dir,
        ["val_char_acc", "test_char_acc"],
        "Character Accuracy",
        "Accuracy",
        "char_acc"
    )

    save_single_plot(
        history_csv,
        plot_dir,
        ["val_edit_distance", "test_edit_distance"],
        "Edit Distance",
        "Edit Distance",
        "edit_distance"
    )

    for i in range(1, 6):
        save_single_plot(
            history_csv,
            plot_dir,
            [f"position_{i}_acc"],
            f"Position {i} Accuracy",
            "Accuracy",
            f"position_{i}_acc"
        )