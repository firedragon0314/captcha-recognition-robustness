import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "_comparisons"
TRAIN_NAME_PATTERN = re.compile(r"train_(\d+)$")
COLOR_PALETTE = [
    "#1d4ed8",
    "#d97706",
    "#059669",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#65a30d",
    "#ea580c",
    "#4f46e5",
    "#be123c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple train_XXX experiments and generate overlay metric charts."
    )
    parser.add_argument(
        "--upto",
        type=str,
        default=None,
        help="Include train_001 through this experiment, for example train_020.",
    )
    parser.add_argument(
        "--start",
        "--from",
        dest="from_experiment",
        type=str,
        default="train_001",
        help="Starting experiment for range mode. Default: train_001",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Explicit experiment names, for example train_003 train_007 train_020.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional custom output folder name under experiments/_comparisons.",
    )
    return parser.parse_args()


def parse_train_number(name: str) -> int:
    match = TRAIN_NAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Invalid experiment name: {name}")
    return int(match.group(1))


def format_train_name(number: int) -> str:
    return f"train_{number:03d}"


def resolve_experiment_names(args: argparse.Namespace) -> List[str]:
    if args.experiments:
        return sorted(set(args.experiments), key=parse_train_number)

    if args.upto is None:
        raise ValueError("Provide either --experiments or --upto.")

    start_num = parse_train_number(args.from_experiment)
    end_num = parse_train_number(args.upto)
    if end_num < start_num:
        raise ValueError("--upto must be greater than or equal to --start.")

    return [format_train_name(number) for number in range(start_num, end_num + 1)]


def load_history_csv(history_path: Path) -> List[Dict[str, float]]:
    with history_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            parsed_row: Dict[str, float] = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed_row[key] = int(value)
                else:
                    parsed_row[key] = float(value)
            rows.append(parsed_row)
        return rows


def load_experiment_histories(experiment_names: List[str]) -> Dict[str, List[Dict[str, float]]]:
    histories: Dict[str, List[Dict[str, float]]] = {}

    for name in experiment_names:
        history_path = EXPERIMENT_ROOT / name / "history.csv"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file: {history_path}")
        histories[name] = load_history_csv(history_path)

    return histories


def series_bounds(histories: Dict[str, List[Dict[str, float]]], metric_key: str) -> Tuple[float, float]:
    values = [
        row[metric_key]
        for history in histories.values()
        for row in history
        if metric_key in row
    ]

    if not values:
        return 0.0, 1.0

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        padding = 1.0 if min_value == 0 else abs(min_value) * 0.1
        return min_value - padding, max_value + padding

    padding = (max_value - min_value) * 0.1
    return min_value - padding, max_value + padding


def max_epoch(histories: Dict[str, List[Dict[str, float]]]) -> int:
    return max(int(row["epoch"]) for history in histories.values() for row in history)


def write_overlay_line_plot(
    histories: Dict[str, List[Dict[str, float]]],
    out_path: Path,
    title: str,
    y_label: str,
    metric_key: str,
) -> None:
    width = 1080
    height = 620
    left = 90
    right = 30
    top = 70
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    max_epoch_value = max_epoch(histories)
    min_y, max_y = series_bounds(histories, metric_key)

    def x_to_svg(epoch: int) -> float:
        if max_epoch_value <= 1:
            return left + plot_width / 2
        return left + ((epoch - 1) / (max_epoch_value - 1)) * plot_width

    def y_to_svg(value: float) -> float:
        if max_y == min_y:
            return top + plot_height / 2
        ratio = (value - min_y) / (max_y - min_y)
        return top + plot_height - (ratio * plot_height)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" font-size="24" text-anchor="middle" fill="#111111">{escape(title)}</text>',
        f'<text x="22" y="{height / 2}" font-size="16" transform="rotate(-90 22,{height / 2})" text-anchor="middle" fill="#333333">{escape(y_label)}</text>',
        f'<text x="{width / 2}" y="{height - 20}" font-size="16" text-anchor="middle" fill="#333333">Epoch</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
    ]

    for tick_index in range(6):
        y_value = min_y + (max_y - min_y) * (tick_index / 5)
        y = y_to_svg(y_value)
        svg_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{left - 10}" y="{y + 5:.2f}" font-size="12" text-anchor="end" fill="#555555">{y_value:.4f}</text>'
        )

    epoch_step = max(1, max_epoch_value // 10)
    for epoch in range(1, max_epoch_value + 1, epoch_step):
        x = x_to_svg(epoch)
        svg_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 22}" font-size="12" text-anchor="middle" fill="#555555">{epoch}</text>'
        )
    if max_epoch_value not in range(1, max_epoch_value + 1, epoch_step):
        x = x_to_svg(max_epoch_value)
        svg_lines.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 22}" font-size="12" text-anchor="middle" fill="#555555">{max_epoch_value}</text>'
        )

    legend_y = top + plot_height + 45
    legend_x = left

    for index, (experiment_name, history) in enumerate(histories.items()):
        color = COLOR_PALETTE[index % len(COLOR_PALETTE)]
        points = " ".join(
            f"{x_to_svg(int(row['epoch'])):.2f},{y_to_svg(row[metric_key]):.2f}"
            for row in history
        )
        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>'
        )
        for row in history:
            x = x_to_svg(int(row["epoch"]))
            y = y_to_svg(row[metric_key])
            svg_lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" fill-opacity="0.9"/>'
            )

        svg_lines.append(
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="18" height="6" fill="{color}"/>'
        )
        svg_lines.append(
            f'<text x="{legend_x + 24}" y="{legend_y - 3}" font-size="13" fill="#333333">{escape(experiment_name)}</text>'
        )
        legend_x += 115
        if legend_x > width - 140:
            legend_x = left
            legend_y += 20

    svg_lines.append("</svg>")
    out_path.write_text("\n".join(svg_lines), encoding="utf-8")


def summarize_histories(histories: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, float]]:
    summary = []

    for experiment_name, history in histories.items():
        best_val_seq = max(history, key=lambda row: row["val_seq_acc"])
        best_test_seq = max(history, key=lambda row: row["test_seq_acc"])
        best_val_char = max(history, key=lambda row: row["val_char_acc"])
        best_test_char = max(history, key=lambda row: row["test_char_acc"])
        best_val_edit = min(history, key=lambda row: row["val_edit_distance"])
        best_test_edit = min(history, key=lambda row: row["test_edit_distance"])

        summary.append(
            {
                "experiment": experiment_name,
                "epochs": len(history),
                "best_val_seq_acc_epoch": int(best_val_seq["epoch"]),
                "best_val_seq_acc": best_val_seq["val_seq_acc"],
                "best_test_seq_acc_epoch": int(best_test_seq["epoch"]),
                "best_test_seq_acc": best_test_seq["test_seq_acc"],
                "best_val_char_acc_epoch": int(best_val_char["epoch"]),
                "best_val_char_acc": best_val_char["val_char_acc"],
                "best_test_char_acc_epoch": int(best_test_char["epoch"]),
                "best_test_char_acc": best_test_char["test_char_acc"],
                "best_val_edit_distance_epoch": int(best_val_edit["epoch"]),
                "best_val_edit_distance": best_val_edit["val_edit_distance"],
                "best_test_edit_distance_epoch": int(best_test_edit["epoch"]),
                "best_test_edit_distance": best_test_edit["test_edit_distance"],
            }
        )

    return summary


def write_summary_csv(summary: List[Dict[str, float]], out_path: Path) -> None:
    if not summary:
        return

    fieldnames = list(summary[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def write_metadata(
    experiment_names: List[str],
    output_dir: Path,
    summary: List[Dict[str, float]],
) -> None:
    metadata = {
        "experiments": experiment_names,
        "generated_at": str(output_dir),
        "experiment_count": len(experiment_names),
        "summary_rows": len(summary),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def plot_all_metrics(histories: Dict[str, List[Dict[str, float]]], output_dir: Path) -> None:
    plot_specs = [
        ("train_loss.svg", "Train Loss", "Loss", "train_loss"),
        ("val_loss.svg", "Validation Loss", "Loss", "val_loss"),
        ("test_loss.svg", "Test Loss", "Loss", "test_loss"),
        ("val_seq_acc.svg", "Validation Sequence Accuracy", "Accuracy", "val_seq_acc"),
        ("test_seq_acc.svg", "Test Sequence Accuracy", "Accuracy", "test_seq_acc"),
        ("val_char_acc.svg", "Validation Character Accuracy", "Accuracy", "val_char_acc"),
        ("test_char_acc.svg", "Test Character Accuracy", "Accuracy", "test_char_acc"),
        ("val_edit_distance.svg", "Validation Edit Distance", "Distance", "val_edit_distance"),
        ("test_edit_distance.svg", "Test Edit Distance", "Distance", "test_edit_distance"),
    ]

    for position in range(1, 6):
        plot_specs.append(
            (
                f"val_position_{position}_acc.svg",
                f"Validation Position {position} Accuracy",
                "Accuracy",
                f"val_position_{position}_acc",
            )
        )
        plot_specs.append(
            (
                f"test_position_{position}_acc.svg",
                f"Test Position {position} Accuracy",
                "Accuracy",
                f"test_position_{position}_acc",
            )
        )

    for filename, title, y_label, metric_key in plot_specs:
        write_overlay_line_plot(histories, output_dir / filename, title, y_label, metric_key)


def main() -> None:
    args = parse_args()
    experiment_names = resolve_experiment_names(args)
    histories = load_experiment_histories(experiment_names)

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"{experiment_names[0]}_to_{experiment_names[-1]}"

    output_dir = DEFAULT_OUTPUT_ROOT / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_all_metrics(histories, output_dir)
    summary = summarize_histories(histories)
    write_summary_csv(summary, output_dir / "summary.csv")
    write_metadata(experiment_names, output_dir, summary)

    print(f"Compared {len(experiment_names)} experiments.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
