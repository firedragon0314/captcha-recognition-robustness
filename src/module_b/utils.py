import numpy as np


CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

char_to_idx = {char: idx + 1 for idx, char in enumerate(CHARS)}
idx_to_char = {idx + 1: char for idx, char in enumerate(CHARS)}
idx_to_char[0] = ""


def encode_label(label: str):
    label = label.upper()
    encoded = []

    for char in label:
        if char not in char_to_idx:
            raise ValueError(f"Unknown character in label: {char}, label={label}")
        encoded.append(char_to_idx[char])

    return encoded


def decode_prediction(indices):
    decoded = []
    previous = None

    for idx in indices:
        idx = int(idx)

        if idx != previous and idx != 0:
            decoded.append(idx_to_char.get(idx, ""))

        previous = idx

    return "".join(decoded)


def edit_distance(a: str, b: str):
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return int(dp[len(a)][len(b)])


def calculate_metrics(predictions, labels, captcha_length=5):
    total = len(labels)

    if total == 0:
        return {
            "seq_acc": 0.0,
            "char_acc": 0.0,
            "edit_distance": 0.0,
            "position_1_acc": 0.0,
            "position_2_acc": 0.0,
            "position_3_acc": 0.0,
            "position_4_acc": 0.0,
            "position_5_acc": 0.0,
        }

    seq_correct = 0
    char_correct = 0
    char_total = total * captcha_length
    edit_distances = []
    position_correct = [0 for _ in range(captcha_length)]

    for pred, gt in zip(predictions, labels):
        pred = pred.upper()
        gt = gt.upper()

        if pred == gt:
            seq_correct += 1

        edit_distances.append(edit_distance(pred, gt))

        for i in range(captcha_length):
            pred_char = pred[i] if i < len(pred) else ""
            gt_char = gt[i] if i < len(gt) else ""

            if pred_char == gt_char:
                char_correct += 1
                position_correct[i] += 1

    metrics = {
        "seq_acc": seq_correct / total,
        "char_acc": char_correct / char_total,
        "edit_distance": float(np.mean(edit_distances)),
    }

    for i in range(captcha_length):
        metrics[f"position_{i + 1}_acc"] = position_correct[i] / total

    return metrics