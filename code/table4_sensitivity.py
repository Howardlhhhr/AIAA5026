"""Recompute paired calibration-threshold sensitivity from the main-run score CSV."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


TEST_COUNTS = {"bottle": (20, 63), "grid": (21, 57), "screw": (41, 119)}
SEEDS = range(11, 21)
SHOTS = (2, 4, 8)
POLICIES = ("random", "diverse")
PERCENTILES = (90, 95, 97.5)
FIELDS = (
    "calibration_percentile", "shots", "n_paired_seeds", "delta_fpr_pp", "delta_recall_pp",
    "delta_fpr_sd_pp", "delta_recall_sd_pp",
)


def read_scores(path):
    configurations = {
        (category, seed, shots, policy)
        for category in TEST_COUNTS for seed in SEEDS for shots in SHOTS for policy in POLICIES
    }
    groups = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"category", "seed", "shots", "policy", "split", "image", "label", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Score CSV is missing columns {sorted(missing)}")
        for row in reader:
            key = row["category"], int(row["seed"]), int(row["shots"]), row["policy"]
            split, image, label = row["split"], row["image"], int(row["label"])
            score = float(row["score"])
            if key not in configurations or (split, label) not in {("calibration", 0), ("test", 0), ("test", 1)}:
                raise ValueError(f"Unexpected configuration or split/label on line {reader.line_num}")
            if not image or image in groups[key] or not math.isfinite(score):
                raise ValueError(f"Missing or duplicate image, or nonfinite score on line {reader.line_num}")
            groups[key][image] = split, label, score
    if groups.keys() != configurations:
        raise ValueError(f"Missing configurations {sorted(configurations - groups.keys())}")

    expected_images = {}
    scores = {}
    for key, rows in groups.items():
        category = key[0]
        normal, defect = TEST_COUNTS[category]
        counts = Counter((split, label) for split, label, _ in rows.values())
        if counts != {("calibration", 0): 40, ("test", 0): normal, ("test", 1): defect}:
            raise ValueError(f"Unexpected calibration or test counts for {key}: {counts}")
        image_labels = {image: (split, label) for image, (split, label, _) in rows.items()}
        if expected_images.setdefault(category, image_labels) != image_labels:
            raise ValueError(f"Image IDs, splits, or labels are inconsistent for {key}")
        scores[key] = (
            sorted(score for split, _, score in rows.values() if split == "calibration"),
            [score for split, label, score in rows.values() if (split, label) == ("test", 0)],
            [score for split, label, score in rows.values() if (split, label) == ("test", 1)],
        )
    tests = [image for images in expected_images.values() for image, (split, _) in images.items() if split == "test"]
    if len(set(tests)) != len(tests):
        raise ValueError("Test image IDs overlap across categories")
    return scores, len(tests)


def linear_percentile(sorted_scores, percentile):
    index = (len(sorted_scores) - 1) * (percentile / 100)
    lower = math.floor(index)
    upper = math.ceil(index)
    return sorted_scores[lower] + (index - lower) * (sorted_scores[upper] - sorted_scores[lower])


def sensitivity(scores):
    for percentile in PERCENTILES:
        rates = {}
        for key, (calibration, normal, defect) in scores.items():
            threshold = linear_percentile(calibration, percentile)
            rates[key] = (sum(score > threshold for score in normal) / len(normal),
                          sum(score > threshold for score in defect) / len(defect))
        for shots in SHOTS:
            fpr_deltas, recall_deltas = [], []
            for seed in SEEDS:
                fpr_by_category, recall_by_category = [], []
                for category in TEST_COUNTS:
                    random_fpr, random_recall = rates[category, seed, shots, "random"]
                    diverse_fpr, diverse_recall = rates[category, seed, shots, "diverse"]
                    fpr_by_category.append(diverse_fpr - random_fpr)
                    recall_by_category.append(diverse_recall - random_recall)
                fpr_deltas.append(100 * mean(fpr_by_category))
                recall_deltas.append(100 * mean(recall_by_category))
            yield {
                "calibration_percentile": percentile, "shots": shots, "n_paired_seeds": len(SEEDS),
                "delta_fpr_pp": mean(fpr_deltas), "delta_recall_pp": mean(recall_deltas),
                "delta_fpr_sd_pp": stdev(fpr_deltas), "delta_recall_sd_pp": stdev(recall_deltas),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Main-run scores.csv")
    parser.add_argument("--output", type=Path, required=True, help="Destination for table4_sensitivity.csv")
    args = parser.parse_args()
    if args.scores.resolve() == args.output.resolve():
        parser.error("Use a different path for the output CSV")
    try:
        scores, test_images = read_scores(args.scores)
        rows = list(sensitivity(scores))
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Wrote {len(rows)} paired comparisons from {len(scores)} configurations and "
          f"{test_images} distinct test images to {args.output}")


if __name__ == "__main__":
    main()
