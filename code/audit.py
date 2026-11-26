import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from protocol import (COUNTS, SETTINGS, config_key, configs, read_csv, read_json, require,
                      sha256, write_json)


def group(records):
    result = defaultdict(list)
    for row in records:
        result[config_key(row)].append(row)
    return result


def check_split(rows, category):
    train, good, defective = COUNTS[category]
    require(len({row["image"] for row in rows}) == len(rows) == train + good + defective,
            f"Split size or uniqueness failure: {category}")
    training, test = [], []
    for row in rows:
        image = Path(row["image"]).parts
        require(len(image) == 4 and image[0] == category and image[2] == row["defect_type"],
                f"Incorrect image metadata: {row}")
        require(int(row["label"]) == int(image[2] != "good"), f"Incorrect image label: {row}")
        if row["split"] == "test":
            require(image[1] == "test", f"Test split contains a non-test image: {row}")
            test.append(row)
        else:
            require(image[1:3] == ("train", "good") and row["split"] in ("calibration", "candidate"),
                    f"Train split incorrect: {row}")
            training.append(row)
    require(sum(row["label"] == "0" for row in test) == good and
            sum(row["label"] == "1" for row in test) == defective, f"Test count mismatch: {category}")
    training.sort(key=lambda row: row["image"])
    expected = {training[i]["image"] for i in np.random.default_rng(5026).permutation(train)[:40]}
    require(expected == {row["image"] for row in training if row["split"] == "calibration"},
            f"Calibration split mismatch: {category}")


def close(actual, expected, field, key):
    require(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12),
            f"{key} {field}: {actual} vs {expected}")


def recompute(rows, metric, expected_images, key):
    require(len(rows) == len(expected_images) and {row["image"] for row in rows} == expected_images.keys(),
            f"Query file set mismatch: {key}")
    by_label = defaultdict(list)
    for row in rows:
        expected = expected_images[row["image"]]
        require(all(row[field] == expected[field] for field in ("split", "label", "defect_type")),
                f"Query metadata mismatch: {key}, {row['image']}")
        require(0 <= float(row["score"]) <= 2, f"Score out of range: {key}")
        by_label[row["split"], row["label"]].append(float(row["score"]))
    calibration = np.sort(by_label["calibration", "0"])
    require(len(calibration) == 40, f"Wrong calibration size: {key}")
    threshold = calibration[37] + 0.05 * (calibration[38] - calibration[37])
    for row in rows:
        close(row["threshold"], threshold, "threshold", key)
        require(int(row["predicted"]) == int(float(row["score"]) > float(row["threshold"])),
                f"Prediction mismatch: {key} {row['image']}")
    positive = np.array(by_label["test", "1"])
    negative = np.array(by_label["test", "0"])
    comparisons = positive[:, None] - negative
    test_threshold = next(t for t in sorted(set(positive) | set(negative), reverse=True)
                          if np.mean(positive >= t) >= 0.95)
    actual_threshold = float(metric["calibration_threshold"])
    expected = {
        "auroc": np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0),
        "fpr_at_95": np.mean(negative >= test_threshold),
        "recall_at_95": np.mean(positive >= test_threshold),
        "calibrated_fpr": np.mean(negative > actual_threshold),
        "calibrated_recall": np.mean(positive > actual_threshold),
        "roc_threshold": test_threshold, "calibration_threshold": threshold,
    }
    for field, value in expected.items():
        close(metric[field], value, field, key)
    require(all(actual_threshold == float(row["threshold"]) for row in rows),
            f"Stored thresholds differ: {key}")
    require(0 <= float(metric["calibration_mean_patch_distance"]) <= 2, f"Invalid local coverage: {key}")


def audit(inventory_dir, results, cache, categories, seeds, legacy_smoke=False):
    manifest_path = inventory_dir / "input_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["settings"] == SETTINGS, "Audit protocol and recorded protocol differ")
    split_path = inventory_dir / "splits.csv"
    require(sha256(split_path) == manifest["split_sha256"], "Split CSV changed")
    splits = [row for row in read_csv(split_path) if row["category"] in categories]
    require(not legacy_smoke or (categories == ["bottle"] and seeds == [11]),
            "Legacy smoke mode applies only to the single-seed Bottle trial")
    require(len({row["image"] for row in splits}) == len(splits), "Global duplicate image IDs")
    tables = {name: [] for name in ("scores", "references", "metrics")}
    collected = {}
    for directory in results:
        info = read_json(directory / "result.json")
        category = info["category"]
        require(category in categories and category not in collected, f"Unexpected or duplicate result: {category}")
        require(info["state"] == "finished" and sorted(info["seeds"]) == sorted(seeds),
                f"Result state or seed mismatch: {directory}")
        require(info["input_manifest_sha256"] == sha256(manifest_path), f"Manifest mismatch: {directory}")
        rows = [row for row in splits if row["category"] == category]
        check_split(rows, category)
        queries = {row["image"]: row for row in rows if row["split"] != "candidate"}
        candidates = sorted(row["image"] for row in rows if row["split"] == "candidate")
        feature = cache / info["feature_cache"]
        require(sha256(feature) == info["feature_cache_sha256"], f"Feature archive changed: {category}")
        with np.load(feature, allow_pickle=False) as archive:
            images = archive["images"].tolist()
            tokens = archive["tokens"]
            require(images == [row["image"] for row in rows] and
                    list(tokens.shape) == info["feature_shape"] == [len(rows), 257, 384] and
                    tokens.dtype == np.float32 and
                    np.allclose(np.linalg.norm(tokens, axis=-1), 1, atol=2e-6),
                    f"Feature set, shape, or normalization mismatch: {category}")
            cls = dict(zip(images, tokens[:, 0].astype(np.float64)))
        del tokens
        candidate_cls = np.stack([cls[image] for image in candidates])
        distances = 1 - np.clip(candidate_cls @ candidate_cls.T, -1, 1)
        reference_index = {image: i for i, image in enumerate(candidates)}
        data = {name: read_csv(directory / f"{name}.csv") for name in tables}
        grouped = {name: group(records) for name, records in data.items()}
        expected = configs([category], seeds)
        require(all(groups.keys() == expected for groups in grouped.values()), f"Configuration mismatch: {category}")
        require([len(data[k]) for k in ("metrics", "scores", "references")] ==
                [info[k] for k in ("configs", "query_rows", "reference_rows")],
                f"Result cardinality mismatch: {category}")
        calibration_images = [row["image"] for row in rows if row["split"] == "calibration"]
        calibration_maps = {}
        calibration_path = directory / "calibration_maps.npz"
        if calibration_path.exists() or not legacy_smoke:
            with np.load(calibration_path, allow_pickle=False) as maps:
                require(maps["images"].tolist() == calibration_images and
                        set(maps.files) == {"images", *(f"{seed}_{shots}_{policy}" for _, seed, shots, policy in expected)},
                        f"Calibration map keys mismatch: {category}")
                calibration_maps = {key: maps[f"{key[1]}_{key[2]}_{key[3]}"] for key in expected}
        orders = {}
        for key in sorted(expected):
            _, seed, shots, policy = key
            refs = sorted(grouped["references"][key], key=lambda row: int(row["position"]))
            order = [reference_index[row["image"]] for row in refs]
            require([int(row["position"]) for row in refs] == list(range(1, shots + 1)) and len(set(order)) == shots,
                    f"Incorrect reference set: {key}")
            orders[seed, shots, policy] = order
            metrics = grouped["metrics"][key]
            require(len(metrics) == 1, f"Duplicate metrics: {key}")
            recompute(grouped["scores"][key], metrics[0], queries, key)
            if calibration_maps:
                calibration_map = calibration_maps[key]
                require(calibration_map.shape == (40, 256) and
                        np.all((calibration_map >= 0) & (calibration_map <= 2)), f"Invalid calibration maps: {key}")
                close(metrics[0]["calibration_mean_patch_distance"], calibration_map.mean(dtype=np.float64),
                      "calibration patch mean", key)
                saved = {row["image"]: float(row["score"]) for row in grouped["scores"][key]}
                calculated = np.sort(calibration_map, axis=-1)[:, -3:].mean(axis=-1, dtype=np.float64)
                require(np.allclose(calculated, [saved[i] for i in calibration_images], rtol=0, atol=1e-12),
                        f"Calibration patch-to-image score mismatch: {key}")
            expected_cls = distances[np.ix_(order, order)][np.triu_indices(shots, 1)].mean()
            close(metrics[0]["mean_pairwise_cls_distance"], expected_cls, "CLS coverage", key)
        for seed in seeds:
            random = np.random.default_rng(seed).permutation(len(candidates))[:8].tolist()
            diverse = orders[seed, 8, "diverse"]
            require(orders[seed, 8, "random"] == random and diverse[0] == random[0],
                    f"Random order or shared first mismatch: {category}, {seed}")
            for position in range(1, len(diverse)):
                prefix = diverse[:position]
                remaining = sorted(set(range(len(candidates))) - set(prefix))
                choice = max(remaining, key=lambda i: distances[i, prefix].min())
                require(diverse[position] == choice, f"Greedy choice mismatch: {category}, {seed}, {position}")
            for policy, order in (("random", random), ("diverse", diverse)):
                for shots in SETTINGS["shots"]:
                    require(orders[seed, shots, policy] == order[:shots], f"Reference prefix mismatch: {category}, {seed}")
        if 11 in seeds:
            with np.load(directory / "case_maps.npz", allow_pickle=False) as maps:
                map_images = maps["images"].tolist()
                require(len(map_images) == len(queries) and set(map_images) == queries.keys(),
                        f"Case map image set differs: {category}")
                for policy in ("random", "diverse"):
                    key = (category, 11, 4, policy)
                    by_image = {row["image"]: row for row in grouped["scores"][key]}
                    matrix = maps[policy]
                    require(matrix.shape == (len(map_images), 16, 16) and np.isfinite(matrix).all() and
                            np.all((matrix >= 0) & (matrix <= 2)), f"Invalid case map: {key}")
                    calibration_maps = []
                    for image, patch_map in zip(map_images, matrix):
                        close(by_image[image]["score"], np.sort(patch_map.ravel())[-3:].astype(float).mean(),
                              "case-map top-three score", key)
                        if queries[image]["split"] == "calibration":
                            calibration_maps.append(patch_map)
                    close(grouped["metrics"][key][0]["calibration_mean_patch_distance"],
                          np.mean(calibration_maps, dtype=np.float64), "case-map local coverage", key)
        for name in tables:
            tables[name].extend(data[name])
        collected[category] = info
    require(collected.keys() == set(categories), f"Missing category results: {set(categories) - collected.keys()}")
    counts = {"test_images": sum(row["split"] == "test" for row in splits),
              **{name: len(rows) for name, rows in tables.items()}}
    if set(categories) == set(COUNTS) and set(seeds) == set(SETTINGS["seeds"]):
        require(counts == {"test_images": 321, "metrics": 180, "scores": 26460, "references": 840},
                f"Full experiment counts differ: {counts}")
    return manifest, splits, tables, collected, counts


def main():
    parser = argparse.ArgumentParser(description="Audit exact experiment inputs and outputs")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", choices=COUNTS, default=list(COUNTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SETTINGS["seeds"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--legacy-smoke", action="store_true")
    args = parser.parse_args()
    _, _, _, info, counts = audit(args.inventory, args.results, args.cache, args.categories, args.seeds,
                                 legacy_smoke=args.legacy_smoke)
    evidence = {"status": "passed", "counts": counts, "runs": info, "legacy_smoke": args.legacy_smoke}
    if args.out:
        write_json(args.out, evidence)
    print(f"Audit passed: {counts}")


if __name__ == "__main__":
    main()
