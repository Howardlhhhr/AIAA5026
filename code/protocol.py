import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


COUNTS = {"bottle": (209, 20, 63), "grid": (264, 21, 57), "screw": (320, 41, 119)}
DATA_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads"
ARCHIVES = {
    "bottle": "https://www.mydrive.ch/shares/163640/9a565b9a02f45cd135e46b1c36e01ec5/download/420937370-1629958698/bottle.tar.xz",
    "grid": "https://www.mydrive.ch/shares/163644/8c7fc7cba43ef5275c87ca750a6d0004/download/420937487-1629959044/grid.tar.xz",
    "screw": "https://www.mydrive.ch/shares/163649/cfd394b4e3a49013745fab9e2084988a/download/420938130-1629960389/screw.tar.xz",
}
MODEL = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
WEIGHT_SHA = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
CONFIG_SHA = "5664b325e6258d3960fad8c4c1cff958f3cc2272"  # Git blob ID
SETTINGS = {
    "calibration_seed": 5026, "calibration_count": 40,
    "seeds": list(range(11, 21)), "shots": [2, 4, 8],
    "size": 224, "interpolation": "PIL RGB bicubic direct resize",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "token_source": "final last_hidden_state, row-major patches, L2 per token",
    "dtype": "float32", "tf32": False, "attention": "eager", "top_k": 3,
    "roc_min_recall": 0.95, "calibration_quantile": 0.95,
    "bootstrap_seed": 5026, "bootstrap_draws": 10000,
}
POLICIES = ("random", "diverse")
SPLIT_FIELDS = ["category", "image", "split", "label", "defect_type"]
CONFIG_FIELDS = ["category", "seed", "shots", "policy"]
SCORE_FIELDS = CONFIG_FIELDS + ["split", "image", "label", "defect_type", "score", "threshold", "predicted"]
REF_FIELDS = CONFIG_FIELDS + ["position", "image"]
RATES = ["auroc", "fpr_at_95", "recall_at_95", "calibrated_fpr", "calibrated_recall"]
DIAGNOSTICS = ["mean_pairwise_cls_distance", "calibration_mean_patch_distance"]
METRIC_FIELDS = CONFIG_FIELDS + RATES + ["roc_threshold", "calibration_threshold"] + DIAGNOSTICS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path, fields, rows):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def configs(categories, seeds):
    return {(category, seed, shots, policy)
            for category in categories for seed in seeds
            for shots in SETTINGS["shots"] for policy in POLICIES}


def config_key(row):
    return row["category"], int(row["seed"]), int(row["shots"]), row["policy"]


def cls_cosine_distances(cls):
    cls = cls.astype(np.float64)
    return 1 - np.clip(cls @ cls.T, -1, 1)


def reference_orders(distances, seed):
    random = np.random.default_rng(seed).permutation(len(distances))[:8]
    selected = [int(random[0])]
    nearest = distances[selected[0]].copy()
    for _ in range(7):
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))
        np.minimum(nearest, distances[selected[-1]], out=nearest)
    return {"random": random, "diverse": np.array(selected)}


def candidate_patch_distances(query, references, query_block=2048, ref_block=8):
    """Return a float32 array shaped (references, query images, query patches)."""
    import torch

    images, patches, width = query.shape
    flat_query = query.reshape(-1, width)
    result = np.empty((len(references), len(flat_query)), dtype=np.float32)
    for r in range(0, len(references), ref_block):
        bank = references[r:r + ref_block]
        flat_bank = bank.reshape(-1, width).T.contiguous()
        for q in range(0, len(flat_query), query_block):
            block = flat_query[q:q + query_block]
            cosine = (block @ flat_bank).reshape(len(block), len(bank), patches)
            distance = 1 - cosine.amax(dim=2).clamp(-1, 1)
            result[r:r + len(bank), q:q + len(block)] = distance.T.cpu().numpy()
    return result.reshape(len(references), images, patches)


def image_scores(maps):
    top = np.partition(maps, -3, axis=-1)[..., -3:]
    return np.mean(top, axis=-1, dtype=np.float64)


def score_metrics(scores, labels, calibration):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    require(len(pos) and len(neg) and len(calibration) == 40, "Missing test or calibration scores")
    require(np.isfinite(scores).all() and np.isfinite(calibration).all(), "Nonfinite scores")
    # Each positive contributes one for a smaller negative and one half for a tie.
    sorted_neg = np.sort(neg)
    below = np.searchsorted(sorted_neg, pos, side="left")
    at_or_below = np.searchsorted(sorted_neg, pos, side="right")
    needed = math.ceil(0.95 * len(pos))
    roc_threshold = float(np.sort(pos)[-needed])
    threshold = float(np.quantile(calibration, 0.95, method="linear"))
    return {
        "auroc": float(np.mean(below + at_or_below) / (2 * len(neg))),
        "fpr_at_95": float(np.mean(neg >= roc_threshold)),
        "recall_at_95": float(np.mean(pos >= roc_threshold)),
        "calibrated_fpr": float(np.mean(neg > threshold)),
        "calibrated_recall": float(np.mean(pos > threshold)),
        "roc_threshold": roc_threshold,
        "calibration_threshold": threshold,
    }
