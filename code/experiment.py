import argparse
import importlib.metadata as metadata
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np

from data import check_inputs, prepare, verified_model
from protocol import (CONFIG_FIELDS, COUNTS, METRIC_FIELDS, REF_FIELDS, SCORE_FIELDS, SETTINGS,
                      candidate_patch_distances, cls_cosine_distances, digest, image_scores, read_csv, read_json,
                      reference_orders, require, score_metrics, sha256, write_csv, write_json)


def runtime(device):
    import torch

    torch.manual_seed(5026)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but unavailable")
    return {
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in
                     ("numpy", "pillow", "torch", "transformers", "safetensors", "matplotlib")},
        "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "threads": torch.get_num_threads(), "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def slurm():
    return {name: os.environ.get(f"SLURM_{name}") for name in
            ("JOB_ID", "JOB_PARTITION", "CPUS_PER_TASK", "MEM_PER_NODE", "JOB_GPUS", "JOB_NODELIST")}


def category_inputs(args):
    manifest = read_json(args.inventory / "input_manifest.json")
    require(manifest["settings"] == SETTINGS, "Input settings differ from current protocol")
    require(manifest["split_sha256"] == sha256(args.inventory / "splits.csv"), "Split CSV changed")
    rows = [row for row in read_csv(args.inventory / "splits.csv") if row["category"] == args.category]
    require(rows, f"No manifest rows for {args.category}")
    return manifest, rows


def cache_key(args, manifest, environment):
    return digest({
        "data": manifest["categories"][args.category]["data_digest"],
        "model": manifest["model"], "settings": SETTINGS,
        "extractor": sha256(Path(__file__)), "environment": environment,
        "batch_size": args.batch_size,
    })


def validate_features(data, rows, key):
    images = [row["image"] for row in rows]
    require(data["cache_key"].item() == key and data["images"].tolist() == images,
            "Feature cache key or file order mismatch")
    tokens = data["tokens"]
    require(tokens.shape == (len(rows), 257, 384) and tokens.dtype == np.float32,
            f"Unexpected feature shape or dtype: {tokens.shape}, {tokens.dtype}")
    require(np.isfinite(tokens).all() and np.allclose(np.linalg.norm(tokens, axis=-1), 1, atol=2e-6),
            "Feature cache contains invalid or unnormalized tokens")
    return tokens


def extract(args):
    import torch
    from PIL import Image
    from transformers import Dinov2Model

    manifest, rows = category_inputs(args)
    check_inputs(args.inventory, args.data, args.category, rows, manifest)
    require(verified_model(args.model) == manifest["model"], "Pinned model differs from manifest")
    environment = runtime(args.device)
    key = cache_key(args, manifest, environment)
    args.cache.mkdir(parents=True, exist_ok=True)
    path = args.cache / f"{args.category}-{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            validate_features(data, rows, key)
        print(f"Cache hit {path.name}", flush=True)
    else:
        model = Dinov2Model.from_pretrained(args.model, local_files_only=True, use_safetensors=True,
                                             attn_implementation="eager", torch_dtype=torch.float32)
        model.requires_grad_(False).eval().to(args.device)
        tokens = []
        mean = np.asarray(SETTINGS["mean"], dtype=np.float32)
        std = np.asarray(SETTINGS["std"], dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                images = []
                for row in rows[start:start + args.batch_size]:
                    with Image.open(args.data / row["image"]) as pic:
                        rgb = pic.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                        pixels = (np.asarray(rgb, dtype=np.float32) / 255 - mean) / std
                        images.append(pixels.transpose(2, 0, 1))
                batch = torch.from_numpy(np.stack(images)).to(args.device)
                result = model(pixel_values=batch).last_hidden_state
                tokens.append(torch.nn.functional.normalize(result, p=2, dim=-1).cpu().numpy())
        payload = {"cache_key": np.array(key), "tokens": np.concatenate(tokens),
                   "images": np.array([row["image"] for row in rows])}
        validate_features(payload, rows, key)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(path)
        print(f"Wrote {path.name} shape {payload['tokens'].shape}", flush=True)
    return path, environment


def evaluate(args, feature_path, environment):
    import torch

    start = time.perf_counter()
    manifest, rows = category_inputs(args)
    with np.load(feature_path, allow_pickle=False) as data:
        tokens = validate_features(data, rows, cache_key(args, manifest, environment))
    candidates = sorted((i for i, row in enumerate(rows) if row["split"] == "candidate"),
                        key=lambda i: rows[i]["image"])
    calibration = [i for i, row in enumerate(rows) if row["split"] == "calibration"]
    test = [i for i, row in enumerate(rows) if row["split"] == "test"]
    queries = calibration + test
    labels = [int(rows[i]["label"]) for i in test]
    cls_distances = cls_cosine_distances(tokens[candidates, 0])
    plans = {}
    for seed in args.seeds:
        orders = reference_orders(cls_distances, seed)
        plans.update({(seed, policy): order for policy, order in orders.items()})
    union = sorted({int(ref) for order in plans.values() for ref in order})
    distance_row = {ref: i for i, ref in enumerate(union)}
    print(f"Scoring {len(queries)} queries against {len(union)} distinct references for {args.category}", flush=True)
    with torch.inference_mode():
        patch_distances = candidate_patch_distances(
            torch.from_numpy(tokens[queries, 1:]).to(args.device),
            torch.from_numpy(tokens[np.array(candidates)[union], 1:]).to(args.device))
    references, scores, metrics, case_maps, calibration_maps = [], [], [], {}, {}
    for (seed, policy), order in plans.items():
        for shots in SETTINGS["shots"]:
            config = dict(zip(CONFIG_FIELDS, (args.category, seed, shots, policy)))
            selected = order[:shots]
            for position, ref in enumerate(selected, start=1):
                references.append(dict(config, position=position, image=rows[candidates[ref]]["image"]))
            nearest = patch_distances[[distance_row[ref] for ref in selected]].min(axis=0)
            values = image_scores(nearest)
            split = len(calibration)
            stats = score_metrics(values[split:], labels, values[:split])
            threshold = stats["calibration_threshold"]
            cls_upper = np.triu_indices(shots, k=1)
            stats["mean_pairwise_cls_distance"] = float(cls_distances[np.ix_(selected, selected)][cls_upper].mean())
            stats["calibration_mean_patch_distance"] = float(nearest[:split].mean(dtype=np.float64))
            calibration_maps[f"{seed}_{shots}_{policy}"] = nearest[:split]
            metrics.append(dict(config, **stats))
            for i, value in zip(queries, values):
                row = rows[i]
                scores.append(dict(config, split=row["split"], image=row["image"], label=row["label"],
                                   defect_type=row["defect_type"], score=float(value),
                                   threshold=threshold, predicted=int(value > threshold)))
            if seed == 11 and shots == 4:
                case_maps[policy] = nearest.reshape(len(queries), 16, 16)
    write_csv(args.out / "references.csv", REF_FIELDS, references)
    write_csv(args.out / "scores.csv", SCORE_FIELDS, scores)
    write_csv(args.out / "metrics.csv", METRIC_FIELDS, metrics)
    np.savez(args.out / "calibration_maps.npz", images=[rows[i]["image"] for i in calibration], **calibration_maps)
    if case_maps:
        np.savez(args.out / "case_maps.npz", images=[rows[i]["image"] for i in queries], **case_maps)
    write_json(args.out / "result.json", {
        "state": "finished", "category": args.category, "seeds": args.seeds, "job": slurm(),
        "configs": len(metrics), "query_rows": len(scores), "reference_rows": len(references),
        "feature_cache": feature_path.name, "feature_shape": list(tokens.shape),
        "feature_cache_sha256": sha256(feature_path),
        "input_manifest_sha256": sha256(args.inventory / "input_manifest.json"),
        "environment": environment, "batch_size": args.batch_size, "scoring_seconds": time.perf_counter() - start,
        "query_block": 2048, "reference_block": 8, "candidate_union_size": len(union),
    })
    print(f"Finished {args.category}: {len(metrics)} configurations", flush=True)


def main():
    parser = argparse.ArgumentParser(description="MVTec AD reference selection experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest")
    for option in ("data", "model", "out"):
        p.add_argument(f"--{option}", type=Path, required=True)
    p.add_argument("--archives", type=Path)
    p.add_argument("--unpack", action="store_true")
    p.add_argument("--source", default="User-provided original MVTec AD directory")
    p.add_argument("--categories", nargs="+", choices=COUNTS, default=list(COUNTS))
    for command in ("features", "evaluate", "run"):
        p = sub.add_parser(command)
        for option in ("data", "model", "inventory", "cache"):
            p.add_argument(f"--{option}", type=Path, required=True)
        p.add_argument("--category", choices=COUNTS, required=True)
        p.add_argument("--device", choices=("cpu", "cuda"), required=True)
        p.add_argument("--batch-size", type=int, default=16)
        if command != "features":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--seeds", nargs="+", type=int, default=SETTINGS["seeds"], choices=SETTINGS["seeds"])
    args = parser.parse_args()
    if args.command == "manifest":
        require(not args.unpack or args.archives is not None, "--unpack requires --archives")
        require(len(set(args.categories)) == len(args.categories), "Duplicate category")
        prepare(args)
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    require(args.batch_size > 0, "Batch size must be positive")
    if args.command == "features":
        extract(args)
        return
    require(len(set(args.seeds)) == len(args.seeds), "Duplicate seed")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        if args.command == "run":
            path, environment = extract(args)
        else:
            manifest, _ = category_inputs(args)
            environment = runtime(args.device)
            path = args.cache / f"{args.category}-{cache_key(args, manifest, environment)}.npz"
        evaluate(args, path, environment)
    except Exception:
        write_json(args.out / "result.json", {
            "state": "failed", "category": args.category, "seeds": args.seeds, "job": slurm(),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
