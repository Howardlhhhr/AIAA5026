import hashlib
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from protocol import (ARCHIVES, CONFIG_SHA, COUNTS, DATA_URL, MODEL, REVISION, SETTINGS,
                      SPLIT_FIELDS, WEIGHT_SHA, digest, read_json, require, sha256,
                      write_csv, write_json)


def unpack(archives, data, categories):
    data.mkdir(parents=True, exist_ok=True)
    for category in categories:
        if (data / category).exists():
            continue
        archive = archives / f"{category}.tar.xz"
        with tarfile.open(archive, "r:xz") as stream:
            members = []
            for member in stream.getmembers():
                path = Path(member.name)
                require(not path.is_absolute() and ".." not in path.parts,
                        f"Unexpected archive path {path}")
                if path.parts and path.parts[0] == category:
                    require(member.isfile() or member.isdir(), f"Unexpected archive entry {path}")
                    members.append(member)
            require(members, f"No {category} entries in {archive}")
            stream.extractall(data, members=members, filter="data")


def verified_model(model):
    config = (model / "config.json").read_bytes()
    git_blob = f"blob {len(config)}\0".encode() + config
    require(hashlib.sha1(git_blob).hexdigest() == CONFIG_SHA, "Unexpected model config at pinned revision")
    weights = sha256(model / "model.safetensors")
    require(weights == WEIGHT_SHA, f"Unexpected model SHA-256 {weights}")
    return {"id": MODEL, "revision": REVISION, "weight_sha256": weights,
            "config_sha256": sha256(model / "config.json"), "format": "safetensors"}


def inventory(data, category):
    train = sorted((data / category / "train" / "good").glob("*.png"))
    test = sorted((data / category / "test").glob("*/*.png"))
    masks = sorted((data / category / "ground_truth").glob("*/*.png"))
    train_count, normal, defective = COUNTS[category]
    require((len(train), sum(p.parent.name == "good" for p in test),
             sum(p.parent.name != "good" for p in test)) == (train_count, normal, defective),
            f"Unexpected train/test counts for {category}: {len(train)}, {Counter(p.parent.name for p in test)}")
    all_paths = {path.relative_to(data).as_posix() for path in train + test + masks}
    expected_paths = {path.relative_to(data).as_posix() for path in (data / category).rglob("*")
                      if path.suffix.lower() in {".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"}}
    require(all_paths == expected_paths, f"Unexpected image paths in {category}: {all_paths ^ expected_paths}")
    require(len({path.resolve(strict=True) for path in train + test + masks}) == len(all_paths),
            f"Image path aliases in {category}")
    calibration = set(np.random.default_rng(SETTINGS["calibration_seed"]).permutation(len(train))[:40])
    rows, files = [], []
    for i, path in enumerate(train + test):
        image = path.relative_to(data).as_posix()
        defect = path.parent.name
        split = "test" if i >= len(train) else "calibration" if i in calibration else "candidate"
        rows.append(dict(category=category, image=image, split=split,
                         label=int(defect != "good"), defect_type=defect))
        files.append({"image": image, "sha256": sha256(path)})
    expected_masks = {f"{category}/ground_truth/{p.parent.name}/{p.stem}_mask.png"
                      for p in test if p.parent.name != "good"}
    require({p.relative_to(data).as_posix() for p in masks} == expected_masks,
            f"Defect mask set differs from test defects for {category}")
    files += [{"image": path.relative_to(data).as_posix(), "sha256": sha256(path)} for path in masks]
    return rows, files


def prepare(args):
    if args.unpack:
        unpack(args.archives, args.data, args.categories)
    model = verified_model(args.model)
    args.out.mkdir(parents=True, exist_ok=False)
    splits, categories = [], {}
    for category in args.categories:
        rows, files = inventory(args.data, category)
        splits.extend(rows)
        expected = COUNTS[category]
        archive = args.archives / f"{category}.tar.xz" if args.archives else None
        categories[category] = {
            "counts": dict(zip(("train_good", "test_good", "test_defect"), expected)),
            "defect_types": dict(Counter(row["defect_type"] for row in rows if row["label"])),
            "archive_sha256": sha256(archive) if archive and archive.is_file() else None,
            "source": ARCHIVES[category] if args.unpack else args.source,
            "data_digest": digest({"splits": rows, "files": files}),
        }
        write_json(args.out / f"{category}_files.json", files)
        print(f"{category}: {expected}, {len(files)} verified paths", flush=True)
    manifest = {
        "dataset": "Original MVTec AD", "data_homepage": DATA_URL,
        "data_owner": "MVTec Software GmbH", "data_license": "CC BY-NC-SA 4.0",
        "categories": categories, "model": model, "settings": SETTINGS,
    }
    write_csv(args.out / "splits.csv", SPLIT_FIELDS, splits)
    manifest["split_sha256"] = sha256(args.out / "splits.csv")
    write_json(args.out / "input_manifest.json", manifest)


def check_inputs(inventory_dir, data, category, rows, manifest):
    entries = read_json(inventory_dir / f"{category}_files.json")
    for entry in entries:
        path = data / entry["image"]
        require(sha256(path) == entry["sha256"], f"Input file changed: {entry['image']}")
    normalized_rows = [dict(row, label=int(row["label"])) for row in rows]
    require(digest({"splits": normalized_rows, "files": entries}) ==
            manifest["categories"][category]["data_digest"], f"Input manifest mismatch for {category}")
