import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
from PIL import Image

from audit import audit
from protocol import (COUNTS, DIAGNOSTICS, METRIC_FIELDS, POLICIES, RATES, REF_FIELDS, SCORE_FIELDS,
                      SETTINGS, SPLIT_FIELDS, config_key, read_json, require, sha256, write_csv, write_json)


VARIABLES = RATES + DIAGNOSTICS
COMPARISONS = [(2, 2), (4, 4), (8, 8), (4, 8)]
SCOPES = [*COUNTS, "macro"]
CASE_TYPES = {
    (0, 1, 0): "normal_false_positive_corrected",
    (1, 1, 0): "defect_newly_missed",
    (1, 0, 1): "defect_recovered",
    (0, 0, 1): "normal_new_false_positive",
}
ATTRIBUTION = "Images and masks from MVTec AD © MVTec Software GmbH · CC BY-NC-SA 4.0"
CASE_RULE = "Maximum absolute difference in score / calibration threshold, then ascending relative image path"


def summarize(metrics):
    values = {config_key(row): np.array([float(row[key]) for key in VARIABLES]) for row in metrics}
    for seed in SETTINGS["seeds"]:
        for shots in SETTINGS["shots"]:
            for policy in POLICIES:
                values["macro", seed, shots, policy] = np.mean(
                    [values[category, seed, shots, policy] for category in COUNTS], axis=0)
    rng = np.random.default_rng(SETTINGS["bootstrap_seed"])
    seeds = SETTINGS["seeds"]
    draws = rng.integers(len(seeds), size=(SETTINGS["bootstrap_draws"], len(seeds)))
    summary, paired = [], []
    for scope in SCOPES:
        for shots in SETTINGS["shots"]:
            for policy in POLICIES:
                sample = np.stack([values[scope, seed, shots, policy] for seed in seeds])
                for metric, mean, sd in zip(VARIABLES, sample.mean(axis=0), sample.std(axis=0, ddof=1)):
                    summary.append(dict(scope=scope, shots=shots, policy=policy, metric=metric,
                                        n_seeds=len(seeds), mean=mean, sample_sd=sd))
        for diverse, random in COMPARISONS:
            diff = np.stack([values[scope, seed, diverse, "diverse"] - values[scope, seed, random, "random"]
                             for seed in seeds])
            intervals = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975], axis=0)
            for j, metric in enumerate(VARIABLES):
                scale = 100 if metric in RATES else 1
                paired.append(dict(scope=scope, comparison=f"diverse_{diverse}_minus_random_{random}",
                                   diverse_shots=diverse, random_shots=random, metric=metric,
                                   n_paired_seeds=len(seeds), unit="pp" if scale == 100 else "cosine_distance",
                                   mean_difference=diff[:, j].mean() * scale,
                                   sample_sd=diff[:, j].std(ddof=1) * scale,
                                   ci95_low=intervals[0, j] * scale, ci95_high=intervals[1, j] * scale))
    return summary, paired


def defect_recall(scores):
    defects = defaultdict(list)
    for row in scores:
        if row["split"] == "test" and row["label"] == "1":
            defects[(*config_key(row), row["defect_type"])].append(int(row["predicted"]))
    return [
        dict(category=category, seed=seed, shots=shots, policy=policy, defect_type=kind,
             n_defects=len(predict), n_detected=sum(predict), calibrated_recall=float(np.mean(predict)))
        for (category, seed, shots, policy, kind), predict in sorted(defects.items())
    ]


def budget_figure(summary, out):
    colors = {"random": "#386cb0", "diverse": "#b84825"}
    labels = {"fpr_at_95": "FPR at ≥95% defect recall", "calibrated_fpr": "Calibrated test FPR",
              "calibrated_recall": "Calibrated test defect recall"}
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), layout="constrained")
    for axis, (metric, label) in zip(axes, labels.items()):
        for policy in POLICIES:
            points = [row for row in summary if row["scope"] == "macro" and
                      row["metric"] == metric and row["policy"] == policy]
            axis.errorbar([row["shots"] for row in points], [row["mean"] for row in points],
                          yerr=[row["sample_sd"] for row in points], marker="o", capsize=4,
                          color=colors[policy], label=policy.capitalize())
        axis.set(title=label, xlabel="Normal reference images", xticks=SETTINGS["shots"])
        axis.set_ylim(bottom=max(0, axis.get_ylim()[0]), top=min(1, axis.get_ylim()[1]))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Bottle, Grid and Screw macro-average · Mean ± sample SD across 10 selection seeds")
    figure.savefig(out)
    plt.close(figure)


def change_cases(scores, references):
    selected_scores = defaultdict(dict)
    for row in scores:
        if int(row["seed"]) == 11 and int(row["shots"]) == 4 and row["split"] == "test":
            selected_scores[row["image"]][row["policy"]] = row
    refs = {}
    for category in COUNTS:
        for policy in POLICIES:
            rows = [row for row in references if config_key(row) == (category, 11, 4, policy)]
            refs[category, policy] = "|".join(row["image"] for row in sorted(rows, key=lambda r: int(r["position"])))
    changes = defaultdict(list)
    for image, policies in selected_scores.items():
        random, diverse = policies["random"], policies["diverse"]
        kind = CASE_TYPES.get((int(random["label"]), int(random["predicted"]), int(diverse["predicted"])))
        if kind is None:
            continue
        case = dict(image=image, category=random["category"], label=random["label"], defect_type=random["defect_type"])
        ratios = {}
        for policy, row in policies.items():
            case[f"{policy}_references"] = refs[row["category"], policy]
            for field in ("score", "threshold", "predicted"):
                case[f"{policy}_{field}"] = row[field]
            require(float(row["threshold"]) > 0, f"Case ratio requires positive threshold: {image}")
            ratios[policy] = float(row["score"]) / float(row["threshold"])
        case["ratio_difference"] = abs(ratios["diverse"] - ratios["random"])
        changes[kind].append(case)
    cases = []
    for kind in CASE_TYPES.values():
        options = sorted(changes[kind], key=lambda row: (-row["ratio_difference"], row["image"]))
        case = dict(change_type=kind, count=len(options), seed=11, shots=4, rule=CASE_RULE)
        if options:
            case.update(options[0])
        cases.append(case)
    return cases


def case_figure(cases, dirs, data, inventory, out):
    with PdfPages(out) as pdf:
        for case in cases:
            title = case["change_type"].replace("_", " ").capitalize()
            if not case["count"]:
                fig = plt.figure(figsize=(11.0, 4.7))
                fig.text(0.07, 0.78, title, fontsize=18)
                fig.text(0.07, 0.63, "0 changes across Bottle, Grid and Screw at seed 11 and four shots.")
                fig.text(0.07, 0.12, ATTRIBUTION, fontsize=9)
            else:
                image = case["image"]
                file_hashes = {entry["image"]: entry["sha256"] for entry in
                               read_json(inventory / f"{case['category']}_files.json")}
                with np.load(dirs[case["category"]] / "case_maps.npz", allow_pickle=False) as maps:
                    i = maps["images"].tolist().index(image)
                    random, diverse = (maps[policy][i] for policy in POLICIES)
                require(sha256(data / image) == file_hashes[image], f"Case image changed: {image}")
                with Image.open(data / image) as pic:
                    rgb = np.asarray(pic.convert("RGB"))
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                if int(case["label"]):
                    mask_path = Path(case["category"]) / "ground_truth" / case["defect_type"] / (Path(image).stem + "_mask.png")
                    require(sha256(data / mask_path) == file_hashes[mask_path.as_posix()],
                            f"Case mask changed: {mask_path}")
                    with Image.open(data / mask_path) as annotation:
                        mask = np.asarray(annotation.convert("L")) > 0
                    require(mask.shape == rgb.shape[:2] and mask.any(), f"Invalid displayed mask: {mask_path}")
                w, h = rgb.shape[1], rgb.shape[0]
                extent = (-0.5, w - 0.5, h - 0.5, -0.5)
                limits = (min(random.min(), diverse.min()), max(random.max(), diverse.max()))
                fig, axes = plt.subplots(1, 4, figsize=(12.4, 5.1))
                fig.subplots_adjust(left=0.04, right=0.96, bottom=0.21, top=0.76, wspace=0.15)
                axes[0].imshow(rgb)
                axes[0].set_title("Original image")
                axes[1].imshow(rgb)
                axes[1].set_title("Ground truth" if mask.any() else "Normal · no defect mask")
                for axis, policy, heat in zip(axes[2:], POLICIES, (random, diverse)):
                    rendered = axis.imshow(heat, extent=extent, interpolation="nearest", cmap="magma",
                                           vmin=limits[0], vmax=limits[1])
                    label = "defect" if int(case[f"{policy}_predicted"]) else "normal"
                    axis.set_title(f"{policy.capitalize()} · {label}\n"
                                   f"Score {float(case[f'{policy}_score']):.4f} / threshold {float(case[f'{policy}_threshold']):.4f}",
                                   fontsize=10)
                if mask.any():
                    for axis in axes[1:]:
                        axis.contour(mask, levels=[0.5], colors=["#27edc2"], linewidths=1.0)
                for axis in axes:
                    axis.set_axis_off()
                cax = fig.add_axes([0.56, 0.15, 0.36, 0.025])
                fig.colorbar(rendered, cax=cax, orientation="horizontal", label="Nearest patch cosine distance")
                change_word = "change" if case["count"] == 1 else "changes"
                fig.suptitle(f"{title} · {case['count']} matching {change_word}", x=0.04, y=0.95, ha="left")
                contour = " · Mint contour marks the defect mask" if mask.any() else ""
                fig.text(0.04, 0.86, f"{image} · Seed 11 · Four references{contour}", fontsize=9)
                fig.text(0.04, 0.065, ATTRIBUTION + " · Derived heatmaps under the same license", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


def findings(paired, defects, cases, counts):
    keyed = {(r["scope"], r["diverse_shots"], r["random_shots"], r["metric"]): r for r in paired}

    def d(scope, left, right, metric):
        return keyed[scope, left, right, metric]["mean_difference"]

    def signed(value, digits=2):
        return f"{value:+.{digits}f}" if round(value, digits) else f"{0:.{digits}f}"

    def interval(scope, left, right, metric):
        row = keyed[scope, left, right, metric]
        digits = 2 if row["unit"] == "pp" else 4
        suffix = " pp" if row["unit"] == "pp" else ""
        return (f"{signed(row['mean_difference'], digits)}{suffix} "
                f"[{signed(row['ci95_low'], digits)}, {signed(row['ci95_high'], digits)}]")

    lines = [
        "# AIAA 5026 Topic 3 results",
        "",
        f"The audited main run contains {counts['metrics']} configurations and {counts['test_images']} distinct "
        "test images. Each macro estimate averages Bottle, Grid and Screw equally within each seed, then averages "
        "the ten seeds. Brackets show paired percentile 95% bootstrap intervals in the same unit as the estimate.",
        "", "## 1. Normal false positives at equal budgets", "",
        "| References | Macro FPR@95 change | Macro calibrated FPR change |",
        "| ---: | ---: | ---: |",
    ]
    for shots in SETTINGS["shots"]:
        lines.append(f"| {shots} | {interval('macro', shots, shots, 'fpr_at_95')} | "
                     f"{interval('macro', shots, shots, 'calibrated_fpr')} |")
    if all(keyed["macro", s, s, "calibrated_fpr"]["ci95_high"] < 0 for s in SETTINGS["shots"]):
        lines += ["", "Diverse selection reduced macro calibrated FPR at all three budgets. "
                  "All three paired intervals were below zero."]
    if all(keyed["macro", s, s, "fpr_at_95"]["ci95_low"] <= 0 <=
           keyed["macro", s, s, "fpr_at_95"]["ci95_high"] for s in SETTINGS["shots"]):
        lines += ["", "All FPR@95 intervals included zero. In this experiment the retrospective ROC comparison "
                  "therefore supports a less consistent reduction."]
    lines += [
        "",
        "| Category | 2 refs FPR@95 / calibrated FPR | 4 refs FPR@95 / calibrated FPR | 8 refs FPR@95 / calibrated FPR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for cat in COUNTS:
        cells = [" / ".join(signed(d(cat, shots, shots, key)) for key in ("fpr_at_95", "calibrated_fpr"))
                 for shots in SETTINGS["shots"]]
        lines.append(f"| {cat.capitalize()} | {' | '.join(cells)} |")
    lines += ["", "Category differences above are in percentage points. Negative entries support lower false positives. "
              "Positive entries identify settings where diverse selection raised false positives.", "",
              "| References | Macro CLS distance change | Macro normal calibration patch distance change |",
              "| ---: | ---: | ---: |"]
    for shots in SETTINGS["shots"]:
        lines.append(f"| {shots} | {interval('macro', shots, shots, DIAGNOSTICS[0])} | "
                     f"{interval('macro', shots, shots, DIAGNOSTICS[1])} |")
    if all(d("macro", s, s, metric) > 0 for s in SETTINGS["shots"] for metric in DIAGNOSTICS):
        lines += ["", "Both distances increased at every budget. Diverse selection spread reference CLS embeddings "
                  "farther apart, and normal calibration patches had farther nearest neighbors on average."]
    lines += [
        "",
        "Smaller calibration patch distance indicates greater local coverage. The category rows in `paired.csv` "
        "show whether the coverage change is shared across classes.",
        "",
        "Evidence in `paired.csv`, keyed by `scope`, `comparison` (for example `diverse_4_minus_random_4`), "
        "and `metric`. `metrics.csv` contains achieved test recall at the ROC threshold by "
        "`category,seed,shots,policy`. FPR@95 uses defect labels to select a retrospective ROC point.",
        "", "## 2. Diverse four versus random eight", "",
        "| Metric | Macro paired change [95% interval] | Sample SD of paired changes |",
        "| --- | ---: | ---: |",
    ]
    for metric in RATES:
        row = keyed["macro", 4, 8, metric]
        lines.append(f"| {metric} | {interval('macro', 4, 8, metric)} | {row['sample_sd']:.2f} pp |")
    lines += [
        "",
        "Four references require 1,024 stored patch descriptors (1.50 MiB as float32 × 384) and 262,144 "
        "384-dimensional patch dot products per scored image with direct matching. Eight require 2,048 "
        "descriptors (3.00 MiB) and 524,288 dot products. These counts exclude the query descriptors. "
        "Diverse selection also requires candidate CLS descriptors and a greedy search. The shared offline "
        "implementation computes a CLS distance matrix for 169 Bottle, 224 Grid and 280 Screw candidates, "
        "then reuses per-reference query distances across configurations. The run timings cover shared, "
        "batched experiment work. The study specifies no equivalence margin for this comparison.",
        "",
        "Evidence in `paired.csv` at `scope=macro,comparison=diverse_4_minus_random_8`.",
        "", "## 3. Defect recall under independent calibration", "",
    ]
    for shots in SETTINGS["shots"]:
        lines.append(f"- **{shots} references.** Calibrated macro defect recall changed by "
                     f"{interval('macro', shots, shots, 'calibrated_recall')}.")
    recall_drops = [str(s) for s in SETTINGS["shots"] if keyed["macro", s, s, "calibrated_recall"]["ci95_high"] < 0]
    if recall_drops:
        lines += ["", f"The calibrated recall interval was entirely below zero at the following reference "
                  f"counts ({', '.join(recall_drops)})."]
    lines += ["", "| Category | 2 refs FPR / recall | 4 refs FPR / recall | 8 refs FPR / recall |",
              "| --- | ---: | ---: | ---: |"]
    for cat in COUNTS:
        cells = [" / ".join(signed(d(cat, shots, shots, key)) for key in ("calibrated_fpr", "calibrated_recall"))
                 for shots in SETTINGS["shots"]]
        lines.append(f"| {cat.capitalize()} | {' | '.join(cells)} |")
    lines += ["", "Entries are paired percentage-point changes. Two negative entries in a cell indicate "
              "that lower false positives coincided with lower defect recall.", "",
              "Calibrated prediction changes for seed 11 and four references across all three categories:", ""]
    for case in cases:
        image = f" · `{case['image']}`" if case["count"] else ""
        lines.append(f"- {case['change_type'].replace('_', ' ').capitalize()} · {case['count']}{image}")
    type_pairs = defaultdict(dict)
    for row in defects:
        type_pairs[row["category"], row["defect_type"], row["shots"], row["seed"]][row["policy"]] = row
    type_changes = defaultdict(list)
    for (category, defect, shots, _), policy in type_pairs.items():
        difference = policy["diverse"]["calibrated_recall"] - policy["random"]["calibrated_recall"]
        type_changes[category, defect, shots, policy["random"]["n_defects"]].append(difference)
    lines += ["", "The largest decreases and increases by defect type across equal-budget comparisons are shown "
              "below when present. Each mean uses ten paired seeds.", "",
              "| Category | Defect type | References | Test defects | Recall change |",
              "| --- | --- | ---: | ---: | ---: |"]
    ordered = sorted((float(np.mean(values)), key) for key, values in type_changes.items())
    extremes = [row for row in ordered[:2] if row[0] < 0] + [row for row in reversed(ordered[-2:]) if row[0] > 0]
    for change, (category, defect, shots, n_defects) in extremes:
        lines.append(f"| {category.capitalize()} | {defect} | {shots} | {n_defects} | {signed(100 * change)} pp |")
    if not extremes:
        lines.append("| All | All | All | — | 0.00 pp |")
    lines += [
        "",
        "Evidence in `paired.csv` (`metric=calibrated_recall` and `metric=calibrated_fpr`), "
        "`defect_recall.csv` (`category,seed,shots,policy,defect_type`), and `cases.csv` (`change_type`). "
        "Calibration uses 40 separate normal images per class, with a strict score > threshold rule.",
        "",
        "The bootstrap resamples the ten paired selection seeds 10,000 times, with images and categories "
        "fixed. Its intervals describe selection variability for these three categories and test images. "
        "Test denominators are discrete, some defect types have few examples, and operational condition labels "
        "are unavailable. The design therefore supports descriptions by category and defect type.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Audit and render the complete main experiment")
    for path in ("inventory", "cache", "data", "out", "run-record", "run-status"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--logs", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    manifest, splits, tables, info, counts = audit(
        args.inventory, args.results, args.cache, list(COUNTS), SETTINGS["seeds"])
    run_record = read_json(args.run_record)
    require(all(str(run_record["main_jobs"][cat]["job_id"]) == info[cat]["job"]["JOB_ID"] and
                run_record["main_jobs"][cat]["state"] == "COMPLETED" and
                run_record["main_jobs"][cat]["exit_code"] == "0:0" for cat in COUNTS),
            "Slurm record does not confirm successful main jobs")
    dirs = {read_json(path / "result.json")["category"]: path for path in args.results}
    summary, paired = summarize(tables["metrics"])
    defects = defect_recall(tables["scores"])
    cases = change_cases(tables["scores"], tables["references"])
    args.out.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).parent
    manifest.update({"state": "complete_and_audited", "counts": counts, "main_runs": info,
                     "scheduler": run_record, "bootstrap_unit": "paired selection seed with fixed categories",
                     "code_sha256": {p.name: sha256(p) for p in sorted(code.iterdir())
                                     if p.suffix in {".py", ".slurm"} or p.name == "PROTOCOL.md"}})
    for name, rows, fields in (
        ("splits", splits, SPLIT_FIELDS),
        ("scores", tables["scores"], SCORE_FIELDS),
        ("metrics", tables["metrics"], METRIC_FIELDS),
        ("references", tables["references"], REF_FIELDS),
        ("summary", summary, list(summary[0])),
        ("paired", paired, list(paired[0])),
        ("defect_recall", defects, list(defects[0])),
        ("cases", cases, ["change_type", "count", "image", "category", "label", "defect_type", "seed", "shots",
                          "random_references", "diverse_references", "random_score", "diverse_score",
                          "random_threshold", "diverse_threshold", "random_predicted", "diverse_predicted",
                          "ratio_difference", "rule"]),
    ):
        write_csv(args.out / f"{name}.csv", fields, rows)
    budget_figure(summary, args.out / "reference_budget.pdf")
    case_figure(cases, dirs, args.data, args.inventory, args.out / "case_studies.pdf")
    (args.out / "findings.md").write_text(findings(paired, defects, cases, counts), encoding="utf-8")
    shutil.copyfile(args.run_status, args.out / "RUN_STATUS.md")
    readme = code / "README.md"
    shutil.copyfile(readme if readme.exists() else code.parent / "README.md", args.out / "README.md")
    shutil.copytree(code, args.out / "code", ignore=shutil.ignore_patterns("__pycache__", "README.md"))
    (args.out / "logs").mkdir()
    for path in args.logs:
        shutil.copyfile(path, args.out / "logs" / path.name)
    write_json(args.out / "manifest.json", manifest)
    print(f"Wrote complete handoff to {args.out}")


if __name__ == "__main__":
    main()
