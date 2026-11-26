# AIAA 5026 Topic 3

**Diversity or Quantity in Few-Shot Industrial Anomaly Detection**

This project compares random and greedy farthest-first selection of normal reference images for MVTec AD Bottle, Grid, and Screw. It uses frozen DINOv2 ViT-S/14 patch embeddings, a separate normal calibration split, and paired evaluation across ten seeds. The [registered protocol](code/PROTOCOL.md) defines the experiment.

## Recompute Table 4 from the released scores

From the project root, use the existing `report_handoff/scores.csv`. The script uses only the Python standard library and the score CSV.

```bash
python3 code/table4_sensitivity.py --scores report_handoff/scores.csv --output report_handoff/table4_sensitivity.csv
```

From inside the unpacked `report_handoff/` directory, use paths relative to that directory.

```bash
python3 code/table4_sensitivity.py --scores scores.csv --output table4_sensitivity.csv
```

For each category, seed, budget, and policy, the script linearly interpolates the 90th, 95th, and 97.5th percentiles of the 40 normal calibration scores. It flags a test image when `score > threshold`. Within each seed, Bottle, Grid, and Screw contribute equally to the diverse-minus-random difference at the same budget. The CSV reports the mean and sample standard deviation over the ten paired seeds in percentage points. There is one row per percentile and budget. Negative FPR differences indicate fewer false positives. Negative recall differences indicate more missed defects. The 95th percentile rows use the original main-run calibration rule.

## CPU reproduction of the main experiment

The paper results came from CPU array `14141239` in `emergency_cpu` on HKUST(GZ) HPC Phase II. The array ran one category at a time with 12 CPU cores, 12 GiB of memory, and an 18-minute limit per task. Historical A40 job `14140781` checked the Bottle pipeline. The CPU smoke trial below checks a fresh CPU installation. [Slurm guide](https://docs.hpc.hkust-gz.edu.cn/en/docs/hpc12/slurm/job2/) · [Queue limits](https://docs.hpc.hkust-gz.edu.cn/en/docs/hpc12/slurm/queue/)

Use a shared working directory visible to compute nodes. For exact software matching, set `py` to the recorded Python 3.11 interpreter. The main run used PyTorch `2.4.0+cu121` in CPU mode, Transformers `4.47.0.dev0`, safetensors `0.4.4`, NumPy `1.26.4`, Pillow `13.0.0`, and Matplotlib `3.9.2`. This independent Linux CPU setup uses public builds. Different package builds can change numerical outputs. [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/)

```bash
python3.11 -m venv .venv
py="$PWD/.venv/bin/python"
"$py" -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
"$py" -m pip install transformers==4.47.1 safetensors==0.4.4 numpy==1.26.4 Pillow==10.4.0 matplotlib==3.9.2
```

Use an approved original MVTec AD copy or stage `bottle.tar.xz`, `grid.tar.xz`, and `screw.tar.xz` in `inputs/archives/`. Place the pinned `config.json` and `model.safetensors` in `inputs/dinov2-small/`. The manifest command verifies the model checksum and dataset inventory, records archive checksums, and creates the seeded splits. Compute jobs read the staged files offline. [MVTec AD downloads](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads) · [Pinned DINOv2 revision](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)

```bash
set -euo pipefail
root="$PWD"
py="${py:-$(command -v python3)}"
mkdir -p logs runs
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 USE_TF=0
"$py" -B -m unittest discover -s code -v >logs/unit_tests.log 2>&1
"$py" -B code/experiment.py manifest \
    --data inputs/mvtec --model inputs/dinov2-small --archives inputs/archives \
    --unpack --out runs/inventory-main
```

For an unpacked copy, make the dataset available at `inputs/mvtec`, omit `--unpack` and `--archives`, and record its provenance with `--source`. Then submit the Bottle CPU smoke trial. Use the account, partitions, and QoS values allowed by your cluster. Run the following blocks in the same Bash session.

```bash
account=root
smoke_partition=debug
cpu_partition=emergency_cpu
cpu_qos=emergency_cpu

smoke_options=(
    --parsable --partition="$smoke_partition" --account="$account" --time=00:15:00
    --output=logs/smoke-%j.out --error=logs/smoke-%j.err
)
smoke=$(sbatch "${smoke_options[@]}" code/run.slurm \
    "$root" "$py" runs/inventory-main runs/smoke-01 bottle cpu --seeds 11)
sacct -j "${smoke%%;*}" -X --format=JobID,State,ExitCode,Elapsed
```

After the smoke job shows `COMPLETED` with `0:0`, audit it and submit the CPU production array. The array indices select Bottle, Grid, and Screw in order. Monitor all three tasks to completion.

```bash
"$py" -B code/audit.py --inventory runs/inventory-main --cache runs/features \
    --results runs/smoke-01/bottle --categories bottle --seeds 11 \
    --out runs/smoke_audit.json | tee logs/smoke_audit.log
main_options=(
    --parsable --partition="$cpu_partition" --qos="$cpu_qos" --account="$account"
    --array=0-2%1 --cpus-per-task=12 --mem=12G --time=00:18:00
    --output=logs/main-%A_%a.out --error=logs/main-%A_%a.err
)
main=$(sbatch "${main_options[@]}" code/run.slurm \
    "$root" "$py" runs/inventory-main runs/main-01 array cpu)
squeue -j "${main%%;*}"
sacct -j "${main%%;*}" -X --format=JobID,JobIDRaw,State,ExitCode,Elapsed,AllocTRES
```

If a category fails, keep the logs and resubmit that category with a fresh attempt directory. Omit the array option and pass the category name where the command passes `array`. Use the successful output directory in the audit.

## Audit and generate the handoff

Run these commands after the three main jobs have completed. `report.py` creates a fresh output directory. Use a different destination if you already have an unpacked handoff there.

```bash
results=(runs/main-01/bottle runs/main-01/grid runs/main-01/screw)
"$py" -B code/audit.py --inventory runs/inventory-main --cache runs/features \
    --results "${results[@]}" --out runs/main_audit.json | tee logs/main_audit.log
"$py" -B code/record_jobs.py --smoke-audit runs/smoke_audit.json --main "${results[@]}" \
    --tests logs/unit_tests.log --out runs/job_accounting.json --status runs/RUN_STATUS.md
"$py" -B code/report.py --inventory runs/inventory-main --cache runs/features --data inputs/mvtec \
    --results "${results[@]}" --run-record runs/job_accounting.json --run-status runs/RUN_STATUS.md \
    --out report_handoff --logs logs/unit_tests.log logs/smoke_audit.log logs/main_audit.log
"$py" -B code/table4_sensitivity.py \
    --scores report_handoff/scores.csv --output report_handoff/table4_sensitivity.csv
```

The audited handoff contains 180 configurations in `metrics.csv`, 321 distinct test images in `splits.csv`, 26,460 rows in `scores.csv`, and 840 rows in `references.csv`. `manifest.json` and `RUN_STATUS.md` record provenance and Slurm accounting. `paired.csv` contains the original equal-budget comparisons for checking the 95th-percentile rows of Table 4. The `threshold` and `predicted` columns in `scores.csv` record the original calibration.

The original images, annotations, and derived heatmaps in `case_studies.pdf` are based on MVTec AD © MVTec Software GmbH and are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The dataset archives and model weights are staged separately.
