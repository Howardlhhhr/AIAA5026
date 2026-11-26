# AIAA 5026 Topic 3

**Diversity or Quantity in Few-Shot Industrial Anomaly Detection**

This project compares random selection and greedy farthest-first selection of normal reference images for original MVTec AD Bottle, Grid, and Screw. It uses frozen DINOv2 ViT-S/14 patch embeddings, a separate normal calibration split, and paired evaluation across ten reference-selection seeds. The [registered protocol](code/PROTOCOL.md) defines the budgets, thresholds, metrics, and case-selection rule.

## Setup

Use Python 3.11. On HPC, point `py` to the Python executable of an environment with PyTorch, Transformers, safetensors, NumPy, Pillow, and Matplotlib. The audited run used PyTorch `2.4.0+cu121`, Transformers `4.47.0.dev0`, safetensors `0.4.4`, NumPy `1.26.4`, Pillow `13.0.0`, and Matplotlib `3.9.2`. A released Transformers 4.47 build can run the workflow in a fresh environment. Software builds and CPU versus GPU arithmetic can affect the last digits of scores.

For an independent CPU installation, use the following commands. Select the matching PyTorch CUDA wheel instead for GPU jobs. [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/)

```bash
python3.11 -m venv .venv
py="$PWD/.venv/bin/python"
"$py" -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
"$py" -m pip install transformers==4.47.1 safetensors==0.4.4 numpy==1.26.4 Pillow==10.4.0 matplotlib==3.9.2
"$py" -B -m unittest discover -s code -v
```

Stage `bottle.tar.xz`, `grid.tar.xz`, and `screw.tar.xz` from the [official MVTec AD download page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads) in `inputs/archives/`. Stage `config.json` and `model.safetensors` from the [fixed DINOv2 revision](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056) in `inputs/dinov2-small/`. The manifest command verifies the weight checksum, inventories the official image and mask files, and creates the seeded splits.

```bash
mkdir -p logs runs
"$py" -B -m unittest discover -s code -v >logs/unit_tests.log 2>&1
"$py" -B code/experiment.py manifest +    --data inputs/mvtec --model inputs/dinov2-small --archives inputs/archives +    --unpack --out runs/inventory
```

To use an approved unpacked copy, point `--data` to the dataset root, omit `--unpack` and `--archives`, and record its provenance with `--source`. Make that root available at `inputs/mvtec` for the Slurm script.

## Run on Slurm

Use a shared working directory visible to compute nodes. The jobs load the staged dataset and model offline. Replace the partition variables with values that are available to your Slurm account. Set the account and QoS in the submission command if your cluster requires them.

```bash
smoke_partition=debug
main_partition=YOUR_GPU_PARTITION
smoke=$(sbatch --parsable -p "$smoke_partition" --gres=gpu:1 +    --time=00:15:00 --output=logs/smoke-%j.out --error=logs/smoke-%j.err +    code/run.slurm "$PWD" "$py" runs/inventory runs/smoke-01 bottle cuda --seeds 11)
squeue -j "${smoke%%;*}"
sacct -j "${smoke%%;*}" -X --format=JobID,State,ExitCode,Elapsed
```

Once the Bottle smoke job completes with `0:0`, audit it and submit the main experiment. The array runs at most one category at a time. For CPU-only Slurm, choose a CPU partition, omit `--gres`, and pass `cpu` in place of `cuda`.

```bash
"$py" -B code/audit.py --inventory runs/inventory --cache runs/features +    --results runs/smoke-01/bottle --categories bottle --seeds 11 +    --out runs/smoke_audit.json | tee logs/smoke_audit.log
main=$(sbatch --parsable -p "$main_partition" --gres=gpu:1 --array=0-2%1 +    --output=logs/main-%A_%a.out --error=logs/main-%A_%a.err +    code/run.slurm "$PWD" "$py" runs/inventory runs/main-01 array cuda)
squeue -j "${main%%;*}"
sacct -j "${main%%;*}" -X --format=JobID,JobIDRaw,State,ExitCode,Elapsed
```

If a category fails, preserve its logs and resubmit that category with a fresh attempt directory. Pass its name in place of `array` and use its successful output path in the commands below.

## Audit and generate results

After all three main tasks complete, audit the configurations and regenerate the tables and PDF figures.

```bash
results=(runs/main-01/bottle runs/main-01/grid runs/main-01/screw)
"$py" -B code/audit.py --inventory runs/inventory --cache runs/features +    --results "${results[@]}" --out runs/main_audit.json | tee logs/main_audit.log
"$py" -B code/record_jobs.py --smoke-audit runs/smoke_audit.json --main "${results[@]}" +    --tests logs/unit_tests.log --out runs/job_accounting.json --status runs/RUN_STATUS.md
"$py" -B code/report.py --inventory runs/inventory --cache runs/features --data inputs/mvtec +    --results "${results[@]}" --run-record runs/job_accounting.json --run-status runs/RUN_STATUS.md +    --out report_handoff --logs logs/unit_tests.log logs/smoke_audit.log logs/main_audit.log
```

The completed audit expects 180 configurations, 321 distinct test images, 26,460 score rows, and 840 reference rows. The generated `report_handoff/` includes `findings.md`, `manifest.json`, the CSV tables, and both figures. Model weights, the original dataset, and experiment outputs are prepared or generated separately from this source repository. MVTec AD images and annotations are © MVTec Software GmbH under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
