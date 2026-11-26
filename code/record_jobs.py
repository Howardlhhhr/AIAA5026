import argparse
import subprocess
from pathlib import Path

from protocol import read_json, require, write_json


FIELDS = {
    "JobIDRaw": "job_id", "JobID": "slurm_reference", "Partition": "partition",
    "State": "state", "ExitCode": "exit_code", "Elapsed": "elapsed",
    "AllocTRES": "resources", "NodeList": "node", "Start": "start", "End": "end",
}


def accounting(job_ids):
    query = subprocess.run(
        ["sacct", "-j", ",".join(job_ids), "-nP", "-X", f"--format={','.join(FIELDS)}"],
        check=True, capture_output=True, text=True,
    )
    rows = [dict(zip(FIELDS.values(), line.split("|"))) for line in query.stdout.splitlines() if line]
    return {row["job_id"]: row for row in rows}, rows


def main():
    parser = argparse.ArgumentParser(description="Collect Slurm accounting and write handoff run status")
    parser.add_argument("--smoke-audit", type=Path, required=True)
    parser.add_argument("--main", nargs="+", type=Path, required=True)
    parser.add_argument("--attempts", nargs="*", default=[])
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    smoke = read_json(args.smoke_audit)
    require(smoke["status"] == "passed", "Smoke audit is not marked passed")
    smoke_id = smoke["runs"]["bottle"]["job"]["JOB_ID"]
    main_runs = [read_json(path / "result.json") for path in args.main]
    main_ids = {run["category"]: run["job"]["JOB_ID"] for run in main_runs}
    by_job, rows = accounting(sorted(set([smoke_id, *main_ids.values(), *args.attempts])))
    completed = [by_job[job_id] for job_id in [smoke_id, *main_ids.values()]]
    require(all(row["state"] == "COMPLETED" and row["exit_code"] == "0:0" for row in completed),
            "Slurm has not confirmed all audited jobs complete")
    tests = args.tests.read_text(encoding="utf-8")
    require(tests.strip().endswith("OK"), "Tests did not complete successfully")
    metadata = {
        "smoke_job": by_job[smoke_id], "smoke_audit": smoke,
        "main_jobs": {category: by_job[job_id] for category, job_id in main_ids.items()},
        "other_accounting_rows": [row for row in rows if row["job_id"] not in {j["job_id"] for j in completed}],
        "unit_tests": "passed",
    }
    write_json(args.out, metadata)
    lines = [
        "# Run status", "", "## Bottle smoke trial", "",
        f"Job `{smoke_id}` ran on `{by_job[smoke_id]['node']}` in `{by_job[smoke_id]['partition']}` "
        f"and completed with exit `0:0` in {by_job[smoke_id]['elapsed']}. "
        f"The audit confirmed {smoke['counts']['metrics']} configurations, {smoke['counts']['test_images']} "
        f"test images, {smoke['counts']['scores']} score rows, and {smoke['counts']['references']} reference rows.",
        "",
        "## Main experiment", "",
        "| Category | Slurm job | Partition | Node | State / exit | Elapsed | Configurations |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for run in main_runs:
        cat, job = run["category"], by_job[run["job"]["JOB_ID"]]
        lines.append(f"| {cat.capitalize()} | {job['slurm_reference']} (raw {job['job_id']}) | {job['partition']} | "
                     f"{job['node']} | {job['state']} / {job['exit_code']} | {job['elapsed']} | {run['configs']} |")
    if metadata["other_accounting_rows"]:
        lines += ["", "Additional scheduler accounting is included in `manifest.json`."]
    lines += ["", "## Test and audit evidence", "",
              "The unit test log records passing selection, prefix, cosine and threshold checks. The full audit "
              "recomputed per-configuration metrics and checked the split, feature shape, normalized tokens, reference "
              "order, calibration maps, thresholds, predictions, and case-map scores. Logs are in `logs/`.",
              "", "## Failures and recovery", "",
              args.notes.read_text(encoding="utf-8").strip() if args.notes else "See the scheduler accounting and job logs.",
              "", "## Follow-up", "", "The main experiment and handoff are complete.", ""]
    args.status.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
