"""Persist experiment comparisons and publish completion reports to W&B."""

import argparse
import csv
import json
import statistics
from pathlib import Path

ENTITY = "sjun0803-seoul-national-university"
PROJECT = "f1tenth-ppo-drift"


def report_results(results, output_dir, title, *, publish=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["track", "formulation", "scale", "seed", "return", "cvar10",
               "collision_rate", "budget", "budget_satisfied", "lambda", "run_url", "tuning_label",
               "tuning_parameters", "joint_goal_satisfied", "steps"]
    rows = []
    for r in results:
        collision = r.get("collision_rate", r.get("collision"))
        budget = r.get("budget", 0.30)
        run_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{r['run_id']}"
        rows.append([r["track"], r.get("formulation", r.get("sampler", "unknown")),
                     r.get("cost_score_scale", 1.0), r.get("seed", 1), r["reward"], r["cvar10"],
                     collision, budget, collision <= budget, r["lambda_final"], run_url, r.get("tuning_label"),
                     json.dumps(r["candidate"], sort_keys=True) if "candidate" in r else None,
                     r.get("eligible"), r.get("steps")])
    report_path = output_dir / "report.txt"
    lines = [title, "", "Track | Formulation | Scale | Seed | Return | CVaR10 | Collision | Budget | Satisfied"]
    for row in rows:
        track, formulation, scale, seed, reward, cvar, collision, budget, satisfied, _, _ = row[:11]
        lines.append(f"{track} | {formulation} | {scale:g} | {seed} | {reward:.1f} | {cvar:.1f} | {collision:.2%} | {budget:.2%} | {satisfied} | {row[11]}")
        if row[12] is not None:
            lines.append(f"  tuning_parameters={row[12]} | joint_goal_satisfied={row[13]} | steps={row[14]}")
    groups = {}
    for r in results:
        formulation = r.get("formulation", r.get("sampler", "unknown"))
        key = (r["track"], r.get("formulation", r.get("sampler", "unknown")),
               r.get("cost_score_scale", 1.0), r.get("budget", 0.30), r.get("horizon"),
               r.get("cost_dual_ema_decay", 0.9) if formulation in {"cost", "reward_cost"} else None,
               r.get("cost_dual_lr", 0.1) if formulation in {"cost", "reward_cost"} else None,
               r.get("cost_dual_update", "linear") if formulation in {"cost", "reward_cost"} else None,
               r.get("cost_dual_direction", "budget_reducing") if formulation in {"cost", "reward_cost"} else None,
               r.get("rollout_length", 6144), r.get("reset_state_on_rollout", True),
               r.get("lagrangian_update_mode", "first_episode"), r.get("ppo_batch_size", 8192), r.get("tuning_label"))
        groups.setdefault(key, []).append(r)
    aggregate_columns = ["track", "formulation", "scale", "budget", "horizon", "cost_ema_decay", "cost_dual_lr", "cost_dual_update", "cost_dual_direction", "rollout_length", "reset_state_on_rollout", "lagrangian_update_mode", "ppo_batch_size", "tuning_label", "n_seeds",
                         "return_mean", "return_std", "collision_mean", "collision_std",
                         "cvar10_mean", "budget_satisfied_seed_count"]
    aggregate_rows = []
    for key, records in sorted(groups.items(), key=lambda item: str(item[0])):
        seeds = [r.get("seed", 1) for r in records]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Repeated training seeds in comparison group {key}: {seeds}")
        rewards = [r["reward"] for r in records]
        collisions = [r.get("collision_rate", r.get("collision")) for r in records]
        aggregate_rows.append([*key, len(seeds), statistics.mean(rewards),
            statistics.stdev(rewards) if len(seeds) > 1 else None,
            statistics.mean(collisions), statistics.stdev(collisions) if len(seeds) > 1 else None,
            statistics.mean(r["cvar10"] for r in records), sum(c <= key[3] for c in collisions)])
    lines.extend(["", "Across training seeds (sample standard deviation):"])
    for row in aggregate_rows:
        track, formulation, scale, budget, horizon, cost_ema, cost_lr, cost_update, cost_direction, rollout_length, state_reset, lag_mode, ppo_batch_size, tuning_label, n, mean_reward, std_reward, mean_cost, std_cost, _, passed = row
        reward_spread = f" +/- {std_reward:.1f}" if std_reward is not None else " (std unavailable)"
        cost_spread = f" +/- {std_cost:.2%}" if std_cost is not None else " (std unavailable)"
        lines.append(f"{track} | {formulation} | scale={scale:g} | cost_ema={cost_ema} | cost_lr={cost_lr} | update={cost_update} | direction={cost_direction} | rollout={rollout_length} | batch={ppo_batch_size} | state_reset={state_reset} | lag_mode={lag_mode} | label={tuning_label} | seeds={n} | return={mean_reward:.1f}{reward_spread} | collision={mean_cost:.2%}{cost_spread} | passed={passed}/{n}")
    lines.extend(["", "Do not infer statistical significance from small seed counts.",
                  "Constraint satisfaction is evaluated from mean episode collision cost."])
    report_path.write_text("\n".join(lines) + "\n")
    with (output_dir / "report.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    with (output_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(aggregate_columns)
        writer.writerows(aggregate_rows)
    if not publish:
        return None
    try:
        import wandb
        run = wandb.init(entity=ENTITY, project=PROJECT, job_type="report",
                         name=output_dir.name + "_report", tags=["experiment-report"],
                         settings={"disable_git": True, "disable_code": True})
        run.log({"report/comparison": wandb.Table(columns=columns, data=rows),
                 "report/seed_aggregate": wandb.Table(columns=aggregate_columns, data=aggregate_rows),
                 "report/run_count": len(rows),
                 "report/budget_satisfied_count": sum(row[8] for row in rows)})
        artifact = wandb.Artifact(output_dir.name + "-report", type="experiment-report")
        artifact.add_file(str(report_path))
        artifact.add_file(str(output_dir / "report.csv"))
        artifact.add_file(str(output_dir / "aggregate.csv"))
        run.log_artifact(artifact)
        summary_start = lines.index("Across training seeds (sample standard deviation):")
        alert_summary = "\n".join([title, f"Completed runs: {len(rows)}", *lines[summary_start:]])
        alert_text = alert_summary[:3700] + f"\nFull comparison: {run.url}"
        run.alert(title="Experiment comparison completed", text=alert_text, level="INFO", wait_duration=0)
        url = run.url
        (output_dir / "report_publication.json").write_text(json.dumps(
            {"report_url": url, "alert_submitted": True}, indent=2))
        run.finish()
        print(f"Completion report: {url}", flush=True)
        return url
    except Exception as exc:
        print(f"Local report saved; W&B report failed: {exc}", flush=True)
        (output_dir / "report_publication.json").write_text(json.dumps(
            {"alert_submitted": False, "error": str(exc)}, indent=2))
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="Experiment comparison")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    results = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        results.extend(data if isinstance(data, list) else data["results"])
    report_results(results, args.output_dir, args.title, publish=not args.no_publish)


if __name__ == "__main__":
    main()
