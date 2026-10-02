#!/usr/bin/env python3
"""Persistent nominal PPO-Lag search, with reward-gated long-horizon confirmation."""

import copy
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml
from report_experiments import report_results

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/silverstone_nominal_lambda_autotune"
SOURCE = ROOT / "outputs/silverstone_ppo_lag_nominal_rollout256_batch1024"
REWARD_FLOOR = 1500.0
BUDGET = 0.05
SCREEN_STEPS = 60000000
CONFIRM_STEPS = 200000000


def eligible(record):
    return record["reward"] >= REWARD_FLOOR and record["collision_rate"] <= BUDGET


def wilson_upper(rate, n, z=1.96):
    den = 1.0 + z*z/n
    return (rate + z*z/(2*n) + z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))) / den


def refinement_center(records):
    racing = [r for r in records if r["reward"] >= REWARD_FLOOR]
    if racing:
        return min(racing, key=lambda r: (max(0., r["collision_rate"]-BUDGET), -r["reward"]))
    # Before any candidate races meaningfully, optimize progress, not idle safety.
    return max(records, key=lambda r: (r["reward"], -r["collision_rate"]))


def search_parameters(candidate):
    """Continuation metadata must not leak into fresh 60M/200M trials."""
    return {key: value for key, value in candidate.items()
            if key not in {"resume_checkpoint", "resume_from_update", "std_start", "fixed_optimizer_lr",
                           "continuation_target_steps"}}


def continuation_job(record):
    if not (CONFIRM_STEPS <= record["steps"]
            and record["reward"] >= REWARD_FLOOR and record["collision_rate"] <= .1):
        return None
    candidate = dict(search_parameters(record["candidate"]),
                     resume_checkpoint=str(OUT / record["tuning_label"] / "checkpoint.msgpack"),
                     resume_from_update=int(record["steps"]) // (96*256),
                     std_start=record["candidate"]["std_end"], fixed_optimizer_lr=5e-5)
    target = (int(record["steps"]) // CONFIRM_STEPS + 1) * CONFIRM_STEPS
    return dict(gpu=2, candidate=candidate, label=record["tuning_label"]+f"_extend{target//1000000}m",
                steps=target, eval_seed=3000)


def stabilization_candidates(record):
    job = continuation_job(record)
    if job is None:
        raise ValueError("No near-feasible racing checkpoint for normalization stabilization")
    return [dict(job["candidate"], lr=1., normalize_cost_advantage=True,
                 cost_advantage_std_floor=floor, continuation_target_steps=job["steps"])
            for floor in (.001, .01, .03)]


def continuation_for_gpu(record, gpu):
    job = continuation_job(record)
    if job is None:
        return None
    if gpu not in (2, 3, 4):
        raise ValueError("Continuation is restricted to GPUs 2/3/4")
    job["gpu"] = gpu
    if gpu != 2:
        rate, suffix = (.1, "lr0p1") if gpu == 3 else (1., "lr1p0")
        job["candidate"]["lr"] = rate
        job["label"] += "_"+suffix
    return job


def training_pids(config_path):
    """Match exact argv entries; a PID or stale status file alone is not live work."""
    matches = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / "cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if "train/jax_sampler_ppo.py" in argv and str(config_path) in argv:
            matches.append(int(directory.name))
    return matches


def training_environment(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                JAX_COMPILATION_CACHE_DIR=str(OUT / "jax_compilation_cache"),
                JAX_COMPILATION_CACHE_MAX_SIZE=str(2 * 1024**3))
    # A report run can leave a parent-owned W&B service socket in os.environ.
    # Each independent training/recovery process must own its own service.
    env.pop("WANDB_SERVICE", None)
    return env


def confirmation_jobs(generation, screened, records):
    # Warm continuations already include the full long-horizon confirmation.
    # If their success gates fail, advance the search instead of cold-starting
    # the old candidate and discarding the completed checkpoint comparison.
    if screened and all(r.get("steps", 0) >= CONFIRM_STEPS
                        and "resume_checkpoint" in r["candidate"] for r in screened):
        return []
    feasible = sorted([r for r in screened if eligible(r)], key=lambda r: -r["reward"])
    chosen = feasible
    if not chosen and generation % 3 == 2:
        chosen = [refinement_center(records)]
    jobs = [dict(gpu=2+i, candidate=search_parameters(r["candidate"]), label=f"g{generation:02d}_confirm{i}",
                 steps=CONFIRM_STEPS, eval_seed=2000) for i, r in enumerate(chosen)]
    if jobs and len(jobs) < 3:
        center = search_parameters(chosen[0]["candidate"])
        side_candidates = [dict(center, normalize_cost_advantage=not center.get("normalize_cost_advantage", True)),
                           dict(center, normalize_cost_advantage=False,
                                cost_discounting=(1. if center.get("cost_discounting", .99) != 1. else .999))]
        for i, candidate in enumerate(side_candidates[:3-len(jobs)]):
            jobs.append(dict(gpu=2+len(jobs), candidate=candidate, label=f"g{generation:02d}_side{i}",
                             steps=SCREEN_STEPS, eval_seed=1000))
    return jobs


def confirmed_success(record):
    return (record["steps"] >= CONFIRM_STEPS and eligible(record) and record["stable_last_two"]
            and record["collision_wilson_upper95"] <= BUDGET)


def finite_episode_candidates(center):
    center = search_parameters(center)
    return [dict(center, cost_discounting=1., normalize_cost_advantage=False,
                 cost_terminal_at_time_limit=True, lr=min(10., center["lr"]*factor))
            for factor in (1., 5., 10.)]


def credit_horizon_candidates(center):
    """Separate sparse cost credit from the raw-unit multiplier ceiling."""
    center = search_parameters(center)
    return [dict(center, cost_discounting=gamma, normalize_cost_advantage=False,
                 cost_terminal_at_time_limit=True, max_lambda=1000., lr=2.5,
                 warmup=30000000)
            for gamma in (.99, .999, 1.)]


def next_screen_job(generation, gpu, candidates):
    """Use a freed GPU for the planned timeout-credit experiment, not a duplicate."""
    if generation not in (2, 3):
        return None
    index = gpu - 2
    return dict(gpu=gpu, candidate=candidates[index], label=f"g{generation+1:02d}_screen{index}",
                steps=SCREEN_STEPS, eval_seed=1000)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controller_lock = (OUT / ".controller.lock").open("a")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base_path = OUT / "base_rl_config.yaml"
    base = yaml.safe_load((base_path if base_path.exists() else
                          SOURCE / "Silverstone_nominal/rl_config.yaml").read_text())
    gym_path = OUT / "gym_config.yaml"
    if not gym_path.exists():
        gym_path.write_text((SOURCE / "gym_config.yaml").read_text())
    if not base_path.exists():
        base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    records_path = OUT / "results_so_far.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    status_path = OUT / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(status="running", generation=0, completed_trials=0,
                  reward_floor=REWARD_FLOOR, collision_budget=BUDGET,
                  training_seed=1, gpus=[2, 3, 4], rollout_length=256, batch_size=1024,
                  domain_randomization=False, screen_steps=SCREEN_STEPS,
                  confirmation_steps=CONFIRM_STEPS)
    if status["status"] == "completed":
        print("Search already completed; retained winner.json", flush=True)
        return

    def save():
        (OUT / "status.json").write_text(json.dumps(status, indent=2))
        (OUT / "results_so_far.json").write_text(json.dumps(records, indent=2))

    save()

    def trial(gpu, candidate, label, steps, eval_seed=1000):
        directory = OUT / label
        directory.mkdir(exist_ok=True)
        result_path = directory / "result.json"
        if result_path.exists():
            cached = json.loads(result_path.read_text())
            if cached["candidate"] != candidate or cached["eval_seed"] != eval_seed:
                raise RuntimeError(f"Resume specification mismatch: {label}")
            return cached
        config = copy.deepcopy(base)
        config.update(seed=1, eval_seed=eval_seed, total_timesteps=steps,
                      rollout_length=256, batch_size=1024)
        config.pop("n_steps", None)
        jc = config["jax_sampler_ppo"]
        jc.update(sampler="uniform", domain_randomization=False, eval_domain_randomization=False,
                  num_envs=96, num_eval_envs=1024, eval_episodes_per_dynamics=10,
                  num_evals=6, eval_episode_steps=12288, reset_state_on_rollout=False,
                  use_ppo_lag=True, constraint_cost_type="collision", safety_bound=BUDGET,
                  lagrangian_update_mode="completed_episode", allow_partial_first_episode=False,
                  lagrangian_coef_rate=candidate["lr"], initial_lambda_lagr=candidate["initial"],
                  lagrangian_ema_decay=candidate["ema"], lagrangian_max=float(candidate.get("max_lambda", 100.)),
                  lagrangian_min_completed_episodes=candidate["episode_batch"],
                  lagrangian_warmup_steps=candidate["warmup"], eval_video=False,
                  eval_render=True, sampler_plot_samples=0)
        config["log_std_schedule"] = dict(init=-0.4, end=candidate["std_end"])
        if "resume_checkpoint" in candidate:
            config["resume_checkpoint"] = candidate["resume_checkpoint"]
            config["resume_from_update"] = candidate["resume_from_update"]
            config["log_std_schedule"]["init"] = candidate["std_start"]
            config["start_learning_rate"] = candidate["fixed_optimizer_lr"]
            config["end_learning_rate"] = candidate["fixed_optimizer_lr"]
            jc["num_evals"] = int(math.ceil(steps / 40000000)) + 1
        if "cost_discounting" in candidate:
            jc["cost_discounting"] = candidate["cost_discounting"]
        if "normalize_cost_advantage" in candidate:
            jc["normalize_cost_advantage"] = candidate["normalize_cost_advantage"]
        if "cost_terminal_at_time_limit" in candidate:
            jc["cost_terminal_at_time_limit"] = candidate["cost_terminal_at_time_limit"]
        if "cost_advantage_std_floor" in candidate:
            jc["cost_advantage_std_floor"] = candidate["cost_advantage_std_floor"]
        config_path = directory / "rl_config.yaml"
        if config_path.exists():
            if yaml.safe_load(config_path.read_text()) != config:
                raise RuntimeError(f"Resume configuration mismatch: {label}")
        else:
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        started = config_path.stat().st_mtime
        log_path = directory / "train.log"
        active = training_pids(config_path)
        if active:
            print(f"Adopting {label} existing PID={active}; training remains uninterrupted", flush=True)
            while training_pids(config_path):
                time.sleep(10)
            returncode = 0 if (directory / "checkpoint.msgpack").exists() else 1
        elif log_path.exists():
            # Harvest a finished orphan, but never silently overwrite failed run logs.
            returncode = 0 if (directory / "checkpoint.msgpack").exists() else 1
        else:
            print(f"Starting {label} GPU={gpu}: {candidate}", flush=True)
            with log_path.open("w") as log:
                run = subprocess.run([sys.executable, "-u", "train/jax_sampler_ppo.py",
                    "--rl-config", str(config_path), "--gym-config", str(gym_path),
                    "--track", "Silverstone", "--checkpoint-output", str(directory / "checkpoint.msgpack")], cwd=ROOT,
                    env=training_environment(gpu), stdout=log,
                    stderr=subprocess.STDOUT)
            returncode = run.returncode
        log_text = (directory / "train.log").read_text()
        ids = re.findall(r"View run at .*?/runs/([a-z0-9]+)", log_text)
        if returncode or not ids:
            raise RuntimeError(f"Failed {label}: exit={returncode}; inspect retained train.log")
        rid = ids[-1]
        summary_path = list((ROOT / "wandb").glob(f"run-*-{rid}/files/wandb-summary.json"))
        if len(summary_path) != 1:
            raise RuntimeError(f"Missing summary for {label}")
        s = json.loads(summary_path[0].read_text())
        expected_updates = math.ceil(steps / (96 * 256))
        original_run_id = rid
        logging_recovered = False
        if (s.get("training/progress/env_steps", 0) < expected_updates * 96 * 256
                or s.get("evaluation/meta/update") != expected_updates):
            if not (directory / "checkpoint.msgpack").exists():
                raise RuntimeError(f"Incomplete training or final evaluation: {label}")
            recovery_path = directory / "recovered_summary.json"
            if not recovery_path.exists():
                print(f"Recovering final evaluation from retained checkpoint: {label}", flush=True)
                with (directory / "recovery.log").open("w") as recovery_log:
                    recovery = subprocess.run([sys.executable, "-u", "recover_nominal_trial.py",
                        "--trial-dir", str(directory), "--gym-config", str(gym_path),
                        "--original-run-id", rid], cwd=ROOT, env=training_environment(gpu),
                        stdout=recovery_log, stderr=subprocess.STDOUT)
                if recovery.returncode:
                    raise RuntimeError(f"Checkpoint evaluation recovery failed: {label}")
            recovered = json.loads(recovery_path.read_text())
            s = recovered["metrics"]
            rid = recovered["run_id"]
            logging_recovered = True
        history = re.findall(r"eval_reward=([^ ]+) eval_cost=([^ ]+)", log_text)
        result = dict(tuning_label=label, candidate=candidate, track="Silverstone",
            formulation="nominal", seed=1, run_id=rid, reward=s["evaluation/reward/episode_return_mean"],
            original_run_id=original_run_id, logging_recovered=logging_recovered,
            cvar10=s["evaluation/reward/cvar10"], collision_rate=s["evaluation/collision/rate"],
            lambda_final=s["training/constraint/lambda_lagr"], budget=BUDGET, horizon=12288,
            steps=s["training/progress/env_steps"], rollout_length=256, ppo_batch_size=1024,
            reset_state_on_rollout=False, lagrangian_update_mode="completed_episode",
            domain_randomization=False, eval_domain_randomization=False,
            eval_seed=eval_seed, elapsed_seconds=time.time()-started,
            stable_last_two=len(history)>=2 and all(float(r)>=REWARD_FLOOR and float(c)<=BUDGET
                                                    for r,c in history[-2:]))
        result["collision_wilson_upper95"] = wilson_upper(result["collision_rate"], 10240)
        result["eligible"] = eligible(result)
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result

    generation = status["generation"]
    winner = None
    try:
        while True:
            status.update(generation=generation, status="screening")
            save()
            if generation == 0:
                candidates = [dict(lr=lr, initial=0., ema=.5, episode_batch=384,
                                   warmup=10000000, std_end=-.4) for lr in (.1, .5, 1.)]
            elif generation == 1:
                candidates = [dict(lr=lr, initial=0., ema=.5, episode_batch=384,
                                   warmup=10000000, std_end=-1.5) for lr in (.1, .5, 1.)]
            elif generation == 2:
                center = search_parameters(refinement_center(records)["candidate"])
                candidates = [dict(center, cost_discounting=gamma) for gamma in (.99, .999, 1.)]
            elif generation == 3:
                center = refinement_center(records)["candidate"]
                # Undiscounted finite-episode risk needs credit for safe timeouts.
                # Raw cost advantages have different multiplier units: sweep rates.
                candidates = finite_episode_candidates(center)
            elif generation == 4:
                center = refinement_center(records)["candidate"]
                candidates = credit_horizon_candidates(center)
            elif generation == 5:
                long_records = [r for r in records if r["steps"] >= CONFIRM_STEPS]
                candidates = stabilization_candidates(refinement_center(long_records))
            else:
                # Refine the best racing candidate; do not select idle safety as success.
                center = search_parameters(refinement_center(records)["candidate"])
                candidates = []
                for i, factor in enumerate((.5, 1., 2.)):
                    c = dict(center)
                    c["lr"] = min(10., max(.001, center["lr"]*factor))
                    c["episode_batch"] = (96, 384, 1536)[(generation+i)%3]
                    c["ema"] = (0., .5, .9)[(generation+2*i)%3]
                    c["warmup"] = (0, 10000000, 30000000)[(generation+i)%3]
                    c["std_end"] = (-.7, -1.2, -1.8)[(generation+i)%3]
                    candidates.append(c)
            plan_path = OUT / f"g{generation:02d}_plan.json"
            if plan_path.exists():
                candidates = json.loads(plan_path.read_text())
            else:
                plan_path.write_text(json.dumps(candidates, indent=2))
            screened = []
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(trial, gpu, c, f"g{generation:02d}_screen{i}",
                                       c.get("continuation_target_steps", SCREEN_STEPS),
                                       3000 if "resume_checkpoint" in c else 1000)
                           for i, (gpu, c) in enumerate(zip((2,3,4), candidates))]
                for future in as_completed(futures):
                    record = future.result()
                    screened.append(record)
                    if confirmed_success(record) and (winner is None or record["reward"] > winner["reward"]):
                        winner = record
                    if any(r["tuning_label"] == record["tuning_label"] for r in records):
                        continue
                    records.append(record)
                    status["completed_trials"] = len(records)
                    save()
                    report_results(records, OUT / f"report_trial{len(records):03d}",
                                   "Silverstone nominal PPO-Lag automatic tuning (reward gate >=1500)")
            if winner:
                break
            # Do not discard a promising learner solely because a 60M screen
            # is too short. Every third generation also studies the best at 200M;
            # the final success gates remain unchanged.
            confirmation_plan = OUT / f"g{generation:02d}_confirmation_plan.json"
            if confirmation_plan.exists():
                jobs = json.loads(confirmation_plan.read_text())
                if jobs and "label" not in jobs[0]:
                    # Preserve old persisted sequential confirmation specifications.
                    jobs = [dict(gpu=2+i, candidate=r["candidate"], label=f"g{generation:02d}_confirm{i}",
                                 steps=CONFIRM_STEPS, eval_seed=2000) for i, r in enumerate(jobs)]
            else:
                jobs = confirmation_jobs(generation, screened, records)
                confirmation_plan.write_text(json.dumps(jobs, indent=2))
            if jobs:
                status.update(status="confirming_and_tuning", active_confirmation_jobs=jobs)
                save()
                with ThreadPoolExecutor(max_workers=3) as pool:
                    pending = {pool.submit(trial, job["gpu"], job["candidate"], job["label"],
                                           job["steps"], job["eval_seed"]): job for job in jobs}
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            job = pending.pop(future)
                            confirmed = future.result()
                            is_new = not any(r["tuning_label"] == confirmed["tuning_label"] for r in records)
                            if is_new:
                                records.append(confirmed)
                            status["completed_trials"] = len(records)
                            save()
                            if is_new:
                                report_results(records, OUT / f"report_trial{len(records):03d}",
                                               "PPO-Lag long confirmation and parallel planned screening")
                            if confirmed_success(confirmed) and (winner is None or confirmed["reward"] > winner["reward"]):
                                winner = confirmed
                            job_generation = int(job["label"].split("_")[0][1:])
                            if (generation == 2 and job_generation in (2, 3) and winner is None
                                    and "resume_checkpoint" not in job["candidate"]):
                                # Persist before launching; resume adopts the same exact trials.
                                next_plan = OUT / f"g{job_generation+1:02d}_plan.json"
                                if next_plan.exists():
                                    planned = json.loads(next_plan.read_text())
                                else:
                                    factory = finite_episode_candidates if job_generation == 2 else credit_horizon_candidates
                                    planned = factory(refinement_center(records)["candidate"])
                                    next_plan.write_text(json.dumps(planned, indent=2))
                                probe = next_screen_job(job_generation, job["gpu"], planned)
                                if job_generation == 3 and job["gpu"] == 2:
                                    extension_plan = OUT / "continuation_plan.json"
                                    if extension_plan.exists():
                                        extension = json.loads(extension_plan.read_text())
                                    else:
                                        long_records = [r for r in records if r["steps"] >= CONFIRM_STEPS]
                                        extension = continuation_job(refinement_center(long_records)) if long_records else None
                                        if extension is not None:
                                            extension_plan.write_text(json.dumps(extension, indent=2))
                                    if extension is not None:
                                        probe = extension
                                status["active_confirmation_jobs"].append(probe)
                                save()
                                pending[pool.submit(trial, probe["gpu"], probe["candidate"], probe["label"],
                                                    probe["steps"], probe["eval_seed"])] = probe
                            elif (generation == 2 and job_generation == 4 and job["gpu"] in (3, 4)
                                  and winner is None and "resume_checkpoint" not in job["candidate"]):
                                extension_plan = OUT / f"continuation_gpu{job['gpu']}_plan.json"
                                if extension_plan.exists():
                                    extension = json.loads(extension_plan.read_text())
                                else:
                                    long_records = [r for r in records if r["steps"] >= CONFIRM_STEPS]
                                    extension = continuation_for_gpu(refinement_center(long_records), job["gpu"]) if long_records else None
                                    if extension is not None:
                                        extension_plan.write_text(json.dumps(extension, indent=2))
                                if extension is not None:
                                    status["active_confirmation_jobs"].append(extension)
                                    save()
                                    pending[pool.submit(trial, extension["gpu"], extension["candidate"], extension["label"],
                                                        extension["steps"], extension["eval_seed"])] = extension
            if winner:
                break
            # Gen 3 and the remaining cold gen 4 control are superseded by the
            # completed pipelined screens and the near-feasible checkpoint.
            generation = 5 if generation == 2 else generation + 1
        (OUT / "winner.json").write_text(json.dumps(winner, indent=2))
        (OUT / "final.json").write_text(json.dumps(records, indent=2))
        status.update(status="completed", winner=winner)
        save()
        report_results(records, OUT / "final_report", "PPO-Lag tuning target achieved and confirmed")
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
