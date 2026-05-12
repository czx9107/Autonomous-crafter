import argparse
import datetime as _datetime
import json
import pathlib
import re
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "01-basic_reward.py"
EXPERIMENT_ROOT = ROOT / "saves" / "auto_experiments"
EXPERIMENT_LOG = ROOT / "docx" / "progress" / "01-basic_reward_experiment_log.md"


EXPERIMENTS = {
    "E00_baseline": {
        "description": "Record 26 oldmap baseline; no parameter changes.",
        "changes": [],
    },
    "E01_entropy": {
        "description": "Increase entropy coefficient to reduce early action collapse.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
        ],
    },
    "E02_conservative_clip": {
        "description": "Lower PPO clip range to reduce overly abrupt policy drift.",
        "changes": [
            {
                "name": "PPO_CLIP_RANGE",
                "from": "PPO_CLIP_RANGE = 0.2",
                "to": "PPO_CLIP_RANGE = 0.15",
            },
        ],
    },
    "E03_entropy_plus_clip": {
        "description": "Combine entropy pressure and conservative clip range.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
            {
                "name": "PPO_CLIP_RANGE",
                "from": "PPO_CLIP_RANGE = 0.2",
                "to": "PPO_CLIP_RANGE = 0.15",
            },
        ],
    },
    "E04_entropy_low_alive": {
        "description": "Keep entropy pressure, but reduce dense alive bonus to weaken passive survival bias.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
            {
                "name": "alive_bonus",
                "from": "    alive_bonus: float = 0.04",
                "to": "    alive_bonus: float = 0.02",
            },
        ],
    },
    "E05_entropy_food_weight": {
        "description": "Keep entropy pressure and increase food status potential weight to address starvation dominance.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
            {
                "name": "food_delta",
                "from": "    food_delta: float = 1.0 * rate",
                "to": "    food_delta: float = 1.5 * rate",
            },
        ],
    },
    "E06_entropy_long_rollout": {
        "description": "Keep entropy pressure and double PPO rollout length to improve long-horizon advantage estimates.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
        ],
        "arg_overrides": {
            "n_steps": 1024,
        },
    },
    "E07_entropy_gamma_0999": {
        "description": "Keep entropy pressure and increase PPO/potential gamma for longer survival credit assignment.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
            {
                "name": "PPO_GAMMA",
                "from": "PPO_GAMMA = 0.997",
                "to": "PPO_GAMMA = 0.999",
            },
        ],
    },
    "E08_entropy_history8": {
        "description": "Keep entropy pressure and expand recent observation history from 4 to 8 frames.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
            {
                "name": "OBS_HISTORY_LENGTH",
                "from": "OBS_HISTORY_LENGTH = 4",
                "to": "OBS_HISTORY_LENGTH = 8",
            },
        ],
    },
    "E09_entropy_memory48": {
        "description": "Keep entropy pressure and enlarge old-map local memory window from 32 to 48.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
            {
                "name": "MEMORY_MAP_SIZE",
                "from": "MEMORY_MAP_SIZE = 32",
                "to": "MEMORY_MAP_SIZE = 48",
            },
        ],
    },
    "E10_entropy_new_map25": {
        "description": "Keep entropy pressure and switch to the token memory-map observation branch.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
        ],
        "arg_overrides": {
            "new_map": True,
        },
    },
    "E11_long_rollout_gamma_1m": {
        "description": "Use E06 as baseline, combine long rollout with gamma 0.999, and train for 1M steps.",
        "changes": [
            {
                "name": "PPO_ENT_COEF",
                "from": "PPO_ENT_COEF = 0.00",
                "to": "PPO_ENT_COEF = 0.01",
            },
            {
                "name": "PPO_GAMMA",
                "from": "PPO_GAMMA = 0.997",
                "to": "PPO_GAMMA = 0.999",
            },
        ],
        "arg_overrides": {
            "n_steps": 1024,
        },
    },
    "E12_record9_lr_decay": {
        "description": "Strictly return to Record 9 PPO/reward settings and start linear learning-rate decay after 500k of a 1M run.",
        "changes": [
            {
                "name": "PPO_ENABLE_LR_DECAY",
                "from": "PPO_ENABLE_LR_DECAY = False",
                "to": "PPO_ENABLE_LR_DECAY = True",
            },
        ],
        "arg_overrides": {
            "n_steps": 512,
            "batch_size": 1024,
            "learning_rate": 0.00015,
        },
    },
    "E13_food_low_penalty_015": {
        "description": "Keep the current Record 9-style architecture/reward baseline and only raise food low-status pressure.",
        "changes": [
            {
                "name": "food_low_status_penalty",
                "from": "    food_low_status_penalty: float = 0.1",
                "to": "    food_low_status_penalty: float = 0.15",
            },
        ],
    },
}


SUCCESS_RE = re.compile(
    r"\[success\]\s+episodes=(?P<episodes>\d+)\s+timesteps=(?P<timesteps>\d+)\s+"
    r"recent_success_rate=(?P<recent>[-+0-9.eE]+)\s+"
    r"total_success_rate=(?P<total>[-+0-9.eE]+)"
)
AVG_EATS_RE = re.compile(
    r"recent_avg_eats=(?P<recent>[-+0-9.eE]+)\s+total_avg_eats=(?P<total>[-+0-9.eE]+)"
)
AVG_DRINKS_RE = re.compile(
    r"recent_avg_drinks=(?P<recent>[-+0-9.eE]+)\s+total_avg_drinks=(?P<total>[-+0-9.eE]+)"
)
AVG_VISITED_RE = re.compile(
    r"recent_avg_visited_area=(?P<recent>[-+0-9.eE]+)\s+"
    r"total_avg_visited_area=(?P<total>[-+0-9.eE]+)"
)
AVG_MOVES_RE = re.compile(
    r"recent_avg_move_actions=(?P<recent>[-+0-9.eE]+)\s+"
    r"total_avg_move_actions=(?P<total>[-+0-9.eE]+)"
)
DEATH_RE = re.compile(
    r"recent_deaths=(?P<deaths>\d+)\s+recent_death_ratios="
    r"starvation:(?P<starvation>[-+0-9.eE]+),"
    r"dehydration:(?P<dehydration>[-+0-9.eE]+),"
    r"monster_attack:(?P<monster>[-+0-9.eE]+)"
)
SB3_ROW_RE = re.compile(r"\|\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*\|\s*(?P<value>[-+0-9.eE]+)\s*\|")
REWARD_COMPONENT_RE = re.compile(
    r"terminal=(?P<terminal_avg>[-+0-9.eE]+)/(?P<terminal_share>[-+0-9.eE]+),\s+"
    r"low_status=(?P<low_avg>[-+0-9.eE]+)/(?P<low_share>[-+0-9.eE]+),\s+"
    r"alive=(?P<alive_avg>[-+0-9.eE]+)/(?P<alive_share>[-+0-9.eE]+),\s+"
    r"potential=(?P<potential_avg>[-+0-9.eE]+)/(?P<potential_share>[-+0-9.eE]+)"
)


def parse_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def now_stamp():
    return _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class MetricsParser:
    def __init__(self):
        self.reports = []
        self.last_report = None
        self.best_recent_success = 0.0
        self.best_report = None
        self.sb3_latest = {}
        self._expect_move_ratios = None
        self._expect_reward_components = None
        self._expect_supply_rows = None

    def parse_line(self, line):
        success_match = SUCCESS_RE.search(line)
        if success_match:
            report = {
                "episodes": int(success_match.group("episodes")),
                "timesteps": int(success_match.group("timesteps")),
                "recent_success_rate": float(success_match.group("recent")),
                "total_success_rate": float(success_match.group("total")),
            }
            self.reports.append(report)
            self.last_report = report
            if report["recent_success_rate"] >= self.best_recent_success:
                self.best_recent_success = report["recent_success_rate"]
                self.best_report = report
            return

        if self.last_report is not None:
            self._parse_episode_stat_line(line, self.last_report)
        self._parse_sb3_line(line)

    def _parse_episode_stat_line(self, line, report):
        match = AVG_EATS_RE.search(line)
        if match:
            report["recent_avg_eats"] = float(match.group("recent"))
            report["total_avg_eats"] = float(match.group("total"))
            return
        match = AVG_DRINKS_RE.search(line)
        if match:
            report["recent_avg_drinks"] = float(match.group("recent"))
            report["total_avg_drinks"] = float(match.group("total"))
            return
        match = AVG_VISITED_RE.search(line)
        if match:
            report["recent_avg_visited_area"] = float(match.group("recent"))
            report["total_avg_visited_area"] = float(match.group("total"))
            return
        match = AVG_MOVES_RE.search(line)
        if match:
            report["recent_avg_move_actions"] = float(match.group("recent"))
            report["total_avg_move_actions"] = float(match.group("total"))
            return
        if "recent_move_ratios left/right/up/down" in line:
            self._expect_move_ratios = "recent"
            return
        if "total_move_ratios left/right/up/down" in line:
            self._expect_move_ratios = "total"
            return
        if self._expect_move_ratios and "/" in line:
            values = [parse_float(part.strip()) for part in line.strip().split("/")]
            if len(values) == 4 and all(value is not None for value in values):
                report[f"{self._expect_move_ratios}_move_ratios"] = {
                    "left": values[0],
                    "right": values[1],
                    "up": values[2],
                    "down": values[3],
                }
            self._expect_move_ratios = None
            return
        if "recent_reward_components avg_per_episode/abs_share" in line:
            self._expect_reward_components = "recent"
            return
        if "total_reward_components avg_per_episode/abs_share" in line:
            self._expect_reward_components = "total"
            return
        if self._expect_reward_components:
            match = REWARD_COMPONENT_RE.search(line)
            if match:
                report[f"{self._expect_reward_components}_reward_components"] = {
                    "terminal": {
                        "avg": float(match.group("terminal_avg")),
                        "abs_share": float(match.group("terminal_share")),
                    },
                    "low_status": {
                        "avg": float(match.group("low_avg")),
                        "abs_share": float(match.group("low_share")),
                    },
                    "alive": {
                        "avg": float(match.group("alive_avg")),
                        "abs_share": float(match.group("alive_share")),
                    },
                    "potential": {
                        "avg": float(match.group("potential_avg")),
                        "abs_share": float(match.group("potential_share")),
                    },
                }
            self._expect_reward_components = None
            return
        if "recent_supply_table" in line:
            self._expect_supply_rows = ("recent", [])
            return
        if "total_supply_table" in line:
            self._expect_supply_rows = ("total", [])
            return
        if self._expect_supply_rows:
            values = [parse_float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", line)]
            key, rows = self._expect_supply_rows
            if len(values) == 2:
                rows.append(values)
                if len(rows) == 2:
                    report[f"{key}_supply_table"] = {
                        "no_eat_no_drink": rows[0][0],
                        "no_eat_drink": rows[0][1],
                        "eat_no_drink": rows[1][0],
                        "eat_drink": rows[1][1],
                    }
                    self._expect_supply_rows = None
            return
        match = DEATH_RE.search(line)
        if match:
            report["recent_deaths"] = int(match.group("deaths"))
            report["recent_death_ratios"] = {
                "starvation": float(match.group("starvation")),
                "dehydration": float(match.group("dehydration")),
                "monster_attack": float(match.group("monster")),
            }

    def _parse_sb3_line(self, line):
        match = SB3_ROW_RE.search(line)
        if not match:
            return
        key = match.group("key")
        value = parse_float(match.group("value"))
        if value is not None:
            self.sb3_latest[key] = value


class EarlyStopper:
    def __init__(self, args):
        self.args = args
        self.best_recent_success = 0.0
        self.no_improvement_reports = 0
        self.direction_collapse_reports = 0

    def check(self, report):
        timesteps = int(report.get("timesteps", 0))
        recent_success = float(report.get("recent_success_rate", 0.0))
        if recent_success >= self.args.target_success_rate and timesteps <= self.args.target_timesteps:
            return "target_success_rate_reached"

        if timesteps < self.args.min_timesteps_before_early_stop:
            return None

        if recent_success >= self.best_recent_success + self.args.min_success_improvement:
            self.best_recent_success = recent_success
            self.no_improvement_reports = 0
        else:
            self.no_improvement_reports += 1

        ratios = report.get("recent_move_ratios") or {}
        max_direction = max(ratios.values()) if ratios else 0.0
        visited_area = float(report.get("recent_avg_visited_area", 0.0))
        if (
            max_direction > self.args.direction_collapse_threshold
            and visited_area < self.args.min_visited_area_for_direction_check
        ):
            self.direction_collapse_reports += 1
        else:
            self.direction_collapse_reports = 0

        if self.direction_collapse_reports >= self.args.direction_collapse_patience:
            return "direction_collapse"
        if self.no_improvement_reports >= self.args.patience_reports:
            return "no_recent_success_improvement"
        return None


def apply_changes(script_path, changes):
    if not changes:
        return []
    text = script_path.read_text(encoding="utf-8")
    applied = []
    for change in changes:
        old = change["from"]
        new = change["to"]
        if old not in text:
            raise RuntimeError(
                f"Cannot apply {change['name']}: expected source line not found: {old}"
            )
        text = text.replace(old, new, 1)
        applied.append(change)
    script_path.write_text(text, encoding="utf-8")
    return applied


def restore_source_if_needed(script_path, original_text, applied_changes, keep_patch):
    if applied_changes and not keep_patch:
        script_path.write_text(original_text, encoding="utf-8")


def run_command(command, cwd, log_path=None, parser=None, stopper=None):
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    stop_reason = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            if log_file:
                log_file.write(line)
                log_file.flush()
            if parser is not None:
                parser.parse_line(line)
                report_complete = "total_deaths=" in line and parser.reports
                if stopper is not None and report_complete:
                    stop_reason = stopper.check(parser.reports[-1])
                    if stop_reason:
                        print(f"[auto_stop] reason={stop_reason}")
                        if log_file:
                            log_file.write(f"[auto_stop] reason={stop_reason}\n")
                        proc.terminate()
                        break
        try:
            return_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            return_code = proc.wait()
            stop_reason = stop_reason or "forced_kill_after_timeout"
    finally:
        if log_file:
            log_file.close()
    return return_code, stop_reason


def append_experiment_log_start(run_id, experiment_id, experiment, args, command):
    if args.no_log_write:
        return
    lines = [
        "",
        f"## 自动实验 {run_id}：{experiment_id}",
        "",
        f"时间：{_datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### 启动配置",
        "",
        f"- 基线：记录 26 oldmap 当前最优基线",
        f"- 实验说明：{experiment['description']}",
        f"- total_timesteps: `{args.total_timesteps}`",
        f"- target_success_rate: `{args.target_success_rate}`",
        f"- target_timesteps: `{args.target_timesteps}`",
        f"- dry_run: `{args.dry_run}`",
        "",
        "修改变量：",
        "",
    ]
    if experiment["changes"]:
        for change in experiment["changes"]:
            lines.append(f"- `{change['name']}`: `{change['from']}` -> `{change['to']}`")
    else:
        lines.append("- 无")
    arg_overrides = experiment.get("arg_overrides") or {}
    lines.extend(["", "训练参数覆盖：", ""])
    if arg_overrides:
        for name, value in arg_overrides.items():
            lines.append(f"- `{name}`: `{getattr(args, name)}`")
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "训练命令：",
            "",
            "```text",
            " ".join(command),
            "```",
            "",
        ]
    )
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))


def append_experiment_log_result(run_id, summary, args):
    if args.no_log_write:
        return
    last = summary.get("last_report") or {}
    best = summary.get("best_report") or {}
    lines = [
        "### 实验结果",
        "",
        f"- 退出原因：`{summary.get('stop_reason')}`",
        f"- 进程返回码：`{summary.get('return_code')}`",
        f"- 是否达标：`{summary.get('target_reached')}`",
        f"- 最后 timesteps：`{last.get('timesteps')}`",
        f"- 最后 recent_success_rate：`{last.get('recent_success_rate')}`",
        f"- 最佳 recent_success_rate：`{best.get('recent_success_rate')}`",
        f"- 最后 visited_area：`{last.get('recent_avg_visited_area')}`",
        f"- 最后 move_ratios：`{last.get('recent_move_ratios')}`",
        f"- 最后 avg_eats/drinks：`{last.get('recent_avg_eats')}` / `{last.get('recent_avg_drinks')}`",
        f"- summary：`{summary.get('summary_path')}`",
        "",
    ]
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))


def make_train_command(args, output_dir):
    command = [
        sys.executable,
        "-B",
        str(TRAIN_SCRIPT),
        "--total-timesteps",
        str(args.total_timesteps),
        "--episode-steps",
        str(args.episode_steps),
        "--n-envs",
        str(args.n_envs),
        "--vec-env",
        args.vec_env,
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--log-interval",
        str(args.log_interval),
        "--render-every",
        str(args.render_every),
        "--report-every",
        str(args.report_every),
        "--output-dir",
        str(output_dir / "model"),
    ]
    if args.new_map:
        command.append("--new-map")
    return command


def make_effective_args(args, experiment):
    effective_args = argparse.Namespace(**vars(args))
    for name, value in (experiment.get("arg_overrides") or {}).items():
        setattr(effective_args, name, value)
    return effective_args


def run_smoke(args, run_dir):
    command = [sys.executable, "-B", str(TRAIN_SCRIPT), "--smoke-test"]
    if args.new_map:
        command.append("--new-map")
    return run_command(command, ROOT.parent, run_dir / "smoke_stdout.log")


def run_experiment(args, experiment_id):
    if experiment_id not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment: {experiment_id}")
    experiment = EXPERIMENTS[experiment_id]
    effective_args = make_effective_args(args, experiment)
    run_id = f"{now_stamp()}_{experiment_id}"
    run_dir = EXPERIMENT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "description": experiment["description"],
        "changes": experiment["changes"],
        "arg_overrides": experiment.get("arg_overrides", {}),
        "args": vars(effective_args),
    }
    (run_dir / "params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    original_source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    applied_changes = []
    if not args.no_patch:
        applied_changes = apply_changes(TRAIN_SCRIPT, experiment["changes"])

    train_command = make_train_command(effective_args, run_dir)
    append_experiment_log_start(run_id, experiment_id, experiment, effective_args, train_command)

    if effective_args.dry_run:
        summary = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "dry_run": True,
            "applied_changes": applied_changes,
            "arg_overrides": experiment.get("arg_overrides", {}),
            "train_command": train_command,
            "stop_reason": "dry_run",
            "return_code": 0,
            "target_reached": False,
        }
        summary_path = run_dir / "train_summary.json"
        summary["summary_path"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        append_experiment_log_result(run_id, summary, effective_args)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        restore_source_if_needed(TRAIN_SCRIPT, original_source, applied_changes, effective_args.keep_patch)
        return summary

    smoke_code, smoke_stop = run_smoke(effective_args, run_dir)
    if smoke_code != 0:
        summary = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "dry_run": False,
            "applied_changes": applied_changes,
            "arg_overrides": experiment.get("arg_overrides", {}),
            "stop_reason": smoke_stop or "smoke_failed",
            "return_code": smoke_code,
            "target_reached": False,
        }
        summary_path = run_dir / "train_summary.json"
        summary["summary_path"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        append_experiment_log_result(run_id, summary, effective_args)
        restore_source_if_needed(TRAIN_SCRIPT, original_source, applied_changes, effective_args.keep_patch)
        return summary

    parser = MetricsParser()
    stopper = EarlyStopper(effective_args)
    start = time.time()
    return_code, stop_reason = run_command(
        train_command,
        ROOT.parent,
        run_dir / "train_stdout.log",
        parser=parser,
        stopper=stopper,
    )
    elapsed = time.time() - start
    last_report = parser.reports[-1] if parser.reports else None
    target_reached = bool(
        last_report
        and last_report.get("recent_success_rate", 0.0) >= args.target_success_rate
        and last_report.get("timesteps", 10**12) <= effective_args.target_timesteps
    )
    summary = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "dry_run": False,
        "applied_changes": applied_changes,
        "arg_overrides": experiment.get("arg_overrides", {}),
        "train_command": train_command,
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason or ("completed" if return_code == 0 else "process_failed"),
        "return_code": return_code,
        "target_reached": target_reached,
        "reports": parser.reports,
        "last_report": last_report,
        "best_report": parser.best_report,
        "sb3_latest": parser.sb3_latest,
    }
    summary_path = run_dir / "train_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    append_experiment_log_result(run_id, summary, effective_args)
    restore_source_if_needed(TRAIN_SCRIPT, original_source, applied_changes, effective_args.keep_patch)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="E00_baseline", choices=sorted(EXPERIMENTS))
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-patch", action="store_true")
    parser.add_argument("--keep-patch", action="store_true")
    parser.add_argument("--no-log-write", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=250000)
    parser.add_argument("--target-timesteps", type=int, default=250000)
    parser.add_argument("--target-success-rate", type=float, default=0.10)
    parser.add_argument("--episode-steps", type=int, default=512)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--render-every", type=int, default=64)
    parser.add_argument("--report-every", type=int, default=64)
    parser.add_argument("--new-map", action="store_true")
    parser.add_argument("--min-timesteps-before-early-stop", type=int, default=80000)
    parser.add_argument("--patience-reports", type=int, default=4)
    parser.add_argument("--min-success-improvement", type=float, default=0.01)
    parser.add_argument("--direction-collapse-threshold", type=float, default=0.70)
    parser.add_argument("--direction-collapse-patience", type=int, default=2)
    parser.add_argument("--min-visited-area-for-direction-check", type=float, default=12.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summaries = []
    for _ in range(args.max_runs):
        summaries.append(run_experiment(args, args.experiment))
        if summaries[-1].get("target_reached"):
            break
    print("[auto_summary]")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
