"""Optional SwanLab reporting shared by the trainer and historical log import."""

from collections import deque
import logging
import math
from pathlib import Path


_LOGGER = logging.getLogger(__name__)
_DECISION_FIELDS = ("n", "accuracy", "nll", "brier", "ece", "score_mae", "score_rps")
_BOX_FIELDS = ("evaluated", "single_n", "acc_iou50", "miou", "presence_accuracy", "set_f1_50")


def _scalars(prefix, values, fields):
    return {f"{prefix}/{key}": value for key in fields
            if isinstance((value := values.get(key)), (int, float)) and math.isfinite(value)}


def development_metrics(report):
    decision = report.get("decisions", {})
    metrics = _scalars("dev/decision", decision.get("overall", {}), _DECISION_FIELDS)
    for section, name in (("by_modality", "modality"), ("by_type", "type")):
        for label, scores in decision.get(section, {}).items():
            metrics.update(_scalars(f"dev/{name}/{label}", scores, _DECISION_FIELDS))
    metrics.update(_scalars("dev/box", report, _BOX_FIELDS))
    modalities = [value for key, value in metrics.items()
                  if key.startswith("dev/modality/") and key.endswith("/accuracy")]
    if modalities:
        macro = sum(modalities) / len(modalities)
        metrics["dev/decision/macro_accuracy"] = macro
        box = metrics.get("dev/box/acc_iou50")
        if box is not None and report.get("single_n"):
            metrics["dev/joint/macro_harmonic_accuracy"] = (
                2 * macro * box / (macro + box) if macro + box else 0.
            )
    for kind in ("decision", "grounding"):
        passed = report.get(f"{kind}_retention_passed")
        if isinstance(passed, bool):
            metrics[f"dev/joint/{kind}_retention_passed"] = int(passed)
    return metrics


def progress_metrics(step, loss, elapsed, ranks, history, total_steps, batch_size):
    metrics = {"train/loss": loss, "train/optimizer_step": step,
               "train/elapsed_hours": elapsed / 3600, "train/ranks": ranks}
    if total_steps:
        metrics["train/progress_pct"] = 100 * step / total_steps
    history.append((step, elapsed))
    while len(history) > 1 and step - history[1][0] >= 500:
        history.popleft()
    first_step, first_elapsed = history[0]
    if step > first_step and elapsed > first_elapsed:
        seconds = (elapsed - first_elapsed) / (step - first_step)
        metrics["train/seconds_per_step_500"] = seconds
        if batch_size:
            metrics["train/examples_per_second"] = batch_size / seconds
        if total_steps:
            metrics["train/estimated_hours_left"] = max(0, total_steps - step) * seconds / 3600
    return metrics


def init_dashboard(swanlab, *, mode, logdir, project, name, group, config, on_online_error="raise"):
    settings = swanlab.Settings(interactive=False, probe={"hardware": False, "runtime": False,
                                                          "requirements": False, "git": False, "monitor": False})
    config = {key: value for key, value in config.items() if key != "video_cache_dir"}
    return swanlab.init(mode=mode, log_dir=str(logdir), project=project, public=False, name=name, group=group,
                        config=config, settings=settings, on_online_error=on_online_error)


class SwanLabReporter:
    """Record scalars from rank zero and isolate SDK failures from training."""

    def __init__(self, output, *, mode, project, name=None):
        self.output = Path(output).resolve()
        self.mode = mode
        self.project = project
        self.group = self.output.parent.name
        self.name = name or f"{self.group}-{self.output.name}"
        self.history = deque()
        self.run = None
        self.failed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            state = "aborted" if exc_type is KeyboardInterrupt else "crashed" if exc_type or self.failed else "success"
            try:
                self.run.finish(state=state)
            except Exception as error:
                self._failure("finish", error)

    def _failure(self, action, error):
        if not self.failed:
            _LOGGER.warning("SwanLab %s failed (%s). Training uses the standard log.", action, type(error).__name__)
        self.failed = True

    def _log(self, metrics, step):
        if self.run is not None and not self.failed:
            try:
                self.run.log(metrics, step=step)
            except Exception as error:
                self._failure("logging", error)

    def start(self, config, baseline):
        starting_step = config["starting_step"]
        self.total_steps = config["planned_steps"]
        self.ranks = config["world_size"]
        self.batch_size = config["global_batch_size"]
        # Training can roll back to a step already shown in the previous experiment.
        name = f"{self.name}-resume-{starting_step}" if starting_step else self.name
        try:
            import swanlab

            self.run = init_dashboard(swanlab, mode=self.mode, logdir=self.output / "swanlog",
                                      project=self.project, name=name, group=self.group, config=config,
                                      on_online_error="offline")
        except Exception as error:
            self._failure("initialization", error)
        if not starting_step:
            self.evaluation(0, baseline)

    def training_step(self, step, loss, elapsed, grad_norm, window_size, learning_rates):
        metrics = progress_metrics(step, loss, elapsed, self.ranks, self.history, self.total_steps, self.batch_size)
        metrics.update({"train/gradient_norm": grad_norm, "train/window_size": window_size})
        metrics.update({f"lr/{name}": value for name, value in learning_rates.items()})
        self._log(metrics, step)

    def evaluation(self, step, report, *, score=None, selected=None, final=False):
        metrics = development_metrics(report)
        if score is not None:
            metrics["dev/joint/selection_score"] = score
        if selected is not None:
            metrics["dev/joint/checkpoint_selected"] = int(selected)
        if final:
            metrics = {"final/" + key.removeprefix("dev/"): value for key, value in metrics.items()}
        self._log(metrics, step)
