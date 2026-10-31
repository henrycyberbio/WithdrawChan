"""Command line: `lab fetch`, `lab snapshot`, `lab run`, `lab train-logit`, `lab summarize`,
`lab compare`, `lab figures`."""

from __future__ import annotations

import json
from pathlib import Path

import fire

from lab import chain
from lab.report.compare import compare as compare_experiments
from lab.report.figures import figures as draw_figures
from lab.report.summary import summarize as summarize_experiment
from lab.run import run as run_experiment
from lab.schema import Manifest, PinError, canonical, sha256
from lab.train import train as train_experiment


def _manifest(path: str) -> Manifest:
    return Manifest.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _dir(manifest: str, data: str) -> Path:
    return Path(data) / Path(manifest).stem


class Lab:
    """Reproducible experiments on withdrawal-amount privacy."""

    def fetch(self, manifest: str, *, rpc: str, data: str) -> str:
        """Download the manifest's raw logs from the JSON-RPC endpoint `rpc` into `data`.

        Args:
            manifest: The manifest JSON file to fetch the range of.
            rpc: The JSON-RPC endpoint URL the logs are read from.
            data: The data directory; the logs land in `<data>/<manifest stem>/logs.json`.

        Returns:
            The number of logs fetched and the output path.
        """
        out = _dir(manifest, data) / "logs.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        logs = chain.fetch(rpc, _manifest(manifest))
        out.write_text(json.dumps(logs), encoding="utf-8")
        return f"{len(logs)} logs -> {out}"

    def snapshot(self, manifest: str, *, data: str) -> str:
        """Decode the raw logs into snapshot.json and check its sha256 against the manifest.

        A manifest without a hash gets the computed one written into it.

        Args:
            manifest: The manifest whose pinned snapshot is decoded and checked.
            data: The data directory holding `<manifest stem>/logs.json`.

        Returns:
            The snapshot sha256, with whether it was written to or matched the manifest.
        """
        pinned = _manifest(manifest)
        folder = _dir(manifest, data)
        snap = chain.build(pinned, json.loads((folder / "logs.json").read_text(encoding="utf-8")))
        (folder / "snapshot.json").write_bytes(canonical(snap))
        digest = sha256(snap)
        if pinned.snapshot_sha256 is None:
            raw = json.loads(Path(manifest).read_text(encoding="utf-8"))
            Path(manifest).write_text(json.dumps({**raw, "snapshot_sha256": digest}, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
            return f"{digest} written to {manifest}"
        if digest != pinned.snapshot_sha256:
            raise PinError("snapshot_sha256", digest, pinned.snapshot_sha256)
        return f"{digest} matches {manifest}"

    def run(self, config: str, *, data: str, out: str, workers: int = 1,
            logit_beta: str | None = None) -> dict:
        """Run an experiment config into `out`; `--workers` parallelizes paths.

        `--logit-beta` is the path to the `train-logit` JSON artifact: its `beta` holds the
        coefficients and its `whole_unit` and `logit_units` pin the rounding unit and tail axis they
        were fitted on.  Each path's amounts use the config's `p`, the probability of adding the
        random decimal; a path already holding a `path.json` is skipped, so a rerun resumes.

        Args:
            config: The experiment config JSON file.
            data: The data directory holding `<manifest stem>/snapshot.json`.
            out: The output directory for the per-path results and `run.json`.
            workers: The number of spawned worker processes; the results do not depend on it.
            logit_beta: Path to the `train-logit` JSON artifact (its `beta`, `whole_unit`,
                `logit_units`), or None.

        Returns:
            The output directory, the path and written counts and the participant count.
        """
        return run_experiment(config, data=data, out=out, workers=workers, logit_beta_path=logit_beta)

    def train_logit(self, config: str, *, data: str, out: str) -> dict:
        """Fit the trained-logit observer on the config's simulated paths into `<out>/logit_beta.json`.

        The fit is taken at `p = 0`, the probability of adding the random decimal, whatever the
        config's `p` is.

        Args:
            config: The experiment config JSON file.
            data: The data directory holding `<manifest stem>/snapshot.json`.
            out: The output directory for `logit_beta.json`.

        Returns:
            The output directory and the fitted path and group counts.
        """
        return train_experiment(config, data=data, out=out)

    def summarize(self, out: str, *, best_observers: list[str] | tuple[str, ...] | None = None
                  ) -> dict:
        """Write an experiment output's summary tables (groups, best, paired, cost) into `<out>/summary`.

        `--best-observers` names the candidates of a combined `best.csv` envelope, as a JSON list
        literal such as `--best-observers='["historical", "trained-logit", "combinatorial",
        "group"]'`; without it `best.csv` keeps the strongest public observer's envelope.

        Args:
            out: The experiment output holding the path directories to summarize.
            best_observers: The candidate observers of the combined `best` table, or None for
                the default strongest-public envelope.

        Returns:
            The summary directory and the row counts of groups, best and paired.
        """
        return summarize_experiment(out, best_observers=best_observers)

    def compare(self, base: str, alt: str, *, dest: str) -> dict:
        """Compare two runs on a shared grid and seed root, writing `compare.csv` and `compare.json`.

        `dest` is the directory that receives both files.

        Args:
            base: The baseline experiment output.
            alt: The varied output; it must share `base`'s grid, seed root, replicate count, `N`
                and snapshot, differing only in the swept config values.
            dest: The directory that receives `compare.csv` and `compare.json`.

        Returns:
            The destination directory, the `compare.csv` row count and the changed config keys.
        """
        return compare_experiments(base, alt, dest=dest)

    def figures(self, out: str, *, rule: str = "decsmooth", m_model: str = "double",
                comparison_rho: str = "1/2", comparison_observer: str = "combinatorial",
                figure1_observer: str | None = None) -> dict:
        """Draw the figures of a summarized experiment output into `<out>/figures` as PNG.

        `--rule` and `--m-model` set figure 1; `--m-model` also sets figure 3's maximum model.
        `--comparison-rho` and `--comparison-observer` select figures 2 and 3;
        `--comparison-observer best` reads the composite paired rows for figure 2 and the combined
        `best.csv` cells for figure 3, where each cell may have a different winning observer.
        `--figure1-observer` reads figure 1 from that observer's `groups.csv` rows; without it
        figure 1 plots the strongest-observer envelope of `best.csv`.

        Args:
            out: The summarized experiment output.
            rule: The tool rule of figure 1.
            m_model: The maximum model of figures 1 and 3.
            comparison_rho: The adoption ratio of figures 2 and 3.
            comparison_observer: The observer of figures 2 and 3, or `best` for the composite
                statistic.
            figure1_observer: The observer of figure 1, or `None` for the `best.csv` envelope.

        Returns:
            The written PNG paths keyed by figure number (`fig1`, `fig2`, `fig3`).
        """
        return draw_figures(out, rule=rule, m_model=m_model, comparison_rho=comparison_rho,
                            comparison_observer=comparison_observer,
                            figure1_observer=figure1_observer)


def main() -> None:
    fire.Fire(Lab)
