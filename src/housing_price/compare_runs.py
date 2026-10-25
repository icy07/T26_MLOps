"""Сравнение запусков эксперимента в MLflow с учетом разброса (ЛР4).

    python -m housing_price.compare_runs --experiment housing-price

Берет завершенные запуски эксперимента через ``mlflow.search_runs``, отказывается сравнивать
запуски на разных версиях данных (``data_sha256``), считает по каждому варианту модели среднее
и стандартное отклонение MAE и сравнивает варианты попарно по критерию 2σ: разница средних
MAE значима, только если она больше ``2 x max(σ1, σ2)``. Результат печатается Markdown-таблицами.

Коды завершения: 0 - сравнение выполнено; 1 - запуски на разных версиях данных;
2 - нет запусков, нет нужных столбцов или эксперимент не читается.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass

import pandas as pd

DEFAULT_EXPERIMENT = "housing-price"
DEFAULT_BASELINE = "median_baseline"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
SIGMA_FACTOR = 2.0

MODEL = "tags.model"
SEED = "tags.seed"
DATA_SHA = "tags.data_sha256"
MAE = "metrics.mae"
BASELINE_MAE = "metrics.baseline_mae"
SKILL = "metrics.mae_skill"
REQUIRED_COLUMNS = (MODEL, SEED, DATA_SHA, MAE, BASELINE_MAE, SKILL)


class ComparisonError(ValueError):
    """Запуски нельзя сравнивать: нет данных, нужных столбцов или эталона."""


class DataVersionError(ComparisonError):
    """У запусков больше одной версии данных: сравнение на разных данных недопустимо."""


@dataclass(frozen=True)
class Comparison:
    """Сравнение двух вариантов: ``better`` - кандидат (A), ``other`` - с чем сравниваем (B)."""

    kind: str
    better: str
    other: str
    diff: float  # MAE(B) - MAE(A): положительная разница означает, что A лучше
    threshold: float  # 2 x max(sigma_A, sigma_B); NaN, если нет оценки разброса (n < 2)
    significant: bool | None  # None - оценить нельзя

    @property
    def verdict(self) -> str:
        if self.significant is None:
            return "нельзя оценить (n < 2)"
        if not self.significant:
            return "не значимо"
        return f"значимо: {self.better} лучше" if self.diff > 0 else f"значимо: {self.better} хуже"


def load_runs(experiment: str, tracking_uri: str) -> pd.DataFrame:
    """Завершенные запуски эксперимента (MLflow подключается только здесь)."""
    import mlflow
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri)
    try:
        return mlflow.search_runs(
            experiment_names=[experiment],
            filter_string="attributes.status = 'FINISHED'",
            output_format="pandas",
        )
    except MlflowException as exc:
        raise ComparisonError(f"не удалось прочитать эксперимент {experiment!r}: {exc}") from exc


def check_runs(runs: pd.DataFrame) -> str:
    """Проверить запуски и вернуть единую версию данных; при расхождении - DataVersionError."""
    if runs.empty:
        raise ComparisonError("нет завершенных запусков в эксперименте")
    missing = [column for column in REQUIRED_COLUMNS if column not in runs.columns]
    if missing:
        raise ComparisonError(f"у запусков нет столбцов: {', '.join(missing)}")

    versions = runs[DATA_SHA]
    if versions.isna().any():
        count = int(versions.isna().sum())
        raise DataVersionError(
            f"у {count} запусков нет тега data_sha256, версию данных не доказать"
        )
    distinct = sorted(versions.unique())
    if len(distinct) > 1:
        raise DataVersionError(
            f"запуски выполнены на {len(distinct)} версиях данных (data_sha256): "
            + ", ".join(value[:12] for value in distinct)
            + "; сравнение на разных данных недопустимо"
        )

    metrics = runs[[MAE, BASELINE_MAE, SKILL]]
    if metrics.isna().any().any():
        raise ComparisonError("у части запусков нет метрик mae, baseline_mae или mae_skill")
    return str(distinct[0])


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """По каждому варианту: число запусков, среднее и σ (выборочное, ddof=1) MAE, точка отсчета."""
    summary = runs.groupby(MODEL).agg(
        n=(MAE, "size"),
        mae_mean=(MAE, "mean"),
        mae_std=(MAE, "std"),
        baseline_mae_mean=(BASELINE_MAE, "mean"),
        mae_skill_mean=(SKILL, "mean"),
    )
    summary.index.name = "model"
    return summary.sort_values("mae_mean")


def compare(summary: pd.DataFrame, better: str, other: str, kind: str) -> Comparison:
    """Критерий 2σ: разница средних MAE значима, если она больше 2 x max(σ_A, σ_B)."""
    diff = float(summary.loc[other, "mae_mean"] - summary.loc[better, "mae_mean"])
    sigmas = [float(summary.loc[name, "mae_std"]) for name in (better, other)]
    if any(math.isnan(sigma) for sigma in sigmas):
        return Comparison(kind, better, other, diff, math.nan, None)
    threshold = SIGMA_FACTOR * max(sigmas)
    return Comparison(kind, better, other, diff, threshold, abs(diff) > threshold)


def pairwise(summary: pd.DataFrame, baseline: str) -> list[Comparison]:
    """Каждый вариант против baseline и лучший вариант против следующего за ним."""
    if baseline not in summary.index:
        raise ComparisonError(f"в эксперименте нет запусков точки отсчета {baseline!r}")

    result = [
        compare(summary, name, baseline, "вариант против baseline")
        for name in summary.index
        if name != baseline
    ]
    if len(summary) >= 2:
        best, following = summary.index[0], summary.index[1]
        for index, item in enumerate(result):
            if (item.better, item.other) == (best, following):
                result[index] = Comparison(
                    f"{item.kind}; лучший против следующего",
                    item.better, item.other, item.diff, item.threshold, item.significant,
                )
                break
        else:
            result.append(compare(summary, best, following, "лучший против следующего"))
    return result


def _plus_minus(mean: float, std: float) -> str:
    return f"{mean:.1f} ± н/д" if math.isnan(std) else f"{mean:.1f} ± {std:.1f}"


def render_summary(summary: pd.DataFrame) -> str:
    lines = [
        "| Вариант | n | MAE среднее ± σ | Точка отсчета (baseline MAE, среднее) | mae_skill |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in summary.iterrows():
        lines.append(
            f"| {name} | {int(row['n'])} | {_plus_minus(row['mae_mean'], row['mae_std'])} "
            f"| {row['baseline_mae_mean']:.1f} | {row['mae_skill_mean']:.4f} |"
        )
    return "\n".join(lines)


def render_pairwise(comparisons: list[Comparison]) -> str:
    lines = [
        "| Сравнение | Разница средних MAE | Порог 2σ | Вывод |",
        "|---|---:|---:|---|",
    ]
    for item in comparisons:
        threshold = "н/д" if math.isnan(item.threshold) else f"{item.threshold:.1f}"
        lines.append(
            f"| {item.better} против {item.other} ({item.kind}) "
            f"| {item.diff:+.1f} | {threshold} | {item.verdict} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнение запусков эксперимента в MLflow")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="имя эксперимента")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="вариант - точка отсчета")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="адрес MLflow (по умолчанию MLFLOW_TRACKING_URI или sqlite:///mlflow.db)",
    )
    args = parser.parse_args(argv)

    try:
        runs = load_runs(args.experiment, args.tracking_uri)
        data_sha256 = check_runs(runs)
        summary = summarize(runs)
        comparisons = pairwise(summary, args.baseline)
    except DataVersionError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1
    except ComparisonError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2

    unique_pairs = len(runs[[MODEL, SEED]].drop_duplicates())
    print(
        f"Эксперимент `{args.experiment}`: запусков {len(runs)}, "
        f"версия данных `data_sha256` {data_sha256}"
    )
    if unique_pairs != len(runs):
        print(
            f"\nПредупреждение: уникальных пар model × seed {unique_pairs} "
            f"из {len(runs)} запусков, есть повторы - они входят в среднее и σ.",
            file=sys.stderr,
        )
    print("\n### Сводка по вариантам\n")
    print(render_summary(summary))
    print("\n### Попарные сравнения (критерий 2σ)\n")
    if comparisons:
        print(render_pairwise(comparisons))
    else:
        print(f"В эксперименте только `{args.baseline}`: сравнивать не с чем.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
