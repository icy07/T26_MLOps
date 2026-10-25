"""Тесты ЛР4: серия запусков (run_experiment) и сравнение запусков (compare_runs)."""

import math
from pathlib import Path

import pandas as pd
import pytest

from housing_price import compare_runs as cr
from housing_price import run_experiment as re_

PARAMS_TEXT = (
    "# общий комментарий\n"
    "split:\n"
    "  random_state: 42\n"
    "  seed: 1  # не трогать: другая секция\n"
    "train:\n"
    "  # какая модель обучается\n"
    "  model: median_baseline  # кандидат\n"
    "  seed: 42\n"
    "  params: {}\n"
    "evaluate:\n"
    "  seed: 5\n"
)


def write_params(path: Path, text: str = PARAMS_TEXT, eol: str = "\n") -> bytes:
    data = text.replace("\n", eol).encode("utf-8")
    path.write_bytes(data)
    return data


# ---------- run_experiment ----------


def test_set_train_params_changes_only_train_values():
    result = re_.set_train_params(PARAMS_TEXT, "random_forest", 7)

    assert "  model: random_forest  # кандидат\n" in result
    assert "  seed: 7\n" in result
    assert "  seed: 1  # не трогать: другая секция\n" in result  # секция split
    assert "  seed: 5\n" in result  # секция evaluate
    assert result.count("\n") == PARAMS_TEXT.count("\n")


def test_set_train_params_keeps_crlf():
    text = PARAMS_TEXT.replace("\n", "\r\n")

    result = re_.set_train_params(text, "linear_regression", 13)

    assert "  model: linear_regression  # кандидат\r\n" in result
    assert "\r\n" in result and "\n" not in result.replace("\r\n", "")


@pytest.mark.parametrize(
    "text",
    [
        "split:\n  seed: 1\n",  # нет секции train
        "train:\n  model: median_baseline\n",  # нет seed
        "train:\n  seed: 1\n",  # нет model
        "train:\n  model: a\n  model: b\n  seed: 1\n",  # model дважды
    ],
)
def test_set_train_params_rejects_unexpected_layout(text):
    with pytest.raises(re_.ParamsError):
        re_.set_train_params(text, "random_forest", 7)


def test_run_series_runs_every_pair_and_restores_params(tmp_path):
    params = tmp_path / "params.yaml"
    original = write_params(params)
    seen: list[tuple[str, str]] = []

    def runner(argv):
        assert argv == ["--params", str(params), "--mlflow"]
        text = params.read_text(encoding="utf-8")
        model = next(line for line in text.splitlines() if "model:" in line)
        seed = text.split("train:")[1].split("seed:")[1].split()[0]
        seen.append((model.split(":")[1].split()[0], seed))
        return 0

    code = re_.run_series(params, [42, 7], ["median_baseline", "random_forest"], runner)

    assert code == 0
    assert seen == [
        ("median_baseline", "42"),
        ("random_forest", "42"),
        ("median_baseline", "7"),
        ("random_forest", "7"),
    ]
    assert params.read_bytes() == original


def test_run_series_restores_crlf_params_byte_for_byte(tmp_path):
    params = tmp_path / "params.yaml"
    original = write_params(params, eol="\r\n")

    re_.run_series(params, [1, 2], ["random_forest"], lambda argv: 0)

    assert params.read_bytes() == original


def test_run_series_stops_on_failure_and_still_restores(tmp_path, capsys):
    params = tmp_path / "params.yaml"
    original = write_params(params)
    calls = []

    def runner(argv):
        calls.append(argv)
        return 3 if len(calls) == 2 else 0

    code = re_.run_series(params, [1, 2, 3], ["median_baseline"], runner)

    assert code == 3
    assert len(calls) == 2
    assert params.read_bytes() == original
    assert "завершился с кодом 3" in capsys.readouterr().err


def test_run_series_restores_params_when_runner_raises(tmp_path):
    params = tmp_path / "params.yaml"
    original = write_params(params)

    def runner(argv):
        raise RuntimeError("сбой обучения")

    with pytest.raises(RuntimeError):
        re_.run_series(params, [1], ["median_baseline"], runner)

    assert params.read_bytes() == original


def test_main_rejects_unknown_model(tmp_path, capsys):
    params = tmp_path / "params.yaml"
    write_params(params)

    code = re_.main(["--seeds", "1", "--models", "no_such_model", "--params", str(params)])

    assert code == 2
    assert "неизвестные модели" in capsys.readouterr().err


def test_main_rejects_duplicate_seeds(tmp_path, capsys):
    params = tmp_path / "params.yaml"
    write_params(params)

    code = re_.main(
        ["--seeds", "1", "1", "--models", "median_baseline", "--params", str(params)]
    )

    assert code == 2
    assert "не должны повторяться" in capsys.readouterr().err


def test_main_reports_missing_params_file(tmp_path, capsys):
    code = re_.main(
        ["--seeds", "1", "--models", "median_baseline", "--params", str(tmp_path / "nope.yaml")]
    )

    assert code == 2
    assert "нет файла параметров" in capsys.readouterr().err


# ---------- compare_runs ----------

SHA = "a" * 64


def make_runs(values: dict[str, list[float]], baseline: float = 100.0, sha: str = SHA):
    """Таблица запусков в формате mlflow.search_runs: по одному запуску на значение MAE."""
    rows = []
    for model, maes in values.items():
        for seed, mae in enumerate(maes):
            rows.append(
                {
                    cr.MODEL: model,
                    cr.SEED: str(seed),
                    cr.DATA_SHA: sha,
                    cr.MAE: mae,
                    cr.BASELINE_MAE: baseline,
                    cr.SKILL: 1 - mae / baseline,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_uses_sample_std_and_sorts_by_mean_mae():
    runs = make_runs({"median_baseline": [100, 100, 100], "random_forest": [60, 70, 80]})

    summary = cr.summarize(runs)

    assert list(summary.index) == ["random_forest", "median_baseline"]
    assert summary.loc["random_forest", "n"] == 3
    assert summary.loc["random_forest", "mae_mean"] == pytest.approx(70.0)
    assert summary.loc["random_forest", "mae_std"] == pytest.approx(10.0)  # ddof=1
    assert summary.loc["median_baseline", "mae_std"] == pytest.approx(0.0)


def test_difference_larger_than_two_sigma_is_significant():
    runs = make_runs({"median_baseline": [100, 100, 100], "random_forest": [60, 70, 80]})
    summary = cr.summarize(runs)

    (item,) = cr.pairwise(summary, "median_baseline")

    assert item.diff == pytest.approx(30.0)
    assert item.threshold == pytest.approx(20.0)  # 2 x max(10, 0)
    assert item.significant is True
    assert "random_forest лучше" in item.verdict


def test_difference_smaller_than_two_sigma_is_not_significant():
    runs = make_runs({"median_baseline": [100, 110, 90], "random_forest": [95, 105, 85]})
    summary = cr.summarize(runs)

    (item,) = cr.pairwise(summary, "median_baseline")

    assert item.diff == pytest.approx(5.0)
    assert item.threshold == pytest.approx(20.0)
    assert item.significant is False
    assert item.verdict == "не значимо"


def test_difference_exactly_two_sigma_is_not_significant():
    runs = make_runs({"median_baseline": [100, 100], "random_forest": [90, 110]})
    summary = cr.summarize(runs)
    summary.loc["random_forest", "mae_mean"] = 100 - 2 * summary.loc["random_forest", "mae_std"]

    item = cr.compare(summary, "random_forest", "median_baseline", "тест")

    assert item.diff == pytest.approx(item.threshold)
    assert item.significant is False  # строго больше 2σ


def test_single_run_per_model_cannot_be_judged():
    runs = make_runs({"median_baseline": [100.0], "random_forest": [50.0]})
    summary = cr.summarize(runs)

    (item,) = cr.pairwise(summary, "median_baseline")

    assert math.isnan(item.threshold)
    assert item.significant is None
    assert "нельзя оценить" in item.verdict


def test_pairwise_compares_best_with_runner_up_too():
    runs = make_runs(
        {
            "median_baseline": [100, 100, 100],
            "linear_regression": [70, 72, 74],
            "random_forest": [50, 52, 54],
        }
    )
    summary = cr.summarize(runs)

    items = cr.pairwise(summary, "median_baseline")

    pairs = {(item.better, item.other): item for item in items}
    assert set(pairs) == {
        ("random_forest", "median_baseline"),
        ("linear_regression", "median_baseline"),
        ("random_forest", "linear_regression"),
    }
    assert pairs[("random_forest", "linear_regression")].significant is True


def test_pairwise_when_runner_up_is_baseline_does_not_duplicate():
    runs = make_runs({"median_baseline": [100, 101, 99], "random_forest": [60, 61, 59]})

    items = cr.pairwise(cr.summarize(runs), "median_baseline")

    assert len(items) == 1
    assert "лучший против следующего" in items[0].kind


def test_pairwise_requires_baseline_runs():
    runs = make_runs({"random_forest": [60, 61, 59]})

    with pytest.raises(cr.ComparisonError, match="точки отсчета"):
        cr.pairwise(cr.summarize(runs), "median_baseline")


def test_check_runs_returns_single_data_version():
    runs = make_runs({"median_baseline": [100, 100]})

    assert cr.check_runs(runs) == SHA


def test_check_runs_refuses_different_data_versions():
    runs = pd.concat(
        [make_runs({"median_baseline": [100]}), make_runs({"random_forest": [60]}, sha="b" * 64)],
        ignore_index=True,
    )

    with pytest.raises(cr.DataVersionError, match="разных данных"):
        cr.check_runs(runs)


def test_check_runs_refuses_missing_data_version_tag():
    runs = make_runs({"median_baseline": [100, 100]})
    runs.loc[0, cr.DATA_SHA] = None

    with pytest.raises(cr.DataVersionError, match="data_sha256"):
        cr.check_runs(runs)


def test_check_runs_rejects_empty_and_incomplete_tables():
    with pytest.raises(cr.ComparisonError, match="нет завершенных"):
        cr.check_runs(pd.DataFrame())
    with pytest.raises(cr.ComparisonError, match="нет столбцов"):
        cr.check_runs(make_runs({"median_baseline": [100]}).drop(columns=[cr.SKILL]))


def test_check_runs_rejects_missing_metric_values():
    runs = make_runs({"median_baseline": [100, 100]})
    runs.loc[1, cr.MAE] = float("nan")

    with pytest.raises(cr.ComparisonError, match="нет метрик"):
        cr.check_runs(runs)


def test_render_tables_are_markdown():
    runs = make_runs({"median_baseline": [100, 100, 100], "random_forest": [60, 70, 80]})
    summary = cr.summarize(runs)

    table = cr.render_summary(summary)
    pairs = cr.render_pairwise(cr.pairwise(summary, "median_baseline"))

    assert table.splitlines()[0].startswith("| Вариант |")
    assert "| random_forest | 3 | 70.0 ± 10.0 |" in table
    assert "значимо: random_forest лучше" in pairs
    assert "+30.0" in pairs


def test_main_end_to_end_with_real_mlflow(tmp_path, capsys):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("exp")
    for model, maes in {"median_baseline": [100.0, 100.0], "random_forest": [60.0, 62.0]}.items():
        for seed, mae in enumerate(maes):
            with mlflow.start_run(run_name=f"{model}-seed{seed}"):
                mlflow.set_tags({"model": model, "seed": str(seed), "data_sha256": SHA})
                mlflow.log_metrics({"mae": mae, "baseline_mae": 100.0, "mae_skill": 1 - mae / 100})

    code = cr.main(["--experiment", "exp", "--tracking-uri", uri])

    out = capsys.readouterr().out
    assert code == 0
    assert "### Сводка по вариантам" in out
    assert "значимо: random_forest лучше" in out


def test_main_returns_1_for_mixed_data_versions(tmp_path, capsys):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("exp")
    for model, sha in (("median_baseline", SHA), ("random_forest", "b" * 64)):
        with mlflow.start_run():
            mlflow.set_tags({"model": model, "seed": "0", "data_sha256": sha})
            mlflow.log_metrics({"mae": 1.0, "baseline_mae": 1.0, "mae_skill": 0.0})

    code = cr.main(["--experiment", "exp", "--tracking-uri", uri])

    assert code == 1
    assert "разных данных" in capsys.readouterr().err


def test_main_returns_2_for_empty_experiment(tmp_path, capsys):
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    code = cr.main(["--experiment", "nothing", "--tracking-uri", uri])

    assert code == 2
    assert "ошибка" in capsys.readouterr().err


def test_main_reports_when_only_baseline_exists(tmp_path, capsys):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("exp")
    for seed in range(2):
        with mlflow.start_run():
            mlflow.set_tags({"model": "median_baseline", "seed": str(seed), "data_sha256": SHA})
            mlflow.log_metrics({"mae": 100.0 + seed, "baseline_mae": 100.0, "mae_skill": 0.0})

    code = cr.main(["--experiment", "exp", "--tracking-uri", uri])

    assert code == 0
    assert "сравнивать не с чем" in capsys.readouterr().out
