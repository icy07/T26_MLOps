"""Серия запусков эксперимента ЛР4: фактор меняется только через params.yaml.

    python -m housing_price.run_experiment --seeds 42 7 13 21 99 \
        --models median_baseline linear_regression random_forest

Для каждой пары seed x model скрипт записывает ``train.model`` и ``train.seed`` в params.yaml
и вызывает ``housing_price.train --mlflow``. Код обучения между запусками не меняется.
В конце, в том числе при ошибке, params.yaml возвращается в исходное состояние байт в байт.
"""

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

DEFAULT_PARAMS = Path("params.yaml")

Runner = Callable[[list[str]], int]


class ParamsError(ValueError):
    """В params.yaml нет секции train или ключей model/seed."""


def _train_section(text: str) -> tuple[int, int]:
    """Границы секции ``train:`` верхнего уровня: от строки после заголовка до следующей секции."""
    header = re.search(r"(?m)^train:[ \t]*(?:#.*)?\r?$", text)
    if header is None:
        raise ParamsError("в params.yaml нет секции train")
    start = header.end()
    following = re.search(r"(?m)^[^\s#]", text[start:])
    end = start + following.start() if following else len(text)
    return start, end


def set_train_params(text: str, model: str, seed: int) -> str:
    """Заменить значения ``model:`` и ``seed:`` в секции train, не трогая остальной текст.

    YAML не перезаписывается целиком: комментарии, порядок ключей и окончания строк
    остаются прежними, меняются только сами значения (комментарий после значения сохраняется).
    """
    start, end = _train_section(text)
    section = text[start:end]
    for key, value in (("model", model), ("seed", str(seed))):
        pattern = rf"(?m)^([ \t]+{key}:[ \t]*)[^\s#]+"
        section, count = re.subn(pattern, lambda match, v=value: match.group(1) + v, section)
        if count != 1:
            raise ParamsError(f"в секции train ключ {key}: найден {count} раз(а), нужен ровно один")
    return text[:start] + section + text[end:]


def run_series(
    params_path: Path,
    seeds: Sequence[int],
    models: Sequence[str],
    runner: Runner | None = None,
) -> int:
    """Выполнить все пары seed x model и вернуть params.yaml в исходное состояние.

    Возвращает 0, если все запуски успешны, иначе код ошибки первого неуспешного запуска.
    """
    if runner is None:
        from housing_price import train  # MLflow подключается внутри train только при --mlflow

        runner = train.main

    original = params_path.read_bytes()  # байты, чтобы вернуть файл без изменений
    text = original.decode("utf-8")
    pairs = [(seed, model) for seed in seeds for model in models]
    try:
        for number, (seed, model) in enumerate(pairs, start=1):
            updated = set_train_params(text, model, seed)
            params_path.write_bytes(updated.encode("utf-8"))  # без перевода строк ОС
            print(f"[{number}/{len(pairs)}] {model}-seed{seed}", flush=True)
            code = runner(["--params", str(params_path), "--mlflow"])
            if code != 0:
                print(f"запуск {model}-seed{seed} завершился с кодом {code}", file=sys.stderr)
                return code
    finally:
        params_path.write_bytes(original)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Серия запусков seed x model для MLflow")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="значения train.seed")
    parser.add_argument("--models", nargs="+", required=True, help="значения train.model")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS, help="путь к params.yaml")
    args = parser.parse_args(argv)

    from housing_price.models import MODEL_KINDS

    unknown = [name for name in args.models if name not in MODEL_KINDS]
    if unknown:
        print(
            f"неизвестные модели: {', '.join(unknown)}; допустимые: {', '.join(MODEL_KINDS)}",
            file=sys.stderr,
        )
        return 2
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        print("значения --seeds и --models не должны повторяться", file=sys.stderr)
        return 2

    if not args.params.is_file():
        print(f"нет файла параметров: {args.params}", file=sys.stderr)
        return 2
    try:
        code = run_series(args.params, args.seeds, args.models)
    except ParamsError as exc:
        print(f"ошибка params.yaml: {exc}", file=sys.stderr)
        return 2
    if code == 0:
        print(f"выполнено запусков: {len(args.seeds) * len(args.models)}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
